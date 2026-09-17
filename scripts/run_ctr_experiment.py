#!/usr/bin/env python3
"""First ranking-stage experiment: impression-level CTR classification.

Implements the "Recommended First Ranking Experiment" from
docs/ranking_problem_design.md: P(click | impression), trained and evaluated
only on real, already-served impressions -- see docs/ctr_model.md for the
full write-up.

Pipeline:
    1. Load the Attribution dataset, split chronologically (train/val/test).
    2. Build point-in-time CTR features on the FULL dataset (train-period-
       only for frozen campaign aggregates; full cross-split history for
       user/pair features -- see src/features/attribution_features.py).
    3. Split the feature-augmented dataset using the same day boundaries.
    4. Check for missing values in the raw contextual fields (cat1-9).
    5. Fit the global-CTR baseline on train, evaluate on test.
    6. Fit LightGBM on train (early-stopped on validation), evaluate on test.
    7. Report both side by side: PR-AUC, ROC-AUC, LogLoss, calibration,
       feature importance.

Usage:
    python scripts/run_ctr_experiment.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.attribution import load, split_by_day  # noqa: E402
from src.evaluation.classification_metrics import (  # noqa: E402
    compute_auc,
    compute_calibration,
    compute_logloss,
    compute_pr_auc,
)
from src.features.attribution_features import (  # noqa: E402
    CATEGORICAL_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    assert_no_post_outcome_leakage,
    build_ctr_features,
)
from src.models.ctr_model import CTRModel, GlobalCTRBaseline  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def main() -> None:
    config = load_config()
    attr_cfg = config["attribution"]
    ctr_cfg = attr_cfg["ranking"]["ctr_model"]

    assert_no_post_outcome_leakage(FEATURE_COLUMNS)
    print(f"Feature columns ({len(FEATURE_COLUMNS)}): {FEATURE_COLUMNS}")
    print("Verified: no post-outcome columns among them.\n")

    print("Loading Attribution dataset...")
    df = load(config)
    print(f"Loaded {len(df):,} rows.")

    train_df, val_df, test_df = split_by_day(
        df, attr_cfg["split"]["train_end_day"], attr_cfg["split"]["val_end_day"]
    )
    print(f"Split -> train: {len(train_df):,} | val: {len(val_df):,} | test: {len(test_df):,} rows\n")

    contextual_cols = [f"cat{i}" for i in range(1, 10)]
    n_missing = df[contextual_cols].isna().sum().sum()
    print(f"Missing values in cat1..cat9 (checked directly, not assumed): {n_missing}")
    if n_missing == 0:
        print("Confirmed: no missing-value handling needed for cat1..cat9.\n")
    else:
        print("WARNING: missing values found in contextual features -- see docs/ctr_model.md.\n")

    print("Building point-in-time CTR features on the full dataset...")
    features = build_ctr_features(df, train_df)
    train_f, val_f, test_f = split_by_day(
        features, attr_cfg["split"]["train_end_day"], attr_cfg["split"]["val_end_day"]
    )
    assert list(train_f.columns) == list(val_f.columns) == list(test_f.columns)
    print("Train/val/test feature schemas confirmed identical.\n")

    X_train, y_train = train_f[FEATURE_COLUMNS], train_f[TARGET_COLUMN]
    X_val, y_val = val_f[FEATURE_COLUMNS], val_f[TARGET_COLUMN]
    X_test, y_test = test_f[FEATURE_COLUMNS], test_f[TARGET_COLUMN]

    print("=" * 80)
    print("BASELINE: global training-period CTR")
    print("=" * 80)
    baseline = GlobalCTRBaseline().fit(X_train, y_train)
    print(f"Global training CTR: {baseline.global_ctr_:.4f}")
    baseline_preds = baseline.predict(X_test)
    baseline_metrics = {
        "pr_auc": compute_pr_auc(y_test, baseline_preds),
        "roc_auc": compute_auc(y_test, baseline_preds),
        "logloss": compute_logloss(y_test, baseline_preds),
    }
    print(f"Baseline is a constant prediction -- it induces no ordering at all, so ROC-AUC/PR-AUC")
    print(f"are expected to sit at the uninformative floor (~0.5 / ~test-period CTR respectively).")
    print(f"Reported here only as the floor LightGBM must beat by actually using features.")
    for k, v in baseline_metrics.items():
        print(f"  {k}: {v:.4f}")

    print()
    print("=" * 80)
    print(f"LIGHTGBM: {ctr_cfg['params']}")
    print("=" * 80)
    model = CTRModel(
        params=ctr_cfg["params"],
        categorical_features=ctr_cfg["categorical_features"],
        num_boost_round=ctr_cfg["training"]["num_boost_round"],
        early_stopping_rounds=ctr_cfg["training"]["early_stopping_rounds"],
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val))
    print(f"Best iteration (selected via validation set): {model.booster.best_iteration}")

    lgb_preds = model.predict(X_test)
    lgb_metrics = {
        "pr_auc": compute_pr_auc(y_test, lgb_preds),
        "roc_auc": compute_auc(y_test, lgb_preds),
        "logloss": compute_logloss(y_test, lgb_preds),
    }
    for k, v in lgb_metrics.items():
        print(f"  {k}: {v:.4f}")

    print()
    print("=" * 80)
    print("BASELINE vs. LIGHTGBM (test period, side by side)")
    print("=" * 80)
    print(f"{'metric':<10}{'baseline':>12}{'lightgbm':>12}{'abs_improvement':>18}{'rel_improvement':>18}")
    for metric in ("pr_auc", "roc_auc", "logloss"):
        b, l = baseline_metrics[metric], lgb_metrics[metric]
        abs_imp = l - b
        rel_imp = (abs_imp / b * 100.0) if b != 0 else float("nan")
        # For logloss, LOWER is better -- flip the sign so "improvement" reads consistently.
        if metric == "logloss":
            abs_imp, rel_imp = -abs_imp, -rel_imp
        print(f"{metric:<10}{b:>12.4f}{l:>12.4f}{abs_imp:>+18.4f}{rel_imp:>+17.2f}%")

    print()
    print("=" * 80)
    print("CALIBRATION (LightGBM, test period, quantile-binned)")
    print("=" * 80)
    calibration = compute_calibration(y_test, lgb_preds, n_bins=10)
    print(calibration.to_string(index=False))

    print()
    print("=" * 80)
    print("FEATURE IMPORTANCE (gain-based -- reflects what the trained model relies on,")
    print("NOT a causal effect estimate; cat1..cat9 semantics remain undisclosed)")
    print("=" * 80)
    importance = model.feature_importance(importance_type="gain")
    print(importance.to_string())


if __name__ == "__main__":
    main()
