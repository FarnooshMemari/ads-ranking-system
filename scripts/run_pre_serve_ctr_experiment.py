#!/usr/bin/env python3
"""Part A of the ranking-integration design: a pre-serve CTR model.

Retrains the LightGBM CTR classifier excluding `cat1`-`cat9` -- the feature
subset genuinely available BEFORE an impression is served (see
docs/pre_serve_ctr_model.md and docs/ranking_integration_design.md). Same
temporal protocol, same target (`P(click | impression)`), same
hyperparameters as the full-feature model (`scripts/run_ctr_experiment.py`)
-- only the feature/categorical list differs, so any performance delta is
attributable to the removed features, not a different configuration.

Saves the trained model to `configs/config.yaml`:
`attribution.ranking.ctr_model_pre_serve.model_output_path` for reuse by
Part B (`scripts/run_ranking_integration_experiment.py`).

Usage:
    python scripts/run_pre_serve_ctr_experiment.py
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
    IMPRESSION_CONTEXT_ONLY_COLUMNS,
    PRE_SERVE_FEATURE_COLUMNS,
    TARGET_COLUMN,
    assert_no_post_outcome_leakage,
    build_ctr_features,
)
from src.models.ctr_model import CTRModel, GlobalCTRBaseline  # noqa: E402
from src.utils.config import load_config, resolve_path  # noqa: E402


def main() -> None:
    config = load_config()
    attr_cfg = config["attribution"]
    ctr_cfg = attr_cfg["ranking"]["ctr_model_pre_serve"]

    assert_no_post_outcome_leakage(PRE_SERVE_FEATURE_COLUMNS)
    assert not set(PRE_SERVE_FEATURE_COLUMNS) & set(IMPRESSION_CONTEXT_ONLY_COLUMNS), (
        "cat1..cat9 must be absent from the pre-serve feature set."
    )
    print(f"Pre-serve feature columns ({len(PRE_SERVE_FEATURE_COLUMNS)}): {PRE_SERVE_FEATURE_COLUMNS}")
    print("Verified: no post-outcome columns and no cat1..cat9 among them.\n")

    print("Loading Attribution dataset...")
    df = load(config)
    print(f"Loaded {len(df):,} rows.")

    train_df, val_df, test_df = split_by_day(
        df, attr_cfg["split"]["train_end_day"], attr_cfg["split"]["val_end_day"]
    )
    print(f"Split -> train: {len(train_df):,} | val: {len(val_df):,} | test: {len(test_df):,} rows\n")

    print("Building point-in-time CTR features on the full dataset (same pipeline as the")
    print("full-feature model -- only the column selection below differs)...")
    features = build_ctr_features(df, train_df)
    train_f, val_f, test_f = split_by_day(
        features, attr_cfg["split"]["train_end_day"], attr_cfg["split"]["val_end_day"]
    )
    assert list(train_f.columns) == list(val_f.columns) == list(test_f.columns)
    print("Train/val/test feature schemas confirmed identical.\n")

    X_train, y_train = train_f[PRE_SERVE_FEATURE_COLUMNS], train_f[TARGET_COLUMN]
    X_val, y_val = val_f[PRE_SERVE_FEATURE_COLUMNS], val_f[TARGET_COLUMN]
    X_test, y_test = test_f[PRE_SERVE_FEATURE_COLUMNS], test_f[TARGET_COLUMN]

    print("=" * 80)
    print("BASELINE: global training-period CTR (same as the full-feature experiment)")
    print("=" * 80)
    baseline = GlobalCTRBaseline().fit(X_train, y_train)
    baseline_preds = baseline.predict(X_test)
    baseline_metrics = {
        "pr_auc": compute_pr_auc(y_test, baseline_preds),
        "roc_auc": compute_auc(y_test, baseline_preds),
        "logloss": compute_logloss(y_test, baseline_preds),
    }
    print(f"Global training CTR: {baseline.global_ctr_:.4f}")
    for k, v in baseline_metrics.items():
        print(f"  {k}: {v:.4f}")

    print()
    print("=" * 80)
    print(f"PRE-SERVE LIGHTGBM: {ctr_cfg['params']}")
    print("=" * 80)
    model = CTRModel(
        params=ctr_cfg["params"],
        categorical_features=ctr_cfg["categorical_features"],
        num_boost_round=ctr_cfg["training"]["num_boost_round"],
        early_stopping_rounds=ctr_cfg["training"]["early_stopping_rounds"],
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val))
    print(f"Best iteration (selected via validation set): {model.booster.best_iteration}")

    preds = model.predict(X_test)
    metrics = {
        "pr_auc": compute_pr_auc(y_test, preds),
        "roc_auc": compute_auc(y_test, preds),
        "logloss": compute_logloss(y_test, preds),
    }
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    print()
    print("=" * 80)
    print("PRE-SERVE MODEL vs. ITS OWN BASELINE (test period)")
    print("=" * 80)
    print(f"{'metric':<10}{'baseline':>12}{'pre_serve':>12}{'abs_improvement':>18}{'rel_improvement':>18}")
    for metric in ("pr_auc", "roc_auc", "logloss"):
        b, m = baseline_metrics[metric], metrics[metric]
        abs_imp, rel_imp = m - b, ((m - b) / b * 100.0) if b != 0 else float("nan")
        if metric == "logloss":
            abs_imp, rel_imp = -abs_imp, -rel_imp
        print(f"{metric:<10}{b:>12.4f}{m:>12.4f}{abs_imp:>+18.4f}{rel_imp:>+17.2f}%")

    print()
    print("=" * 80)
    print("CALIBRATION (pre-serve model, test period, quantile-binned)")
    print("=" * 80)
    calibration = compute_calibration(y_test, preds, n_bins=10)
    print(calibration.to_string(index=False))

    print()
    print("=" * 80)
    print("FEATURE IMPORTANCE (gain-based, pre-serve model)")
    print("=" * 80)
    print(model.feature_importance(importance_type="gain").to_string())

    output_path = resolve_path(ctr_cfg["model_output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))
    print(f"\nSaved pre-serve model to: {output_path}")


if __name__ == "__main__":
    main()
