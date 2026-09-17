#!/usr/bin/env python3
"""Warm-cohort collaborative filtering vs. popularity — lift experiment.

Tests whether personalized (item-based collaborative filtering) retrieval
provides measurable lift over the popularity baseline for the warm cohort
specifically. This is the direct empirical test the segmented architecture
in docs/attribution_modeling_design.md §3B/§4 was designed to enable — the
goal is to find out whether personalization helps, not to prove that it does.

Pipeline:
    1. Load the Attribution dataset, split chronologically (train/val/test).
    2. Classify users into cold/warm cohorts from train-period history only.
    3. Fit the popularity baseline on the full training set (identical
       procedure to scripts/run_popularity_baseline.py, so the comparison
       is apples-to-apples with the already-committed baseline numbers).
    4. Fit collaborative filtering ONLY on the warm cohort's own
       training-period click history (see docs/collaborative_filtering.md
       for why: the item-item similarity matrix should reflect what a warm
       user's own history could plausibly inform, not be diluted by the
       cold majority's single-touch interactions).
    5. For warm test users, generate top-K from both methods and evaluate
       Recall@K / Hit Rate@K / NDCG@K against test-period ground truth.
    6. Report absolute and relative lift, plus coverage diagnostics.

Usage:
    python scripts/run_collaborative_filtering_experiment.py [--k K] [--similarity jaccard|cooccurrence]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.attribution import load, split_by_day  # noqa: E402
from src.evaluation.retrieval_metrics import (  # noqa: E402
    hit_rate_at_k,
    ndcg_at_k,
    recall_at_k,
)
from src.retrieval.candidate_generator import (  # noqa: E402
    CollaborativeFilteringCandidateGenerator,
    PopularityCandidateGenerator,
)
from src.retrieval.cohort import COLD, WARM, classify_users, cohort_counts  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def _lift(cf_value: float, pop_value: float) -> tuple[float, float]:
    absolute = cf_value - pop_value
    relative = (absolute / pop_value * 100.0) if pop_value > 0 else float("nan")
    return absolute, relative


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=None, help="Override attribution.retrieval.top_k")
    parser.add_argument(
        "--similarity", choices=["jaccard", "cooccurrence"], default="jaccard",
        help="Item-item similarity metric for collaborative filtering.",
    )
    args = parser.parse_args()

    config = load_config()
    attr_cfg = config["attribution"]
    k = args.k if args.k is not None else attr_cfg["retrieval"]["top_k"]

    print("Loading Attribution dataset...")
    df = load(config)
    print(f"Loaded {len(df):,} rows.")

    train_df, val_df, test_df = split_by_day(
        df, attr_cfg["split"]["train_end_day"], attr_cfg["split"]["val_end_day"]
    )
    print(f"Split -> train: {len(train_df):,} | val: {len(val_df):,} | test: {len(test_df):,} rows")

    cohorts = classify_users(
        train_df, warm_min_distinct_campaigns=attr_cfg["cohort"]["warm_min_distinct_clicked_campaigns"]
    )
    counts = cohort_counts(cohorts)
    n_total = counts["cold"] + counts["warm"]
    print(f"Cohorts (train-period only): cold={counts['cold']:,} ({counts['cold']/n_total:.2%}), "
          f"warm={counts['warm']:,} ({counts['warm']/n_total:.2%})")

    warm_uids = set(cohorts[cohorts == WARM].index)

    # --- Popularity baseline: fit on the FULL training set (matches the
    # already-committed popularity baseline exactly -- apples-to-apples). ---
    pop_gen = PopularityCandidateGenerator().fit(train_df, item_col="campaign", positive_col="click")
    pop_top_k = pop_gen.generate(user=None, context=None, k=k)

    # --- Collaborative filtering: fit ONLY on the warm cohort's own
    # training interactions (task requirement -- no information available
    # only to other users leaks into a warm user's recommendations). ---
    warm_train_df = train_df[train_df["uid"].isin(warm_uids)]
    cf_gen = CollaborativeFilteringCandidateGenerator(similarity_type=args.similarity).fit(
        warm_train_df, user_col="uid", item_col="campaign", positive_col="click"
    )
    cf_catalog = cf_gen.coverage_stats()
    print(f"CF fit on warm-cohort training data only: {len(warm_train_df):,} rows, "
          f"{cf_catalog['n_users_with_affinity']:,} users with affinity, "
          f"{cf_catalog['n_items_in_catalog']:,} campaigns in CF's catalog.")

    # --- Ground truth and evaluation universe: warm test users with >=1
    # test-period click, identical filter to the popularity baseline run. ---
    test_clicks = test_df[test_df["click"] == 1]
    y_true_all = test_clicks.groupby("uid")["campaign"].apply(set).to_dict()
    warm_test_uids = [u for u in y_true_all if cohorts.get(u, COLD) == WARM]
    y_true = {u: y_true_all[u] for u in warm_test_uids}
    print(f"Warm test users with >=1 test-period click (evaluation universe): {len(y_true):,}")

    # Popularity predictions: same list for everyone.
    y_pred_pop = {u: pop_top_k for u in y_true}

    # CF predictions, two variants.
    y_pred_cf = {u: cf_gen.generate(u, None, k=k, exclude_seen=False) for u in y_true}
    y_pred_cf_excl = {u: cf_gen.generate(u, None, k=k, exclude_seen=True) for u in y_true}

    def report(label, y_pred):
        r = recall_at_k(y_true, y_pred, k)
        h = hit_rate_at_k(y_true, y_pred, k)
        n = ndcg_at_k(y_true, y_pred, k)
        return {"recall@k": r, "hit_rate@k": h, "ndcg@k": n}

    pop_metrics = report("popularity", y_pred_pop)
    cf_metrics = report("cf", y_pred_cf)
    cf_excl_metrics = report("cf_exclude_seen", y_pred_cf_excl)

    print(f"\n=== Warm-cohort retrieval comparison (K={k}, similarity={args.similarity}) ===")
    print(f"{'method':<20}{'Recall@k':>12}{'HitRate@k':>12}{'NDCG@k':>12}")
    for label, m in [
        ("popularity", pop_metrics),
        ("cf", cf_metrics),
        ("cf_exclude_seen", cf_excl_metrics),
    ]:
        print(f"{label:<20}{m['recall@k']:>12.4f}{m['hit_rate@k']:>12.4f}{m['ndcg@k']:>12.4f}")

    print("\n--- Lift: CF (primary, exclude_seen=False) vs. popularity, same warm cohort ---")
    for metric in ("recall@k", "hit_rate@k", "ndcg@k"):
        abs_lift, rel_lift = _lift(cf_metrics[metric], pop_metrics[metric])
        print(f"{metric:<12} popularity={pop_metrics[metric]:.4f}  cf={cf_metrics[metric]:.4f}  "
              f"abs_lift={abs_lift:+.4f}  rel_lift={rel_lift:+.2f}%")

    print("\n--- Lift: CF (exclude_seen=True) vs. popularity, same warm cohort ---")
    for metric in ("recall@k", "hit_rate@k", "ndcg@k"):
        abs_lift, rel_lift = _lift(cf_excl_metrics[metric], pop_metrics[metric])
        print(f"{metric:<12} popularity={pop_metrics[metric]:.4f}  cf_excl={cf_excl_metrics[metric]:.4f}  "
              f"abs_lift={abs_lift:+.4f}  rel_lift={rel_lift:+.2f}%")

    # --- Coverage diagnostics ---
    n_users_with_cf_recs = sum(1 for u in y_true if len(y_pred_cf[u]) > 0)
    pct_users_with_recs = n_users_with_cf_recs / len(y_true) if y_true else 0.0

    recommended_campaigns = set()
    for recs in y_pred_cf.values():
        recommended_campaigns.update(recs)
    full_catalog_size = df["campaign"].nunique()
    cf_catalog_size = cf_catalog["n_items_in_catalog"]

    n_excluded_users = len(y_true) - n_users_with_cf_recs
    n_campaigns_never_in_warm_training = full_catalog_size - cf_catalog_size

    print("\n=== Coverage ===")
    print(f"Warm test users receiving >=1 CF recommendation: {n_users_with_cf_recs:,} / {len(y_true):,} "
          f"({pct_users_with_recs:.2%})")
    print(f"Warm test users receiving ZERO CF recommendations: {n_excluded_users:,} "
          f"(no positive-similarity evidence for any campaign, or absent from CF's training affinity)")
    print(f"Distinct campaigns ever recommended by CF: {len(recommended_campaigns):,} / "
          f"{cf_catalog_size:,} in CF's training catalog "
          f"({len(recommended_campaigns)/cf_catalog_size:.2%}) / "
          f"{full_catalog_size:,} in the full dataset catalog "
          f"({len(recommended_campaigns)/full_catalog_size:.2%})")
    print(f"Campaigns with zero warm-cohort training clicks (never eligible for CF recommendation): "
          f"{n_campaigns_never_in_warm_training:,} / {full_catalog_size:,}")


if __name__ == "__main__":
    main()
