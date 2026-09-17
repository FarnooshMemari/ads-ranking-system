#!/usr/bin/env python3
"""Cold-start popularity retrieval baseline experiment.

This is a **cold-start popularity retrieval baseline**, not a personalized
recommendation system — see docs/attribution_modeling_design.md §3B/§4 and
docs/popularity_baseline.md.

Pipeline:
    1. Load the Criteo Attribution dataset.
    2. Split it chronologically into train/validation/test by day
       (configs/config.yaml: attribution.split).
    3. Classify users into cold/warm cohorts using ONLY train-period click
       history (configs/config.yaml: attribution.cohort).
    4. Compute a campaign popularity ranking from train-period clicks only.
    5. Retrieve the same top-K campaigns for every user (both cohorts — no
       collaborative filtering exists yet for the warm cohort, so this run
       establishes the popularity reference number a future warm-cohort
       model must be measured against).
    6. Evaluate Recall@K, Hit Rate@K, and NDCG@K against test-period
       ground truth, reported separately per cohort.

Usage:
    python scripts/run_popularity_baseline.py [--k K]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.attribution import load, split_by_day  # noqa: E402
from src.evaluation.retrieval_metrics import evaluate_retrieval_by_cohort  # noqa: E402
from src.retrieval.candidate_generator import PopularityCandidateGenerator  # noqa: E402
from src.retrieval.cohort import COLD, classify_users, cohort_counts  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--k", type=int, default=None, help="Override configs/config.yaml: attribution.retrieval.top_k"
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
    print(
        f"Split -> train: {len(train_df):,} rows | val: {len(val_df):,} rows | "
        f"test: {len(test_df):,} rows"
    )

    cohorts = classify_users(
        train_df,
        warm_min_distinct_campaigns=attr_cfg["cohort"]["warm_min_distinct_clicked_campaigns"],
    )
    counts = cohort_counts(cohorts)
    n_total = counts["cold"] + counts["warm"]
    print(
        f"Cohorts (train-period only): cold={counts['cold']:,} "
        f"({counts['cold']/n_total:.2%}), warm={counts['warm']:,} ({counts['warm']/n_total:.2%})"
    )

    generator = PopularityCandidateGenerator().fit(train_df, item_col="campaign", positive_col="click")
    top_k_campaigns = generator.generate(user=None, context=None, k=k)
    print(f"Top-{k} campaigns by train-period popularity: {top_k_campaigns}")

    test_clicks = test_df[test_df["click"] == 1]
    y_true = test_clicks.groupby("uid")["campaign"].apply(set).to_dict()
    # Same retrieved list for every user -- this baseline is not personalized.
    y_pred = {u: top_k_campaigns for u in y_true}

    results = evaluate_retrieval_by_cohort(
        y_true, y_pred, cohorts.to_dict(), k=k, default_cohort=COLD
    )

    print(f"\n=== Cold-start popularity retrieval baseline (K={k}) ===")
    for label in sorted(results):
        r = results[label]
        print(
            f"{label:>5}: n_users={r['n_users']:>9,}  "
            f"Recall@{k}={r['recall@k']:.4f}  "
            f"HitRate@{k}={r['hit_rate@k']:.4f}  "
            f"NDCG@{k}={r['ndcg@k']:.4f}"
        )


if __name__ == "__main__":
    main()
