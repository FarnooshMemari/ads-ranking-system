#!/usr/bin/env python3
"""Assembled pipeline demonstration (docs/next_phase_design.md).

Fits RecommendationPipeline (src/pipeline.py) from the full Attribution
dataset and:

    1. Runs a retrieval-level consistency check: recommend_batch's candidate
       sets, evaluated with the existing retrieval_metrics functions, must
       reproduce the already-published cold/warm Recall@10 and Hit Rate@10
       from popularity_baseline.md / collaborative_filtering.md exactly --
       re-sorting candidates by predicted CTR does not change top-K set
       membership, so this is a strict correctness check, not an
       approximation.
    2. Computes the pipeline's own pairwise ranking-agreement accuracy on
       the same verified-exposure population used in
       ranking_integration_results.md, and reports it alongside (NOT as an
       expected exact match with) that document's published number -- the
       pipeline scores every candidate using a single as-of-day-25 snapshot
       of each user's history, while the original Part B script scored each
       verified candidate using that specific real row's own point-in-time
       features (which could fall anywhere in the day 25-30 test window).
       These are two different, both legitimate, questions -- see
       docs/pipeline_demo.md for the full explanation.
    3. Produces a deterministic (never cherry-picked) qualitative
       demonstration for one cold and one warm example user.

This is an offline, batch replay against already-logged data -- not a live
system. See src/pipeline.py's module docstring for the full scope statement.

Usage:
    python scripts/run_pipeline_demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.attribution import load, split_by_day  # noqa: E402
from src.evaluation.ranking_integration import (  # noqa: E402
    build_exposure_maps,
    find_contrast_users,
    pairwise_accuracy,
    verified_candidates,
)
from src.evaluation.retrieval_metrics import hit_rate_at_k, recall_at_k  # noqa: E402
from src.models.ctr_model import CTRModel  # noqa: E402
from src.pipeline import RecommendationPipeline  # noqa: E402
from src.retrieval.cohort import COLD, WARM  # noqa: E402
from src.utils.config import load_config, resolve_path  # noqa: E402


def main() -> None:
    config = load_config()
    attr_cfg = config["attribution"]
    k = attr_cfg["retrieval"]["top_k"]
    min_users = attr_cfg["ranking_integration"]["min_users_for_ranking_check"]

    model_path = resolve_path(attr_cfg["ranking"]["ctr_model_pre_serve"]["model_output_path"])
    if not model_path.exists():
        print(f"Error: pre-serve model not found at {model_path}.", file=sys.stderr)
        print("Run scripts/run_pre_serve_ctr_experiment.py first.", file=sys.stderr)
        sys.exit(1)
    pre_serve_model = CTRModel(params={}).load(str(model_path))

    print("Loading Attribution dataset...")
    df = load(config)
    train_end_day = attr_cfg["split"]["train_end_day"]
    val_end_day = attr_cfg["split"]["val_end_day"]
    _, _, test_df = split_by_day(df, train_end_day, val_end_day)
    print(f"Test period: {len(test_df):,} rows")

    print("\nFitting RecommendationPipeline (reuses existing retrieval fits, existing pre-serve model)...")
    pipeline = RecommendationPipeline(k=k).fit(
        df, train_end_day=train_end_day, val_end_day=val_end_day, pre_serve_model=pre_serve_model
    )

    # --- Part 1: retrieval-level consistency check ---
    test_clicks = test_df[test_df["click"] == 1]
    y_true = test_clicks.groupby("uid")["campaign"].apply(set).to_dict()
    cohort_of = {uid: pipeline.cohort_for(uid) for uid in y_true}

    uids_by_cohort = {COLD: [], WARM: []}
    for uid, cohort in cohort_of.items():
        uids_by_cohort[cohort].append(uid)

    print("\n" + "=" * 80)
    print("PART 1: RETRIEVAL-LEVEL CONSISTENCY CHECK")
    print("(must exactly match the already-published Recall@10 / Hit Rate@10 -- reordering")
    print(" candidates by predicted CTR does not change top-K set membership)")
    print("=" * 80)
    for cohort_label in (COLD, WARM):
        uids = uids_by_cohort[cohort_label]
        batch = pipeline.recommend_batch(uids)
        y_pred = {uid: [r.campaign for r in recs] for uid, recs in batch.items()}
        sub_true = {uid: y_true[uid] for uid in uids}
        r = recall_at_k(sub_true, y_pred, k)
        h = hit_rate_at_k(sub_true, y_pred, k)
        print(f"{cohort_label:>5}: n_users={len(uids):,}  Recall@{k}={r:.4f}  HitRate@{k}={h:.4f}")

    print("\nCompare against:")
    print("  popularity_baseline.md -- cold Recall@10=0.1959, warm Recall@10=0.1731")
    print("  collaborative_filtering.md -- warm (unfiltered) Recall@10=0.3983")
    print("  (warm here uses cf_gen exclude_seen=False, matching the 'cf' unfiltered variant,")
    print("   not the 'cf_exclude_seen' fair-comparison variant, since recommend_batch's")
    print("   retrieval call does not exclude seen campaigns -- see src/pipeline.py)")

    # --- Part 2: pipeline's own pairwise ranking-agreement accuracy ---
    print("\n" + "=" * 80)
    print("PART 2: PIPELINE PAIRWISE RANKING-AGREEMENT ACCURACY")
    print("(a NEW number from this pipeline's own single as-of-day-25 snapshot scoring --")
    print(" NOT expected to exactly match ranking_integration_results.md's per-real-row-timing")
    print(" number; see docs/pipeline_demo.md for why these are different, both legitimate, checks)")
    print("=" * 80)
    shown_map, click_lookup = build_exposure_maps(test_df)

    for cohort_label in (COLD, WARM):
        uids = uids_by_cohort[cohort_label]
        batch = pipeline.recommend_batch(uids)
        verified_per_user = {}
        for uid, recs in batch.items():
            retrieved = [r.campaign for r in recs]
            verified = verified_candidates(retrieved, shown_map.get(uid, set()))
            if verified:
                verified_per_user[uid] = verified
        contrast_users = find_contrast_users(verified_per_user, click_lookup)
        print(f"\n--- {cohort_label} ---")
        print(f"users with a verified click-vs-non-click contrast: {len(contrast_users):,}")
        if len(contrast_users) < min_users:
            print(f"BELOW the documented minimum ({min_users}) -- not statistically defensible, not computed.")
            continue

        scores_cache = {}

        def score_fn(uid, campaign, _batch=batch):
            key = (uid, campaign)
            if key not in scores_cache:
                scores_cache.update(
                    {(uid, r.campaign): r.predicted_ctr for r in _batch[uid]}
                )
            return scores_cache.get(key)

        acc, n_pairs, n_users_used = pairwise_accuracy(contrast_users, score_fn)
        print(f"pairwise accuracy = {acc:.4f} over {n_pairs:,} pairs from {n_users_used:,} users")

    # --- Part 3: deterministic qualitative demonstration ---
    print("\n" + "=" * 80)
    print("PART 3: QUALITATIVE DEMONSTRATION (deterministic selection -- never cherry-picked)")
    print("NOT a claim of a real served slate. Offline replay against already-logged data only.")
    print("=" * 80)
    cold_batch = pipeline.recommend_batch(uids_by_cohort[COLD])
    warm_batch = pipeline.recommend_batch(uids_by_cohort[WARM])
    cold_verified = {
        uid: verified_candidates([r.campaign for r in recs], shown_map.get(uid, set()))
        for uid, recs in cold_batch.items()
    }
    warm_verified = {
        uid: verified_candidates([r.campaign for r in recs], shown_map.get(uid, set()))
        for uid, recs in warm_batch.items()
    }
    cold_contrast = find_contrast_users(cold_verified, click_lookup)
    warm_contrast = find_contrast_users(warm_verified, click_lookup)

    for label, contrast, batch in [("COLD", cold_contrast, cold_batch), ("WARM", warm_contrast, warm_batch)]:
        example_uid = sorted(contrast.keys())[0]  # deterministic: smallest uid with a verified contrast
        print(f"\n--- Example {label} user: uid={example_uid} ---")
        print(f"cohort: {pipeline.cohort_for(example_uid)}")
        print(f"{'rank':<6}{'campaign':<14}{'predicted_ctr':<16}{'verified_shown':<16}{'real_click'}")
        for rank, r in enumerate(batch[example_uid], start=1):
            verified_shown = r.campaign in shown_map.get(example_uid, set())
            real_click = click_lookup.get((example_uid, r.campaign), "-") if verified_shown else "-"
            print(f"{rank:<6}{r.campaign:<14}{r.predicted_ctr:<16.4f}{str(verified_shown):<16}{real_click}")


if __name__ == "__main__":
    main()
