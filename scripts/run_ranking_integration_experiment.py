#!/usr/bin/env python3
"""Part B of the ranking-integration design: a restricted retrieval/ranking check.

Requires Part A (`scripts/run_pre_serve_ctr_experiment.py`) to have already run
and saved a trained pre-serve model -- this script loads it rather than
retraining.

For each cohort, using the EXISTING, UNCHANGED retrieval system (popularity
for cold, collaborative filtering for warm -- see src/retrieval/):

    1. Generate each user's retrieved top-K candidates.
    2. Intersect those candidates with campaigns INDEPENDENTLY CONFIRMED as
       actually shown to that same user in the test period (any outcome --
       never an unobserved campaign treated as a negative). Core logic in
       src/evaluation/ranking_integration.py, unit-tested separately.
    3. Report the size and coverage of that intersection BEFORE computing
       any ranking metric.
    4. Only if enough users have a genuine click-vs-non-click contrast among
       their verified candidates (>= `attribution.ranking_integration.
       min_users_for_ranking_check`, a documented, pragmatic threshold) does
       this script compute pairwise ranking accuracy with the pre-serve
       model. Otherwise it stops and reports explicitly that quantitative
       ranking evaluation is not defensible for that cohort.

This does NOT claim to reproduce a real served slate -- see
docs/ranking_integration_results.md.

Usage:
    python scripts/run_ranking_integration_experiment.py [--k K]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.attribution import load, split_by_day  # noqa: E402
from src.evaluation.ranking_integration import (  # noqa: E402
    build_exposure_maps,
    classify_test_users,
    find_contrast_users,
    pairwise_accuracy,
    verified_candidates,
)
from src.features.attribution_features import (  # noqa: E402
    PRE_SERVE_FEATURE_COLUMNS,
    build_ctr_features,
)
from src.models.ctr_model import CTRModel  # noqa: E402
from src.retrieval.candidate_generator import (  # noqa: E402
    CollaborativeFilteringCandidateGenerator,
    PopularityCandidateGenerator,
)
from src.retrieval.cohort import COLD, WARM, classify_users  # noqa: E402
from src.utils.config import load_config, resolve_path  # noqa: E402


def build_key_to_index(test_features):
    """(uid, campaign) -> first matching row's index label in `test_features`,
    used to slice real, properly-dtyped feature rows for scoring (never
    reconstructed from a Series, which would lose the `campaign` category
    dtype LightGBM was trained with)."""
    key_to_index = {}
    for idx, uid, campaign in zip(test_features.index, test_features["uid"], test_features["campaign"]):
        key = (uid, campaign)
        if key not in key_to_index:
            key_to_index[key] = idx
    return key_to_index


def analyze_cohort(label, user_ids, retrieved_fn, shown_map, click_lookup, min_users_for_ranking_check, k):
    """Compute intersection size/coverage for one cohort; return the
    click/non-click contrast set for later scoring plus whether the
    documented minimum sample size was met."""
    n_users = len(user_ids)
    verified_per_user = {}
    n_retrieved_slots = 0
    n_verified_slots = 0

    for uid in user_ids:
        retrieved = retrieved_fn(uid)
        n_retrieved_slots += len(retrieved)
        verified = verified_candidates(retrieved, shown_map.get(uid, set()))
        n_verified_slots += len(verified)
        if verified:
            verified_per_user[uid] = verified

    n_with_ge1 = sum(1 for v in verified_per_user.values() if len(v) >= 1)
    n_with_ge2 = sum(1 for v in verified_per_user.values() if len(v) >= 2)
    contrast_users = find_contrast_users(verified_per_user, click_lookup)

    print(f"\n--- {label} cohort ---")
    print(f"users evaluated: {n_users:,}")
    print(f"retrieved candidate slots (total, k={k}): {n_retrieved_slots:,}")
    if n_retrieved_slots:
        print(f"verified slots (candidate AND independently shown in test period): {n_verified_slots:,} "
              f"({n_verified_slots / n_retrieved_slots:.4%} of retrieved slots)")
    if n_users:
        print(f"users with >=1 verified candidate: {n_with_ge1:,} ({n_with_ge1 / n_users:.4%})")
        print(f"users with >=2 verified candidates: {n_with_ge2:,} ({n_with_ge2 / n_users:.4%})")
    print(f"users with a genuine click-vs-non-click CONTRAST among verified candidates: "
          f"{len(contrast_users):,}")

    if len(contrast_users) < min_users_for_ranking_check:
        print(f"BELOW the documented minimum ({min_users_for_ranking_check}) for a ranking-agreement "
              f"metric to be reported as reliable.")
        print("=> Quantitative ranking evaluation is NOT statistically defensible for this cohort. "
              "Reporting size/coverage only, per design.")
        return contrast_users, False

    print(f"MEETS the documented minimum ({min_users_for_ranking_check}) -- proceeding to score.")
    return contrast_users, True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=None, help="Override attribution.retrieval.top_k")
    args = parser.parse_args()

    config = load_config()
    attr_cfg = config["attribution"]
    k = args.k if args.k is not None else attr_cfg["retrieval"]["top_k"]
    min_users = attr_cfg["ranking_integration"]["min_users_for_ranking_check"]

    model_path = resolve_path(attr_cfg["ranking"]["ctr_model_pre_serve"]["model_output_path"])
    if not model_path.exists():
        print(f"Error: pre-serve model not found at {model_path}.", file=sys.stderr)
        print("Run scripts/run_pre_serve_ctr_experiment.py first (Part A).", file=sys.stderr)
        sys.exit(1)
    model = CTRModel(params={}).load(str(model_path))
    print(f"Loaded pre-serve model from {model_path}")

    print("Loading Attribution dataset...")
    df = load(config)
    train_df, val_df, test_df = split_by_day(
        df, attr_cfg["split"]["train_end_day"], attr_cfg["split"]["val_end_day"]
    )
    print(f"Split -> train: {len(train_df):,} | val: {len(val_df):,} | test: {len(test_df):,} rows")

    cohorts = classify_users(
        train_df, warm_min_distinct_campaigns=attr_cfg["cohort"]["warm_min_distinct_clicked_campaigns"]
    )
    warm_uids = set(cohorts[cohorts == WARM].index)

    # --- Retrieval: reused exactly as built, unchanged (popularity on the
    # full training set; CF on the warm cohort's own training interactions,
    # exclude_seen=False -- the primary variant already established in
    # collaborative_filtering.md, matched here for consistency). ---
    pop_gen = PopularityCandidateGenerator().fit(train_df, item_col="campaign", positive_col="click")
    pop_top_k = pop_gen.generate(user=None, context=None, k=k)

    warm_train_df = train_df[train_df["uid"].isin(warm_uids)]
    cf_gen = CollaborativeFilteringCandidateGenerator(similarity_type="jaccard").fit(
        warm_train_df, user_col="uid", item_col="campaign", positive_col="click"
    )

    shown_map, click_lookup = build_exposure_maps(test_df)

    # Evaluation universe: users present in the test period at all (any real
    # impression, not just clickers -- broader than retrieval's own recall
    # evaluation, since exposure coverage is the question here, not clicks).
    #
    # FIXED (population-coverage bug, not leakage -- see
    # src.evaluation.ranking_integration.classify_test_users and
    # docs/ranking_integration_results.md): a user absent from `cohorts`
    # entirely (zero training-period history) must default to COLD, the
    # same convention already used correctly elsewhere in this project.
    # The previous version of this script built cold_test_uids as
    # `test_uids & set(cohorts[cohorts == COLD].index)`, which silently
    # excluded any such user from both cohorts -- neither cold nor warm,
    # just dropped. Verified: 353,170 users with real test-period activity
    # had zero training-period history and were missing as a result.
    test_uids = set(test_df["uid"].unique())
    test_user_cohorts = classify_test_users(cohorts, test_uids, default_cohort=COLD)
    cold_test_uids = sorted(uid for uid, c in test_user_cohorts.items() if c == COLD)
    warm_test_uids = sorted(uid for uid, c in test_user_cohorts.items() if c == WARM)

    print("\nBuilding pre-serve features for test-period rows (for scoring verified candidates)...")
    features = build_ctr_features(df, train_df)
    _, _, test_features = split_by_day(
        features, attr_cfg["split"]["train_end_day"], attr_cfg["split"]["val_end_day"]
    )
    key_to_index = build_key_to_index(test_features)

    cold_contrast, cold_ok = analyze_cohort(
        "COLD", cold_test_uids, lambda uid: pop_top_k, shown_map, click_lookup, min_users, k
    )
    warm_contrast, warm_ok = analyze_cohort(
        "WARM", warm_test_uids,
        lambda uid: cf_gen.generate(uid, None, k=k, exclude_seen=False),
        shown_map, click_lookup, min_users, k,
    )

    def make_score_fn():
        cache = {}

        def score_fn(uid, campaign):
            key = (uid, campaign)
            if key not in key_to_index:
                return None
            if key not in cache:
                row = test_features.loc[[key_to_index[key]], PRE_SERVE_FEATURE_COLUMNS]
                cache[key] = float(model.predict(row)[0])
            return cache[key]

        return score_fn

    print("\n" + "=" * 80)
    print("RANKING-AGREEMENT RESULT (pairwise accuracy: does the pre-serve model score the")
    print("clicked verified candidate above the non-clicked one? NOT a claim of a real served slate.)")
    print("=" * 80)
    score_fn = make_score_fn()
    for label, contrast, ok in [("COLD", cold_contrast, cold_ok), ("WARM", warm_contrast, warm_ok)]:
        if not ok:
            print(f"{label}: NOT statistically defensible (see above) -- not computed.")
            continue
        acc, n_pairs, n_users_used = pairwise_accuracy(contrast, score_fn)
        print(f"{label}: pairwise accuracy = {acc:.4f} over {n_pairs:,} verified pairs "
              f"from {n_users_used:,} users.")


if __name__ == "__main__":
    main()
