# Cold-Start Popularity Retrieval Baseline

This document describes the **implemented** cold-start popularity retrieval baseline — the first
concrete piece of the recommendation-system component, following the architecture reassessment in
[`attribution_modeling_design.md`](attribution_modeling_design.md) §3B/§4.

**What this is**: a training-period campaign popularity ranking, retrieved identically for every
user, evaluated separately for the cold and warm cohorts against test-period ground truth.

**What this is not**: a personalized recommendation system. No user-specific signal drives the
retrieved list in this baseline — every user in the evaluated population receives the same top-K
campaigns. Collaborative filtering (the warm-cohort-only personalization layer from the design
doc) is explicitly **not** implemented here; this baseline exists specifically to give that future
work a number to be measured against.

## Cohort definition

Implemented in [`src/retrieval/cohort.py`](../src/retrieval/cohort.py).

A user is classified **warm** if they clicked on at least `warm_min_distinct_campaigns` (default
2, configurable at `configs/config.yaml`: `attribution.cohort.warm_min_distinct_clicked_campaigns`)
distinct campaigns during the **training period only**; otherwise **cold**. This threshold is not
arbitrary — it is the verified minimum below which a user provides no relative-preference signal
(a single clicked campaign is one data point, not a preference between alternatives; see design
doc §3B).

Classification uses **only** the `train` split produced by
[`src.data.attribution.split_by_day`](../src/data/attribution.py). The function itself performs no
time filtering — the leakage boundary is enforced once, explicitly, at the split step, and is
covered by an end-to-end test (`tests/test_cohort.py::test_no_temporal_leakage_end_to_end`) that
constructs a user who would be misclassified warm if test-period clicks were allowed to leak in,
and confirms the correctly-split pipeline avoids it.

A user who never appears in the training period at all (a genuinely new user by the time of
evaluation) is not assigned a cohort by `classify_users` — callers default such users to **cold**,
the coldest possible case, via `evaluate_retrieval_by_cohort`'s `default_cohort` argument.

## Popularity calculation

Implemented in `PopularityCandidateGenerator` in
[`src/retrieval/candidate_generator.py`](../src/retrieval/candidate_generator.py).

Campaigns are ranked by their **training-period click count** (`train_df.groupby("campaign")
["click"].sum()`), descending. Ties are broken deterministically — campaigns are first sorted by
ascending campaign id, then stably sorted by descending click count, so equal-count campaigns
always appear in the same, reproducible ascending-id order across runs. This is unit-tested
directly (`tests/test_popularity_retrieval.py::test_ties_are_broken_deterministically_by_ascending_campaign_id`).

A campaign that never appears in the training period (zero clicks, or absent entirely) is simply
absent from the ranking — it is never retrieved. No smoothing, minimum-volume floor, or recency
decay is applied in this first version (see Limitations).

## Retrieval procedure

`PopularityCandidateGenerator.generate(user, context, k)` ignores `user` and `context` entirely
and returns the same top-`k` campaign ids for every call — this is the literal mechanism by which
"not personalized" is enforced in code, not just in documentation. If the catalog has fewer than
`k` campaigns with any training-period clicks, fewer than `k` are returned (no padding, no error).

`k` is configurable at `configs/config.yaml`: `attribution.retrieval.top_k` (default 10), or via
`--k` on `scripts/run_popularity_baseline.py`.

## Evaluation methodology

Implemented in [`src/evaluation/retrieval_metrics.py`](../src/evaluation/retrieval_metrics.py).

- **Ground truth**: each user's distinct clicked campaigns in the **test period** (days ≥
  `val_end_day`, per the design doc's day-based split — train 0–20, validation 20–25, test 25–30).
- **Metrics**: Recall@K, Hit Rate@K, and NDCG@K (binary relevance), computed per user and averaged
  over users who have at least one test-period positive (a user with zero test-period clicks
  contributes no signal to any of these metrics and is excluded from the denominator, not
  penalized).
- **Reported strictly per cohort, never blended**: cohort membership (cold/warm) is determined
  once, from train-period history, and results are reported as separate `{cohort: {metric: value}}`
  entries. A single population-wide number would be dominated by the cold majority (94.6% of
  users), making it impossible to tell whether anything beyond "recommend what's popular" is
  happening — see design doc §8 for the full rationale. This baseline currently applies the *same*
  popularity list to both cohorts (since no warm-cohort-specific method exists yet), so the
  warm-cohort numbers reported here are exactly the reference point a future collaborative-filtering
  model's *lift* must be measured against.

## Limitations

- **No smoothing or confidence adjustment.** A campaign with very few training impressions but a
  high click rate by chance could rank higher than warranted; this is not corrected for in v1.
- **No recency weighting.** All 20 days of the training period are weighted equally; a campaign
  popular only at the start of training is treated the same as one popular throughout.
- **No context adjustment.** The `cat1`–`cat9` fields are not used at all in this baseline — per
  the design doc §1, none is a reliable static campaign attribute, so no content-based adjustment
  is attempted here (this was `PopularityCandidateGenerator`'s explicit scope: pure popularity).
  Whether a weak context adjustment for the cold cohort is worth adding is left as a documented
  open question in the design doc's baseline progression (§9, step 2) — not implemented here.
- **Warm-cohort numbers here are a baseline reference, not a claim of personalization.** Applying
  the identical popularity list to warm users produces their Recall@K/Hit Rate@K/NDCG@K under "no
  personalization" — this is the number a future CF model has to beat, not a result about
  personalization itself.
- **Catalog size caveat carried over from the design doc**: with only 675 campaigns, Recall@K here
  is an easier task than the same metric on a large-catalog production system.
