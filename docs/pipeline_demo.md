# Assembled Pipeline Demonstration

Implements the recommended next step from [`next_phase_design.md`](next_phase_design.md):
`src/pipeline.py`, connecting cohort classification → retrieval → pre-serve CTR scoring into one
callable artifact, `RecommendationPipeline`. This document reports the real results of running it
against the full dataset (`scripts/run_pipeline_demo.py`), including one genuine, previously
undiscovered issue this exercise surfaced in earlier, already-published work — reported here rather
than smoothed over, consistent with this project's existing standard.

**This is not a live serving system.** Everything below is an offline, batch replay against
already-logged data, fit once from historical data up to the existing day-25 validation/test
boundary — no new cutoff, no online component, no API.

## Part 1: Retrieval-level consistency check — exact match, as required

| Cohort | n_users | Recall@10 | Hit Rate@10 |
|---|---|---|---|
| Cold | 693,737 | **0.1959** | 0.2007 |
| Warm | 56,353 | **0.3983** | 0.4350 |

These match the already-published numbers **exactly**:
[`popularity_baseline.md`](popularity_baseline.md)'s cold Recall@10 (0.1959), and
[`collaborative_filtering.md`](collaborative_filtering.md)'s warm *unfiltered* Recall@10/Hit
Rate@10 (0.3983/0.4350 — the pipeline's retrieval call does not exclude already-seen campaigns, the
same choice used for that variant). This is the correctness bar this step existed to clear:
re-sorting a fixed candidate set by predicted CTR does not change top-K set membership, so an exact
match confirms the pipeline's retrieval fitting is faithful to the original, separately-validated
experiments — not a new, drifted implementation.

## Part 2: Pipeline's own pairwise ranking-agreement accuracy — a new, different number, explained

| Cohort | Verified contrast users | Pairwise accuracy (this pipeline) | Published ([`ranking_integration_results.md`](ranking_integration_results.md)) |
|---|---|---|---|
| Cold | 1,091 | **0.5982** | 0.5345 (n=1,091) |
| Warm | 3,479 | **0.5783** | 0.5298 (n=3,479) |

Both numbers are new and were never expected to exactly reproduce the earlier ones — this was
stated explicitly before running anything (`scripts/run_pipeline_demo.py`'s own header comment),
because this pipeline scores every candidate using a **single as-of-day-25 snapshot** of a user's
history, while the Part B script scores each verified candidate using **that specific real row's own
point-in-time features**, computed wherever in the day 25–30 test window that row actually fell.
These are two different, both legitimate, questions: "what would we recommend right now, at a
single coherent moment" versus "was the model's prediction good at the exact moment each real
impression happened."

### A population-coverage bug this comparison surfaced — found and fixed

Both cohorts' contrast-user counts now match exactly between this pipeline and the (corrected) Part
B script: warm at 3,479 = 3,479, and cold at 1,091 = 1,091. That wasn't true when this comparison was
first run — cold showed 1,091 (pipeline) vs. 759 (`run_ranking_integration_experiment.py`), while
warm already matched. That asymmetry was investigated directly rather than assumed, and traced to a
real bug rather than the expected scoring-timing difference alone:

`run_ranking_integration_experiment.py` had built its cold-cohort population as `test_uids &
cold_uids`, where `cold_uids = set(cohorts[cohorts == COLD].index)` — this **silently excluded any
user absent from the training-period cohort assignment entirely** (a user who first appears during
the test period, with zero training history). This project's own
`src.evaluation.retrieval_metrics.evaluate_retrieval_by_cohort` — used correctly by the *original*
popularity baseline script — instead defaults such users to `COLD` via `cohorts.get(uid,
default_cohort)`, the correct convention (a never-before-seen user is the coldest possible case,
explicitly documented in `src/retrieval/cohort.py`). `RecommendationPipeline.cohort_for` uses this
same correct default. Verified directly: **353,170 users** with a real test-period click had zero
training-period history at all, and were silently dropped from `ranking_integration_results.md`'s
original cold-cohort evaluation entirely — not misclassified, simply never counted in either cohort.

This never affected the warm cohort (a never-seen user can default to `cold`, never to `warm`, under
any correct convention), and it was **not a leakage issue** — no future information was ever used;
it was a population-coverage gap, silently undercounting one cohort's evaluated population. It has
since been **fixed**: `src.evaluation.ranking_integration.classify_test_users` now applies the same
default-to-cold convention used here, `run_ranking_integration_experiment.py` was updated to use it,
regression coverage was added (`tests/test_ranking_integration.py::TestClassifyTestUsers`), and
`ranking_integration_results.md` was rerun and corrected — full detail and corrected numbers there.
With the population fixed, the remaining gap between this pipeline's cold accuracy (0.5982) and the
corrected Part B figure (0.5345) is now cleanly attributable to the scoring-timing difference alone,
the same single factor that already explained the warm cohort's gap.

## Part 3: Qualitative demonstration (deterministic selection)

Selected as the smallest-`uid` user with a verified click/non-click contrast in each cohort — not
hand-picked. Full ranked output, `k=10`:

**Cold example, uid=51600** (top prediction 0.7162, verified real click confirms rank 4 at
predicted CTR 0.5805 — a real hit, though not the top-ranked item):

| rank | campaign | predicted_ctr | verified_shown | real_click |
|---|---|---|---|---|
| 1 | 28351001 | 0.7162 | False | — |
| 2 | 10341182 | 0.6291 | False | — |
| 3 | 31772643 | 0.5845 | False | — |
| **4** | **15398570** | **0.5805** | **True** | **1** |
| 5 | 18975823 | 0.5375 | False | — |
| 6 | 497593 | 0.5297 | False | — |
| 7 | 15184511 | 0.5276 | False | — |
| 8 | 30801593 | 0.5254 | False | — |
| 9 | 5061834 | 0.4668 | **True** | 0 |
| 10 | 17686799 | 0.4542 | False | — |

**Warm example, uid=5110** (real click confirmed at rank 10, the lowest-ranked verified candidate —
a real miss for this specific user, reported as-is, not omitted):

| rank | campaign | predicted_ctr | verified_shown | real_click |
|---|---|---|---|---|
| 1 | 25119476 | 0.4516 | False | — |
| ... | ... | ... | ... | ... |
| 8 | 12121528 | 0.3797 | **True** | 0 |
| 9 | 29427842 | 0.2959 | False | — |
| **10** | **11321105** | **0.2903** | **True** | **1** |

Both examples are reported in full, including the warm example's real miss — a cherry-picked demo
would have quietly chosen a different user. `verified_shown`/`real_click` come only from real,
independently logged test-period rows (`src.evaluation.ranking_integration.build_exposure_maps`);
"False"/"—" means unknown exposure, never an inferred negative.

## Leakage and feature-set verification (checked directly, not assumed)

- `cat1`–`cat9` never reach the scoring call — enforced by an explicit assertion in
  `RecommendationPipeline._score` (`tests/test_pipeline.py::TestPreServeOnlyScoring` covers this
  with a fake model that records exactly which columns it received).
- No post-outcome column (`cost`, `cpo`, `conversion`, `click_pos`, `click_nb`,
  `conversion_timestamp`) reaches scoring — same assertion, reusing
  `assert_no_post_outcome_leakage` unchanged.
- The as-of-day-25 snapshot excludes any row at or after the cutoff
  (`tests/test_pipeline.py::TestTemporalIntegrity`), and the snapshot aggregation is verified
  numerically equivalent to the already-tested per-row point-in-time functions
  (`TestSnapshotEquivalence`).
- `src/retrieval/` is unchanged (verified via `git diff --stat` before committing) — this pipeline
  only orchestrates it.

## Summary

The pipeline reproduces already-published retrieval numbers exactly, confirming it's a faithful
assembly rather than a re-implementation. Its own ranking-agreement numbers (0.60 cold, 0.58 warm)
are modestly higher than, but not directly comparable to, the (corrected) published ones, for one
remaining, fully explained reason: the scoring-timing difference (single as-of-day-25 snapshot vs.
per-row point-in-time features). Comparing the two independently-built code paths also surfaced a
real population-coverage bug in the earlier Part B script — unrelated to leakage — which has since
been found, fixed, tested, and corrected in `ranking_integration_results.md`. The qualitative
examples show the assembled system producing real, checkable output for individual users —
including an honestly-reported miss, not just a hit.
