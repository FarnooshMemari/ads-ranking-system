# Ranking Integration Results

Results from running Parts A and B of [`ranking_integration_design.md`](ranking_integration_design.md)
against the full 16,468,027-row Attribution dataset. Part A's full writeup is in
[`pre_serve_ctr_model.md`](pre_serve_ctr_model.md); this document covers Part B and the overall
conclusion.

## Part A summary (full detail in `pre_serve_ctr_model.md`)

The pre-serve model (excluding `cat1`–`cat9`) reached PR-AUC 0.5802 / ROC-AUC 0.6911 / LogLoss
0.5987 on real test-period impressions — a real but moderate ~3–4% relative cost versus the
full-feature model (PR-AUC 0.6069 / ROC-AUC 0.7209 / LogLoss 0.5816), and still a large, genuine
improvement over the constant-prediction baseline (+58%/+38%/+9%). This is the only model variant
valid for Part B, since `cat1`–`cat9` don't exist for a not-yet-served retrieval candidate.

## Correction (population-coverage bug, not leakage)

The numbers below were corrected after this project's own next-phase pipeline work
(`docs/pipeline_demo.md`) surfaced an inconsistency: the cold cohort's contrast-user count didn't
match between the two independently-built code paths (1,091 vs. 759), while the warm cohort's did
match exactly. Investigating that asymmetry found the root cause in
`scripts/run_ranking_integration_experiment.py`: the cold-cohort evaluation population was built as
`test_uids & set(cohorts[cohorts == COLD].index)`, which silently **excludes any user entirely
absent from the training-period cohort assignment** — i.e. a user with zero train-period history who
first appears during the test period. Such a user is neither cold nor warm under that construction;
they were simply dropped from evaluation, never counted in either cohort.

This is a **population-coverage bug, not temporal leakage** — no future information was ever used.
It undercounted who was evaluated, not when their features were computed. The fix
(`src.evaluation.ranking_integration.classify_test_users`) applies the same default-to-cold
convention already used correctly elsewhere in this project (
`src.evaluation.retrieval_metrics.evaluate_retrieval_by_cohort`'s `default_cohort` parameter, and
`src.pipeline.RecommendationPipeline.cohort_for`): a user absent from training-period cohort data is
the coldest possible case, not an omitted one. Verified directly: 353,170 users with real
test-period click activity — and more with any real test-period impression — had zero
training-period history and were missing from this document's original cold-cohort numbers as a
result. The warm cohort was never affected (a never-seen user can default to cold, never to warm,
under any correct convention), and its numbers below are unchanged from the original publication —
confirmed by rerunning the corrected script and finding the warm figures bit-for-bit identical.

Regression coverage for this exact scenario (a test-period user absent from train-period cohort
data must default to cold, never be dropped) lives in
`tests/test_ranking_integration.py::TestClassifyTestUsers`.

## Part B: intersection size and coverage (reported before any ranking metric, as designed)

| | Cold cohort | Warm cohort |
|---|---|---|
| Users evaluated (any real test-period impression) | 1,761,268 | 90,912 |
| Retrieved candidate slots (K=10) | 17,612,680 | 909,120 |
| **Verified slots** (retrieved AND independently confirmed shown) | 327,977 (1.86% of retrieved slots) | 49,115 (5.40% of retrieved slots) |
| Users with ≥1 verified candidate | 325,466 (18.48%) | 38,974 (42.87%) |
| Users with ≥2 verified candidates | 2,502 (0.14%) | 8,952 (9.85%) |
| **Users with a genuine click-vs-non-click contrast** | **1,091** | **3,479** |
| Documented minimum (`min_users_for_ranking_check`) | 30 | 30 |
| **Meets minimum?** | **Yes** | **Yes** |

Both cohorts cleared the documented minimum with real margin — contrary to the real possibility
(anticipated explicitly in the design doc) that this could turn out too small to be meaningful. The
warm cohort shows visibly better coverage than cold on every measure (2.9× the verified-slot rate,
~69× the ≥2-verified-candidate rate) — consistent with warm users, by their training-period
definition, having more interaction history overall, which plausibly extends into the test period
too.

**What this table already shows, before any ranking number**: even with real coverage clearing the
statistical minimum, the *absolute* overlap between retrieval's fixed candidate lists and what
users were actually independently shown is small — under 2% of retrieved slots for cold and ~5% for
warm, and well under 1% of cold users ever have a genuine preference contrast to check. This is the
honest, quantified shape of the sparsity problem this whole project has tracked since the
retrieval-stage distributional analysis, now measured at the specific point where it matters for
ranking evaluation.

## Part B: ranking-agreement result

| Cohort | Pairwise accuracy | Verified pairs | Users contributing |
|---|---|---|---|
| Cold | **0.5345** | 1,100 | 1,091 |
| Warm | **0.5298** | 4,209 | 3,479 |

**Reading this honestly**: both are above the 0.50 chance level, but only modestly — the pre-serve
model gets the clicked candidate ranked above the non-clicked one about 53% of the time for cold,
53% for warm, versus 50% for a coin flip. This is a weak, real, positive signal — not a strong
validation that this pipeline ranks well, and not a null result either. It sits between the two
outcomes the design doc anticipated ("meaningful lift" vs. "not statistically defensible") — a
third, more honest outcome: statistically computable, but small.

**The cold and warm cohorts' accuracy are now close enough (0.5345 vs. 0.5298) that the earlier
narrative of cold noticeably outperforming warm no longer holds** — that gap (0.5535 vs. 0.5298) was
partly an artifact of the smaller, biased pre-fix cold population. With the population-coverage bug
fixed, the two cohorts' pairwise accuracy is not meaningfully distinguishable at this pair count —
1,100 and 4,209 pairs are enough to clear the design's minimum-sample bar, but not enough to treat a
half-point difference between cohorts as a confident finding on its own.

## What this does and does not demonstrate

**Demonstrates**: the pipeline (cohort classification → retrieval → pre-serve CTR scoring) runs
end to end, produces real, checkable outputs, and — on the only population this dataset can support
a check for — shows a small but real positive ranking signal, not zero and not fabricated.

**Does not demonstrate**: that this pipeline would perform well as a real served ranking system.
Per `ranking_integration_design.md`'s formulation 3, this dataset provides no serving propensities,
so no claim about *live* performance is possible, and a ~53% pairwise accuracy on a thousand to a
few thousand verified pairs is a narrow diagnostic, not a production readiness signal.

**Does not claim to reproduce a real served slate.** Every verified candidate in this check came
from an actual, independently logged impression — never a fabricated exposure or an invented
negative for an unobserved campaign, consistent with the constraint that has governed this entire
line of work.

## Conclusion

A full, statistically powerful, end-to-end offline ranking evaluation remains not achievable with
this dataset, exactly as `ranking_integration_design.md` concluded before this experiment ran. What
*is* now available, and wasn't before: a real, quantified, small-but-positive ranking-agreement
signal, computed only on genuinely observed data, with its sample size and coverage reported
honestly alongside it — the ceiling this dataset supports, demonstrated rather than assumed.
