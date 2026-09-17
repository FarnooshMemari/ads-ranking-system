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

## Part B: intersection size and coverage (reported before any ranking metric, as designed)

| | Cold cohort | Warm cohort |
|---|---|---|
| Users evaluated (any real test-period impression) | 822,446 | 90,912 |
| Retrieved candidate slots (K=10) | 8,224,460 | 909,120 |
| **Verified slots** (retrieved AND independently confirmed shown) | 165,276 (2.01% of retrieved slots) | 49,115 (5.40% of retrieved slots) |
| Users with ≥1 verified candidate | 163,499 (19.88%) | 38,974 (42.87%) |
| Users with ≥2 verified candidates | 1,770 (0.22%) | 8,952 (9.85%) |
| **Users with a genuine click-vs-non-click contrast** | **759** | **3,479** |
| Documented minimum (`min_users_for_ranking_check`) | 30 | 30 |
| **Meets minimum?** | **Yes** | **Yes** |

Both cohorts cleared the documented minimum with real margin — contrary to the real possibility
(anticipated explicitly in the design doc) that this could turn out too small to be meaningful. The
warm cohort shows visibly better coverage than cold on every measure (2.7× the verified-slot rate,
~45× the ≥2-verified-candidate rate) — consistent with warm users, by their training-period
definition, having more interaction history overall, which plausibly extends into the test period
too.

**What this table already shows, before any ranking number**: even with real coverage clearing the
statistical minimum, the *absolute* overlap between retrieval's fixed candidate lists and what
users were actually independently shown is small — 2–5% of retrieved slots, and well under 1% of
cold users ever have a genuine preference contrast to check. This is the honest, quantified shape of
the sparsity problem this whole project has tracked since the retrieval-stage distributional
analysis, now measured at the specific point where it matters for ranking evaluation.

## Part B: ranking-agreement result

| Cohort | Pairwise accuracy | Verified pairs | Users contributing |
|---|---|---|---|
| Cold | **0.5535** | 766 | 759 |
| Warm | **0.5298** | 4,209 | 3,479 |

**Reading this honestly**: both are above the 0.50 chance level, but only modestly — the pre-serve
model gets the clicked candidate ranked above the non-clicked one about 55% of the time for cold,
53% for warm, versus 50% for a coin flip. This is a weak, real, positive signal — not a strong
validation that this pipeline ranks well, and not a null result either. It sits between the two
outcomes the design doc anticipated ("meaningful lift" vs. "not statistically defensible") — a
third, more honest outcome: statistically computable, but small.

**The cold cohort's slightly higher accuracy than warm's is worth noting, not over-reading.** It
could reflect the pre-serve model generalizing more easily on the (typically simpler, more
popularity-driven) cold population, or could simply be noise at this pair count — 766 and 4,209
pairs are enough to clear the design's minimum-sample bar, but not enough to treat a ~2-point
percentage difference between cohorts as a confident finding on its own.

## What this does and does not demonstrate

**Demonstrates**: the pipeline (cohort classification → retrieval → pre-serve CTR scoring) runs
end to end, produces real, checkable outputs, and — on the only population this dataset can support
a check for — shows a small but real positive ranking signal, not zero and not fabricated.

**Does not demonstrate**: that this pipeline would perform well as a real served ranking system.
Per `ranking_integration_design.md`'s formulation 3, this dataset provides no serving propensities,
so no claim about *live* performance is possible, and a ~53–55% pairwise accuracy on a few hundred
to a few thousand verified pairs is a narrow diagnostic, not a production readiness signal.

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
