# Retrieval → CTR Ranking Integration — Design

This document designs (does not implement) how the two already-validated, separately-built
components — retrieval (`src/retrieval/`, [`popularity_baseline.md`](popularity_baseline.md),
[`collaborative_filtering.md`](collaborative_filtering.md)) and the CTR model
(`src/models/ctr_model.py`, [`ctr_model.md`](ctr_model.md)) — can be legitimately combined, given
the same constraint that has governed every design decision so far
([`attribution_modeling_design.md`](attribution_modeling_design.md),
[`ranking_problem_design.md`](ranking_problem_design.md)):

**This dataset has no request/session/candidate-slate ID. An unobserved `(user, campaign)` pair is
unknown, never a negative.** Every option below is checked against that constraint before anything
else.

## What's already validated, reused here without re-deriving it

- **Retrieval**: cohort-segmented (cold: popularity; warm: CF, which did not beat popularity once
  seen campaigns were excluded — see `collaborative_filtering.md`). Cold Recall@10 = 0.1959
  (n=693,737 test users), warm Recall@10 = 0.1731 (n=56,353), both against test-period ground
  truth, popularity top-10 fixed and identical for every user in a cohort.
- **CTR model**: LightGBM, PR-AUC 0.6069 / ROC-AUC 0.7209 / LogLoss 0.5816 on real, logged
  test-period impressions, well-calibrated, trained on the feature set in `ctr_model.md` — which
  **includes `cat1`–`cat9`**, fields verified (`attribution_dataset_verification.md` §1;
  `ranking_problem_design.md` Q5) to exist only *after* an impression has actually happened. This
  fact is the crux of the integration problem and is addressed explicitly below.

---

## Comparing three integration formulations

### 1. Offline evaluation only on actually observed impressions

Score real, already-logged impressions with the CTR model; report classification metrics. This is
**exactly what `ctr_model.md` already did** — not a new formulation, the reference point the other
two are compared against.

- **What can be measured**: discrimination and calibration quality of the scoring function itself
  (PR-AUC, ROC-AUC, LogLoss, calibration) — already measured, real numbers exist.
- **What cannot be measured**: anything about ordering a *set* of candidates. There is no list here
  — each scored row is independent. This formulation answers "is the classifier good," not "is the
  ranking good," and was never claimed to answer the second question (`ranking_problem_design.md`
  Q1 already drew this line).
- **Assumptions required**: none beyond what `ctr_model.md` already validated.
- **Leakage risk**: none — already enforced (point-in-time features, excluded post-outcome
  columns).
- **Is NDCG/ranking metrics defensible?** **Not applicable in this narrow form** — there is no list
  structure to rank. (`ranking_problem_design.md`'s Option A — grouping a user's own real
  impressions into a list — is a *different*, already-analyzed formulation that could produce an
  NDCG number from this same data; it is not what "option 1" as posed here is asking about, and is
  cross-referenced rather than repeated.)

### 2. Retrieval-generated candidate analysis, restricted to observed exposure

Take retrieval's actual output (the fixed popularity top-10, or the warm-cohort CF top-10) for each
user, then restrict to the subset of those campaigns that **also** correspond to a real, separately
logged test-period impression for that same user — whether clicked or not — and check whether the
CTR model's predicted ordering agrees with the real, observed outcomes on that verified subset.

- **What can be measured**: for users who have **two or more** retrieved campaigns independently
  confirmed as real impressions, a local ranking check — e.g. did the model score the one that was
  actually clicked above the one(s) that weren't. This is a genuinely evidence-backed comparison:
  every item in it is a real row with a real outcome.
- **What cannot be measured**: overall recall/precision of "what the model would have served" in a
  live system — that requires a real feedback loop connecting this project's retrieval choice to
  what actually gets shown, which doesn't exist here (this project never served anything; it only
  analyzes an already-fixed log). This is the same off-policy evaluation gap already excluded from
  scope in the retrieval design work.
- **Assumptions required**: that users with two-plus verified-exposed retrieved candidates are not
  wildly unrepresentative of the broader population. They likely are *somewhat* unrepresentative —
  by construction, they are users who happened to receive multiple impressions concentrated on
  already-popular campaigns, which correlates with simply being shown more ads overall — so any
  conclusion here generalizes to "verified multi-exposure users," not to the full cohort. This must
  be stated as a limitation, not glossed over.
- **How big is this verified subset, honestly estimated from numbers already in hand?** Recall@10
  itself already measures part of this: cold Recall@10 = 0.1959 over 693,737 users implies roughly
  135,900 cold users have their one *known positive* inside the popularity top-10 — a large enough
  raw count to work with for the "at least one verified item" case. But a **local ranking check**
  needs a *second* verified item (clicked or not) from the same retrieved list, which recall alone
  doesn't tell us. Popularity concentration works partly in our favor here: the top-10 campaigns by
  clicks already capture ~20.17% of all click volume (verified in the retrieval-stage sparsity
  analysis), so a nontrivial share of *any* user's real impressions — not just their click — will
  coincide with the popularity list purely because those campaigns are shown so often. Still, given
  the base rate of 2.68 impressions per user overall, the number of users with **two or more**
  verified, retrieved-and-actually-shown campaigns is very likely a small fraction of the already-
  filtered cohort populations. This must be measured, not assumed, before relying on it — flagged
  explicitly in the proposed experiment below rather than assumed away.
- **Leakage risk**: none new, if discipline already in place is kept — retrieval's candidate list
  is still computed from train-period data only; any CTR-model features used to score the verified
  test-period rows still follow point-in-time discipline. The one *new* risk to guard against:
  don't let "was this campaign retrieved" (a train-period-derived fact) or "was it verified as
  shown" (a test-period fact used only to *select which rows to check*, never as a *feature*)
  leak into the feature vector itself — a discipline note for the smallest-experiment section below.
- **Is NDCG/ranking metrics defensible?** **Only in a heavily restricted sense** — NDCG computed
  over the tiny verified sub-list (not the full top-10), for the subset of users where such a
  sub-list has 2+ items. Legitimate, but should be reported as a narrow diagnostic with its sample
  size stated prominently, not presented as a robust benchmark.

### 3. Serving-time conceptual architecture (retrieval → CTR scoring), offline limits acknowledged

Describe (and, in a future step, build) the actual inference-time flow: classify a user's cohort →
retrieve candidates (popularity or CF, per cohort) → score each candidate with the CTR model → sort
by predicted CTR → return the ranked list. This is a real, buildable pipeline, not a data
formulation — but **it surfaces a concrete blocker not fully resolved until now.**

- **What can be measured**: each stage's own, already-validated metrics (retrieval's per-cohort
  Recall@K/Hit Rate@K; the CTR model's PR-AUC/ROC-AUC/LogLoss on logged impressions) — nothing new
  here beyond what's already built. Additionally, a **leakage-free, outcome-independent diagnostic**
  is legitimate and useful: does the CTR-model-scored order actually *differ* from plain popularity
  order for a sample of retrieved lists? This requires no ground truth at all (it's a property of
  the model, not a claim about quality) and is safe to compute.
- **What cannot be measured**: whether the re-ordering actually *improves* outcomes relative to
  serving in popularity order. That would require either real A/B testing (no live system exists),
  the restricted check in Option 2 (small-sample, as discussed), or an off-policy/counterfactual
  estimator (e.g. inverse propensity scoring) — which requires knowing the real serving system's
  selection propensities. **This dataset does not provide them** (verified nowhere in the schema or
  documentation), so proper off-policy evaluation is not constructible here, not just deprioritized.
- **A blocker this analysis surfaces concretely, not just in the abstract**: the CTR model
  currently deployed in `ctr_model.md` was trained *with* `cat1`–`cat9`, and per its own feature
  importance table, several of those fields rank in the top 10 by gain (`cat8`, `cat3`, `cat6`,
  `cat9`, `cat7`, `cat1`) — a meaningful share of its predictive power. But `cat1`–`cat9` **do not
  exist for a not-yet-served retrieval candidate** (`ranking_problem_design.md` Q5). This means the
  *existing trained model is not directly usable* for genuine pre-serve candidate scoring — using
  it anyway (e.g. imputing zeros or NaN for the missing context) would silently evaluate it outside
  the feature distribution it was trained and validated on, a real train/serve skew risk, not a
  hypothetical one. This must be resolved with a **separate, cat-free model variant** before any
  serving-time architecture can be honestly exercised, even conceptually.
- **Assumptions required**: that scoring ≤10–20 retrieved candidates per user with LightGBM at
  serve time is computationally trivial (true — no scale concern here, consistent with this
  project's prototype framing).
- **Leakage risk**: the train/serve feature mismatch above *is* a leakage-adjacent risk in the
  sense that it's an unacknowledged distribution shift, not classical label leakage — still worth
  naming precisely rather than lumping it in with the dataset's other leakage risks.
- **Is NDCG/ranking metrics defensible?** **No, not for a real quality claim** — there is no ground
  truth for the full candidate set once we're purely conceptual/serving-time. The only legitimate
  measurement here is the ordering-difference diagnostic above, which is not NDCG and does not
  claim to measure quality.

---

## Summary table

| | 1. Offline-observed-only | 2. Retrieval ∩ observed | 3. Serving-time conceptual |
|---|---|---|---|
| Measures | Classifier quality | Local ranking agreement, small verified sample | Ordering-difference diagnostic only |
| Cannot measure | Any ranking/ordering quality | Full-list recall/precision; live-system outcomes | Real quality improvement (no propensities) |
| New assumptions | None | Verified-subset users are representative of "engaged" users only | Serving is computationally feasible (true, trivial here) |
| Leakage risk | None (already enforced) | None new, if discipline kept | Train/serve feature mismatch (`cat1`–`cat9`) — real, must be fixed first |
| NDCG defensible? | Not applicable (no list) | Only on tiny verified sub-lists | No — diagnostic only, not a quality claim |

---

## Smallest technically defensible integration experiment

Two parts, in this order, because the second depends on the first existing:

### Part A — a `cat1`–`cat9`-free CTR model variant

Retrain `CTRModel` using only the feature subset genuinely available before an impression happens:
`campaign`, `user_*` history, `pair_*` history, `campaign_*_train` aggregates —
i.e. `FEATURE_COLUMNS` minus the nine contextual fields, already a strict subset of what's
implemented (no new feature engineering required, only a different column list passed to the
existing pipeline). Evaluate it with the exact same offline protocol already validated in
`ctr_model.md` (PR-AUC/ROC-AUC/LogLoss/calibration on real test-period impressions) — this alone
answers a real, previously-unmeasured question: **how much predictive power is lost without
post-serve context**, and whether what's left is still usable at all for candidate scoring. Fully
leakage-safe, reuses existing, already-tested infrastructure, requires no new labels or candidate
construction — the lowest-risk piece of this whole integration.

### Part B — the restricted verified-intersection check (Option 2), using Part A's model

Using the cat-free model from Part A (the only one actually valid for scoring retrieval
candidates): for each cohort's retrieved top-10, identify test-period users with **two or more**
of those campaigns independently confirmed as real logged impressions, score them, and check local
ranking agreement against real outcomes. **Report the size of this verified population first,
before reporting any metric computed from it** — if it turns out too small to be informative (a
real possibility flagged above, not hidden), that finding itself is the honest result of this
experiment, not a failure to paper over.

### Explicit fallback, stated up front rather than discovered after the fact

If Part B's verified population is too small for a meaningful metric (plausible, given 2.68
average impressions/user), **the honestly-demonstrable deliverable is Part A alone, plus a
qualitative, outcome-free demonstration**: run the retrieval → cat-free-CTR-scoring pipeline for a
handful of example users, show the resulting ranked candidate list next to plain popularity order,
and report the ordering-difference diagnostic from Option 3 — explicitly labeled as "the pipeline
runs and reorders candidates using learned features" with **no claim that the reordering is
better**, since that claim is exactly what this dataset cannot support offline. This is not a
consolation prize; it is the technically honest ceiling for this dataset, stated as such.

## Does this dataset support a valid offline ranking evaluation?

**Not a full one.** A statistically powerful, representative, end-to-end "does ranking beat
retrieval-alone" metric is not achievable here — no slate ID, no serving propensities, and severe
per-user sparsity combine to make it structurally unavailable, not merely difficult. What *is*
achievable, and is what Part A + B (with its explicit size-first reporting and fallback) delivers:
a validated feature-ablated CTR model appropriate for candidate scoring, and an honest, scoped,
possibly-small ranking-agreement check on the only population this dataset can support one for —
reported as exactly what it is, not inflated into a claim the data doesn't back.

## Not implemented in this document

Per instructions — this is a design document only. No new model was trained, no new features were
built, no experiment was run. `src/retrieval/`, `src/models/ctr_model.py`, and
`src/features/attribution_features.py` are unchanged by this document.

**Update**: Parts A and B have since been implemented and run against the full dataset. Both
cohorts' verified-intersection populations cleared the documented minimum sample size (1,091 cold,
3,479 warm) — the "too small to be meaningful" contingency did not occur, though coverage itself is
small (roughly 2–5% of retrieved slots). (The cold figure was corrected from an initially published
759 after a population-coverage bug — not leakage — was found and fixed; see
[`ranking_integration_results.md`](ranking_integration_results.md)'s Correction section.) Results:
[`pre_serve_ctr_model.md`](pre_serve_ctr_model.md) (Part A) and
[`ranking_integration_results.md`](ranking_integration_results.md) (Part B and overall conclusion).
