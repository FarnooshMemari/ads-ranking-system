# Ranking Stage — Problem Design

This document designs (does not implement) the ranking stage that follows retrieval in
[`attribution_modeling_design.md`](attribution_modeling_design.md)'s architecture. It resolves one
specific, critical issue before any ranking code is written: **this dataset has no
request/session/candidate-slate identifier**, so a naive "everything the user didn't click is a
negative" formulation would fabricate exposure that was never observed. Every recommendation below
is checked against that constraint first, and against the verified facts in
[`attribution_dataset_verification.md`](attribution_dataset_verification.md),
[`popularity_baseline.md`](popularity_baseline.md), and
[`collaborative_filtering.md`](collaborative_filtering.md), and the actual retrieval code in
`src/retrieval/` and `src/evaluation/retrieval_metrics.py`.

## The critical issue, stated precisely

A row in this dataset is `(timestamp, uid, campaign, cat1..cat9, click, conversion, ...)` — one
**actually-served impression**. There is no `display_id`/session/request field grouping multiple
campaigns that competed for the same slot (verified absent in the verification report; also absent
from DAC). This means:

- If campaign X was shown to user U and not clicked: **valid negative** — exposure is directly
  proven by the row's existence.
- If campaign Y was never shown to user U at all: **we know nothing** — Y could have been
  ineligible (targeting, budget, geography, advertiser's own decision), never selected by whatever
  auction/serving logic produced this log, or simply never reached in this 31-day sample. Treating
  this as "user U rejected campaign Y" is fabricating information the data does not contain.

Every formulation below is evaluated against whether it respects this distinction.

---

## Comparing the five formulations

### A. Impression-based ranking (group a user's real impressions into a list)

Group each user's **actually-observed** impressions (across their whole history, or within a
window) into a list, with click as the relevance label, and evaluate/train ranking metrics (e.g.
NDCG) over that list.

- **Label validity**: fully valid — every item in the list is a real impression with a real
  click/no-click outcome.
- **Negative-sampling validity**: no sampling needed or done — negatives are exactly the
  shown-but-not-clicked rows already in the data.
- **Temporal leakage risk**: low, if the list is built only from pre-cutoff impressions (same
  discipline as §6/§7 of the design doc).
- **Alignment with the data-generating process**: **weak** — the "list" is an artificial grouping
  across time. A user's three impressions at hour 2, day 9, and day 17 were never actually
  competing for the same slot; grouping them into one ranked list is a convenient evaluation
  construction, not a reconstruction of a real decision point. This is the same caveat the design
  doc already applies to retrieval-stage NDCG@K (§8) — this formulation inherits it, not introduces
  it.
- **Ability to evaluate ranking fairly**: yes, and it's the only way to get list-based metrics
  (NDCG) with zero invented negatives on this dataset.
- **Compatibility with LightGBM**: yes — `group`-based ranking objectives (`lambdarank`) grouped by
  `uid` work directly on this label set.
- **Usefulness for the TikTok Ads use case**: moderate — demonstrates the ranking-metric machinery
  honestly, but should never be presented as "we ranked competing candidates," because they
  weren't observed as competing.
- **Limitations**: given 2.68 average impressions/user, most "lists" have 1–3 items — too short for
  NDCG to be very discriminating for most users; only useful in aggregate.

### B. Retrieval-based sampled ranking (candidates from retrieval, labels from future interactions)

Take the retrieval stage's top-K output as candidates, then look at future interactions to label
them positive/negative.

- **Label validity**: **valid only for the subset of retrieved candidates that were *also* actually
  shown** to the user in the label window. For retrieved candidates never actually shown, there is
  no real outcome to attach — inventing one (e.g. "not clicked" because it never appears) is exactly
  the fabricated-negative problem this document exists to prevent.
- **Negative-sampling validity**: **invalid if done naively** (retrieved-but-unshown = negative);
  **valid if restricted** to `retrieved candidates ∩ actually-shown campaigns` for that user in the
  evaluation window.
- **Temporal leakage risk**: same as A, manageable with the same point-in-time discipline.
- **Alignment with the data-generating process**: better than a naive version, but still imperfect —
  this project's own retrieval policy (popularity/CF) is not the policy that actually generated the
  logged impressions, so the intersection with real logs is a form of off-policy evaluation without
  the propensity information that field normally requires. Not solved here — see §6 of this
  document.
- **Ability to evaluate ranking fairly**: yes, restricted to the safe intersection — this is the
  correct way to check "does the ranker improve ordering over what retrieval already found,"
  without claiming to have tested candidates that were never actually shown.
- **Compatibility with LightGBM**: yes, once labels are constructed from the safe intersection.
- **Usefulness for the TikTok Ads use case**: high, in the restricted form — this is the realistic
  shape of "evaluate the retrieval→ranking pipeline together" without pretending to have
  counterfactual outcomes this dataset can't provide.
- **Limitations**: the safe intersection may be small, since there's no real serving loop connecting
  this project's retrieval policy to what was actually logged — an explicit, acknowledged
  limitation (§9 of the design doc already excludes any off-policy/counterfactual evaluation claim
  for the same reason).

### C. Pairwise preference learning

Construct preference pairs (`item A preferred over item B`) only where the data supports it.

- **Label validity**: valid **only** when both items in a pair are real, observed impressions for
  the same user (e.g. a clicked campaign vs. a not-clicked campaign from that user's own history).
- **Negative-sampling validity**: same constraint as A — no fabricated pairs, since pairs are drawn
  from real rows only.
- **Temporal leakage risk**: low, with point-in-time discipline.
- **Alignment with the data-generating process**: **the weakest of the five options.** Classical
  pairwise learning-to-rank (the theory behind LambdaMART-style losses) assumes the two compared
  items were genuinely competing for the same slot — that assumption is exactly what this dataset
  cannot support (no slate ID). A pair built from two impressions served at different times is real
  data, but the "preference" it encodes is much weaker evidence than a same-slate comparison.
- **Ability to evaluate ranking fairly**: possible, but evaluation would inherit the same weak-pair
  caveat as training.
- **Compatibility with LightGBM**: yes — LightGBM's `lambdarank` objective is literally built for
  this (and is the same mechanism used in Option A when grouped by user). **C and A are the same
  underlying data with a different loss function**, not two independent options — worth naming
  precisely rather than presenting as unrelated.
- **Usefulness for the TikTok Ads use case**: limited on its own; useful only as an alternative loss
  function on top of A's data, not as an independent data formulation.
- **Limitations**: inherits A's short-list problem, plus the same-slate-assumption mismatch above.

### D. CTR prediction on observed impressions

Predict `P(click = 1 | impression)` directly, one row = one training example, using only real,
already-served impressions.

- **Label validity**: fully valid — the label is a directly observed outcome of a directly observed
  event. No candidate set, no slate, no sampling of any kind is required.
- **Negative-sampling validity**: **not applicable — there is no sampling.** Every row is real.
  This is the formulation with the least room for the false-negative problem to occur at all.
- **Temporal leakage risk**: fully addressed by the existing point-in-time feature discipline
  (§6/§7 of the design doc) — nothing new needed here.
- **Alignment with the data-generating process**: **exact match.** This is literally what the
  dataset is: a log of impressions with observed outcomes. No reconstruction or grouping is
  imposed on it.
- **Ability to evaluate ranking fairly**: this alone does not evaluate *ranking* (ordering a
  candidate set) — it evaluates classification quality (AUC, PR-AUC, LogLoss) on realized
  impressions. Getting from "a good classifier" to "a good ranker of retrieval candidates" requires
  the extension described in the Recommended Formulation section below.
- **Compatibility with LightGBM**: yes — this is exactly the binary classification setup already
  used for the DAC/OSS-side CTR notebook, so the project's existing LightGBM conventions
  (`src/models/ctr_model.py`, `configs/config.yaml`: `model.ctr`) transfer directly.
- **Usefulness for the TikTok Ads use case**: high — this is precisely how production CTR models
  are framed (score a candidate using learned, generalizable features, not a memorized per-pair
  label), and it's the only formulation with zero fabricated-negative risk.
- **Limitations**: by itself, doesn't produce a "ranking" output — needs to be paired with the
  retrieval stage's candidate lists to become one (see below).

### E. Hybrid (recommended — see full rationale below)

**D for labels and training (zero fabricated negatives) + A's grouping for ranking-metric
evaluation on real impressions only + the existing retrieval stage supplying candidates for the
model to score at serving time (not training time).** Not a new data formulation — a specific,
disciplined composition of A and D that avoids B's and C's weaknesses while keeping their
legitimate uses (A for NDCG-style evaluation, D for leakage-free training).

---

## Nine questions

### 1. Can we honestly call the next stage "ads ranking" with this dataset?

**Partially, and the boundary matters.** We can honestly build a CTR/CVR **scoring function** and
use it to **order** the retrieval stage's candidates — that is legitimately "ranking" in the sense
of producing an ordered list from a model. We **cannot** honestly claim to have learned from
observed competitive comparisons between candidates (classical slate-based ranking), because no
slate was ever logged. Any write-up of this stage must say "CTR/CVR-based candidate scoring," not
"we ranked competing ads" — consistent with the project's existing discipline of stating precisely
what a claim means (design doc §11's "no blanket personalization claim" is the same kind of
precision, now applied to ranking).

### 2. What exactly should one training example represent?

**One real, already-served impression**: `(uid, campaign, timestamp, cat1..cat9,
time_since_last_click, point-in-time user/campaign/pair aggregates) → click` (and, for the CVR
model, a second target restricted to the clicked subset — see Q3). Not a `(user, retrieval
candidate)` pair unless that pair also happens to correspond to a real logged impression.

### 3. What is a positive label?

- **CTR**: `click == 1` on a real impression row.
- **CVR**: `conversion == 1` (with the attributed-vs-loose distinction from design doc §5),
  restricted to the **clicked subset** of impressions. This is a refinement beyond what §5
  previously stated: since verification proved every `conversion=1` row is also `click=1` (no
  view-through conversions exist in this data), training a CVR model on the *full* impression
  population (rather than the clicked subset) would just be re-deriving a mix of CTR and CVR, not
  a clean conversion-given-click signal. CVR should be modeled as `P(conversion=1 | click=1)`,
  matching standard ad-tech practice and what the data actually supports.

### 4. What is a defensible negative label?

- **CTR**: `click == 0` on a real impression row — exposure is directly proven.
- **CVR**: `conversion == 0` on a real **clicked** impression row.
- **Never defensible**: a `(uid, campaign)` pair with no impression row at all, for any reason.
  This applies equally to candidates a future retrieval-scoring step might propose — those are
  scored, never labeled.

### 5. Which features are available at ranking time?

Reuses design doc §7's table (user-history, campaign, user-campaign-interaction features, all with
point-in-time discipline; `cost`/`cpo` excluded), with **one addition this document surfaces for
the first time**: there are two different "ranking time" moments, and they don't have the same
features available.

| Moment | `cat1`–`cat9` available? | Why |
|---|---|---|
| **(a) Offline, scoring an already-logged impression** (training, and the D-style evaluation above) | ✅ Yes | The impression already happened — its context fields are recorded on the row. |
| **(b) Scoring a retrieval-stage candidate *before* it has been served** | ❌ No | `cat1`–`cat9` describe the specific serving context (verified impression-level, not campaign-level, in §1 of the verification report) — a candidate that hasn't been served yet has no context fields to read, because the context doesn't exist until serving happens. |

**Implication**: a model trained with `cat1`–`cat9` as required inputs cannot be honestly applied
to score not-yet-served retrieval candidates — it would need those fields imputed or omitted at
that point, which is a different (weaker) feature set than what it was trained and evaluated with.
This document's Recommended Formulation section resolves this explicitly rather than leaving it as
a silent mismatch.

### 6. How should train/validation/test be constructed?

Identical to the already-established day-based split (design doc §6): train days 0–19, validation
20–24, test 25–30, `timestamp // 86400`. No new split logic is needed — reuse
`src.data.attribution.split_by_day`. The existing CVR right-censoring caveat (verification report
§5) applies unchanged: test-period CVR labels are the most censored and should be read as a lower
bound, not a clean comparison to train-period CVR.

### 7. How should retrieval and ranking be evaluated together?

As two separate measurements, not one blended score, for the same reason cohorts are reported
separately in retrieval evaluation (design doc §8):

1. **Retrieval alone** (already implemented): Recall@K/Hit Rate@K/NDCG@K per cohort, against
   test-period ground truth — unchanged by this document.
2. **Ranking alone**: PR-AUC/LogLoss/ROC-AUC on real test-period impressions (Option D), measuring
   whether the model's predicted probabilities discriminate and are calibrated — independent of
   what retrieval would have surfaced.
3. **Pipeline check** (Option B, restricted form): among test-period impressions whose campaign
   also appears in that user's retrieval-stage candidate list, does the ranker's ordering agree
   with real outcomes better than retrieval's own raw order (e.g. popularity rank)? This is the
   only place these two stages are evaluated together, and it must be restricted to the safe
   intersection described in Option B — never extended with fabricated outcomes for candidates
   that were retrieved but never actually shown. **Expect this intersection to be small** (there is
   no real feedback loop connecting this project's retrieval policy to what was actually logged);
   report its size honestly rather than treating a small, possibly noisy result as conclusive.

### 8. Can CTR and CVR both be modeled without leakage?

**Yes, using the discipline already built, plus two additions specific to CVR**:

- Both use only point-in-time features (§6/§7 of the design doc — unchanged).
- Both exclude `cost`, `cpo`, and (new here) `click_pos`, `click_nb`, `conversion_timestamp`,
  `conversion_id`, `attribution` as **inputs** — these are outcome-adjacent or outcome-only fields
  (verification report §2/§7); they may be used only to *construct* the label, never as a feature
  the model sees.
- **CVR-specific**: the 2,910 internally-inconsistent `conversion_id`s (verification report §6)
  must be handled with an explicit, documented rule (e.g. exclude them) before label construction —
  silently ignoring this known anomaly would let 0.67% of conversions be mislabeled in an
  undocumented way.
- **CVR-specific**: because one `conversion_id` can span multiple impression rows (multi-touch),
  the label for an *earlier* touchpoint must never be built using information only available from a
  *later* touchpoint of the same conversion journey (design doc §6, restated here because it
  applies directly to CVR label construction, not just feature aggregation).

### 9. Should the ranking model be trained on the Attribution dataset, the DAC dataset, or both?

**The Attribution dataset only.** This isn't a new decision — it follows directly from the scope
boundary already set when the two datasets were first compared: DAC has no `uid`/`campaign`
entities at all (verified in `criteo_dataset.md`), so it cannot support anything entity-aware —
cohorts, retrieval, or a ranking stage that scores per-user candidates. DAC remains the dataset for
the separate Microsoft Recommenders CTR notebook contribution, using the same *methodology*
(impression-level CTR prediction, Option D) but not the same trained model or joined data — design
doc §11 already rules out cross-dataset joint modeling, and nothing in this analysis changes that.

---

## Recommended Formulation

**Option D (impression-level CTR/CVR classification) for training, with two explicit extensions
that resolve the gaps above rather than leaving them implicit:**

1. **Train** two classifiers (CTR primary, CVR secondary per design doc §5) on real, train-period
   impressions only, using only features available at true pre-serve time (user-history,
   frozen campaign aggregates, user-campaign-pair history) plus `cat1`–`cat9`/
   `time_since_last_click` for the *offline* evaluation use case. CVR trained on the clicked
   subset only (Q3).
2. **Evaluate classification quality** (PR-AUC/LogLoss/ROC-AUC) on real test-period impressions —
   this alone already answers "is the model any good," with zero fabricated negatives, satisfying
   Q7's item 2.
3. **Evaluate ranking quality** two ways, both using only real, observed outcomes:
   - Option A-style: group a user's real test-period impressions and compute NDCG — honest about
     being a constructed grouping, not a real slate.
   - Option B-style, restricted: the safe-intersection pipeline check from Q7's item 3, sized and
     reported honestly.
4. **At serving time** (scoring retrieval candidates before they're served), use only the feature
   subset from Q5's row (b) — user/campaign/pair aggregates, no `cat1`–`cat9`. This should be a
   **separate, explicitly named inference path** (e.g. a `predict_for_candidate()` mode distinct
   from `predict_for_logged_impression()`), not a silent fallback, so the feature-availability gap
   from Q5 is enforced in code, not just documented.

This is chosen over a pure Option D (too narrow — never actually ranks anything) and over B or C as
primary formulations (both carry real fabricated-negative risk unless restricted to exactly what D
already provides safely). It reuses everything already built: the day-based split, the point-in-time
feature discipline, the cohort/retrieval stage as the candidate source, and the project's existing
LightGBM conventions from the DAC side.

---

## Known Limitations

- **No genuine slate-based ranking is possible with this dataset, at all** — stated once more,
  plainly, because it's the central fact this whole document is organized around. Every ranking
  metric produced here is either (a) classification-quality metrics on real impressions, or (b)
  ranking metrics computed on a constructed (not observed-simultaneous) grouping. Neither is a
  substitute for the other, and neither should be described as the other.
- **The pipeline check (Q7 item 3) may have very little data** — there is no real feedback loop
  between this project's retrieval policy and the logged impressions, so the safe intersection
  could be small enough to be indicative at best, not conclusive.
- **CVR modeling inherits the right-censoring caveat** from the design doc — test-period CVR labels
  undercount true conversions that hadn't had time to occur before the dataset's own collection
  cutoff.
- **The `cat1`–`cat9` availability gap (Q5) is a real, unresolved awkwardness**: a model that
  performs well offline (with context features) will necessarily perform differently when scoring
  not-yet-served candidates (without them) — the two feature sets are not interchangeable, and this
  document does not claim a fix, only an explicit, honest split between the two inference modes.
- **Catalog-size and subsampling caveats already established elsewhere carry over unchanged**: 675
  campaigns is a small catalog; observed CTR/CVR rates are inflated by undisclosed subsampling, not
  real-world base rates.

## Why This Formulation Is Appropriate for an Advertising Recommendation Prototype

A real ads-ranking system's ranking stage is, at its core, exactly Option D: a model that scores
"how likely is this candidate to be engaged with," trained on logged impressions, applied to score
new candidates using generalizable features rather than memorized per-pair outcomes — production
systems don't have "ground truth" for candidates that were never shown either; they face this same
gap and solve it the same way (learned, generalizable scoring, not fabricated labels). Building this
stage honestly — including explicitly surfacing where slate-based ranking *isn't* possible, and
where the feature set changes between offline evaluation and candidate scoring — demonstrates the
exact skill this portfolio project exists to show: recognizing what a dataset can and cannot support
before building on top of it, rather than defaulting to a textbook formulation that happens not to
fit.

## Recommended First Ranking Experiment

**Train a single LightGBM binary classifier for CTR** (`P(click=1 | impression)`) on train-period
Attribution impressions, using the feature set from Q5 row (a) minus `cost`/`cpo`, evaluated with
PR-AUC/LogLoss/ROC-AUC on test-period impressions (Q7 item 2) — the narrowest possible first slice
of the Recommended Formulation, deliberately excluding CVR and the two ranking-metric evaluations
(A-style and B-style) from this first pass. This mirrors how the retrieval stage was built
incrementally (popularity baseline before collaborative filtering): get one clean, leakage-checked,
honestly-evaluated classifier working end to end first, confirm it behaves sensibly (calibration,
feature importances consistent with what's plausible given `cat1`–`cat9`'s undisclosed semantics),
*then* add CVR and the ranking-specific evaluations in a follow-up step — not because they're hard,
but because bundling them into the first experiment would make it harder to tell which piece is
responsible if something looks wrong.
