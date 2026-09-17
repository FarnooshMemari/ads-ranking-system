# Attribution Dataset — Recommendation System Design

This document designs (but does not implement) the recommendation-system component of the
personal portfolio project, built on the **Criteo Attribution Modeling for Bidding Dataset**
(`criteo/criteo-attribution-dataset`, CC BY-NC-SA 4.0). This dataset is used **only** for this
component — the Microsoft Recommenders open-source contribution remains scoped to Criteo DAC
(sample) and is unaffected by anything in this document.

**Verified schema**: 30.92 days of Criteo live traffic, one row per impression, 16,468,027
impressions / 435,810 distinct conversions / **675** campaigns (the documentation's "700" is a
rounded approximation), tab-separated fields `timestamp, uid, campaign, conversion,
conversion_timestamp, conversion_id, attribution, click, click_pos, click_nb, cost, cpo,
time_since_last_click, cat1–cat9`. Data is sub-sampled and anonymized (Criteo's own disclosure) —
observed rates are not claimed to be true production base rates, the same caution already
documented for DAC (and confirmed materially true here too: observed CTR is 36%, far above
real-world rates — see the verification report).

**Update (schema verification)**: every item originally marked ⚠️ UNVERIFIED below has since been
checked directly against the downloaded dataset. Full evidence, computed statistics, and everything
that remains genuinely unresolved are in
[`attribution_dataset_verification.md`](attribution_dataset_verification.md).

**Update (architecture reassessment)**: a second, deeper verification pass — computing full
per-user and per-campaign *distributions*, not just averages — found that the pure
collaborative-filtering retrieval design originally proposed in §4 is **not statistically
defensible** as the primary method. See §3B below for the evidence and §4 for the revised
approach. This is a real correction to the architecture, not a cosmetic update — flagged
prominently here so it isn't missed.

---

## 1. Entity Model

| Entity | Identifier | Definition | Notes |
|---|---|---|---|
| **User** | `uid` | An anonymized, persistent user identifier. | No demographic/profile fields exist. Anything known about a user must be *derived* from their own interaction history — there is no separate user table. |
| **Campaign** | `campaign` | An advertiser's campaign, the recommendable unit in this project. | **675** distinct values (verified by direct count). No separate campaign metadata table (name, advertiser, category) exists — `campaign` is an opaque ID, like `uid`. |
| **Impression** | (no explicit ID; a row) | One banner display event: `(timestamp, uid, campaign, cat1..cat9, cost, ...)`. | The base unit of exposure and the unit of the raw dataset. |
| **Click** | `click` (0/1 on the impression row) | Whether that specific impression was clicked. | Directly observed, no ambiguity in the label itself. |
| **Conversion** | `conversion` (0/1), `conversion_id`, `conversion_timestamp`, `attribution` | Whether a purchase occurred within 30 days after the impression. | **Important nuance**: per the dataset's own description, `conversion=1` is set "independently of whether this impression was last click or not" — meaning a single real-world conversion (one `conversion_id`) can be linked to **multiple** impression rows (multi-touch). `attribution=1` marks the subset Criteo's own model credited as causally responsible. These are not the same signal — see §5. |
| **Timestamp** | `timestamp` | Relative time (seconds, starting at 0 for the first impression), dataset sorted by it. | **Not** a calendar timestamp — no day-of-week/date/holiday features are derivable. Only relative ordering and elapsed-time arithmetic are available. |
| **Campaign/context features** | `cat1`–`cat9` | Anonymized categorical features "associated to the display." | **✅ Verified empirically**: none of `cat1`–`cat9` is constant within every campaign (checked directly by grouping all 16.46M rows by `campaign`); `cat5` is a partial exception (constant for 91.11% of campaigns, low cardinality) but not reliably so. Official semantics remain undisclosed — this is an empirical safety check, not a semantic conclusion. Treated as **impression-level contextual features only**. Full breakdown in the verification report. |

### Campaign-level vs. ad-level recommendation

This project explicitly recommends **campaigns, not ad creatives**. There is no field
identifying an individual banner/creative — `campaign` is the finest granularity available. Every
document, metric, and claim in this design refers to "campaign-level recommendation." This is a
real scope limitation (a production system would typically rank creatives within a campaign too)
and is stated here so it is never silently upgraded to an "ad recommendation" claim later.

---

## 2. Interaction Definition

| Interaction | Defined as | Signal density | Use |
|---|---|---|---|
| **Impression** | One row: `(uid, campaign, timestamp)` was shown. | 16.5M rows | The exposure population — the denominator for CTR/CVR, and the source of "what has this user seen." |
| **Click** | `click == 1` on the impression row. | **5,947,563** clicks (verified count) — CTR = 36.1%, far above real-world rates, consistent with the dataset's disclosed (but unspecified) sub-sampling. | Candidate positive signal for retrieval and the primary ranking-stage target (§5). |
| **Conversion** | `conversion == 1`, deduplicated by `conversion_id`. | **435,810** distinct conversions (verified; the row-count of `conversion=1` is 806,196, higher because 32.31% of conversions span multiple impression rows). | Candidate positive signal for retrieval and the (secondary, sparse) ranking-stage target (§5). |

**✅ Verified**: `click_pos`/`click_nb` are populated **only** when a row is both `click=1` and
`conversion=1` (806,196 rows) — for the other 5,141,367 clicked-but-not-converted rows, both
fields are the `-1` sentinel, with zero exceptions found. They are conversion-path-specific
features, exactly as hypothesized, now confirmed rather than assumed. **Additional confirmed
finding**: every `conversion=1` row is also a `click=1` row in this dataset — there are no
view-through (unclicked) conversions here, correcting an earlier open reading of "independently of
whether this impression was last click or not" as potentially allowing them. Full evidence in the
verification report.

### Should retrieval use clicks, conversions, or both?

| Option | Pros | Cons |
|---|---|---|
| **Click-only positives** | Abundant (learnable embeddings); no multi-touch dedup complexity; consistent with how most implicit-feedback recommenders (ALS/BPR) are trained. | Weaker signal of true preference — a click can be curiosity, not intent. |
| **Conversion-only positives** | Strongest business signal. | 435,810 distinct conversions across 675 campaigns and 6,142,256 unique users (verified) — confirmed too sparse for a reliable user×campaign interaction matrix at prototype scale; only 5.34% of users have even one conversion. |
| **Both, conversion-weighted** | Uses all available signal; conversions can up-weight the interactions that matter most. | Adds a design/tuning knob (relative weight) that can't be justified without first seeing empirical density — premature at design stage. |

**Recommendation**: use **click as the positive implicit-feedback signal for the retrieval
stage**, and reserve **conversion (deduplicated by `conversion_id`) as the ranking-stage target**,
not a retrieval input. This is a density-driven choice, not a value judgment — it should be
revisited once the actual click/conversion counts are inspected at implementation time.

---

## 3. Recommendation Formulation: User → Campaign

- **User history** (at a given point in time `t`): the set of that `uid`'s impression rows with
  `timestamp < t`, along with their `click`/`conversion` outcomes. History is always defined
  relative to a specific `t` — there is no single, static "user profile" (see §6 for why this
  matters). **⚠️ New finding from verification, not one of the original four items**: average
  impressions per user across the whole dataset is only **2.68** (6,142,256 users / 16,468,027
  rows), and only 0.1885% of all possible `(uid, campaign)` pairs are ever observed. For most
  users, "history" will be one to a handful of rows — see the feasibility flag in §4.
- **Candidate universe**: the set of `campaign` values. In principle all 675 (verified), but practically
  restricted to campaigns with a minimum impression volume in the training window (to exclude
  campaigns with no learnable signal — a reasonable simplification, not a claim that inactive
  campaigns don't exist).
- **Positive interaction**: a `(uid, campaign)` pair where that user clicked on an impression of
  that campaign at some `timestamp < t`.
- **Negative / non-interaction example**: this dataset offers two genuinely different kinds of
  "negative," which is worth distinguishing explicitly because most implicit-feedback benchmarks
  (e.g. MovieLens-style data) don't have this option:
  1. **Shown-but-not-clicked**: the user *was* exposed to the campaign (a real impression row
     exists) but didn't click. This is an *informative* negative — real disengagement, observed
     directly.
  2. **Never shown**: no impression row exists for that `(uid, campaign)` pair. This is the
     standard implicit-feedback assumption (absence ≠ dislike, just "unknown").

  **Recommendation**: treat (1) as a stronger, explicit negative signal and (2) as the standard
  implicit-feedback unlabeled/weak-negative set used for negative sampling during training
  (consistent with how ALS/BPR-style models are trained in Microsoft Recommenders itself). This
  distinction is a genuine advantage of impression-logged data over pure ratings data and should
  be used, not discarded.

---

## 3B. Sparsity & Collaborative-Filtering Feasibility Reassessment

This section is a **verified-fact-driven correction** to §4 as originally written. All numbers
below were computed directly from the full 16,468,027-row dataset (`uid`, `campaign`, `click`,
`conversion` columns), not estimated or assumed. They are clearly separated from the modeling
assumptions and proposed methodology that follow.

### Verified facts: user-history sparsity

| Statistic | Value |
|---|---|
| Median impressions/user | **1** |
| Mean impressions/user | 2.68 |
| Users with exactly 1 impression ever | 51.69% |
| Users with 1–2 impressions | 71.07% |
| Users with 6+ impressions | 10.19% |
| Users with 0 clicks ever | 54.19% |
| Users with exactly 1 click | 28.46% |
| Users with **distinct campaigns seen == 1** | **79.20%** |
| Users with **distinct campaigns clicked == 0** | 54.19% |
| Users with **distinct campaigns clicked == 1** | 40.41% |
| **Users with distinct campaigns clicked ≥ 2** | **5.39%** (331,297 of 6,142,256) |
| Users with distinct campaigns clicked ≥ 3 | 0.81% |

**Why "distinct campaigns clicked ≥ 2" is the load-bearing number**: collaborative filtering (and
any personalization method) works by learning a *relative* preference — this campaign over that
one — for a given user. A user who has clicked on only one campaign (or none) provides no relative
signal; there is nothing to triangulate a preference against. **94.61% of all users in this
dataset have no basis for a learnable personalized preference at all.** This is not a modeling
limitation — it is a structural property of the data.

### Verified facts: campaign-side density (for contrast)

| Statistic | Value |
|---|---|
| Median impressions/campaign | 10,357 |
| 10th-percentile campaign, impressions | 2,544 |
| Min impressions for any campaign | 20 |
| Top 20 campaigns' share of all clicks | 29.98% |
| Top 100 campaigns' share of all clicks | 63.16% |
| Gini coefficient, campaign click counts | 0.66 |

Campaigns are data-dense (every campaign has thousands of impressions at minimum) and moderately,
not extremely, concentrated (a long tail beyond the top 100 still gets over a third of clicks).
**The asymmetry is stark and important**: campaigns have abundant aggregate signal; individual
users almost never do.

### Verified fact: even the "warm" cohort barely diverges from population-level popularity

For the 331,297 users (5.39%) who *do* have ≥2 distinct clicked campaigns — the only population for
whom personalization is even theoretically definable — **26.99% of their clicks are on the globally
top-20-by-clicks campaigns**, versus **29.98% for the overall click population**. These two numbers
are close enough that there is no strong evidence the "warm" cohort's taste diverges meaningfully
from simply following the crowd. This does not prove personalization is worthless for this group,
but it means it cannot be assumed to help — it has to be measured, and the prior expectation should
be modest.

### What this does and does not prove (fact vs. assumption)

- **Proven**: the data cannot support a general-purpose, per-user collaborative-filtering retrieval
  system as the *primary* mechanism — the eligible population (5.39%) is too small, and even within
  it, the personalization signal (if any) is weak relative to popularity.
- **Not proven, and not claimed**: that collaborative filtering produces *zero* value for the warm
  cohort. That is an empirical question deferred to implementation (§8's evaluation design is
  built specifically to answer it rigorously, without conflating it with the cold majority).
- **A modeling assumption going forward** (stated as such, not as fact): campaign popularity is
  treated as the safe, always-available default signal, because it is dense and stable; any
  personalization layered on top is treated as an unproven enhancement to be validated per cohort,
  not assumed.

---

## 4. Candidate Generation (Retrieval Stage)

**Revised approach (supersedes the original pure-CF design, per §3B's evidence): a segmented,
cold-start-aware retrieval policy, not a single collaborative-filtering model.** Users are split
into two populations using only their pre-prediction-time history, and each gets a different,
honestly-scoped retrieval strategy:

| Population | Definition | Share of users (verified) | Retrieval strategy |
|---|---|---|---|
| **Cold** | 0 or 1 distinct clicked campaigns in history | 94.61% | Campaign popularity (training-period click counts), optionally adjusted by the single campaign the user *has* seen (e.g. mild frequency boost to campaigns whose available `cat_i` context loosely overlaps with what little the user has been shown), as a modeling enhancement — not collaborative filtering, since there is no relative-preference signal to learn. |
| **Warm** | ≥2 distinct clicked campaigns in history | 5.39% | Implicit-feedback collaborative filtering (matrix factorization over this cohort's `uid × campaign` click matrix), evaluated *against* the popularity baseline for this same cohort — not assumed to win (§3B's popularity-overlap finding suggests it may not, by much). |

This is **option E** in the comparison below (a segmented policy, not a single blended formula) —
chosen because it is auditable (each population's strategy and its evaluation are reported
separately, per §8) and because it matches how production cold-start systems are actually built,
rather than assuming one model form fits a population that is 94.6% sparse and 5.4% workable.

### Comparison of candidate-generation formulations

| # | Formulation | Verdict given verified density | Why |
|---|---|---|---|
| A | **Pure implicit collaborative filtering** (original design) | **Rejected as primary** | 94.61% of users have no relative-preference signal (§3B); the eligible 5.39% show clicks barely more concentrated toward non-popular campaigns than the general population. A single per-user CF model applied to everyone would be fitting noise for the vast majority. |
| B | **Popularity / personalized popularity** | **Adopted for the cold population (94.6%)** | Campaign-side data is dense and stable (median 10,357 impressions/campaign); this is the only signal that reliably exists for almost every user. "Personalized" here can mean at most a light adjustment from the 0–1 campaigns a cold user has actually seen — not a learned embedding. |
| C | **Content/context-based retrieval** (using `cat1`–`cat9`) | **Rejected as a standalone method** | §1 already established no `cat_i` is a reliable static campaign attribute, so there is no stable per-campaign content profile to retrieve against. A weak, explicitly-labeled contextual nudge is usable as an enhancement (folded into B above), not as its own retrieval mechanism. |
| D | **Hybrid (blended score across signals)** | **Partially adopted, in segmented form** | A single blended formula (e.g. `α·popularity + β·CF_score`) was considered but rejected in favor of E below — a blend risks quietly averaging away the fact that CF is only meaningful for 5.4% of users; a single α/β would be tuned mostly by the cold majority regardless of what value it has for the warm minority. |
| E | **Segmented, cold-start-aware retrieval policy** (adopted) | **Adopted as the primary architecture** | Explicitly branches strategy by measured user warmth instead of assuming one model fits all. Auditable: each population's approach and evaluation are reported separately (§8), so a claim like "personalization helps" can be checked, not assumed, for the only cohort where it could possibly be true. |

**Why not a full ANN/vector-index retrieval setup**: with only 675 campaigns (verified count),
brute-force scoring is computationally trivial for either population's strategy — an ANN index
would add infrastructure the catalog size doesn't justify. Unchanged from the original design.

**What the retrieval model(s) optimize**:
- *Cold population*: no model is being fit to individual users — this is a frozen,
  training-period-computed popularity ranking (optionally context-adjusted). There is nothing to
  "learn" per user because, per §3B, there is no relative-preference signal available for 94.6% of
  users.
- *Warm population*: an implicit-feedback objective over the `(uid, campaign)` click matrix,
  restricted to the 331,297-user warm cohort — reconstructing observed positive (clicked) pairs
  while pushing down non-interacted pairs (ALS-style weighted factorization or BPR-style pairwise
  loss; the choice between them is an implementation-time detail, not design-blocking). Crucially,
  this model's value is **not assumed** — §8 defines exactly how its lift over the cohort's own
  popularity baseline is measured before any claim is made about it working.

**What information is available at recommendation time**:
- The user's own click/impression history strictly before the prediction timestamp, used first to
  classify them cold vs. warm, then (for warm users) to place them in the collaborative-filtering
  embedding space.
- The full candidate campaign set and training-period popularity/embeddings.
- **Not available as retrieval input**: `cat1`–`cat9` as *campaign* descriptors, because §1
  confirmed none is reliably constant per campaign — they remain impression-level-only signal, and
  any use of them here (e.g. the cold-population context adjustment above) is a coarse, explicitly
  tentative enhancement, not a claim of genuine content-based retrieval (see §5, option C).

---

## 5. Ranking Problem

**Do not treat this as a foregone conclusion — comparing options against the actual fields:**

| Candidate target | Field(s) | Density | Caveats |
|---|---|---|---|
| **CTR** | `click` | High — 36.1% observed (5,947,563 / 16,468,027, verified) | Not itself delayed/windowed — clicks appear to be immediate, so less exposed to the right-censoring issue in §6. Note the rate itself is inflated by undisclosed sub-sampling, not a real-world base rate. |
| **CVR (loose)** | `conversion` | Low relative to CTR — 2.65% of impressions / 7.33% of clicks (435,810 distinct conversions, verified) | Includes impressions within a 30-day window "independently of whether this impression was last click" — a loose, multi-touch-inclusive positive. Verified: every `conversion=1` row is also `click=1`, so this is not a view-through signal — it overstates causal role only in the sense of counting non-last-click touchpoints, not unclicked exposure. |
| **CVR (attributed)** | `conversion == 1 AND attribution == 1` | Lower still than loose CVR | Stricter, causally cleaner positive (Criteo's own attribution model's credited touchpoints) — better construct validity, worse data density. |
| **Multi-objective (utility) score** | combination of predicted CTR, predicted CVR, and `cost`/`cpo` | Depends on above | This is explicitly one of the dataset's stated intended uses ("the data includes cost and value used for computing Utility metrics") — not an invented use case. |

**Recommendation** (direction, not a formula — per instructions, no scoring formula is fixed
here): **CTR is the primary ranking-stage target**, given its far greater data density and lower
exposure to the temporal right-censoring problem in §6. **CVR is a secondary target**, trained
using the **attributed** definition (`conversion AND attribution`) as the primary CVR label for
better causal validity, with the looser `conversion`-only definition reported alongside for
comparison, not used as the primary label. A blended, utility-style score (direction (c) above) is
the intended eventual combination — but the actual weighting between pCTR, pCVR, and cost/value
cannot be responsibly fixed before empirically inspecting real positive rates and score
calibration at implementation time; committing to a formula now would be exactly the kind of
premature, unsupported design choice this document is meant to avoid.

**Critical exclusion from ranking-model *inputs***: `cost` and `cpo` should still **not** be used
as predictive features for the CTR/CVR ranking model — but the reasoning below is corrected from
the original version of this document, which stated the `cpo` claim as settled fact when it was
actually an unverified assumption:

- **Corrected**: `cpo` is *not*, in fact, "defined only when a conversion has already occurred" —
  this was directly disproven by verification: `cpo` is populated with a nonzero value on **100%**
  of rows, including all 15,661,831 non-converting impressions (mean `cpo` on non-converting rows
  is actually *higher* than on converting ones: 0.201 vs. 0.110). The dataset's documentation
  itself is imprecise here relative to the actual file. What remains true, and is the actual reason
  for exclusion: the dataset does not document *when* `cpo` was computed relative to the
  impression-serving decision, so whether it's a legitimate pre-decision feature or something
  computed after the fact is genuinely unknown — exclusion is a precaution against undocumented
  timing, not proven post-hoc leakage.
- `cost` is the price Criteo paid for the display. Whether this was knowable *before* the ranking
  decision or is itself a byproduct of it is **not addressed by the dataset's documentation at
  all** — this reasoning was, and remains, based on general real-time-bidding domain knowledge,
  not a fact this dataset proves. Both `cost` and `cpo` are populated on 100% of rows with no
  missing values, which is a proven fact; their pre/post-decision timing is not.

Both are still kept out of the feature set and used only in **evaluation** (as utility/ROAS-style
metrics, computed from ranking outputs), as the conservative default pending clarification — see
the verification report's §7 for the full evidence and the distinction between what the dataset
proves and what remains a modeling judgment call.

---

## 6. Temporal Splitting and Leakage

**The dataset's own structure creates a specific, non-obvious risk that must be designed around
explicitly**: the conversion attribution window is **30 days**, and the entire collection period is
**30.92 days** (verified: min timestamp 0, max timestamp 2,671,199 seconds). This means impressions
near the end of the collection window have not had their full 30-day opportunity to convert by the
time data collection stopped — their `conversion` label is **right-censored**, undercounted not
because those impressions truly convert less, but because there wasn't time to observe it. This
window-cap mechanism is now directly confirmed: the maximum observed conversion delay in the data
is exactly 30.00 days, and zero rows show a negative delay. This asymmetry does not affect `click`
(confirmed immediate, not delayed), reinforcing the CTR-primary recommendation in §5.

**Verified nuance**: the *theoretical* censoring risk above is mechanically proven, but the raw
daily conversion-rate series does **not** show a strong, visible decline in the final days of the
window (day 30's row-wise CVR is in the middle of the observed range, not the minimum) — most
likely because the median conversion delay is only 3.74 days, far shorter than the 30-day cap, so
most conversions resolve well before censoring would bite. The risk is real for the tail of the
window specifically, not a dominant effect across the whole dataset. Full daily breakdown in the
verification report.

**Split design** (following this project's existing time-based split convention from
`configs/config.yaml`):

```
day 0 ─────────────────── day 20 ──────── day 25 ──────── day 30
│         TRAIN            │   VALIDATION  │      TEST      │
```

- **Train**: days 0–20. Used to fit retrieval embeddings and the ranking model, and as the source
  for all historical/aggregate features (§7).
- **Validation**: days 20–25. Used for model selection and hyperparameter choices.
- **Test**: days 25–30. Final evaluation only.

**Explicit caveat to carry into every evaluation report**: conversion labels in the validation and
especially test periods are more right-censored than train, since those impressions had less
elapsed time (within the observed 30-day window) to convert before collection ended. CVR metrics
on test should be interpreted as a lower bound / noisier estimate, not a clean comparison against
train-period CVR. CTR metrics do not carry this caveat.

**Preventing specific leakage paths**:

- **Future clicks leaking into historical user features**: any "user's past click rate" feature
  for an impression at time `t` must be computed only from that user's impressions with
  `timestamp < t` — an expanding, point-in-time window, never a global aggregate over the whole
  dataset (which would let a user's *future* clicks inform predictions about their *past*
  behavior).
- **Future conversions leaking into training features**: identical discipline for
  `conversion_timestamp < t`. Additionally, because one `conversion_id` can span multiple
  impression rows (multi-touch), an *earlier* touchpoint's features must never be built using the
  outcome of a *later* touchpoint sharing the same `conversion_id` — each row's features must only
  reflect what was knowable strictly before its own timestamp, even relative to other rows in the
  same conversion journey.
- **Campaign-level information leakage**: aggregate campaign features (e.g. "campaign's historical
  CTR") must be computed only from the training period, and frozen before being applied to
  validation/test. The more rigorous alternative — a rolling, point-in-time campaign aggregate
  recomputed per training example — is the technically stronger approach and is noted as a future
  refinement; the frozen-training-aggregate approach is the practical default for the first
  version (see §11).

---

## 7. Feature Design

| Category | Examples | Computed from | Available at inference time? |
|---|---|---|---|
| **User-history** | past impression count, past click count/rate, past conversion count, recency (time since last impression), number of distinct campaigns seen | that user's own rows with `timestamp < t` | ✅ Yes, if computed with strict point-in-time discipline (§6). For a first-seen user, these default to "no history" (cold-start) — not fabricated. |
| **Campaign** | campaign's historical CTR/CVR, historical impression volume (popularity), average cost | training-period rows only, frozen (§6) | ✅ Yes, as a frozen training-period aggregate. `cat_i` fields are **not** used here — §1 confirmed none is reliably campaign-constant. |
| **User–campaign interaction** | has this `(uid, campaign)` pair been shown before, prior click/conversion count for this specific pair, time since last interaction with this campaign | that pair's own rows with `timestamp < t` | ✅ Yes, with the same point-in-time discipline; defaults to "no prior interaction" for a cold pair. |
| **Contextual** | `cat1`–`cat9`, `time_since_last_click` | the current impression row itself | ✅ Yes — these describe the impression being scored, not the future. `time_since_last_click` is backward-looking by definition (time since the user's *last* click), so it does not leak. |
| **Excluded from features** | `cost`, `cpo` | — | ❌ Deliberately excluded — see §5 (leakage/circularity), used only in evaluation. |

---

## 8. Evaluation

Retrieval, ranking, and business evaluation are kept separate — they answer different questions
and should not be collapsed into one leaderboard number.

**Retrieval** (does the candidate set contain the right campaigns?):
- **Recall@K** — primary. Directly measures retrieval's actual job: did the true positive
  campaign(s) make it into the top-K candidate set.
- **Hit Rate@K** — secondary, simpler binary version of Recall@K per user.
- **NDCG@K** — reported, but with a caveat: it rewards *ordering* within the retrieved set, which
  is really the ranking stage's job, so it's a secondary retrieval diagnostic, not primary.
- **Caveat to state alongside every retrieval number**: with only 675 campaigns in the catalog,
  Recall@K here is a substantially easier task than the same metric on a large-catalog production
  system. Results should be read as "does the technique work," not "this recall rate would hold at
  scale."

**Critical addition, required by §3B/§4's segmented design — retrieval metrics must be reported per
cohort, never as one blended number**: a single population-wide Recall@K would be dominated by the
94.6% cold majority, and *any* method — including pure popularity — would score identically on that
majority almost by construction, making it impossible to tell whether personalization contributed
anything. To avoid rewarding a model merely for recommending globally popular campaigns:

1. **Classify users into cold/warm using only pre-test-period history** (the same 0/1 vs. ≥2
   distinct-clicked-campaigns split from §3B, computed strictly from train-period data — never
   using test-period interactions to decide which cohort a user belongs to, which would itself be
   leakage).
2. **Report Recall@K/Hit Rate@K separately for the cold cohort and the warm cohort** — never
   combined into one number.
3. **For the warm cohort specifically, report lift**: `Recall@K(candidate method) −
   Recall@K(popularity baseline computed on the same warm cohort)`. A positive, non-trivial lift is
   the only legitimate evidence that personalization is adding value; a lift near zero — which
   §3B's popularity-overlap finding (27.0% vs. 29.98%) suggests is plausible — is itself a valid,
   reportable result, not a failure to hide.
4. **The cold cohort's evaluation exists to confirm the fallback works, not to claim
   personalization** — recall here should simply be compared against a naive "always-empty" or
   "random-K" floor, not against the warm cohort's numbers, since the two cohorts are answering
   different questions.

Ground truth for all of the above is each user's **test-period** (days 25–30) clicked campaign(s),
using the day-based split already established in §6 — this is standard temporal evaluation: fit
everything on train, classify cohort membership on train, predict candidates using only train
(and, for the ranking stage, validation-tuned) information, and check against test-period truth.

**Ranking** (does the model score correctly within the candidate set?):
- **PR-AUC** — primary, for both CTR and CVR. Preferred over ROC-AUC specifically because both
  labels are imbalanced (CTR less severely, CVR extremely so), and PR-AUC is more diagnostic under
  imbalance.
- **LogLoss** — secondary, important specifically because any future blended/utility score (§5)
  depends on calibrated probabilities, not just correct ordering.
- **ROC-AUC** — reported for comparability with the DAC/OSS-side project's existing metrics
  (`classification_metrics.py` already implements `auc`/`logloss`), but not treated as primary
  here due to its insensitivity to class imbalance.
- **NDCG@K** — reported using per-user grouped batches as the "list," with the same caveat already
  applied to DAC in the OSS notebook design: this is a constructed evaluation grouping, not a real
  per-request candidate set, since no request/session ID exists in this data either.

**Advertising/business** (what does this mean for the business, descriptively):
- **CTR, CVR** (observed) — descriptive, not the primary model-selection metric; carries the same
  "subsampled, not a true base rate" caveat already applied to DAC.
- **cost/`cpo`-derived utility metrics** — legitimate only as a *relative* comparison between model
  variants on Criteo's own transformed unit scale. Must never be reported or interpreted as real
  currency (Criteo's own documentation states `cost`/`cpo` are "not the real price, only a
  transformed version of it").

**What's primary overall**: cold-cohort Recall@K (does the fallback work at all) and warm-cohort
*lift* over its own popularity baseline (does personalization add anything where it's even
possible) for retrieval; PR-AUC (CTR) for ranking. These are the numbers that should gate whether a
stage "works" at all; everything else is diagnostic or descriptive context around them.

---

## 9. Baselines

**Revised progression** (supersedes the original list, which treated collaborative filtering as
the default second step for the whole population — no longer accurate given §3B):

1. **Popularity baseline** (retrieval, both cohorts): recommend the globally most-clicked campaigns
   (training-period aggregate), identical for every user. This is now the **primary retrieval
   method for the cold cohort (94.6% of users)**, not just a floor — per §3B/§4, no other signal
   reliably exists for this population.
2. **Context-adjusted popularity** (retrieval, cold cohort): a light enhancement over (1) using
   whichever `cat_i` weak signal is available from the cold user's one known impression — evaluated
   against (1) on the cold cohort specifically; adopted only if it measurably helps, per §8.
3. **Collaborative filtering** (retrieval, **warm cohort only, 5.4% of users**): implicit-feedback
   matrix factorization on the `uid × campaign` click matrix, restricted to and evaluated only
   against the warm cohort's own popularity baseline (lift, per §8) — explicitly not assumed to
   win, given §3B's popularity-overlap finding for this same cohort.
4. **Learned retrieval** (retrieval, warm cohort only): a richer embedding objective (e.g. BPR
   pairwise loss) — tests whether a more expressive objective helps within the already-small warm
   cohort; may not beat (3), and that result would itself be a legitimate, reportable finding.
5. **Ranking model** (ranking, applied after either cohort's retrieval stage): a gradient-boosted
   CTR model (and a separate, explicitly-caveated sparse CVR model), consistent with the LightGBM
   choice already established for the DAC/Phase-2 work, trained on the features in §7 and applied
   on top of whichever retrieval stage the cohort used — the stage that actually uses `cat1`–`cat9`
   and history features to differentiate candidates beyond popularity/collaborative signal alone.

---

## 10. Final Proposed Architecture

**Revised to reflect the segmented retrieval design in §4** (supersedes the original single-path
diagram, which routed every user through collaborative filtering):

```
historical interactions
        |
        v
cohort classification              (cold: 0-1 distinct clicked campaigns, 94.6% |
                                     warm: >=2 distinct clicked campaigns, 5.4%  — §3B)
        |
        +---------------------------+
        |                           |
        v                           v
cold-cohort retrieval          warm-cohort retrieval
(popularity, optionally        (implicit-feedback matrix
 context-adjusted — §4)         factorization — §4)
        |                           |
        +---------------------------+
        |
        v
candidate ranking                  (feature-based scoring of retrieved candidates, §7)
        |
        v
CTR/CVR prediction                 (LightGBM, PR-AUC/LogLoss primary, §5 §8)
        |
        v
advertising evaluation             (per-cohort Recall@K, warm-cohort lift, PR-AUC,
                                     observed CTR/CVR, cost-based utility — §8)
```

Maps onto the existing repository structure: `src/retrieval/` (cohort classification + both
retrieval branches), `src/models/` (ranking stage), `src/evaluation/` (per-cohort retrieval
metrics, ranking metrics, business metrics) — all currently placeholders, unimplemented until this
design is approved and built.

---

## 11. Scope Control — Not in the First Version

- **No Kafka/Spark/Kubernetes/AWS infrastructure.** Pandas/NumPy/LightGBM on a laptop is
  sufficient for 16.47M rows and a 675-item catalog; adding distributed infra here would be
  infrastructure the project doesn't need, not a capability gap.
- **No production serving system** (no API, no low-latency path, no online feature store).
- **No fabricated ad-creative IDs or request/session IDs.** `campaign` is the recommendable unit;
  nothing finer-grained is claimed to exist.
- **No ANN/vector-index retrieval infrastructure** (e.g. FAISS) — unjustified at a 675-item
  catalog; brute-force scoring is the honest, right-sized choice (§4).
- **No claim of TikTok-scale or production-scale retrieval.** The catalog size and data volume are
  explicitly prototype-scale, stated as such wherever results are reported.
- **No real-money ROAS/CPA claims** — `cost`/`cpo` are Criteo's own disclosed transformed/proxy
  units, usable only for relative model comparison (§8).
- **No multi-touch attribution *modeling*.** The `attribution` field is used as given; re-deriving
  attribution credit is the subject of Criteo's own published research, not this project's scope.
- **No deep sequence models** (e.g. transformers over click sequences) for v1 — classical implicit
  matrix factorization plus a gradient-boosted ranker is the right scope; sequence modeling is a
  plausible *future* extension, not part of the first version.
- **No hyperparameter-tuning infrastructure** (Optuna/NNI) required for v1 — reasonable defaults
  are sufficient at this stage.
- **No cross-dataset joint modeling.** DAC and the Attribution dataset remain two separate
  deliverables (OSS notebook vs. this portfolio component, per the prior feasibility checkpoint)
  — not merged into a single unified model despite sharing a data source company.
- **No blanket "personalized recommendation" claim.** Per §3B/§4, personalization is only
  theoretically supportable for 5.4% of users, and even there it is not assumed to outperform
  popularity — it is measured, per §8. Any write-up of this project must describe the system as
  **cold-start-aware retrieval with measured, cohort-scoped personalization**, not as a general
  personalization engine. This is the single most important scope boundary added by this revision.

---

## Verification Status

All four items originally listed here as open have been checked directly against the downloaded
dataset. Full evidence is in
[`attribution_dataset_verification.md`](attribution_dataset_verification.md); the corrections are
folded into §1, §2, §3, §4, §5, and §6 above. Summary:

| # | Original open question | Status |
|---|---|---|
| 1 | Are `cat1`–`cat9` campaign-level or impression-level? | Resolved: empirically not campaign-constant (except a weak `cat5` correlation); official semantics remain undisclosed. |
| 2 | Are `click_pos`/`click_nb` populated for all clicks or only conversion-path clicks? | Resolved: confirmed conversion-path-only, zero exceptions across all 16.47M rows. |
| 3 | Actual click/user counts and retrieval-matrix density? | Resolved, and surfaced a new, more material risk: average 2.68 impressions/user — flagged in §3/§4 as a feasibility concern for the next design review, not fixed here. |
| 4 | Severity of right-censoring? | Resolved: the 30-day window mechanism is proven (max delay = 30.00 days), but its effect is not strongly visible in raw daily aggregates, likely because median conversion delay (3.74 days) is well under the cap. |

One item not originally on this list was also corrected during verification: §5's claim that `cpo`
"is defined only when a conversion has already occurred" was disproven by the data (populated on
100% of rows) and has been corrected there, along with the reasoning for still excluding
`cost`/`cpo` from ranking features.

Two things remain genuinely unresolved and are **not** design-blocking, but are noted for
implementation time: the exact cause of 2,910 internally-inconsistent `conversion_id`s found during
integrity checking (§6 of the verification report), and the dataset's exact (undisclosed)
sub-sampling methodology.

### Second pass: architecture reassessment (post-density-verification)

A follow-up verification computed full per-user and per-campaign *distributions* (not just
averages) specifically to test whether the original §4 design (collaborative filtering as the
primary retrieval mechanism) was statistically defensible. It was not: only 5.39% of users have
≥2 distinct clicked campaigns, the minimum needed for any relative-preference signal to exist. This
is documented in full in the new §3B, and the architecture in §4, §8, §9, and §10 has been revised
accordingly to a segmented, cold-start-aware retrieval policy. This is a correction to the
architecture based on evidence, not a change of opinion — see §3B for the complete numbers.
