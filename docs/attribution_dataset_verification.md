# Attribution Dataset — Verification Report

This document resolves the four items marked "⚠️ UNVERIFIED" in
[`attribution_modeling_design.md`](attribution_modeling_design.md), plus everything else this
verification step was scoped to check. All numbers below were computed directly from the actual
dataset file (`criteo_attribution_dataset.tsv.gz`, downloaded fresh from the official Hugging Face
mirror `criteo/criteo-attribution-dataset`, 653,015,824 bytes — matching the previously verified
size exactly) using pandas, loaded in full (16,468,027 rows, no sampling). The file was downloaded
to a scratch location for this analysis only; it is not part of the repository and nothing was
implemented, trained, or added to the project's architecture.

No filtering, cleaning, or correction was applied — every statistic below is computed on the raw
file exactly as distributed.

---

## 1. Campaign Feature Semantics (`cat1`–`cat9`)

**What the official documentation establishes**: nothing beyond "contextual features associated to
the display... the semantic of these features is undisclosed." The documentation does **not**
state whether these are static per-campaign attributes or per-impression context — this was
correctly left "unknown" in the design doc and remains formally unknown. No source documents this;
only an empirical pattern can be reported, and it does not prove semantics.

**What the data shows** (per-campaign uniqueness of each field, computed by grouping all 16.46M
rows by `campaign` and checking how many distinct values each `cat_i` takes within each campaign):

| Field | Global distinct values | % of campaigns where constant | Avg distinct values per campaign |
|---|---|---|---|
| `cat1` | 9 | 0.00% | 8.64 |
| `cat2` | 70 | 4.00% | 4.10 |
| `cat3` | 1,829 | 38.07% | 2.71 |
| `cat4` | 21 | 2.52% | 7.86 |
| `cat5` | 51 | **91.11%** | 1.10 |
| `cat6` | 30 | 0.15% | 18.40 |
| `cat7` | 57,196 | 14.96% | 114.80 |
| `cat8` | 11 | 1.48% | 10.82 |
| `cat9` | 30 | 0.00% | 29.23 |

**Conclusion**: none of `cat1`–`cat9` is constant within every campaign, so **none can be treated
as a reliable static campaign attribute**. `cat5` is a partial, imperfect exception — constant for
91.11% of campaigns, with a low global cardinality (51 values) — but this is reported strictly as
an *empirical correlation*, not a semantic conclusion: 8.89% of campaigns still show more than one
`cat5` value, and even a clean correlation would not prove `cat5` *is* a campaign attribute (e.g.
it could be an ad-format field that most campaigns happen to keep fixed, without being "about" the
campaign itself). Per the instruction not to infer semantics from repeated values alone: **the
semantics of every `cat_i` field remain officially unknown**; what is now established is only that
treating any of them as static campaign metadata would be empirically unsafe (except `cat5`, which
is weakly, not reliably, campaign-correlated).

**Side observation**: unlike DAC's `cat00`–`cat25` (32-bit hashed strings, effectively unbounded
cardinality), these `cat1`–`cat9` fields have low-to-moderate cardinality (9 to 57,196), consistent
with small categorical taxonomies rather than raw feature hashes — this doesn't establish meaning,
only that the encoding differs from DAC's.

---

## 2. `click` / `click_pos` / `click_nb` Semantics

**Official documentation** (quoted exactly from the source): `click`: "1 if the impression was
clicked, 0 otherwise." `click_pos`: "the position of the click **before a conversion** (0 for
first-click)." `click_nb`: "number of clicks. More than 1 if there was several clicks **before a
conversion**." The phrasing of the latter two fields' own descriptions already scopes them to
conversion sequences — the data below is used only to cross-check this reading, not to establish it.

**Cross-check against the data** (exhaustive, zero exceptions found):

| Group | Rows | `click_pos` | `click_nb` |
|---|---|---|---|
| `click=0` (all) | 10,520,464 | always `-1` | always `-1` |
| `click=1`, `conversion=0` | 5,141,367 | always `-1` | always `-1` |
| `click=1`, `conversion=1` | 806,196 | real value (0–173) | real value (1–max) |

**Conclusion — fully verified, documentation and data agree with zero exceptions**: `click_pos`
and `click_nb` are populated **only** for rows that are both clicked **and** part of a converting
sequence. For the 5,141,367 rows that were clicked but did not lead to a conversion (86.4% of all
clicks), both fields are the `-1` sentinel. These two fields must **not** be used as general
click-level features — they describe conversion-path structure only, and using them for anything
outside conversion modeling would silently drop or misrepresent the vast majority of clicks.

**Additional finding, not one of the original four but discovered during this check**: every one
of the 806,196 `conversion=1` rows is *also* a `click=1` row — there are zero rows with
`conversion=1` and `click=0` in this dataset. This means the design doc's earlier reading of
"conversion is set independently of whether this impression was last click or not" as possibly
allowing *view-through* (unclicked) conversions is **not what the data shows**: in this file, a
conversion is always attached to a clicked impression. The "independently of last click" phrasing
appears to refer to click *order* within a multi-click sequence (first click vs. a later click),
not to click *presence*. This is now corrected in the design doc (see the changelog at the bottom
of this document).

---

## 3. Dataset Statistics

All denominators stated explicitly per the instructions.

| Statistic | Value | Denominator |
|---|---|---|
| Rows (impressions) | 16,468,027 | — |
| Unique `uid` | 6,142,256 | — |
| Unique `campaign` | **675** | — (documentation's "700 campaigns" is a rounded approximation; 675 is the exact count) |
| Clicks (`click=1`) | 5,947,563 | — |
| `conversion=1` rows (impression-level, multi-touch inclusive) | 806,196 | — |
| Distinct conversions (`conversion_id`, excl. `-1`) | 435,810 | — |
| `attribution=1` rows | 442,424 | — |
| Distinct attributed conversions | 236,331 | — |
| **CTR** | **0.361158** (36.12%) | clicks / impressions = 5,947,563 / 16,468,027 |
| **CVR (row-wise)** | 0.048955 (4.90%) | conversion=1 rows / impressions = 806,196 / 16,468,027 |
| **CVR (distinct, of impressions)** | 0.026464 (2.65%) | distinct conversions / impressions = 435,810 / 16,468,027 |
| **CVR (distinct, of clicks)** | 0.073275 (7.33%) | distinct conversions / clicks = 435,810 / 5,947,563 |
| Users with ≥1 click | 2,813,528 | / 6,142,256 users = 45.81% |
| Users with ≥1 conversion | 327,860 | / 6,142,256 users = 5.34% |
| Campaigns with ≥1 click | 675 | / 675 campaigns = 100.00% |
| Campaigns with ≥1 conversion | 675 | / 675 campaigns = 100.00% |
| Observed `(uid, campaign)` pairs | 7,813,401 | / 4,146,022,800 possible pairs = 0.1885% |
| Clicked `(uid, campaign)` pairs | 3,205,687 | / 4,146,022,800 possible pairs = 0.0773% |
| Avg impressions per user | **2.68** | rows / unique users |
| Avg impressions per campaign | 24,397.08 | rows / unique campaigns |

**⚠️ Critical finding, beyond the original four items**: average impressions per user is **2.68**
— the overwhelming majority of users appear only a handful of times in the entire 31-day window.
Combined with the 0.1885% observed-pair sparsity, this means most users have far too little
individual history to support a conventional per-user collaborative-filtering embedding. This
directly affects the feasibility of §4 of the design doc (candidate generation) and is flagged
there, without redesigning the approach in this document (per instructions).

**Also notable**: CTR (36.1%) and CVR-of-clicks (7.3%) are both far higher than real-world display
advertising rates (typically low single digits and well under 1%, respectively). The dataset's own
documentation states data is "sub-sampled and anonymized," but — unlike DAC, which explicitly
disclosed differential positive/negative subsampling rates — this dataset's documentation does
**not** disclose its exact subsampling scheme. The elevated rates are consistent with this being a
curated, class-balanced-ish sample rather than raw production traffic, but the precise mechanism
cannot be determined from the documentation or the data alone — this remains unresolved (see
Unresolved Questions).

---

## 4. Temporal Structure

| Statistic | Value |
|---|---|
| Min `timestamp` | 0 |
| Max `timestamp` | 2,671,199 (seconds) |
| Span | 2,671,199 s = **30.92 days** |
| Unique day buckets (`timestamp // 86400`) | 31 (day 0 through day 30) |
| Rows per day | ~436K–638K (fairly stable, no extreme outlier days) |
| Clicks per day | ~165K–217K |
| Conversions per day (row-wise) | ~21K–30K |
| Daily CTR range | 33.4%–38.3% (stable, no sharp trend) |
| Daily CVR (row-wise) range | 4.53%–5.34% (stable, no sharp trend) |

**Conclusion**: the documented "30 days" is confirmed as accurate to within rounding — the actual
observed window is 30.92 days (31 calendar-day buckets from timestamp 0). `timestamp` is
consistent with a seconds-based relative clock starting at the first impression, exactly as
documented; no calendar date/day-of-week is recoverable, as already noted in the design doc.

---

## 5. Conversion Censoring

**Documentation**: "1 if there was a conversion in the 30 days after the impression." This 30-day
window is now directly confirmed from the data: the maximum observed conversion delay
(`conversion_timestamp - timestamp`, among the 806,196 rows with a real conversion timestamp) is
**exactly 30.00 days** (2,591,997 seconds) — the window is being enforced, not just described.
Median delay is 3.74 days; mean is 7.14 days; 75th percentile is 11.97 days. Zero rows have a
negative delay (see §7).

**Right-censoring mechanics, confirmed**: since the dataset's own collection window (30.92 days)
is barely longer than the conversion attribution window (30 days), impressions occurring in
roughly the last day of collection have had almost no time to convert before the file was
generated — any "true" conversion that would have happened after the collection cutoff is
structurally absent from this file. This is a provable mechanical fact from the window-cap
alignment, not a modeling assumption.

**What the raw daily aggregate actually shows**: somewhat contrary to a naive expectation of a
sharp decline, the day-by-day row-wise CVR in §4 does **not** show a strong, monotonic drop in the
final days (day 30's CVR of 5.07% is in the middle of the observed 4.53%–5.34% range, not the
minimum). The most likely explanation, consistent with the delay distribution above: since the
median conversion delay is only 3.74 days (far shorter than the 30-day cap), most conversions that
will ever be observed resolve quickly, so the *marginal* impact of losing the last day or so of
"runway" is small relative to ordinary day-to-day variation. This does **not** mean censoring isn't
happening — the window-cap mechanics guarantee it is, for the tail of the collection period — only
that its effect is not the dominant visible pattern in a simple daily aggregate at this scale.

**Implication for evaluation** (not a fix — per instructions, this is not being corrected here):
the last few days of the window are the most censored and should be excluded or treated cautiously
in any validation/test boundary for CVR evaluation specifically; CTR is not subject to this effect
(no delay window applies to `click`, which is either set on the impression row itself or not). A
rigorous censoring-adjusted analysis (e.g. survival-analysis-style handling) is a implementation-time
task, not resolved by this verification.

---

## 6. Data Integrity Checks

| Check | Result |
|---|---|
| Fully duplicate rows | **0** |
| Distinct `conversion_id` (excl. `-1`) | 435,810 |
| `conversion_id`s appearing on >1 row (multi-touch) | 140,818 (32.31% of all conversion IDs) |
| Max rows sharing one `conversion_id` | 164 |
| **`conversion_id`s with inconsistent `conversion_timestamp` across their rows** | **2,910 (0.668%)** |
| **`conversion_id`s spanning more than one distinct `uid`** | **2,910 (0.668%)** |
| Rows with `conversion_timestamp < timestamp` | 0 |
| Null `uid` | 0 |
| Null `campaign` | 0 |
| Invalid values in `conversion`/`attribution`/`click` (not 0/1) | 0 for all three |
| `attribution=1` with `conversion=0` | 0 (fully consistent — attribution always implies conversion) |
| `conversion=1` with `conversion_timestamp`/`conversion_id` still `-1` | 0 |
| `conversion=0` with `conversion_timestamp`/`conversion_id` not `-1` | 0 |
| Negative `timestamp` | 0 |
| Negative `time_since_last_click` (excl. `-1` sentinel) | 0 |
| `time_since_last_click == -1` ("no prior click yet") | 8,770,820 rows (53.3%) |

**Anomaly flagged, not removed**: 2,910 `conversion_id` values (0.668% of all 435,810 distinct
conversions) are internally inconsistent — the same `conversion_id` appears on rows belonging to
more than one `uid`, and (the identical count suggests the same underlying rows) with more than one
`conversion_timestamp`. This violates the expected invariant that one `conversion_id` represents one
real-world conversion event belonging to one user. Root cause is not determinable from the data or
documentation alone — plausible explanations include anonymization hash collisions or an artifact
of how the sample was constructed, but this is speculation, not a verified cause. **This anomaly is
reported, not filtered or corrected**, per instructions. Its scale (0.67% of conversions) is small
but non-negligible and should be handled explicitly (e.g. excluded or deduplicated with a documented
rule) whenever conversion modeling is actually implemented.

Every other integrity check came back fully clean — no missing identifiers, no invalid binary
values, no logical inconsistencies between `conversion`/`attribution`/timestamps, no reverse-time
anomalies.

---

## 7. Leakage-Relevant Checks

**(a) Does `conversion_timestamp` always occur after `timestamp`?**
**Yes, proven, zero exceptions** — confirmed directly in §6 (0 rows with `conversion_timestamp <
timestamp`, across all 806,196 rows with a real conversion timestamp). This is a dataset fact.

**(b) Is `cpo` populated only when a conversion exists?**
**No — this is disproven by the data, contradicting a literal reading of the documentation.**
The documentation describes `cpo` as "the cost-per-order **in case of** attributed conversion,"
which reads as conditional. The data shows otherwise:

| Group | Rows | `cpo == 0`? | `cpo` mean |
|---|---|---|---|
| `conversion=0` (no conversion at all) | 15,661,831 | **0 rows have `cpo == 0`** — every one has a nonzero value | 0.2009 |
| `conversion=1` | 806,196 | 0 rows have `cpo == 0` | 0.1103 |

Every single row in the entire dataset — including all 15,661,831 non-converting impressions — has
a nonzero `cpo` value, and non-converting rows have a *higher* average `cpo` than converting ones.
This directly contradicts treating `cpo` as an outcome-only field populated "in case of" conversion.

**What this does and does not establish**: the dataset **proves** `cpo` is present on every row
regardless of outcome. It does **not** prove *when* that value was computed relative to the
impression being served — it could be a pre-computed/estimated valuation available at bid time
(which would make it a legitimate pre-decision feature), or it could be computed after the fact
for reasons the documentation doesn't explain. The documentation's own wording is now known to be
imprecise relative to the actual file. **This must be corrected in the design doc**, which had
stated the outcome-only interpretation as settled fact — it was not.

**(c) Can `cost`/`cpo` legitimately be considered pre-impression (pre-decision) features?**
This cannot be resolved from the dataset or its documentation — **neither proves nor disproves
timing**. What is proven: both `cost` and `cpo` are populated on 100% of rows with no missing/
sentinel values, and both are continuous, small, plausible-looking values (not obviously
post-hoc artifacts). What is *not* proven, and not addressed anywhere in the source documentation:
whether these values were knowable to a bidding system before the impression was served, or
computed afterward. **This remains a modeling decision under genuine uncertainty, not a fact
established by the data.** The design doc's existing caution (excluding `cost`/`cpo` from
predictive ranking features) is retained as the conservative default, but its *justification* is
corrected: not because the dataset proves post-hoc leakage (for `cpo`, it now disproves the
strict outcome-only reading), but because the timing of both fields relative to the serving
decision is undocumented and unverifiable either way. Treating them as evaluation-only inputs
remains the safer choice pending clarification.

---

## Unresolved Questions

These could not be resolved from the dataset, the bundled documentation, or the official Criteo AI
Lab page, and are explicitly left open rather than guessed at:

1. **Exact subsampling methodology.** The documentation discloses that data is "sub-sampled and
   anonymized" but not the scheme — the unusually high CTR (36%) and CVR-of-clicks (7.3%) are
   consistent with non-uniform sampling, but the precise mechanism is undocumented.
2. **Root cause of the 2,910 internally-inconsistent `conversion_id`s** (§6) — hash collision,
   construction artifact, or something else is not determinable from available sources.
3. **Timing/provenance of `cost` and `cpo`** relative to the impression-serving decision (§7c) —
   genuinely undocumented, not just unverified by this analysis.
4. **Semantic meaning of any individual `cat1`–`cat9` field** — the per-campaign-constancy pattern
   in §1 is an empirical observation only; no source establishes what these fields represent.

---

## Changelog: What This Verification Corrects in `attribution_modeling_design.md`

- §1/§4/§7: `cat1`–`cat9` confirmed empirically as not campaign-constant (except a weak `cat5`
  correlation); the "⚠️ UNVERIFIED" flags are resolved to "empirically not static campaign
  metadata; semantics still formally unknown," not left open-ended.
- §2: `click_pos`/`click_nb` confirmed exactly as originally hypothesized (conversion-path-only) —
  flag resolved to confirmed. Additionally corrects an implicit assumption that view-through
  (unclicked) conversions might exist in this data — they don't; every conversion row is also a
  click row.
- §5/§7: the claim that `cpo` "is defined only when a conversion has already occurred" is
  **factually incorrect** and is corrected — `cpo` is populated on every row. The recommendation to
  exclude `cost`/`cpo` from ranking features is unchanged, but its justification is corrected from
  "proven outcome-only leakage" to "undocumented timing, excluded as a precaution."
- §4/§9 (new flag, not one of the original four): average impressions per user (2.68) and the
  resulting extreme per-user interaction sparsity are flagged as a material risk to the
  collaborative-filtering retrieval approach — not redesigned here, but noted for the next design
  review.
- §6: campaign count corrected from the documentation's rounded "700" to the exact verified value,
  **675**.
