# CTR Model — First Ranking-Stage Experiment

Implements the "Recommended First Ranking Experiment" from
[`ranking_problem_design.md`](ranking_problem_design.md): a single, deliberately narrow
impression-level CTR classifier, `P(click = 1 | impression)`, trained and evaluated only on real,
already-served impressions. **CVR, ranking-specific NDCG evaluation, and combining retrieval with
ranking are explicitly out of scope for this experiment** — see the design doc for why they come
after, not alongside, this first pass.

## Prediction target

`click` (binary) on a real, logged impression row. Not a candidate-ranking target — see
[`ranking_problem_design.md`](ranking_problem_design.md) Q1/Q2 for why this dataset (no
request/session/slate ID) can only support impression-level classification, not slate-based
ranking, and why that's still the right first building block.

## Feature availability

Reuses [`ranking_problem_design.md`](ranking_problem_design.md) Q5, row (a) — offline scoring of an
already-logged impression — implemented in `src/features/attribution_features.py`:

| Category | Features | Computed from |
|---|---|---|
| Campaign identity | `campaign` | the row itself (categorical) |
| Contextual | `cat1`–`cat9`, `time_since_last_click` | the row itself — describes the impression being scored, not the future |
| User history (point-in-time) | `user_impressions_before`, `user_clicks_before`, `user_click_rate_before`, `user_distinct_campaigns_before` | that user's own rows with `timestamp` strictly earlier than the current row — computed over the **full** dataset (all splits together) before splitting, since a test-period row's history correctly includes real train-period activity for that same user |
| User–campaign pair history (point-in-time) | `pair_impressions_before`, `pair_clicks_before` | that specific pair's own prior rows, same point-in-time discipline |
| Campaign aggregate (frozen) | `campaign_impressions_train`, `campaign_clicks_train`, `campaign_ctr_train` | **training-period rows only**, computed once and broadcast unchanged onto validation/test rows — prevents a campaign's future popularity from leaking backward |

**Explicitly excluded as inputs** (post-outcome or outcome-adjacent, per
[`ranking_problem_design.md`](ranking_problem_design.md) Q4/Q8 and
[`attribution_modeling_design.md`](attribution_modeling_design.md) §5/§7): `conversion`,
`conversion_timestamp`, `conversion_id`, `attribution`, `click_pos`, `click_nb`, `cost`, `cpo`.
Enforced in code, not just documented — `src.features.attribution_features.assert_no_post_outcome_leakage`
raises if any of these ever appear in the feature list, and is called at the start of the experiment
script.

**Not the same feature set usable to score a not-yet-served retrieval candidate** — `cat1`–`cat9`
only exist once an impression has actually happened (verified impression-level, not campaign-level,
in [`attribution_dataset_verification.md`](attribution_dataset_verification.md) §1). This
experiment trains and evaluates on real, already-served impressions only, where that's not an
issue; using this same model to score retrieval candidates before they're served is a distinct
inference mode not built in this experiment (design doc Q5).

## Preprocessing

- **Point-in-time features**: built once over the full (train+val+test) dataset via vectorized,
  chronologically-sorted `groupby`/`cumsum`/`cumcount` operations (no per-row or per-group Python
  loops — required for this to run in reasonable time over 16M+ rows), then split using the
  existing `src.data.attribution.split_by_day` day boundaries.
- **Categorical encoding**: `campaign` and `cat1`–`cat9` are cast to pandas `category` dtype once,
  on the full dataset, *before* splitting — guaranteeing the same category-to-code mapping is used
  consistently across train, validation, and test (splitting a pandas `category` column preserves
  its category set; building categories independently per split risks silent code mismatches).
  Passed to LightGBM via its native categorical-feature support, not one-hot encoding.
- **`uid` is deliberately not a model feature.** With 6,142,256 users averaging 2.68 impressions
  each, a raw user-ID categorical would have millions of near-unique levels — not a generalizable
  feature, closer to a row identifier. `uid` is used only for grouping (point-in-time history
  computation) and is never passed to the model.

## Missing-value strategy

Checked directly on the real data (not assumed): **`cat1`–`cat9` have zero missing values** in this
dataset (confirmed by direct null-count at experiment run time — see Results). This is unlike DAC,
where empty fields are a documented convention; the Attribution dataset's categorical fields load
cleanly as dense integers.

Two genuinely different situations exist among the engineered features:

- **Count-based history features never need imputation.** `user_impressions_before`,
  `user_clicks_before`, `user_distinct_campaigns_before`, `pair_impressions_before`,
  `pair_clicks_before` are cumulative counts that start at 0 — for a user's first-ever impression,
  0 *is* the correct value, not a placeholder for something missing.
- **`user_click_rate_before` is genuinely undefined** for a user's first-ever impression (0/0).
  Left as `NaN` deliberately and passed to LightGBM's native missing-value handling (a documented
  LightGBM capability: at each split, missing values are routed to whichever branch improves the
  objective, which is more informative here than an arbitrary sentinel — "no history yet" is itself
  a meaningful signal, not noise to be smoothed away). Configured explicitly in
  `configs/config.yaml`: `attribution.ranking.ctr_model.missing_value_strategy`.
- **`campaign_ctr_train` for a campaign absent from the training period** is filled with the global
  training CTR (not left as `NaN`) — a documented, different choice from the user-history case,
  because "this campaign has no training data at all" is a distinct situation from "this user has
  no history yet," and a neutral prior (the global rate) is a more defensible default than leaving
  it to LightGBM's missing-value routing.

## Baseline

`GlobalCTRBaseline` predicts the constant training-period global CTR for every impression,
regardless of any feature. **This is a baseline, not a personalized ranking model — it cannot
distinguish between any two candidates at all**, since every row gets the identical score. It
exists purely to establish the floor LightGBM must clear by actually using features; a model that
only matches the baseline would indicate the features carry no learnable signal.

## LightGBM configuration

`configs/config.yaml`: `attribution.ranking.ctr_model` (reproduced here for reference):

```yaml
type: lightgbm
categorical_features: [campaign, cat1, cat2, cat3, cat4, cat5, cat6, cat7, cat8, cat9]
training:
  num_boost_round: 500
  early_stopping_rounds: 30
params:
  objective: binary
  metric: [auc, binary_logloss]
  learning_rate: 0.05
  num_leaves: 31
  min_child_samples: 100
  seed: 42
  verbose: -1
```

Deliberately conservative, unremarkable defaults — no hyperparameter search was run, per the task
constraint not to optimize aggressively. `min_child_samples: 100` is set slightly above LightGBM's
default specifically because several features (pair-history counts) are extremely sparse (most
pairs have 0 or 1 prior interactions, per the retrieval-stage sparsity findings), and a higher leaf
minimum reduces the chance of the tree fitting noise in that long tail.

## Evaluation protocol

- **Train**: days 0–19 only.
- **Validation**: days 20–24 — used **exclusively** for early stopping (a model/configuration
  decision), never for the metrics reported below.
- **Test**: days 25–29 — touched only once, for final reporting.

Metrics: PR-AUC (primary, per [`attribution_modeling_design.md`](attribution_modeling_design.md)
§8 — preferred under class imbalance), ROC-AUC, LogLoss, and a quantile-binned calibration table
(predicted vs. observed CTR per bin).

## Results

*(Filled in from the real experiment run against the full 16,468,027-row dataset —
`scripts/run_ctr_experiment.py`.)*

Run against the full 16,468,027-row dataset (`scripts/run_ctr_experiment.py`):

```
Split -> train: 10,852,826 | val: 2,586,597 | test: 3,028,604 rows
Missing values in cat1..cat9: 0 (verified, not assumed)
Global training CTR: 0.3577
Best iteration (selected via validation set): 500
```

| Metric | Baseline (constant) | LightGBM | Abs. improvement | Rel. improvement |
|---|---|---|---|---|
| PR-AUC | 0.3667 | **0.6069** | +0.2401 | **+65.48%** |
| ROC-AUC | 0.5000 | **0.7209** | +0.2209 | **+44.17%** |
| LogLoss | 0.6574 | **0.5816** | −0.0757 (lower is better) | **+11.52%** |

The baseline's PR-AUC (0.3667) landing almost exactly at the test-period click rate, and its
ROC-AUC landing exactly at 0.5000, are both expected and correct for a constant predictor — they
confirm the baseline is behaving as designed (inducing no ordering at all), not that something is
wrong. LightGBM's ROC-AUC of 0.72 is a believable, moderate number for a CTR model on real
impression-level data — not suspiciously close to 1.0, which is itself a useful methodological
signal: a near-perfect score here would be the classic symptom of a leaked feature, and this result
doesn't show that pattern.

### Calibration (quantile-binned, test period)

| mean_predicted | mean_observed | count |
|---|---|---|
| 0.1100 | 0.1074 | 302,861 |
| 0.1732 | 0.1774 | 302,864 |
| 0.2225 | 0.2274 | 302,856 |
| 0.2695 | 0.2731 | 302,866 |
| 0.3159 | 0.3188 | 303,006 |
| 0.3641 | 0.3659 | 302,709 |
| 0.4172 | 0.4150 | 302,913 |
| 0.4818 | 0.4792 | 302,812 |
| 0.5718 | 0.5653 | 302,856 |
| 0.7448 | 0.7377 | 302,861 |

Well-calibrated across the full range — predicted and observed rates agree closely in every bin,
with no systematic over- or under-confidence. No calibration correction (e.g. Platt/isotonic) is
warranted here.

## Feature importance

*(Gain-based, from the trained model — filled in below.)*

| Rank | Feature | Gain |
|---|---|---|
| 1 | `user_click_rate_before` | 5,582,036 |
| 2 | `campaign_ctr_train` | 2,002,921 |
| 3 | `cat8` | 1,348,095 |
| 4 | `cat3` | 1,180,082 |
| 5 | `cat6` | 1,088,787 |
| 6 | `campaign` | 940,941 |
| 7 | `cat9` | 928,312 |
| 8 | `pair_clicks_before` | 695,134 |
| 9 | `cat7` | 600,913 |
| 10 | `pair_impressions_before` | 447,234 |
| 11 | `cat1` | 402,803 |
| 12 | `user_clicks_before` | 372,678 |
| 13 | `time_since_last_click` | 220,102 |
| 14 | `cat5` | 63,912 |
| 15 | `user_impressions_before` | 43,409 |
| 16 | `cat4` | 34,455 |
| 17 | `cat2` | 28,755 |
| 18 | `user_distinct_campaigns_before` | 5,523 |
| 19 | `campaign_clicks_train` | 876 |
| 20 | `campaign_impressions_train` | 854 |

**Reading this sensibly, not over-reading it**: the two most important features are exactly what a
reasonable prior would expect a CTR model to lean on — a user's own historical click propensity
(when they have one) and a campaign's own historical CTR. Several `cat_i` fields rank highly
despite their undisclosed semantics — real, useful signal, whatever it represents. Notably, the raw
campaign volume counts (`campaign_clicks_train`, `campaign_impressions_train`) contribute almost
nothing once the derived rate (`campaign_ctr_train`) is available — the tree doesn't need to
reconstruct a rate combinatorially from the two counts when the rate is already provided directly.

**Caveat, stated explicitly per the task's instruction not to overinterpret**: gain-based
importance reflects how much the *trained model* relied on a feature to reduce loss — it is
**not** a causal effect estimate, and for `cat1`–`cat9` specifically, their real-world semantics
remain officially undisclosed (verification report §1), so even a high-importance `cat_i` cannot be
narrated as "campaigns with property X get more clicks" — only as "this anonymized field was useful
for prediction," full stop.

## Limitations

- **This is impression-level CTR classification, not candidate ranking.** No slate ever existed in
  this data — see [`ranking_problem_design.md`](ranking_problem_design.md) for the full analysis of
  why. This experiment answers "how well can we predict click on a real impression," not "how well
  would this model rank a set of competing candidates."
- **CVR is not modeled in this experiment**, deliberately — see the design doc's recommendation to
  add it as a separate follow-up, conditioned on click (`P(conversion|click=1)`), not alongside CTR
  in the same first pass.
- **No ranking-specific evaluation (NDCG) or retrieval-pipeline check was run** — also deliberately
  deferred, per the design doc's staged plan.
- **Observed CTR is inflated by the dataset's own undisclosed sub-sampling** (verification report
  §3) — reported metrics describe model quality on this sample, not a real-world production base
  rate.
- **Sparsity carries over from the retrieval-stage findings**: most users have only 1–2 impressions
  total, so `user_*` history features are frequently at or near their "no history" default even
  within the training set itself, not just at evaluation time.
- **No hyperparameter search was performed** — the configuration above is a single, reasonable,
  documented choice, not a tuned result.
- **Training likely had not fully converged at the 500-round cap.** The best iteration selected by
  early stopping was exactly 500 — the configured `num_boost_round` ceiling — meaning the
  validation metric was still improving when training stopped, not that early stopping actually
  triggered. The reported results are therefore a legitimate, honestly-evaluated snapshot, but
  likely understate what this same configuration could achieve with a higher round cap. Not
  extended in this experiment, per the instruction not to optimize aggressively — noted here as a
  concrete, verified next step rather than quietly raised.
- **Training is slow** (tens of minutes on this dataset), most plausibly dominated by `cat7`'s high
  cardinality (57,196 distinct values, verified in the dataset verification report) — LightGBM's
  native categorical split-finding scales with category count. Not addressed here since it doesn't
  affect correctness, but worth knowing before re-running this script.
