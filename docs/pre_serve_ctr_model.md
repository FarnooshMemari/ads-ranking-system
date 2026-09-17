# Pre-Serve CTR Model — Part A of the Ranking Integration

Implements Part A from [`ranking_integration_design.md`](ranking_integration_design.md): a
feature-ablated variant of the CTR model ([`ctr_model.md`](ctr_model.md)) that excludes
`cat1`–`cat9`, so it is actually usable to score a retrieval candidate **before** that candidate
has been served — unlike the full-feature model, which relies on context fields that only exist
once an impression has already happened.

## Why this model exists

`ctr_model.md`'s model was trained and evaluated entirely on real, already-logged impressions,
where `cat1`–`cat9` are always present. But those fields describe the *specific serving context* of
an impression (verified impression-level, not campaign-level, in
[`attribution_dataset_verification.md`](attribution_dataset_verification.md) §1) — they don't exist
for a campaign that retrieval has proposed as a candidate but that hasn't been served yet. Using the
full-feature model to score such a candidate anyway (e.g. by imputing the missing context) would
silently evaluate it outside the feature distribution it was trained and validated on. This model
exists specifically to close that gap, identified in
[`ranking_integration_design.md`](ranking_integration_design.md)'s formulation 3.

## Feature set

`src.features.attribution_features.PRE_SERVE_FEATURE_COLUMNS` — derived from `FEATURE_COLUMNS` by
removing `cat1`–`cat9` (not hardcoded separately, so the two lists cannot silently drift apart):

| Kept | Removed |
|---|---|
| `campaign`, `time_since_last_click`, `user_impressions_before`, `user_clicks_before`, `user_click_rate_before`, `user_distinct_campaigns_before`, `pair_impressions_before`, `pair_clicks_before`, `campaign_impressions_train`, `campaign_clicks_train`, `campaign_ctr_train` | `cat1`, `cat2`, `cat3`, `cat4`, `cat5`, `cat6`, `cat7`, `cat8`, `cat9` |

**`time_since_last_click` is deliberately retained, not removed alongside `cat1`–`cat9`.** It is
derived purely from a user's own past click history — knowable before any serving decision, unlike
`cat1`–`cat9`'s impression-specific context. Grouping it with the contextual fields in earlier docs
was about leakage (neither leaks), not about pre-serve availability, which is the distinction that
matters here.

Same preprocessing pipeline as the full model (`build_ctr_features`, unchanged, only a different
column selection), same point-in-time discipline, same excluded post-outcome columns
(`assert_no_post_outcome_leakage` is run against this feature list too).

## Protocol

Identical to `ctr_model.md`: same day-based train/validation/test split, same target
(`P(click | impression)`), same hyperparameters (`configs/config.yaml`:
`attribution.ranking.ctr_model_pre_serve` — same `params`/`training` block as `ctr_model`, only the
`categorical_features` list differs) — so any performance delta below is attributable to the
removed features, not a different configuration. No hyperparameter search was run here either.

## Results

*(Filled in from the real experiment run — `scripts/run_pre_serve_ctr_experiment.py`.)*

Run against the full 16,468,027-row dataset, same split as `ctr_model.md`
(`scripts/run_pre_serve_ctr_experiment.py`):

```
Split -> train: 10,852,826 | val: 2,586,597 | test: 3,028,604 rows
Global training CTR: 0.3577
Best iteration (selected via validation set): 306
```

| Metric | Baseline (constant) | Pre-serve LightGBM | Abs. improvement | Rel. improvement |
|---|---|---|---|---|
| PR-AUC | 0.3667 | **0.5802** | +0.2135 | **+58.21%** |
| ROC-AUC | 0.5000 | **0.6911** | +0.1911 | **+38.22%** |
| LogLoss | 0.6574 | **0.5987** | −0.0586 (lower is better) | **+8.92%** |

The pre-serve model still substantially beats its own baseline — meaningful, learnable signal
survives even without `cat1`–`cat9`, driven mainly by history- and campaign-aggregate features (see
feature importance below).

**Notable methodological difference from the full model**: early stopping actually *triggered*
here (best iteration 306 of the 500-round budget), whereas the full-feature model's best iteration
landed exactly on its 500-round ceiling (`ctr_model.md`, still improving when training stopped). A
smaller, simpler feature space converges faster — a sensible, expected difference, not an anomaly.

### Calibration (quantile-binned, test period)

| mean_predicted | mean_observed | count |
|---|---|---|
| 0.1289 | 0.1383 | 302,861 |
| 0.2020 | 0.2138 | 302,860 |
| 0.2491 | 0.2578 | 303,578 |
| 0.2845 | 0.2901 | 302,143 |
| 0.3163 | 0.3202 | 303,113 |
| 0.3522 | 0.3549 | 302,878 |
| 0.3949 | 0.3966 | 302,590 |
| 0.4516 | 0.4454 | 302,860 |
| 0.5427 | 0.5338 | 302,899 |
| 0.7294 | 0.7165 | 302,822 |

Well-calibrated throughout, similar quality to the full-feature model — removing `cat1`–`cat9` cost
discrimination power (see the comparison below) but not calibration quality.

### Feature importance (gain-based)

| Rank | Feature | Gain |
|---|---|---|
| 1 | `user_click_rate_before` | 6,025,260 |
| 2 | `campaign_ctr_train` | 2,138,875 |
| 3 | `campaign` | 819,077 |
| 4 | `pair_clicks_before` | 799,022 |
| 5 | `time_since_last_click` | 712,871 |
| 6 | `pair_impressions_before` | 416,124 |
| 7 | `user_clicks_before` | 332,805 |
| 8 | `user_impressions_before` | 71,706 |
| 9 | `user_distinct_campaigns_before` | 17,310 |
| 10 | `campaign_impressions_train` | 541 |
| 11 | `campaign_clicks_train` | 339 |

The same two features that dominated the full model (`user_click_rate_before`,
`campaign_ctr_train`) dominate here too, now carrying an even larger share of total gain with no
`cat_i` fields competing for splits. `time_since_last_click` rises notably in relative importance
(rank 5 here vs. rank 13 in the full model) — plausible now that it's one of the few
context-adjacent signals left, though this is a gain-based observation, not a causal one (per the
same caveat as `ctr_model.md`).

## Direct comparison: full-feature vs. pre-serve

*(The full-feature numbers below are reused from the already-validated `ctr_model.md` run — same
code, same deterministically-split data, not re-run here, since nothing about that pipeline
changed.)*

| Metric | Full-feature (`ctr_model.md`) | Pre-serve (this document) | Delta (full − pre-serve) | Relative cost of removing `cat1`–`cat9` |
|---|---|---|---|---|
| PR-AUC | 0.6069 | 0.5802 | −0.0267 | **−4.40%** |
| ROC-AUC | 0.7209 | 0.6911 | −0.0298 | **−4.13%** |
| LogLoss | 0.5816 | 0.5987 | +0.0171 (worse) | **−2.94%** |

Removing nine features costs a real but moderate 3–4% relative discrimination performance — not
catastrophic, and both models remain far above their shared baseline. This is the expected shape of
result for a legitimate feature-ablation study: some information is genuinely lost, but the
remaining features (history and campaign aggregates) carry most of the model's actual predictive
power, consistent with both models' feature importance tables agreeing on the same top-2 features.

## What signal was lost by removing `cat1`–`cat9`

In the full-feature model, six `cat_i` fields ranked in the top 10 by gain (`cat8`, `cat3`, `cat6`,
`cat9`, `cat7`, `cat1` — `ctr_model.md`), collectively contributing a real, non-trivial share of
total split gain even though no individual `cat_i` came close to `user_click_rate_before` or
`campaign_ctr_train`. Removing all nine at once costs ~3–4% relative discrimination (PR-AUC/
ROC-AUC) and ~3% relative calibration quality (LogLoss) — consistent with "real but secondary
signal," not "critical signal," which matches their combined-but-individually-modest standing in
the full model's own importance ranking. Since `cat1`–`cat9`'s real-world semantics remain
officially undisclosed (`attribution_dataset_verification.md` §1), nothing more specific than "some
useful, anonymized contextual signal" can honestly be said about *what* was lost — only *how much*,
which this experiment measures directly rather than guessing at.

## Limitations

- **This model is intentionally worse than the full-feature model at its own task** (offline CTR
  prediction on logged impressions) — that is the expected, correct outcome of removing real
  predictive signal, not a modeling failure. Its purpose is not to be the best possible CTR
  predictor; it's to be the one variant honestly usable for pre-serve candidate scoring.
- **Still trained and evaluated only on real, logged impressions** — this document validates the
  model's classification quality; whether it produces *useful ranking* when applied to retrieval
  candidates is Part B's question, addressed separately in
  [`ranking_integration_results.md`](ranking_integration_results.md), precisely because the two
  questions have different, non-transferable answers.
- **No hyperparameter search was performed** for this variant either.
