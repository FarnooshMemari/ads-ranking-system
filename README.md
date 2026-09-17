# Personalized Ads Ranking System with CTR Prediction

A **prototype advertising ranking system inspired by large-scale recommendation
architectures**, built to demonstrate applied machine learning skills relevant to ads ranking:
user–ad interaction modeling, cold-start-aware retrieval, click-through-rate (CTR) prediction, and
rigorous offline evaluation — including reporting real, honest negative results rather than only
favorable ones.

This is a portfolio project, not a production system: it does not include real-time/low-latency
serving, live traffic, or the infrastructure scale of any specific company's ads platform. It
borrows the *structure* of those systems (retrieval → ranking) to demonstrate ML system-design
understanding, at a scope that runs entirely offline on a single machine. It is intentionally
**not** another ETL/data-pipeline project — the author already has data engineering experience with
Spark, Kafka, Airflow, Dagster, and AWS; the focus here is the ML system itself.

## Status: retrieval and ranking stages implemented and evaluated

Both major stages described in the architecture below are built, tested, and run against the full
real dataset — **not** a toy sample. Every result below is from an actual experiment run, with the
document that produced it linked alongside. See [What's Implemented](#whats-implemented) for the
full breakdown, and [Key Findings](#key-findings-including-honest-negative-results) for what the
data did and didn't support.

## Two datasets, two distinct purposes — never mixed

| Dataset | Used for | Entities available |
|---|---|---|
| **Criteo DAC** (sample) | A separate, planned open-source contribution to [Microsoft Recommenders](https://github.com/recommenders-team/recommenders) — **not yet submitted** (see below) | None — no user/ad/session IDs, pure impression-level CTR features |
| **Criteo Attribution Modeling for Bidding Dataset** | This project's own recommendation pipeline (retrieval, ranking) | `uid`, `campaign`, click, conversion — see [`docs/attribution_dataset_verification.md`](docs/attribution_dataset_verification.md) |

These are deliberately never trained jointly — see
[`docs/attribution_modeling_design.md`](docs/attribution_modeling_design.md) §11 for why.

## What's Implemented

### Retrieval stage (`src/retrieval/`)

Cohort-segmented, not a single model — a train-period-only sparsity analysis
([`docs/attribution_modeling_design.md`](docs/attribution_modeling_design.md) §3B) found only
5.39% of users have enough click history (≥2 distinct clicked campaigns) to support any
personalization signal at all, so the two cohorts get genuinely different, honestly-scoped
strategies:

- **Cold cohort** (~95% of users): a training-period **popularity baseline** — the same top-K
  campaigns for every user. [`docs/popularity_baseline.md`](docs/popularity_baseline.md).
- **Warm cohort** (~5% of users): **item-based collaborative filtering** (SAR similarity design
  pattern), evaluated *against* its own cohort's popularity baseline rather than assumed to win.
  [`docs/collaborative_filtering.md`](docs/collaborative_filtering.md).

| Method | Cohort | Recall@10 |
|---|---|---|
| Popularity | Cold | 0.1959 |
| Popularity | Warm | 0.1731 |
| Collaborative filtering (excluding already-seen campaigns — the fair test) | Warm | 0.0952 |

### Ranking stage (`src/models/`, `src/features/`)

Impression-level CTR classification (`P(click | impression)`) — the only formulation this dataset's
schema supports without fabricating exposure information (no request/session/slate ID exists; see
[`docs/ranking_problem_design.md`](docs/ranking_problem_design.md)). Two model variants:

- **Full-feature model** — includes impression context (`cat1`–`cat9`).
  [`docs/ctr_model.md`](docs/ctr_model.md).
- **Pre-serve model** — excludes `cat1`–`cat9`, since those fields don't exist for a retrieval
  candidate that hasn't been served yet. The only variant valid for scoring retrieval output.
  [`docs/pre_serve_ctr_model.md`](docs/pre_serve_ctr_model.md).

| Model | PR-AUC | ROC-AUC | LogLoss |
|---|---|---|---|
| Baseline (constant prediction) | 0.3667 | 0.5000 | 0.6574 |
| Pre-serve LightGBM (cat1-9 excluded) | 0.5802 | 0.6911 | 0.5987 |
| Full-feature LightGBM | **0.6069** | **0.7209** | **0.5816** |

### Retrieval → ranking integration (`src/evaluation/ranking_integration.py`)

A restricted check of whether the pre-serve CTR model's scores agree with real outcomes, computed
**only** on campaigns independently confirmed as actually shown to the same user — never on
unobserved candidates treated as negatives. Intersection size and coverage are reported before any
metric. [`docs/ranking_integration_design.md`](docs/ranking_integration_design.md) /
[`docs/ranking_integration_results.md`](docs/ranking_integration_results.md).

| Cohort | Verified users with a click/non-click contrast | Pairwise ranking accuracy |
|---|---|---|
| Cold | 759 | 0.5535 |
| Warm | 3,479 | 0.5298 |

## Key Findings, Including Honest Negative Results

This project reports what the data actually showed, not only what looks favorable:

- **Collaborative filtering does not beat popularity** for the warm cohort once already-seen
  campaigns are excluded from its recommendations (the fair comparison) — Recall@10 drops from
  0.1731 (popularity) to 0.0952 (CF). Kept in the project as a documented experiment, not presented
  as the retrieval solution.
- **Removing impression context (`cat1`–`cat9`) costs ~3–4% relative** CTR model performance — real
  but moderate, needed because those fields don't exist before a candidate is served.
- **The retrieval→ranking agreement signal is real but modest** — pairwise accuracy of 0.53–0.55,
  meaningfully above the 0.50 chance level but not a strong validation of end-to-end ranking
  quality, computed on a small (hundreds-to-low-thousands), explicitly non-representative verified
  subset of users.
- **A full, statistically powerful, slate-based ranking evaluation is not achievable with this
  dataset at all** — there is no request/session/candidate-slate ID anywhere in the schema, so an
  unobserved `(user, campaign)` pair can never be treated as a negative. This is a structural
  property of the data, not a modeling limitation, and is the single most important scope boundary
  in this project — see [`docs/ranking_problem_design.md`](docs/ranking_problem_design.md) and
  [`docs/ranking_integration_design.md`](docs/ranking_integration_design.md).

## What's NOT Implemented (Deferred, Not Claimed)

- **CVR (conversion rate) modeling** — designed (`P(conversion | click)`,
  [`docs/ranking_problem_design.md`](docs/ranking_problem_design.md) Q3) but not built.
- **Slate-based / listwise ranking evaluation (NDCG over real competing candidates)** — not
  possible with this dataset (no slate ID); an artificial per-user grouping was identified as a
  weaker alternative and deliberately not pursued as a headline metric.
- **Live/online serving, A/B testing, or any real-time system.** Everything in this project is
  offline, batch evaluation against a fixed, already-logged dataset.
- **Embedding-based retrieval** (`EmbeddingCandidateGenerator` in
  `src/retrieval/candidate_generator.py`) — remains an unimplemented placeholder; item-based CF was
  built and evaluated instead (and didn't beat popularity — see above), so this wasn't prioritized.
- **Hyperparameter tuning** — every model in this project uses one reasonable, documented
  configuration, not a tuned result, by deliberate choice (avoiding overfitting to this specific
  offline sample).
- **The Microsoft Recommenders open-source contribution is designed but not submitted.** A
  framework analysis and a draft GitHub issue exist in a separate sibling project
  (`recommenders-ads-extension`, not part of this repository) — no PR has been opened, and no code
  from that plan has been implemented anywhere.

## Architecture

```
historical interactions
        |
        v
cohort classification              (cold: 0-1 distinct clicked campaigns |
                                     warm: >=2 distinct clicked campaigns)
        |
        +---------------------------+
        |                           |
        v                           v
cold-cohort retrieval          warm-cohort retrieval
(popularity)                   (collaborative filtering)
        |                           |
        +---------------------------+
        |
        v
CTR prediction                     (LightGBM, pre-serve feature set for
                                     scoring retrieval candidates)
        |
        v
evaluation                         (per-cohort Recall@K, PR-AUC/ROC-AUC/
                                     LogLoss, restricted ranking-agreement check)
```

## Repository Structure

```
ads-ranking-system/
├── configs/              # config.yaml -- dataset paths, split boundaries, cohort
│                          # threshold, LightGBM configs, all seeded/reproducible
├── data/
│   ├── raw/               # Raw, immutable input data (git-ignored)
│   └── processed/         # Processed data (git-ignored)
├── outputs/               # Trained model artifacts (git-ignored)
├── notebooks/              # (unused -- all experiments are reproducible scripts, see below)
├── src/
│   ├── data/               # Criteo DAC + Attribution dataset acquisition/loading/splitting
│   ├── features/           # Point-in-time CTR feature engineering (Attribution)
│   ├── retrieval/           # Cohort classification, popularity + collaborative filtering
│   ├── models/              # CTR model (baseline + LightGBM)
│   ├── evaluation/          # Classification, retrieval, and ranking-integration metrics
│   └── utils/               # Config loading, logging
├── scripts/                # One reproducible script per experiment (see Getting Started)
├── experiments/            # Per-stage artifact directories (git-ignored contents)
├── tests/                  # 120 unit tests, synthetic fixtures only -- no real data required
├── docs/                   # Full design + results documentation (see index below)
├── LICENSE
├── requirements.txt
├── Makefile
└── README.md
```

## Documentation Index

**Foundational / conceptual** (written early, before the Attribution dataset was selected —
`attribution_modeling_design.md` below is the current, data-grounded design that was actually
implemented):
- [`docs/architecture.md`](docs/architecture.md), [`docs/data_model.md`](docs/data_model.md)

**Datasets**:
- [`docs/criteo_dataset.md`](docs/criteo_dataset.md) — DAC, verified schema and license
- [`docs/attribution_dataset_verification.md`](docs/attribution_dataset_verification.md) —
  Attribution dataset, verified schema, sparsity statistics, and license

**Recommendation system design and results** (implemented, in build order):
- [`docs/attribution_modeling_design.md`](docs/attribution_modeling_design.md) — the core design
  document, including the cohort-segmentation decision and its rationale
- [`docs/popularity_baseline.md`](docs/popularity_baseline.md),
  [`docs/collaborative_filtering.md`](docs/collaborative_filtering.md) — retrieval stage
- [`docs/ranking_problem_design.md`](docs/ranking_problem_design.md),
  [`docs/ctr_model.md`](docs/ctr_model.md) — ranking stage
- [`docs/ranking_integration_design.md`](docs/ranking_integration_design.md),
  [`docs/pre_serve_ctr_model.md`](docs/pre_serve_ctr_model.md),
  [`docs/ranking_integration_results.md`](docs/ranking_integration_results.md) — combining the two

## Getting Started

```bash
make setup                        # create virtual environment and install dependencies
make test                         # run the test suite (120 tests, no real data needed)

make download-data                # DAC sample dataset
make download-attribution-data    # Attribution dataset

make popularity-baseline          # retrieval: popularity baseline experiment
make cf-experiment                # retrieval: collaborative filtering experiment
make ctr-experiment               # ranking: full-feature CTR model
make pre-serve-ctr-experiment     # ranking: pre-serve (cat1-9-excluded) CTR model
make ranking-integration-experiment  # combines retrieval output with the pre-serve model
```

Each `make *-experiment` target is a standalone, reproducible script in `scripts/` — none require a
notebook, and running the full CTR experiments end to end downloads real data and trains real
models (expect the full-feature CTR run in particular to take a substantial amount of time; see
[`docs/ctr_model.md`](docs/ctr_model.md) for why).

## Limitations

- **No live serving, no A/B testing, no real-time system** — everything here is offline batch
  evaluation.
- **No slate-based ranking is possible with this dataset** — no request/session/candidate-set ID
  exists anywhere in the schema (verified, not assumed). Every ranking-adjacent metric in this
  project is either impression-level classification or a restricted, explicitly-scoped check —
  never a claim of slate-competitive ranking.
- **Observed CTR/CVR rates are inflated by each dataset's own undisclosed sub-sampling** — reported
  metrics describe model quality on these samples, not real-world production base rates.
- **Campaign-level, not ad-creative-level.** No individual creative identifier exists in the
  Attribution dataset — `campaign` is the finest recommendable granularity.
- **Catalog size (675 campaigns) is small** relative to a production ad platform — retrieval
  metrics here are an easier task than the same metric would be at real scale.
- **This is a prototype, not a claim of TikTok-scale or any specific company's production
  infrastructure or results.**

## License

Code in this repository is licensed under the [MIT License](LICENSE). This license covers the
code only — it does not apply to, and does not redistribute, either dataset. Both are used under
their own terms (DAC: Criteo Labs' non-commercial, non-redistribution terms; Attribution: CC
BY-NC-SA 4.0) — see the dataset documentation above for details. Neither dataset is committed to
this repository; both are downloaded locally via the scripts above and are git-ignored.
