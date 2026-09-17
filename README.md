# Personalized Ads Ranking System with CTR Prediction

A **prototype advertising ranking system inspired by large-scale
recommendation architectures**, built to demonstrate applied machine
learning skills relevant to ads ranking: user–ad interaction modeling,
click-through-rate (CTR) prediction, ranking, candidate generation, and
offline evaluation.

This is a portfolio project, not a production system: it does not include
real-time/low-latency serving, live traffic, or the infrastructure scale
of any specific company's ads platform. It borrows the *structure* of
those systems (multi-stage retrieval and ranking) to demonstrate ML
system-design understanding, at a scope that runs entirely offline on a
single machine.

## Motivation

Ads platforms generally sit at the intersection of large-scale data
engineering and recommendation/ranking machine learning. Serving a
relevant ad to a user requires a pipeline that goes well beyond a single
model: candidate generation narrows a large ad pool down to a manageable
set, a CTR/conversion model scores each candidate, a ranking layer
combines predicted engagement with business objectives (bid, budget
pacing, relevance, diversity), and the whole system is evaluated and
improved through offline metrics and experimentation.

This project is a from-scratch implementation of that same *structure*,
scaled down to a prototype: an offline, single-machine pipeline over a
public dataset, not a deployed or real-time system. It is intentionally
**not** another ETL/data-pipeline project — the author already has data
engineering experience with Spark, Kafka, Airflow, Dagster, and AWS.
Instead, the focus here is the ML system itself: how user-ad interaction
data is modeled, how CTR is predicted, how ranking decisions are made, and
how such a system is evaluated offline before it would ever be considered
for production.

## Business Problem

Given a user, a context (device, time, surface), and a pool of candidate
ads, predict the probability that the user will click (or otherwise engage
with) each ad, and produce a ranked list that maximizes expected value
(e.g. expected revenue = P(click) × bid) subject to relevance, diversity,
and business constraints. The system must:

- Generalize across cold-start users/ads with sparse interaction history
- Handle highly imbalanced click/no-click data
- Rank, not just classify — absolute calibration matters less than
  relative ordering and expected-value maximization
- Be evaluable offline (AUC, LogLoss, NDCG, calibration) before any
  experiment comparison is made

## High-Level Architecture

```
                        ┌─────────────────────┐
                        │   Raw Interaction    │
                        │   & Ad/User Data     │
                        └──────────┬───────────┘
                                   │
                        ┌──────────▼───────────┐
                        │  Feature Engineering  │
                        │ (user, ad, context,   │
                        │  cross features)      │
                        └──────────┬───────────┘
                                   │
                 ┌─────────────────┼─────────────────┐
                 │                 │                 │
        ┌────────▼───────┐ ┌───────▼───────┐ ┌───────▼────────┐
        │   Candidate     │ │      CTR       │ │    Ranking      │
        │   Generation    │ │   Prediction   │ │  (score + biz    │
        │  (retrieval)    │ │    (model)     │ │   constraints)   │
        └────────┬───────┘ └───────┬───────┘ └───────┬────────┘
                 │                 │                 │
                 └─────────────────┼─────────────────┘
                                   │
                        ┌──────────▼───────────┐
                        │      Evaluation /     │
                        │    Experimentation     │
                        │ (offline metrics, A/B) │
                        └──────────────────────┘
```

## Planned ML Workflow

1. **Data understanding & modeling** — explore a public ads/CTR dataset,
   define the conceptual data model (users, ads, impressions, clicks,
   context), and establish train/validation/test splits that respect time.
2. **CTR prediction** — build baseline (logistic regression) through
   stronger (gradient-boosted trees, e.g. LightGBM) CTR models on
   engineered features; evaluate with AUC, LogLoss, calibration.
3. **Ranking system** — move from pointwise CTR prediction to a ranking
   objective (pairwise/listwise), evaluate with NDCG/MAP, and combine
   predicted CTR with bid/value to produce a final ranked list.
4. **Recommendation / candidate generation** — implement a retrieval stage
   (e.g. embedding-based nearest-neighbor or collaborative filtering) that
   narrows the full ad inventory to a candidate set before ranking.
5. **Experimentation** — simulate an offline A/B evaluation framework to
   compare model variants (e.g. counterfactual/replay evaluation, or a
   simple simulated online environment).

## Repository Structure

```
ads-ranking-system/
├── configs/              # YAML configuration (paths, seeds, model/experiment params)
├── data/
│   ├── raw/              # Raw, immutable input data (not committed)
│   └── processed/        # Processed/feature-engineered data (not committed)
├── notebooks/            # Exploratory analysis notebooks
├── src/
│   ├── data/             # Data loading & splitting
│   ├── features/         # Feature engineering
│   ├── retrieval/        # Candidate generation / retrieval
│   ├── models/           # CTR / ranking model implementations
│   ├── evaluation/       # Classification, ranking & ads business metrics
│   └── utils/            # Config loading, logging, shared helpers
├── experiments/          # Per-stage experiment configs, results & artifacts
├── tests/                # Unit tests
├── docs/                 # Architecture & data model documentation
├── requirements.txt
├── Makefile
└── README.md
```

## Roadmap / Milestones

- [ ] **Phase 1 — Data understanding and modeling**: dataset selection,
      EDA, conceptual data model, train/val/test split strategy
      — ✅ dataset selected and a reproducible acquisition/validation layer
      added (Criteo Display Advertising Challenge, sample variant; see
      [`docs/criteo_dataset.md`](docs/criteo_dataset.md)); EDA and modeling
      still open
- [ ] **Phase 2 — CTR prediction**: baseline and gradient-boosted CTR
      models, offline evaluation (AUC, LogLoss, calibration)
- [ ] **Phase 3 — Ranking system**: pairwise/listwise ranking, NDCG/MAP
      evaluation, value-based ranking (CTR × bid)
- [ ] **Phase 4 — Recommendation / candidate generation**: retrieval stage
      to narrow inventory ahead of ranking
- [ ] **Phase 5 — Experimentation**: offline A/B / counterfactual
      evaluation framework for comparing model variants

## Status

The repository foundation is in place, and Phase 1's dataset foundation is
done: the project uses the **Criteo Display Advertising Challenge dataset
(sample variant, ~100K rows)** as its raw CTR data, with a reproducible
download/validation layer (`src/data/criteo.py`, `src/data/validation.py`,
`scripts/download_criteo.py`). No models are implemented yet, and no
feature engineering has started — that's the next step within Phase 1.

## Dataset

This project uses the [Criteo Display Advertising Challenge](https://www.kaggle.com/c/criteo-display-ad-challenge/)
dataset (sample variant). To download it locally:

```bash
make download-data      # or: python3 scripts/download_criteo.py
```

This fetches the dataset into `data/raw/criteo/` (git-ignored, per the
dataset's non-redistribution terms). See
[`docs/criteo_dataset.md`](docs/criteo_dataset.md) for the verified schema,
label definition, missing-value convention, licensing terms, and why this
dataset does **not** support candidate generation or CVR prediction (no
user/ad/session identifiers or conversion label exist in the raw data).

## Getting Started

```bash
make setup          # create virtual environment and install dependencies
make download-data  # download the Criteo sample dataset
make test           # run the test suite
```

See [`docs/architecture.md`](docs/architecture.md) for system design details,
[`docs/data_model.md`](docs/data_model.md) for the conceptual data model, and
[`docs/criteo_dataset.md`](docs/criteo_dataset.md) for the concrete dataset
this project runs on.
