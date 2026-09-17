# Architecture

This document describes the intended system design for the Personalized Ads
Ranking System: a **prototype** advertising ranking system inspired by the
general shape of large-scale industry recommendation architectures (e.g.
multi-stage retrieval → ranking pipelines used by major ads/feed
platforms), implemented here at a scope appropriate for a portfolio
project. It is not a reproduction of any specific company's production
system, and it does not include real-time serving, online infrastructure,
or production-scale data volumes. It reflects the target design the
project will grow into across its phases; components not yet implemented
are marked as such.

## Overview

An ads ranking system turns raw user activity into a ranked list of ads
through a sequence of stages, each narrowing or refining the set of
candidates before the next:

```
┌────────────────────┐
│   User Events       │  Impressions, clicks, conversions, and other
│                      │  raw interaction records.
└──────────┬──────────┘
           ▼
┌────────────────────┐
│  Data Processing     │  Cleaning, joining, and structuring raw events
│  (src/data/)         │  into the conceptual entities in data_model.md.
└──────────┬──────────┘
           ▼
┌────────────────────┐
│  Feature Engineering │  User, ad, and context features plus cross/
│  (src/features/)     │  interaction features, built into a model-ready
│                      │  matrix.
└──────────┬──────────┘
           ▼
┌────────────────────┐
│  Candidate           │  Cheap, high-recall retrieval (e.g. embedding
│  Generation           │  similarity, collaborative filtering) — narrows
│  (src/retrieval/,     │  a large ad pool down to a small candidate set.
│  Phase 4)             │
└──────────┬──────────┘
           ▼
┌────────────────────┐
│  Ranking Model        │  Scores and orders candidates: CTR/engagement
│  (src/models/,        │  prediction (Phase 2) combined with bid/value
│  Phase 2 & 3)         │  and business constraints (Phase 3).
└──────────┬──────────┘
           ▼
┌────────────────────┐
│  Evaluation           │  Offline model-quality metrics (AUC, NDCG),
│  (src/evaluation/,    │  observed business metrics (CTR, CVR, ROAS),
│  Phase 5)             │  and simulated experimentation.
└────────────────────┘
```

Within the funnel above, candidate generation and ranking together form
the classic two-stage "large ad pool → retrieval → ranking" pattern:

```
Large ad pool
      │
      ▼
Candidate generation / retrieval  (src/retrieval/)
      │
      ▼
Ranking model                     (src/models/)
```

## Components

### 1. Data Layer (`src/data/`)
Loads raw interaction/impression data and produces reproducible,
time-respecting train/validation/test splits. See
[`data_model.md`](data_model.md) for the conceptual schema.

### 2. Feature Engineering (`src/features/`)
Transforms raw user, ad, and context records into a model-ready feature
matrix: categorical encoding, numerical normalization, and cross/interaction
features (e.g. user-category affinity × ad category).

### 3. Candidate Generation / Retrieval (`src/retrieval/`, Phase 4)
A retrieval stage that narrows the full ad inventory to a small candidate
set per request, using techniques such as embedding-based nearest-neighbor
search (`EmbeddingCandidateGenerator`) or collaborative filtering
(`CollaborativeFilteringCandidateGenerator`). Optimized for recall and
latency, not precision — the ranking stage refines the final order.

### 4. CTR Prediction (`src/models/ctr_model.py`, Phase 2)
A pointwise binary classifier estimating P(click | user, ad, context).
Planned progression: logistic regression baseline → gradient-boosted trees
(LightGBM). Evaluated with the classification metrics in
`src/evaluation/classification_metrics.py` (AUC, LogLoss, calibration).

### 5. Ranking (`src/models/ranking_model.py`, Phase 3)
Produces the final ordering of candidates. Two complementary approaches are
planned: (a) value-based ranking, scoring `expected_value = P(click) × bid`
computed from the CTR model, and (b) a learned pairwise/listwise ranking
model trained directly on relative ordering. Evaluated with the ranking
metrics in `src/evaluation/ranking_metrics.py` (NDCG@K, Recall@K, MAP@K).

### 6. Evaluation & Experimentation (`src/evaluation/`, Phase 5)
Three complementary kinds of evaluation: CTR model quality
(`classification_metrics.py`), ranking quality (`ranking_metrics.py`), and
observed business outcomes (`ads_metrics.py` — CTR, CVR, ROAS). Combined
with a simulated experimentation framework (e.g. counterfactual/replay
evaluation) to compare model variants without requiring a live serving
environment. Experiment runs and their configs/results/artifacts are
organized under `experiments/` (see `experiments/README.md`).

## Design Principles

- **Pointwise vs. ranking separation** — CTR prediction (pointwise
  probability) and ranking (relative ordering + business value) are kept as
  distinct stages, mirroring how production ads systems separate scoring
  from ranking/allocation logic.
- **Time-respecting evaluation** — splits and evaluation are done
  chronologically wherever possible to avoid leakage and to reflect
  real-world deployment (train on past, evaluate on future).
- **Config-driven** — data paths, model hyperparameters, and experiment
  settings are centralized in `configs/config.yaml` rather than hardcoded,
  so experiments are reproducible and comparable.
- **Offline-first** — since there is no live traffic, the experimentation
  layer focuses on rigorous offline and counterfactual evaluation rather
  than a live online A/B testing system.

## Out of Scope

This project deliberately does not build a data ingestion/ETL pipeline
(Spark/Kafka/Airflow/Dagster) — that is assumed prior expertise. It also
does not include a real-time serving system, low-latency infrastructure,
or production-scale data volumes; the focus is the ML modeling and
evaluation methodology behind ads ranking, implemented as a prototype.

## A Note on Scale

The stage names and diagrams above ("candidate generation", "ranking",
multi-stage funnel) mirror terminology and structure used by large-scale
industry recommendation systems, because that structure is the clearest
way to demonstrate an understanding of how ads ranking works end to end.
This project, however, operates on a small, offline, single-machine
dataset with no production deployment, no real-time serving, and no
claim of TikTok-scale (or any specific company's) infrastructure.
