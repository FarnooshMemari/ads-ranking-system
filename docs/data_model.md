# Data Model

This document describes the conceptual data model the system is designed
around. It is dataset-agnostic for now; the concrete source dataset(s) and
their exact schemas will be selected and documented here in Phase 1.

## Core Entities

### User
**Purpose**: represents the person seeing ads; the primary subject of
personalization.

| Field              | Description                                      |
|--------------------|---------------------------------------------------|
| `user_id`          | Unique user identifier                             |
| `demographic_*`    | Age bucket, gender, region, language, etc.         |
| `device_type`      | Mobile OS/device class                             |
| `interest_*`       | Declared or inferred interest/category affinities  |
| `historical_ctr`   | User's rolling historical click-through rate       |

**Future ML features**: interest/category embeddings, recent-engagement
sequence features (for candidate generation), rolling CTR/CVR by ad
category, recency/frequency of app usage.

### Advertiser
**Purpose**: the business/entity paying for ads; owns one or more
campaigns. Mainly used for aggregation, budget/business constraints, and
preventing over-exposure to a single advertiser.

| Field              | Description                                      |
|--------------------|---------------------------------------------------|
| `advertiser_id`    | Unique advertiser identifier                       |
| `industry`         | Advertiser's industry/vertical                     |
| `account_tier`     | Spend tier / account size, if relevant             |

**Future ML features**: advertiser-level historical CTR/CVR priors (useful
for cold-start campaigns), industry as a categorical feature.

### Campaign
**Purpose**: a grouping of ads under one advertiser, sharing a budget,
targeting configuration, and objective (e.g. awareness, conversions).

| Field              | Description                                      |
|--------------------|---------------------------------------------------|
| `campaign_id`      | Unique campaign identifier                         |
| `advertiser_id`    | FK → Advertiser                                    |
| `objective`        | Optimization goal (clicks, conversions, reach, ...) |
| `budget`           | Total or daily budget                              |
| `bid_strategy`     | How bids are set (manual, target CPA, etc.)         |

**Future ML features**: campaign objective as context for ranking (e.g.
weight predicted CVR higher for conversion-objective campaigns),
budget-pacing signals for the ranking/allocation stage.

### Ad
**Purpose**: the individual creative/item being ranked and shown to users;
the unit scored by candidate generation and ranking.

| Field              | Description                                      |
|--------------------|---------------------------------------------------|
| `ad_id`            | Unique ad/creative identifier                      |
| `campaign_id`      | FK → Campaign                                      |
| `category`         | Ad/product category (e.g. IAB taxonomy)            |
| `bid`              | Advertiser bid / budget-related value               |
| `creative_features`| Format, text/image embedding features (if available) |
| `historical_ctr`   | Ad's rolling historical click-through rate         |

**Future ML features**: ad/creative embeddings (for embedding-based
retrieval in `src/retrieval/`), category as a categorical feature,
user × ad-category cross features.

### Impression
**Purpose**: records that a specific ad was shown to a specific user in a
specific context. The unit of exposure — every ranking decision produces
impressions, and impressions are the population from which interactions
are (or are not) observed.

| Field              | Description                                      |
|--------------------|---------------------------------------------------|
| `impression_id`    | Unique impression identifier                       |
| `user_id`          | FK → User                                          |
| `ad_id`            | FK → Ad                                            |
| `timestamp`        | Event time (used for time-based splitting)         |
| `surface`          | App surface/placement (e.g. For You feed)          |
| `position`         | Slot position in the feed/session                  |
| `session_id`       | Session grouping for sequential context            |

**Future ML features**: position (for position-bias correction), surface
as a categorical context feature; forms the base rate for CTR (clicks /
impressions).

### Interaction
**Purpose**: records a user's engagement with an impression — click,
conversion, or other signal (e.g. dwell time). This is where CTR/CVR
labels come from.

| Field              | Description                                      |
|--------------------|---------------------------------------------------|
| `interaction_id`   | Unique interaction identifier                      |
| `impression_id`    | FK → Impression                                    |
| `type`             | `click`, `conversion`, or other engagement type    |
| `timestamp`        | Event time                                          |
| `dwell_time`       | Optional engagement signal                          |

**Future ML features**: `clicked` (binary, from presence of a `click`
interaction) is the primary CTR target; `converted` is the CVR target;
time-to-conversion can inform value-based ranking.

## Relationships

```
Advertiser (1) ───< Campaign (1) ───< Ad
                                        │
User (1) ───< Impression >─── (1) ──────┘
                   │
                   ▼
              Interaction (0..N)
        (click / conversion / ...)
```

Each `Impression` row represents a single `(user, ad, context)` exposure.
Zero or more `Interaction` rows may reference that impression (e.g. one
`click` interaction, optionally followed by one `conversion`). Joining
`Impression` with its `Interaction`s (or their absence) produces the
primary training table for the CTR model: one row per impression, with a
derived `clicked` label.

## Label Definition

- **CTR target (`clicked`)**: binary, 1 if the user clicked the impressed
  ad, 0 otherwise. Highly imbalanced (typical CTR is low single-digit
  percent or less) — class imbalance handling (class weighting, threshold
  calibration) is part of Phase 2.
- **Ranking relevance**: derived from engagement signals (click, dwell
  time, conversion) to produce a graded or binary relevance label per
  candidate within a ranking group (Phase 3).
- **Ranking group / query**: for listwise/pairwise ranking evaluation,
  impressions are grouped by `(user_id, session_id)` or a comparable
  request-level key, analogous to a "query" in learning-to-rank.

## Splitting Strategy

Splits are time-based by default (`configs/config.yaml`:
`experiment.train_val_test_split.strategy`) — train on earlier
impressions, validate/test on later ones — to reflect realistic
deployment and avoid leakage from future user behavior into training.

## Cold Start

Both new users (little/no interaction history) and new ads (little/no
serving history) are expected. Feature engineering (Phase 1/2) should
include fallback/default features (e.g. category-level priors) so the
model degrades gracefully rather than failing on unseen IDs.

## Notes

This schema will be finalized against the actual dataset selected in
Phase 1 (a public ads/CTR dataset). Field names/types here are
illustrative and subject to change once real data is in hand.
