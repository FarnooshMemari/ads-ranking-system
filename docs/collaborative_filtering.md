# Warm-Cohort Collaborative Filtering vs. Popularity — Lift Experiment

**Question this document answers**: does personalized retrieval provide measurable lift over the
popularity baseline for the warm cohort — the only population (4.2% of users, in the train split)
for whom personalization is even theoretically possible (see
[`attribution_modeling_design.md`](attribution_modeling_design.md) §3B)?

**Short answer, stated up front because it's easy to miss in a wall of metrics**: **no, not
genuinely.** An unfiltered comparison shows a large apparent lift, but that lift is almost entirely
explained by collaborative filtering re-recommending campaigns a user already clicked in training —
a trivial "show them their own history again" effect, not evidence of transferable, cross-campaign
personalization. Once previously-seen campaigns are excluded (the scientifically meaningful test),
collaborative filtering **underperforms** the popularity baseline. See Results below for the exact
numbers.

## 1. Model choice

### What was inspected

Microsoft Recommenders' CPU-feasible implicit-feedback options were reviewed directly against the
source (verified via GitHub, not from memory):

| Algorithm | Verified dependency footprint | Fit for this data? |
|---|---|---|
| **ALS** (`recommenders.models` via PySpark) | Requires PySpark | **Ruled out** — this project's constraints explicitly exclude Spark infrastructure. |
| **SAR** (`recommenders/models/sar/sar_singlenode.py`) | Verified: only `numpy`, `pandas`, `scipy.sparse`, and internal `recommenders.utils.python_utils` helpers (`cosine_similarity`, `jaccard`, `lift`, `cooccurrence`, ...) — no Spark, no GPU | **Chosen as the design pattern** — see below. |
| **Cornac BPR** (latent-factor, pairwise ranking loss) | CPU-feasible, but learns a per-user embedding | Considered and rejected as primary — see rationale. |
| **NCF / LightGCN** | TensorFlow, more complex, heavier setup | Not considered — this project's scope control already excludes deep sequence/embedding models for v1 (design doc §11). |

### Why SAR's design pattern, self-implemented

**SAR (Simple Algorithm for Recommendation)** is a **memory-based, item-to-item similarity**
method: it computes an item-item similarity matrix from co-occurrence in user histories, then
scores each candidate item for a user as a weighted sum over the user's own historical items,
weighted by their similarity to the candidate. Chosen over Cornac BPR (a latent-factor method that
*learns* a compressed embedding per user) for one specific, verified reason: **BPR needs enough
interactions per user to fit a meaningful embedding; SAR does not need to fit anything per user at
all** — it directly uses whatever history exists, however little. Given this project's own
verified finding that even "warm" users average only a handful of distinct clicked campaigns
(§3B), a method that doesn't require estimating a compressed per-user representation from very few
data points is the better-justified choice — not chosen because it's "the Microsoft one," but
because its assumptions match this data's actual density, which happens to also be a documented
reason SAR performs competitively in the recommender-systems literature despite its simplicity.

**Self-implemented** (`src/retrieval/candidate_generator.py`:
`CollaborativeFilteringCandidateGenerator`) using only numpy/pandas — no new dependency, consistent
with how the rest of this project's data/retrieval code is built (the DAC loader, the Attribution
loader, and the popularity baseline are all self-written rather than imported). This uses the exact
same similarity vocabulary SAR does (`jaccard`, `cooccurrence`), verified against SAR's own source,
so the design choice is directly legible to anyone familiar with that library, without taking on
its dependency tree for a dataset it was never built to handle.

**Alternatives explicitly not chosen, and why**: ALS (Spark dependency, out of scope). Cornac BPR
(per-user embedding, poor fit for this data's sparsity — a plausible future experiment once this
memory-based approach's result is understood, not chosen for the first test). Deep models (scope
control, design doc §11).

## 2. Interaction definition

Identical positive-interaction definition to the popularity baseline (design doc §2's decision,
unchanged): **a click** (`click == 1`) on a training-period impression. No conversions, no
`cost`/`cpo`, no post-outcome fields are used as features or signal — consistent with the design
doc's leakage exclusions (§5, §7). Affinity strength for a `(user, campaign)` pair is the training-
period **click count** (repeated clicks on the same campaign count as stronger affinity), not just
a binary flag.

## 3. Training protocol

Two restrictions are enforced, both directly required by the task and both testable:

1. **Chronological**: only rows in the `train` split (`src.data.attribution.split_by_day`, days
   0–19) are used to fit anything — no validation- or test-period row ever enters `fit()`. Verified
   by `tests/test_collaborative_filtering.py::test_train_only_interaction_construction_no_leakage_end_to_end`,
   which constructs a co-click that only exists in the test period and confirms it produces zero
   similarity when the generator is fit on the correctly-split training data.
2. **Warm-cohort-restricted**: `CollaborativeFilteringCandidateGenerator` is fit **only** on the
   already-classified warm cohort's own training interactions
   (`train_df[train_df["uid"].isin(warm_uids)]`) — 1,500,519 of the 10,852,826 training rows,
   194,893 users. This is a deliberate choice, not the only possible one: fitting on the full
   training set (cold users included) was considered and rejected, because the cold majority's
   overwhelmingly single-touch interactions would dilute the item-item similarity matrix with
   co-click pairs that don't reflect any real multi-item preference (a cold user's one click
   contributes no *relative* signal — see design doc §3B) without adding anything a genuinely
   warm user's own history couldn't already provide.

## 4. Evaluation protocol

- **Evaluation universe**: warm test users (per the already-established train-period cohort
  assignment) with at least one test-period click — 56,353 users, the identical filter used for
  the warm cohort's popularity-baseline numbers, so the comparison is apples-to-apples.
- **Ground truth**: each user's distinct clicked campaigns in the test period (days ≥ 25).
- **Popularity baseline**: fit on the **full** training set (not warm-restricted) — this
  intentionally reproduces the exact, already-committed popularity baseline experiment
  (`scripts/run_popularity_baseline.py`) rather than a new variant, so the lift comparison is
  against a number that was already independently verified, not a moving target.
- **Seen-campaign handling — the decision the task asked to be documented explicitly**: two CF
  variants are run and reported:
  - **`cf` (exclude_seen=False, primary comparison)**: candidates may include campaigns the user
    already clicked in training. Chosen as *primary* specifically because the popularity baseline
    it's compared against also does not exclude previously-seen campaigns — matching protocol is
    what makes the lift number meaningful.
  - **`cf_exclude_seen` (secondary)**: previously-clicked campaigns are removed before ranking —
    the more realistic serving policy (don't keep showing someone what they've already seen), and,
    as the results below show, the more informative test of whether the similarity matrix captures
    anything that generalizes beyond a user's own history.

## 5. Results

Run against the full 16,468,027-row dataset (`scripts/run_collaborative_filtering_experiment.py`,
`similarity=jaccard`, `K=10`):

```
Cohorts (train-period only): cold=4,439,942 (95.80%), warm=194,893 (4.20%)
CF fit on warm-cohort training data only: 1,500,519 rows, 194,893 users with affinity, 674 campaigns.
Warm test users with >=1 test-period click (evaluation universe): 56,353

method                  Recall@k   HitRate@k      NDCG@k
popularity                0.1731      0.2019      0.0914
cf                        0.3983      0.4350      0.2245
cf_exclude_seen           0.0952      0.1229      0.0609
```

| Comparison | Recall@10 lift | Hit Rate@10 lift | NDCG@10 lift |
|---|---|---|---|
| **cf vs. popularity** (unfiltered, primary) | +0.2252 abs, **+130.12%** rel | +0.2332 abs, +115.53% rel | +0.1331 abs, +145.54% rel |
| **cf_exclude_seen vs. popularity** (secondary, fair test) | **−0.0779 abs, −45.01% rel** | −0.0789 abs, −39.11% rel | −0.0305 abs, −33.33% rel |

### Why the two variants tell almost opposite stories

The unfiltered `cf` number looks like a dramatic win. It is not what it appears to be. Because
`exclude_seen=False` allows a campaign the user already clicked in training to be recommended
again, and because many users' test-period clicks are simply a *repeat* of a campaign they'd
already engaged with, a large share of `cf`'s apparent recall comes from an effect indistinguishable
from "recommend the user's own history back to them" — which requires no collaborative signal
between different campaigns at all. The moment previously-seen campaigns are excluded
(`cf_exclude_seen`), forcing the model to recommend something *new* based only on item-item
similarity, performance **drops below the popularity baseline**. This is the scientifically honest
reading of these two numbers together, not a result to average or split the difference on.

### Coverage

```
Warm test users receiving >=1 CF recommendation: 56,353 / 56,353 (100.00%)
Warm test users receiving ZERO CF recommendations: 0
Distinct campaigns ever recommended by CF: 668 / 674 in CF's training catalog (99.11%) /
    675 in the full dataset catalog (98.96%)
Campaigns with zero warm-cohort training clicks (never eligible for CF recommendation): 1 / 675
```

Every evaluated warm user received a non-empty recommendation list (no cold-start-within-warm
gap), and the similarity matrix has near-complete catalog coverage (668/675 campaigns, essentially
the whole catalog) — coverage is not the limiting factor here. Only one campaign (out of 675) had
zero warm-cohort training clicks and was therefore structurally unrecommendable by this method,
which is a negligible exclusion, not a material gap.

## 6. Baseline comparison, restated plainly

| Metric | Popularity (warm cohort) | CF, fair test (exclude_seen) | Verdict |
|---|---|---|---|
| Recall@10 | 0.1731 | 0.0952 | Popularity wins |
| Hit Rate@10 | 0.2019 | 0.1229 | Popularity wins |
| NDCG@10 | 0.0914 | 0.0609 | Popularity wins |

## 7. Does collaborative filtering provide meaningful lift?

**No.** On the fair, seen-campaign-excluded comparison — the only one that actually tests whether
the item-item similarity captures a transferable preference rather than trivially recommending a
user's own click history back to them — item-based collaborative filtering underperforms the
simple popularity baseline on all three metrics, by 33–45% relatively. The unfiltered comparison's
apparent +130% lift is real in the narrow sense that the numbers are correct, but it is not evidence
of personalization; it is evidence that users often re-click campaigns they've already clicked,
which a much simpler "remind them of what they clicked before" heuristic would capture equally
well without any collaborative-filtering machinery.

This is consistent with, and provides direct empirical confirmation of, the concern already raised
in the design doc's §3B: even the warm cohort's click behavior was found to barely diverge from
population-level popularity (26.99% vs. 29.98% of clicks on globally top-20 campaigns) before any
model was built. This experiment tested that concern directly and it holds: at this cohort size
(194,893 training users) and this history density (most warm users have only 2–3 distinct clicked
campaigns), item-based collaborative filtering has too little genuine cross-item signal per user
to beat a non-personalized baseline once the trivial repeat-click effect is removed.

## 8. Limitations

- **Only one similarity metric family is the primary comparison** (Jaccard); `cooccurrence` is
  implemented and unit-tested but not run as a full second experiment here, since the negative
  result on the fair comparison is already conclusive enough not to warrant tuning across
  similarity variants — doing so would risk exactly the "tune aggressively to maximize test
  performance" the task explicitly asked not to do.
- **No latent-factor method (e.g. BPR) was tried.** SAR's memory-based design was chosen
  specifically because it doesn't need to fit a per-user embedding from sparse data — but this
  experiment does not prove a latent-factor method would do equally poorly; it only tests the
  method actually implemented here. A future experiment could test this directly.
- **The repeat-click effect itself is not decomposed further.** This document identifies that
  `cf`'s apparent lift is driven by re-recommending seen campaigns, but does not separately
  quantify what fraction of `cf`'s hits are exactly this vs. some smaller genuine cross-campaign
  signal mixed in — the `cf_exclude_seen` result already answers the practical question (no net
  lift either way), so this decomposition was not pursued further.
- **Single train/val/test split.** No cross-validation or multiple time-window replication was
  run; the result reflects this one chronological split (days 0–19 train, 25–30 test).
- **Catalog-size caveat carried over from the design doc**: with 675 campaigns, this is an easier
  retrieval problem than a large-catalog production system, for both methods equally.
