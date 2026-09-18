# Next Phase Design — Assembling the Pipeline

This document identifies the single highest-value next step for this project, based on a fresh
inspection of the current repository (120 passing tests, clean working tree) and all existing
design/results documentation. It does not implement anything.

## What currently exists (verified against the repo, not assumed)

- **Retrieval** (`src/retrieval/`): cohort classification, a popularity baseline, and item-based
  collaborative filtering — the honest negative result that CF doesn't beat popularity once
  already-seen campaigns are excluded ([`collaborative_filtering.md`](collaborative_filtering.md)).
- **Ranking** (`src/models/ctr_model.py`, `src/features/attribution_features.py`): a full-feature
  CTR model and a pre-serve (cat1-9-excluded) variant, both trained, evaluated, and documented
  ([`ctr_model.md`](ctr_model.md), [`pre_serve_ctr_model.md`](pre_serve_ctr_model.md)).
- **Integration** (`src/evaluation/ranking_integration.py`): a restricted, verified-exposure-only
  check of whether the pre-serve model's scores agree with real outcomes
  ([`ranking_integration_results.md`](ranking_integration_results.md)).

**What does not exist**: any single, callable artifact that actually runs the assembled pipeline —
classify a user, retrieve their candidates, score them, return a ranked list. Every component above
is proven correct and validated *in isolation*, via five separate scripts
(`scripts/run_popularity_baseline.py`, `run_collaborative_filtering_experiment.py`,
`run_ctr_experiment.py`, `run_pre_serve_ctr_experiment.py`, `run_ranking_integration_experiment.py`)
whose outputs have never been connected into one coherent "here is what the system actually
recommends" artifact. This is the gap this document addresses.

Notably, this exact gap was already identified — but never built — in this project's own prior
work: [`ranking_integration_design.md`](ranking_integration_design.md)'s explicit fallback plan was
"a qualitative, outcome-free demonstration: run the retrieval → cat-free-CTR-scoring pipeline for a
handful of example users, show the resulting ranked candidate list." Part B's real result turned out
large enough to report a quantitative metric, so this qualitative assembly step was never actually
produced as a standalone artifact. It remains valuable independent of that.

## Options considered and ruled out

Per the explicit constraints for this review:

- **CVR** — the conversion label is sparse (2.65% of impressions, 7.33% of clicks, per
  `attribution_dataset_verification.md`), and nothing in the current results creates new pressure
  to model it now; it was already deliberately deferred with a clear rationale
  (`ranking_problem_design.md` Q3). No new evidence justifies revisiting that now.
- **Embeddings** — `EmbeddingCandidateGenerator` remains an intentional placeholder. Item-based CF,
  the more classically-suited method for this data's sparsity, already underperformed popularity;
  an embedding method is a strictly higher-complexity approach with no evidence it would do better,
  and building it would violate the "unless existing evidence clearly justifies it" bar.
- **Live serving** — explicitly out of scope for this project throughout every design document; not
  reconsidered here.
- **Aggressive hyperparameter tuning** — the full-feature CTR model's best iteration landed exactly
  on its 500-round ceiling (`ctr_model.md`), suggesting a higher round budget alone (not a search)
  could be a small, well-justified, low-risk follow-up. This is worth doing but is a **minor
  refinement**, not a capability addition — it doesn't fill a gap, it polishes an existing result.
  Noted as a candidate for a *later*, smaller task, not this one.

## Recommended next step: an assembled retrieval → ranking inference pipeline

### 1. Why it matters for the target role

An ads-ranking interview will ask, directly or indirectly, "does this actually work end to end, or
is it five separate scripts?" Right now the honest answer is the latter. A callable pipeline that
takes a user and produces a ranked, scored candidate list is the concrete artifact that turns five
validated experiments into an actual **system** — the thing a Data Engineer / ML / Ads Ranking role
is ultimately evaluated on: can you assemble validated components into something that runs, not
just prove each piece works in isolation. It also directly demonstrates the specific, correct
handling of a subtlety this project has spent significant effort establishing — that a not-yet-served
candidate must be scored with a *different, narrower* feature set than an already-logged impression
(`ranking_problem_design.md` Q5) — by actually enforcing it in running code, not just in a design
document.

### 2. What gap it fills

Every architecture diagram in this project (this README, `attribution_modeling_design.md` §10)
draws the same box-and-arrow pipeline: cohort classification → retrieval → CTR scoring →
evaluation. None of those arrows have ever been code — each stage's script starts from raw data and
produces its own isolated metrics. This step **connects the arrows**, with no new modeling
capability, using only already-validated pieces.

### 3. What data/features are legitimately available

Nothing new. This is the central design constraint, not an afterthought:

- **Cohort assignment**: reuse `src.retrieval.cohort.classify_users`, computed from train-period
  history exactly as already validated. A user absent from the training cohort assignment (a
  genuinely new user) defaults to **cold** — the existing, already-justified convention
  (`evaluate_retrieval_by_cohort`'s `default_cohort` behavior), not a new rule invented for this
  step.
- **Candidates**: reuse `PopularityCandidateGenerator` (cold) / `CollaborativeFilteringCandidateGenerator`
  (warm) exactly as fit in the existing experiments — same training data, same configuration.
- **Scoring features**: **only** `PRE_SERVE_FEATURE_COLUMNS`
  (`src/features/attribution_features.py`) — campaign identity, user/pair history, frozen campaign
  aggregates. Explicitly **not** `cat1`–`cat9`, and not the full-feature model, since the candidates
  being scored have not been served and therefore have no impression-level context to read —
  exactly the distinction `pre_serve_ctr_model.md` exists to enforce. This is the one place in this
  step where getting it wrong would silently reintroduce a mismatch this project already spent a
  full phase resolving.
- **"As of" time**: the pipeline must accept a point-in-time cutoff and compute a user's history
  features strictly before it, reusing `build_ctr_features`'s existing, tested point-in-time logic
  unchanged. The natural, non-arbitrary choice is the existing test-period boundary (day 25) — not
  a new cutoff invented for this demo, which would risk a new, unvalidated leakage surface.

### 4. How it would be evaluated

Two layers, neither inventing a new metric:

- **Consistency check (primary correctness bar)**: running the assembled pipeline for the full
  cold and warm test populations must reproduce the *already-validated* numbers — cold/warm
  Recall@10 from `popularity_baseline.md`/`collaborative_filtering.md`, and the pairwise-agreement
  numbers from `ranking_integration_results.md`. This is a regression/consistency test, not a new
  benchmark: if the assembled version doesn't reproduce numbers already trusted, the assembly is
  wrong, not the numbers.
- **Qualitative demonstration**: for a small, **deterministically selected** (not hand-picked) set
  of example users — e.g. the first N users by `uid` within the already-computed verified-contrast
  set from `ranking_integration_results.md`, one cold and one warm — show the actual ranked output:
  retrieved candidates, pre-serve predicted CTR per candidate, and, where independently verifiable,
  the real outcome. This is exactly the fallback artifact `ranking_integration_design.md` already
  specified and never built.

### 5. What could go wrong or introduce leakage

- **Wrong feature set for candidate scoring** — using the full-feature model or including
  `cat1`–`cat9` when scoring a not-yet-served candidate would silently violate the pre-serve
  distinction this project already established. Mitigation: the pipeline module should make it
  structurally impossible to pass the full feature set to the scoring step (e.g. only import
  `PRE_SERVE_FEATURE_COLUMNS`, never `FEATURE_COLUMNS`, in this module).
- **A new, ad-hoc temporal cutoff** — inventing a fresh "as of now" timestamp for the demo instead
  of reusing the established test boundary would create an unvalidated leakage surface outside
  everything already tested. Mitigation: reuse `split_by_day` and the existing config boundaries
  unchanged; do not add a new cutoff parameter beyond what's already validated.
- **Cherry-picked example users** — hand-selecting users whose output happens to look good would
  misrepresent the (real, modest) results already reported. Mitigation: deterministic selection
  (fixed seed or first-N-by-id), stated explicitly in the output itself.
- **Implying live capability** — presenting this as anything beyond a batch, offline, already-logged
  replay would contradict the project's own scope boundaries. Mitigation: explicit labeling in
  output and docs, consistent with existing "not a real served slate" language.
- **Silent divergence from existing components** — reimplementing cohort/retrieval/scoring logic
  instead of importing it would let this new module drift from the validated originals. Mitigation:
  this module contains **only orchestration** — every substantive computation is a direct call into
  already-tested `src/retrieval/`, `src/features/`, `src/models/` code.

### 6. Exact files/code that would change

**New**:
- `src/pipeline.py` — orchestration only: `recommend_for_user(uid, as_of_day, k)` and a batch
  variant for the consistency check. Imports `classify_users`, `PopularityCandidateGenerator`,
  `CollaborativeFilteringCandidateGenerator`, `PRE_SERVE_FEATURE_COLUMNS`, `build_ctr_features`,
  `CTRModel` — no new modeling logic.
- `scripts/run_pipeline_demo.py` — runs the consistency check against already-published numbers,
  then produces the deterministic, labeled qualitative example-user output.
- `tests/test_pipeline.py` — synthetic-fixture tests: cold/warm routing correctness, pre-serve-only
  feature usage (asserting `cat1`–`cat9` never reach the scoring call), deterministic user
  selection, graceful handling of an unknown user (defaults to cold/popularity), output sorted by
  descending predicted score.
- `docs/pipeline_demo.md` — written when this is implemented, documenting the consistency-check
  result and the example-user output (not created now, per "design only").

**Unchanged** (reused, not modified): `src/retrieval/`, `src/features/attribution_features.py`,
`src/models/ctr_model.py`, `src/evaluation/*`, `src/data/attribution.py`, all existing experiment
scripts, all existing docs. This step's low risk comes directly from touching none of the
already-validated code.

**Possibly**: a small `attribution.pipeline_demo` section in `configs/config.yaml` (e.g. number of
example users, selection seed) — consistent with this project's existing config-driven convention,
not a new pattern.

## Recommended Next Implementation Step

**Build `src/pipeline.py`** — a thin orchestration module implementing `recommend_for_user(uid,
as_of_day, k)`, which classifies the user's cohort, retrieves candidates via the existing,
unmodified retrieval components, scores them with the existing pre-serve CTR model using only
`PRE_SERVE_FEATURE_COLUMNS`, and returns a ranked list — then validate it with a consistency check
against the already-published Recall@10 and pairwise-accuracy numbers before producing the
deterministic, labeled qualitative example-user demonstration this project's own prior design
already called for.
