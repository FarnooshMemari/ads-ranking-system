# Experiments

This directory organizes experiment stages for the ads ranking system. Each
subfolder corresponds to a stage of the project roadmap (see the main
[README.md](../README.md)) and is where that stage's run configurations,
results, and artifacts will be stored once modeling begins.

## Structure

```
experiments/
├── baseline/        # Phase 1/2 — simple baseline models (e.g. logistic
│                     # regression CTR) used as a reference point
├── ctr_model/        # Phase 2 — CTR prediction experiments (e.g. LightGBM
│                     # variants, feature ablations)
└── ranking_model/    # Phase 3 — ranking experiments (pairwise/listwise
                       # models, value-based ranking)
```

Each experiment subfolder is expected to eventually hold, per run:

- The configuration used (typically a copy or override of
  `configs/config.yaml`)
- Evaluation results (metrics from `src/evaluation/`)
- Trained model artifacts and/or logs

## Status

These folders are currently empty placeholders. No experiments have been
run yet — this structure is created ahead of Phase 1 (dataset selection and
data modeling) so that later phases have a consistent place to record
results.
