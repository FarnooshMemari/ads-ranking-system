"""Cold/warm user cohort classification.

Cohort membership is defined **only** from a user's pre-split ("train-period")
click history — see docs/attribution_modeling_design.md §3B/§4. A user is
"warm" if they clicked on at least ``warm_min_distinct_campaigns`` distinct
campaigns during the training period; otherwise "cold". The default threshold
(2) is the verified minimum below which no relative-preference signal exists
to learn from — 94.61% of users fall below it.

This module performs no time filtering itself: callers must pass an
already-restricted training-period DataFrame (e.g. from
``src.data.attribution.split_by_day``), so the leakage boundary stays
explicit and independently testable at the call site rather than hidden
inside this function.
"""

from typing import Dict

import pandas as pd

COLD = "cold"
WARM = "warm"
DEFAULT_WARM_MIN_DISTINCT_CAMPAIGNS = 2


def classify_users(
    train_df: pd.DataFrame,
    warm_min_distinct_campaigns: int = DEFAULT_WARM_MIN_DISTINCT_CAMPAIGNS,
) -> pd.Series:
    """Classify each user present in ``train_df`` as ``'cold'`` or ``'warm'``.

    Args:
        train_df: Training-period impressions only. Must already be
            restricted to timestamps strictly before the evaluation split
            boundary (see module docstring) — no filtering happens here.
        warm_min_distinct_campaigns: Minimum number of distinct campaigns a
            user must have **clicked** in ``train_df`` to be classified
            ``'warm'``. Configurable — see ``configs/config.yaml``:
            ``attribution.cohort.warm_min_distinct_clicked_campaigns``.

    Returns:
        A ``pandas.Series`` indexed by ``uid`` (name ``"uid"``), with values
        ``'cold'`` or ``'warm'``, for every user present in ``train_df`` —
        including users with zero clicks (they are always ``'cold'``). Users
        entirely absent from ``train_df`` (never seen before the split) are
        not included here; callers should treat such users as ``'cold'`` by
        default (the coldest possible case), as
        ``src.evaluation.retrieval_metrics.evaluate_retrieval_by_cohort``
        does via its ``default_cohort`` argument.
    """
    all_users = train_df["uid"].unique()
    clicked = train_df[train_df["click"] == 1]
    distinct_clicked = clicked.groupby("uid")["campaign"].nunique()
    distinct_clicked = distinct_clicked.reindex(all_users, fill_value=0)

    cohort = pd.Series(COLD, index=distinct_clicked.index, dtype=object)
    cohort[distinct_clicked >= warm_min_distinct_campaigns] = WARM
    cohort.index.name = "uid"
    cohort.name = "cohort"
    return cohort


def cohort_counts(cohort: pd.Series) -> Dict[str, int]:
    """Summarize a cohort assignment Series as counts per label.

    Args:
        cohort: Output of ``classify_users``.

    Returns:
        ``{"cold": n_cold, "warm": n_warm}`` (0 for a label with no users).
    """
    counts = cohort.value_counts()
    return {COLD: int(counts.get(COLD, 0)), WARM: int(counts.get(WARM, 0))}
