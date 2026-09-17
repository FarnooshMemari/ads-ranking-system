"""Observed advertising business metrics.

These summarize actual outcomes over a set of impressions/interactions
(e.g. for reporting or comparing experiment groups), as distinct from
model-quality metrics in ``classification_metrics.py`` and
``ranking_metrics.py``. Implemented starting in Phase 2/5 as real
interaction data becomes available.
"""

from typing import Any


def compute_ctr(impressions: Any, clicks: Any) -> float:
    """Compute observed click-through rate (CTR).

    CTR = clicks / impressions.

    Args:
        impressions: Impression records or count.
        clicks: Click records or count.

    Returns:
        The observed CTR.
    """
    raise NotImplementedError("Implemented in Phase 2: CTR prediction.")


def compute_cvr(clicks: Any, conversions: Any) -> float:
    """Compute observed conversion rate (CVR).

    CVR = conversions / clicks.

    Args:
        clicks: Click records or count.
        conversions: Conversion records or count.

    Returns:
        The observed CVR.
    """
    raise NotImplementedError("Implemented in Phase 5: experimentation.")


def compute_roas(revenue: Any, spend: Any) -> float:
    """Compute Return on Ad Spend (ROAS).

    ROAS = revenue attributed to ads / advertising spend.

    Args:
        revenue: Attributed revenue (e.g. from conversions).
        spend: Advertising spend over the same period.

    Returns:
        The ROAS value.
    """
    raise NotImplementedError("Implemented in Phase 5: experimentation.")
