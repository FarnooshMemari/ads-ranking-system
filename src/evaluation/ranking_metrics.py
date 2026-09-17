"""Ranking-quality metrics for the ranking stage.

Evaluate how well a model orders candidates within a ranking group (e.g.
all candidates shown to a user in one request), as distinct from the
pointwise classification metrics in ``classification_metrics.py`` and the
observed business metrics in ``ads_metrics.py``. Implemented in Phase 3:
ranking system.
"""

from typing import Sequence


def compute_ndcg_at_k(y_true: Sequence[float], y_pred: Sequence[float], k: int = 10) -> float:
    """Compute Normalized Discounted Cumulative Gain at rank k.

    Args:
        y_true: Ground-truth relevance/engagement labels, ordered per
            ranking group (e.g. one user's request).
        y_pred: Predicted ranking scores, ordered per ranking group.
        k: Rank cutoff.

    Returns:
        The NDCG@k score.
    """
    raise NotImplementedError("Implemented in Phase 3: ranking system.")


def compute_recall_at_k(y_true: Sequence[float], y_pred: Sequence[float], k: int = 10) -> float:
    """Compute Recall at rank k.

    Measures the fraction of all relevant items (across the full candidate
    set) that appear in the top-k ranked results — useful for evaluating
    candidate generation as well as ranking.

    Args:
        y_true: Ground-truth relevance/engagement labels, ordered per
            ranking group.
        y_pred: Predicted ranking scores, ordered per ranking group.
        k: Rank cutoff.

    Returns:
        The Recall@k score.
    """
    raise NotImplementedError("Implemented in Phase 3: ranking system.")


def compute_map_at_k(y_true: Sequence[float], y_pred: Sequence[float], k: int = 10) -> float:
    """Compute Mean Average Precision at rank k.

    Args:
        y_true: Ground-truth relevance/engagement labels, ordered per
            ranking group.
        y_pred: Predicted ranking scores, ordered per ranking group.
        k: Rank cutoff.

    Returns:
        The MAP@k score.
    """
    raise NotImplementedError("Implemented in Phase 3: ranking system.")
