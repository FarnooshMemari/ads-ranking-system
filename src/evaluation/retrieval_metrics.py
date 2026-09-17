"""Retrieval-stage evaluation metrics: Recall@K, Hit Rate@K, NDCG@K.

These operate on the standard retrieval formulation — a ground-truth set of
positive items per user versus a ranked candidate list per user — as
distinct from the ranking-stage, per-item-scored metrics in
``ranking_metrics.py``. See docs/attribution_modeling_design.md §8 for why
these must be reported per cohort (cold/warm), never as one blended number:
a blended number would be dominated by the cold majority and would credit
any method, including plain popularity, near-identically.
"""

import math
from typing import Any, Dict, List, Set


def _users_with_ground_truth(y_true: Dict[Any, Set[Any]]) -> List[Any]:
    """Users with at least one ground-truth positive (metrics are undefined without one)."""
    return [u for u, positives in y_true.items() if positives]


def recall_at_k(y_true: Dict[Any, Set[Any]], y_pred: Dict[Any, List[Any]], k: int) -> float:
    """Mean Recall@K across users with at least one ground-truth positive.

    Per-user recall = |ground truth ∩ top-k retrieved| / |ground truth|.

    Args:
        y_true: user -> set of ground-truth positive item ids (e.g. campaigns
            clicked in the test period). Users with an empty set are skipped.
        y_pred: user -> ranked list of retrieved item ids. Only the first
            ``k`` are used, regardless of the list's actual length.
        k: Cutoff rank.

    Returns:
        Mean per-user Recall@K, or 0.0 if no user has a ground-truth positive.
    """
    users = _users_with_ground_truth(y_true)
    if not users:
        return 0.0
    scores = []
    for u in users:
        positives = y_true[u]
        top_k = set(y_pred.get(u, [])[:k])
        scores.append(len(positives & top_k) / len(positives))
    return sum(scores) / len(scores)


def hit_rate_at_k(y_true: Dict[Any, Set[Any]], y_pred: Dict[Any, List[Any]], k: int) -> float:
    """Mean Hit Rate@K: fraction of users with >=1 ground-truth positive in the top-k.

    Args:
        y_true: user -> set of ground-truth positive item ids.
        y_pred: user -> ranked list of retrieved item ids.
        k: Cutoff rank.

    Returns:
        Fraction of eligible users (those with >=1 ground-truth positive)
        who got at least one hit in their top-k, or 0.0 if no user is eligible.
    """
    users = _users_with_ground_truth(y_true)
    if not users:
        return 0.0
    hits = sum(1 for u in users if y_true[u] & set(y_pred.get(u, [])[:k]))
    return hits / len(users)


def ndcg_at_k(y_true: Dict[Any, Set[Any]], y_pred: Dict[Any, List[Any]], k: int) -> float:
    """Mean NDCG@K with binary relevance.

    Args:
        y_true: user -> set of ground-truth positive item ids.
        y_pred: user -> ranked list of retrieved item ids.
        k: Cutoff rank.

    Returns:
        Mean per-user NDCG@K, or 0.0 if no user has a ground-truth positive.
    """
    users = _users_with_ground_truth(y_true)
    if not users:
        return 0.0
    scores = []
    for u in users:
        positives = y_true[u]
        ranked = y_pred.get(u, [])[:k]
        dcg = sum(1.0 / math.log2(i + 2) for i, item in enumerate(ranked) if item in positives)
        ideal_hits = min(len(positives), k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
        scores.append(dcg / idcg if idcg > 0 else 0.0)
    return sum(scores) / len(scores)


def evaluate_retrieval_by_cohort(
    y_true: Dict[Any, Set[Any]],
    y_pred: Dict[Any, List[Any]],
    cohorts: Dict[Any, str],
    k: int,
    default_cohort: str = "cold",
) -> Dict[str, Dict[str, float]]:
    """Compute Recall@K, Hit Rate@K, and NDCG@K separately per cohort.

    Never collapses cohorts into one blended number — see module docstring
    for why that would be misleading given this project's user-history
    sparsity (docs/attribution_modeling_design.md §3B).

    Args:
        y_true: user -> ground-truth positive item set (e.g. test-period
            clicked campaigns).
        y_pred: user -> ranked list of retrieved item ids.
        cohorts: user -> cohort label (e.g. ``'cold'``/``'warm'``, from
            ``src.retrieval.cohort.classify_users``). A user absent from this
            mapping (never seen in the training period at all) is assigned
            ``default_cohort`` — the coldest possible case.
        k: Cutoff rank.
        default_cohort: Cohort assigned to users missing from ``cohorts``.

    Returns:
        ``{cohort_label: {"recall@k": float, "hit_rate@k": float,
        "ndcg@k": float, "n_users": int}}`` — ``n_users`` is the number of
        users in that cohort with a ground-truth positive (the metrics'
        actual denominator).
    """
    labels = set(cohorts.values()) | {default_cohort}
    results: Dict[str, Dict[str, float]] = {}
    for label in labels:
        users = [u for u in y_true if y_true[u] and cohorts.get(u, default_cohort) == label]
        sub_true = {u: y_true[u] for u in users}
        sub_pred = {u: y_pred.get(u, []) for u in users}
        results[label] = {
            "recall@k": recall_at_k(sub_true, sub_pred, k),
            "hit_rate@k": hit_rate_at_k(sub_true, sub_pred, k),
            "ndcg@k": ndcg_at_k(sub_true, sub_pred, k),
            "n_users": len(users),
        }
    return results
