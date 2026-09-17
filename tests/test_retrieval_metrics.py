"""Tests for retrieval-stage metrics (Recall@K, Hit Rate@K, NDCG@K), using
tiny hand-computed synthetic examples."""

import pytest

from src.evaluation.retrieval_metrics import (
    evaluate_retrieval_by_cohort,
    hit_rate_at_k,
    ndcg_at_k,
    recall_at_k,
)


def test_recall_at_k_perfect_and_zero():
    y_true = {1: {100}, 2: {200}}
    y_pred = {1: [100, 300], 2: [300, 400]}

    assert recall_at_k(y_true, y_pred, k=2) == 0.5  # user 1 perfect, user 2 zero


def test_recall_at_k_partial_multiple_positives():
    y_true = {1: {100, 200, 300}}
    y_pred = {1: [100, 200, 999]}

    assert recall_at_k(y_true, y_pred, k=3) == pytest.approx(2 / 3)


def test_recall_ignores_users_with_no_ground_truth():
    y_true = {1: {100}, 2: set()}
    y_pred = {1: [100], 2: [999]}

    assert recall_at_k(y_true, y_pred, k=1) == 1.0  # user 2 excluded, not penalized


def test_recall_respects_k_cutoff():
    y_true = {1: {300}}
    y_pred = {1: [100, 200, 300]}  # true positive is rank 3

    assert recall_at_k(y_true, y_pred, k=2) == 0.0
    assert recall_at_k(y_true, y_pred, k=3) == 1.0


def test_hit_rate_at_k():
    y_true = {1: {100}, 2: {200}, 3: {300}}
    y_pred = {1: [100], 2: [999], 3: [300]}

    assert hit_rate_at_k(y_true, y_pred, k=1) == pytest.approx(2 / 3)


def test_ndcg_at_k_rank_sensitive():
    y_true = {1: {100}}
    # positive at rank 1 vs rank 2 should score differently
    high = ndcg_at_k(y_true, {1: [100, 999]}, k=2)
    low = ndcg_at_k(y_true, {1: [999, 100]}, k=2)

    assert high == 1.0  # single relevant item at the top rank -> perfect NDCG
    assert 0 < low < high


def test_ndcg_at_k_no_hits_is_zero():
    y_true = {1: {100}}
    y_pred = {1: [999, 998]}

    assert ndcg_at_k(y_true, y_pred, k=2) == 0.0


def test_k_larger_than_available_candidates_handled_gracefully():
    """y_pred lists shorter than k must not error -- just no more hits possible."""
    y_true = {1: {100}}
    y_pred = {1: [100]}  # only one candidate exists, k asks for far more

    assert recall_at_k(y_true, y_pred, k=1000) == 1.0
    assert hit_rate_at_k(y_true, y_pred, k=1000) == 1.0
    assert ndcg_at_k(y_true, y_pred, k=1000) == 1.0


def test_missing_user_in_y_pred_treated_as_empty_retrieval():
    y_true = {1: {100}}
    y_pred = {}  # user 1 has no entry at all

    assert recall_at_k(y_true, y_pred, k=5) == 0.0
    assert hit_rate_at_k(y_true, y_pred, k=5) == 0.0
    assert ndcg_at_k(y_true, y_pred, k=5) == 0.0


def test_evaluate_retrieval_by_cohort_separates_results():
    y_true = {1: {100}, 2: {100}, 3: {100}}
    y_pred = {1: [100], 2: [100], 3: [999]}
    cohorts = {1: "cold", 2: "warm", 3: "warm"}

    results = evaluate_retrieval_by_cohort(y_true, y_pred, cohorts, k=1)

    assert results["cold"]["n_users"] == 1
    assert results["cold"]["recall@k"] == 1.0
    assert results["warm"]["n_users"] == 2
    assert results["warm"]["recall@k"] == 0.5  # user 2 hit, user 3 miss


def test_evaluate_retrieval_by_cohort_defaults_unmapped_users_to_cold():
    y_true = {1: {100}}
    y_pred = {1: [100]}
    cohorts = {}  # user 1 never seen in training -> unmapped

    results = evaluate_retrieval_by_cohort(y_true, y_pred, cohorts, k=1, default_cohort="cold")

    assert results["cold"]["n_users"] == 1
    assert results["cold"]["recall@k"] == 1.0
