"""Tests for classification-quality metrics, using small hand-checkable examples."""

import pytest

from src.evaluation.classification_metrics import (
    compute_auc,
    compute_calibration,
    compute_logloss,
    compute_pr_auc,
)


def test_compute_auc_perfect_separation():
    y_true = [0, 0, 1, 1]
    y_pred = [0.1, 0.2, 0.8, 0.9]
    assert compute_auc(y_true, y_pred) == 1.0


def test_compute_auc_random_is_around_half():
    y_true = [0, 1, 0, 1]
    y_pred = [0.5, 0.5, 0.5, 0.5]
    assert compute_auc(y_true, y_pred) == pytest.approx(0.5)


def test_compute_pr_auc_perfect_separation():
    y_true = [0, 0, 1, 1]
    y_pred = [0.1, 0.2, 0.8, 0.9]
    assert compute_pr_auc(y_true, y_pred) == 1.0


def test_compute_logloss_confident_correct_predictions_score_low():
    y_true = [0, 1]
    confident_correct = compute_logloss(y_true, [0.01, 0.99])
    unsure = compute_logloss(y_true, [0.5, 0.5])
    assert confident_correct < unsure


def test_compute_logloss_confident_wrong_predictions_score_high():
    y_true = [0, 1]
    confident_wrong = compute_logloss(y_true, [0.99, 0.01])
    unsure = compute_logloss(y_true, [0.5, 0.5])
    assert confident_wrong > unsure


def test_compute_calibration_shape_and_columns():
    y_true = [0, 1, 0, 1, 1, 0, 1, 0, 1, 0]
    y_pred = [0.1, 0.9, 0.2, 0.8, 0.7, 0.3, 0.6, 0.4, 0.85, 0.15]
    table = compute_calibration(y_true, y_pred, n_bins=5)

    assert set(table.columns) == {"mean_predicted", "mean_observed", "count"}
    assert table["count"].sum() == len(y_true)
    assert (table["mean_predicted"] >= 0).all() and (table["mean_predicted"] <= 1).all()


def test_compute_calibration_perfectly_calibrated_example():
    # two bins, each with a clear, distinct predicted level matching observed rate
    y_true = [0, 0, 1, 1] * 5
    y_pred = [0.0, 0.0, 1.0, 1.0] * 5
    table = compute_calibration(y_true, y_pred, n_bins=2)

    assert len(table) == 2
    for _, row in table.iterrows():
        assert row["mean_predicted"] == pytest.approx(row["mean_observed"])
