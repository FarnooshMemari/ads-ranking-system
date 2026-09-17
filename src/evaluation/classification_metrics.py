"""Classification-quality metrics for the CTR prediction model itself.

These evaluate predicted-probability quality (discrimination and
calibration), as distinct from the ranking metrics in
``ranking_metrics.py`` and the observed business metrics in
``ads_metrics.py``. Implemented in Phase 2: CTR prediction.
"""

from typing import Any, Sequence


def compute_auc(y_true: Sequence[int], y_pred: Sequence[float]) -> float:
    """Compute ROC-AUC for CTR predictions.

    Args:
        y_true: Ground-truth binary click labels.
        y_pred: Predicted click probabilities.

    Returns:
        The ROC-AUC score.
    """
    raise NotImplementedError("Implemented in Phase 2: CTR prediction.")


def compute_logloss(y_true: Sequence[int], y_pred: Sequence[float]) -> float:
    """Compute binary log loss (cross-entropy) for CTR predictions.

    Args:
        y_true: Ground-truth binary click labels.
        y_pred: Predicted click probabilities.

    Returns:
        The log loss value.
    """
    raise NotImplementedError("Implemented in Phase 2: CTR prediction.")


def compute_calibration(y_true: Sequence[int], y_pred: Sequence[float], n_bins: int = 10) -> Any:
    """Compute a calibration curve (predicted vs. observed CTR by bucket).

    Args:
        y_true: Ground-truth binary click labels.
        y_pred: Predicted click probabilities.
        n_bins: Number of probability buckets.

    Returns:
        Calibration data (e.g. per-bin predicted vs. observed rates).
    """
    raise NotImplementedError("Implemented in Phase 2: CTR prediction.")
