"""Classification-quality metrics for the CTR prediction model itself.

These evaluate predicted-probability quality (discrimination and
calibration), as distinct from the ranking metrics in
``ranking_metrics.py`` and the observed business metrics in
``ads_metrics.py``.
"""

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score


def compute_auc(y_true: Sequence[int], y_pred: Sequence[float]) -> float:
    """Compute ROC-AUC for CTR predictions.

    Args:
        y_true: Ground-truth binary click labels.
        y_pred: Predicted click probabilities.

    Returns:
        The ROC-AUC score.
    """
    return float(roc_auc_score(y_true, y_pred))


def compute_pr_auc(y_true: Sequence[int], y_pred: Sequence[float]) -> float:
    """Compute PR-AUC (average precision) for CTR predictions.

    Preferred over ROC-AUC when the positive class is imbalanced, since
    ROC-AUC can look optimistic under imbalance in a way PR-AUC does not.

    Args:
        y_true: Ground-truth binary click labels.
        y_pred: Predicted click probabilities.

    Returns:
        The PR-AUC (average precision) score.
    """
    return float(average_precision_score(y_true, y_pred))


def compute_logloss(y_true: Sequence[int], y_pred: Sequence[float]) -> float:
    """Compute binary log loss (cross-entropy) for CTR predictions.

    Args:
        y_true: Ground-truth binary click labels.
        y_pred: Predicted click probabilities.

    Returns:
        The log loss value.
    """
    return float(log_loss(y_true, y_pred, labels=[0, 1]))


def compute_calibration(y_true: Sequence[int], y_pred: Sequence[float], n_bins: int = 10) -> pd.DataFrame:
    """Compute a calibration table: per-bin mean predicted vs. mean observed rate.

    Bins are equal-*frequency* (quantile) rather than equal-width, since CTR
    predictions are typically concentrated in a narrow low-probability range
    where equal-width bins would leave most bins nearly empty.

    Args:
        y_true: Ground-truth binary click labels.
        y_pred: Predicted click probabilities.
        n_bins: Target number of probability buckets (fewer are returned if
            there are not enough distinct predicted values to form ``n_bins``
            quantile groups).

    Returns:
        A DataFrame with one row per bin: ``mean_predicted``, ``mean_observed``,
        ``count``.
    """
    df = pd.DataFrame({"y_true": np.asarray(y_true, dtype=float), "y_pred": np.asarray(y_pred, dtype=float)})
    try:
        df["bin"] = pd.qcut(df["y_pred"], q=n_bins, duplicates="drop")
    except ValueError:
        df["bin"] = pd.cut(df["y_pred"], bins=n_bins)

    grouped = (
        df.groupby("bin", observed=True)
        .agg(mean_predicted=("y_pred", "mean"), mean_observed=("y_true", "mean"), count=("y_true", "size"))
        .reset_index(drop=True)
    )
    return grouped
