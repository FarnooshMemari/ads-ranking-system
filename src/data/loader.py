"""Data loading and train/val/test splitting.

Implementations land in Phase 1 (data understanding and modeling).
"""

from pathlib import Path
from typing import Any, Tuple


def load_raw_data(path: str | Path) -> Any:
    """Load raw ad-interaction data from disk.

    Args:
        path: Path to the raw data file or directory.

    Returns:
        The raw dataset (e.g. a ``pandas.DataFrame``).
    """
    raise NotImplementedError("Implemented in Phase 1: data understanding and modeling.")


def load_processed_data(path: str | Path) -> Any:
    """Load a previously processed/feature-engineered dataset.

    Args:
        path: Path to the processed data file.

    Returns:
        The processed dataset (e.g. a ``pandas.DataFrame``).
    """
    raise NotImplementedError("Implemented in Phase 1: data understanding and modeling.")


def split_train_val_test(
    data: Any, val_size: float = 0.15, test_size: float = 0.15, time_based: bool = True
) -> Tuple[Any, Any, Any]:
    """Split a dataset into train/validation/test sets.

    Args:
        data: The full dataset to split.
        val_size: Fraction of data reserved for validation.
        test_size: Fraction of data reserved for testing.
        time_based: If True, split chronologically (recommended for CTR
            prediction to avoid leakage); otherwise split randomly.

    Returns:
        A ``(train, val, test)`` tuple of datasets.
    """
    raise NotImplementedError("Implemented in Phase 1: data understanding and modeling.")
