"""Criteo Attribution Modeling for Bidding Dataset — download, loading, and
chronological splitting.

Used only for this project's own recommendation-system component (see
docs/attribution_modeling_design.md); entirely separate from the Criteo DAC
dataset used for the Microsoft Recommenders contribution (``src/data/criteo.py``).

The raw file is a single gzip-compressed TSV **with a header row** (unlike
DAC) — verified directly from the source archive. It is treated as immutable
once downloaded: by default, an existing local copy is reused rather than
re-downloaded.
"""

import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from src.utils.config import resolve_path

DTYPES = {
    "timestamp": "int64",
    "uid": "int64",
    "campaign": "int64",
    "conversion": "int8",
    "conversion_timestamp": "int64",
    "conversion_id": "int64",
    "attribution": "int8",
    "click": "int8",
    "click_pos": "int32",
    "click_nb": "int32",
    "cost": "float64",
    "cpo": "float64",
    "time_since_last_click": "int64",
    **{f"cat{i}": "int64" for i in range(1, 10)},
}

SECONDS_PER_DAY = 86400


class AttributionDatasetUnavailableError(RuntimeError):
    """Raised when the Attribution dataset cannot be downloaded or located locally."""


def _archive_path(config: Dict[str, Any]) -> Path:
    cfg = config["attribution"]
    return resolve_path(cfg["raw_dir"]) / cfg["archive_filename"]


def download(config: Dict[str, Any], force: bool = False) -> Path:
    """Download the Attribution dataset archive into the configured raw directory.

    Args:
        config: Project configuration (see ``configs/config.yaml``: ``attribution``).
        force: Re-download even if a local copy already exists.

    Returns:
        Path to the downloaded ``.tsv.gz`` file.

    Raises:
        AttributionDatasetUnavailableError: If the dataset cannot be downloaded.
            The error message includes a manual-download fallback.
    """
    cfg = config["attribution"]
    raw_dir = resolve_path(cfg["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = _archive_path(config)

    if path.exists() and not force:
        return path

    source_url = cfg["source_url"]
    try:
        urllib.request.urlretrieve(source_url, path)
    except (urllib.error.URLError, OSError) as exc:
        raise AttributionDatasetUnavailableError(
            f"Could not download the Attribution dataset from {source_url}.\n"
            f"Check your network connection, or download the file manually and place it at "
            f"'{path}'.\n"
            "See docs/attribution_dataset_verification.md for details and the dataset's "
            f"CC BY-NC-SA 4.0 terms. Original error: {exc}"
        ) from exc

    return path


def load(config: Dict[str, Any], nrows: Optional[int] = None) -> pd.DataFrame:
    """Load the Attribution dataset as a pandas DataFrame, downloading it if needed.

    Values are loaded as-is — no cleaning, imputation, or feature engineering
    is performed here.

    Args:
        config: Project configuration (see ``configs/config.yaml``: ``attribution``).
        nrows: Optional row limit, useful for quick local iteration.

    Returns:
        DataFrame with the dataset's native columns (``timestamp, uid, campaign,
        conversion, conversion_timestamp, conversion_id, attribution, click,
        click_pos, click_nb, cost, cpo, time_since_last_click, cat1..cat9``).
    """
    path = download(config)
    return pd.read_csv(path, sep="\t", dtype=DTYPES, nrows=nrows)


def split_by_day(
    df: pd.DataFrame,
    train_end_day: int,
    val_end_day: int,
    timestamp_col: str = "timestamp",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame chronologically into train/validation/test by day boundary.

    ``day = timestamp // 86400`` (the dataset's own relative-seconds clock).
    This is the single source of temporal truth for the whole pipeline — cohort
    classification (``src/retrieval/cohort.py``) and popularity ranking
    (``src/retrieval/candidate_generator.py``) must only ever be given the
    ``train`` split returned here, never the full DataFrame, to prevent future
    information from leaking into training-time decisions.

    Args:
        df: The full (or a subset of the) Attribution dataset, with a raw
            ``timestamp`` column.
        train_end_day: Rows with ``day < train_end_day`` go to train.
        val_end_day: Rows with ``train_end_day <= day < val_end_day`` go to
            validation; rows with ``day >= val_end_day`` go to test.
        timestamp_col: Name of the timestamp column.

    Returns:
        ``(train_df, val_df, test_df)``, each a view/copy of ``df`` restricted
        to its day range.
    """
    day = df[timestamp_col] // SECONDS_PER_DAY
    train_df = df[day < train_end_day]
    val_df = df[(day >= train_end_day) & (day < val_end_day)]
    test_df = df[day >= val_end_day]
    return train_df, val_df, test_df
