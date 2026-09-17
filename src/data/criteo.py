"""Criteo Display Advertising Challenge sample dataset — download and loading.

This is the Phase 1 raw dataset for the project (see docs/criteo_dataset.md for
the verified schema, licensing terms, and rationale). Only the "sample" variant
(~100K rows) is supported here; it is small enough to download and iterate on
locally, unlike the ~45.8M-row "full" variant.

The raw archive is treated as immutable once downloaded: by default, an
existing extracted file is reused rather than re-downloaded.
"""

import tarfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.utils.config import resolve_path

LABEL_COLUMN = "label"
NUMERICAL_COLUMNS = [f"int{i:02d}" for i in range(13)]
CATEGORICAL_COLUMNS = [f"cat{i:02d}" for i in range(26)]
HEADER = [LABEL_COLUMN] + NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS


class CriteoDatasetUnavailableError(RuntimeError):
    """Raised when the Criteo dataset cannot be downloaded or located locally."""


def _raw_dir(config: Dict[str, Any]) -> Path:
    return resolve_path(config["criteo"]["raw_dir"])


def download(config: Dict[str, Any], force: bool = False) -> Path:
    """Download and extract the Criteo sample archive into the configured raw directory.

    Args:
        config: Project configuration (see ``configs/config.yaml``: ``criteo``).
        force: Re-download and re-extract even if the extracted file already exists.

    Returns:
        Path to the extracted raw Criteo TSV file (tab-separated, no header row).

    Raises:
        CriteoDatasetUnavailableError: If the dataset cannot be downloaded or the
            downloaded archive doesn't contain the expected file. The error message
            includes a manual-download fallback.
    """
    criteo_cfg = config["criteo"]
    raw_dir = _raw_dir(config)
    raw_dir.mkdir(parents=True, exist_ok=True)

    extracted_path = raw_dir / criteo_cfg["extracted_filename"]
    if extracted_path.exists() and not force:
        return extracted_path

    source_url = criteo_cfg["source_url"]
    archive_path = raw_dir / criteo_cfg["archive_filename"]

    try:
        urllib.request.urlretrieve(source_url, archive_path)
    except (urllib.error.URLError, OSError) as exc:
        raise CriteoDatasetUnavailableError(
            f"Could not download the Criteo sample dataset from {source_url}.\n"
            f"Check your network connection, or download the file manually and place it at "
            f"'{archive_path}'.\n"
            "See docs/criteo_dataset.md for manual-download instructions and the Criteo Labs "
            f"data terms of use. Original error: {exc}"
        ) from exc

    _safe_extract(archive_path, raw_dir)

    if not extracted_path.exists():
        raise CriteoDatasetUnavailableError(
            f"Downloaded '{archive_path.name}' but did not find the expected "
            f"'{criteo_cfg['extracted_filename']}' after extracting into '{raw_dir}'. "
            "The upstream archive layout may have changed — see docs/criteo_dataset.md."
        )

    return extracted_path


def _safe_extract(archive_path: Path, dest_dir: Path) -> None:
    """Extract a tar.gz archive, refusing any member that would escape ``dest_dir``."""
    resolved_dest = dest_dir.resolve()
    with tarfile.open(archive_path) as tar:
        for member in tar.getmembers():
            member_path = (dest_dir / member.name).resolve()
            if resolved_dest not in member_path.parents and member_path != resolved_dest:
                raise CriteoDatasetUnavailableError(
                    f"Refusing to extract '{archive_path}': member '{member.name}' would "
                    f"extract outside of '{dest_dir}'."
                )
        tar.extractall(dest_dir)


def load(config: Dict[str, Any], nrows: Optional[int] = None) -> pd.DataFrame:
    """Load the Criteo sample dataset as a pandas DataFrame, downloading it if needed.

    Values are loaded as-is (raw strings/ints, with empty fields as missing) —
    no cleaning, imputation, or encoding is performed here. See ``src/features/``
    for preprocessing, and ``src/data/validation.py`` to check the result before
    using it further.

    Args:
        config: Project configuration (see ``configs/config.yaml``: ``criteo``).
        nrows: Optional row limit, useful for quick local iteration.

    Returns:
        DataFrame with columns ``[label, int00..int12, cat00..cat25]``.
    """
    path = download(config)
    return pd.read_csv(path, sep="\t", header=None, names=HEADER, nrows=nrows)
