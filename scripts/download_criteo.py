#!/usr/bin/env python3
"""Download and extract the Criteo sample dataset for local development.

Usage:
    python scripts/download_criteo.py [--force]

Downloads into the directory configured at ``configs/config.yaml``: ``criteo.raw_dir``
(default: ``data/raw/criteo``), which is git-ignored. If the download fails (e.g. no
network access, or the source becomes unavailable), see docs/criteo_dataset.md for
manual-download instructions and the dataset's terms of use.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.criteo import CriteoDatasetUnavailableError, download  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="Re-download and re-extract even if already present."
    )
    args = parser.parse_args()

    config = load_config()
    try:
        path = download(config, force=args.force)
    except CriteoDatasetUnavailableError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Criteo sample dataset ready at: {path}")


if __name__ == "__main__":
    main()
