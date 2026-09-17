#!/usr/bin/env python3
"""Download the Criteo Attribution Modeling for Bidding Dataset for local development.

Usage:
    python scripts/download_attribution.py [--force]

Downloads into the directory configured at ``configs/config.yaml``:
``attribution.raw_dir`` (default: ``data/raw/attribution``), which is
git-ignored. See docs/attribution_dataset_verification.md for the dataset's
CC BY-NC-SA 4.0 terms and manual-download fallback.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.attribution import AttributionDatasetUnavailableError, download  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="Re-download even if already present."
    )
    args = parser.parse_args()

    config = load_config()
    try:
        path = download(config, force=args.force)
    except AttributionDatasetUnavailableError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Attribution dataset ready at: {path}")


if __name__ == "__main__":
    main()
