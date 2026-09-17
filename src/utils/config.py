"""Configuration loading utilities."""

from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Load the project YAML configuration file.

    Args:
        config_path: Path to a YAML config file. Defaults to
            ``configs/config.yaml`` at the project root.

    Returns:
        The parsed configuration as a nested dictionary.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def resolve_path(path: str | Path) -> Path:
    """Resolve a path from config relative to the project root.

    Paths in ``configs/config.yaml`` are written relative to the project root
    so the config never hardcodes a machine-specific location. An already
    absolute path is returned unchanged.

    Args:
        path: A path string from config, relative or absolute.

    Returns:
        An absolute ``Path``.
    """
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path
