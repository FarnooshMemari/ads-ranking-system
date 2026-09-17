"""Example unit test — verifies the project config loads and is well-formed."""

from src.utils.config import DEFAULT_CONFIG_PATH, load_config


def test_load_config_returns_expected_top_level_sections():
    config = load_config(DEFAULT_CONFIG_PATH)

    assert isinstance(config, dict)
    for section in ("project", "data", "features", "model", "experiment"):
        assert section in config


def test_config_has_random_seed():
    config = load_config(DEFAULT_CONFIG_PATH)

    assert isinstance(config["project"]["random_seed"], int)
