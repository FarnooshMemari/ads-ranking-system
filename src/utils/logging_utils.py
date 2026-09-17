"""Logging setup shared across the project."""

import logging


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create (or retrieve) a configured logger.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.
        level: Logging level, defaults to ``logging.INFO``.

    Returns:
        A configured ``logging.Logger`` instance with a single stream handler.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger
