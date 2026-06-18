"""Logging utilities: configure a module-level logger with consistent formatting."""

import logging
import sys
from pathlib import Path


def get_logger(name: str, level: str = "INFO", log_file: str | None = None) -> logging.Logger:
    """Return a logger with a StreamHandler (and optional FileHandler).

    Args:
        name: Logger name (typically __name__ of the calling module).
        level: One of DEBUG / INFO / WARNING / ERROR / CRITICAL.
        log_file: If given, also write to this file path (appending).
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger
