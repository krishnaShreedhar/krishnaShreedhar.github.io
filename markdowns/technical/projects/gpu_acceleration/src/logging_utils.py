"""Shared logging configuration — reads level and format from a YAML config dict."""

import logging
import logging.handlers
import os
import sys
from typing import Any


def setup_logger(name: str, config: dict[str, Any]) -> logging.Logger:
    """
    Build a named logger from a config dict with a 'logging' key.

    Expected config shape:
        logging:
          level: DEBUG | INFO | WARNING | ERROR
          format: "%(asctime)s ..."
          file: "logs/something.log"
    """
    log_cfg = config.get("logging", {})
    level_str = log_cfg.get("level", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)
    fmt = log_cfg.get("format", "%(asctime)s [%(levelname)s] %(name)s — %(message)s")
    log_file = log_cfg.get("file")

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(fmt)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(level)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=50 * 1024 * 1024, backupCount=3
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger
