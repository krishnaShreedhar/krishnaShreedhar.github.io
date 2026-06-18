"""
Shared configuration loader for the pandas_pyspark project.

Reads config.yaml and returns a nested dict. All modules import from here
so that every constant and hyperparameter is controlled from a single file.
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str | None = None) -> Dict[str, Any]:
    """Load YAML config relative to the project root.

    Parameters
    ----------
    config_path:
        Explicit path to config.yaml.  When *None* the function walks up from
        this file's location until it finds ``config.yaml``.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    if config_path is None:
        here = Path(__file__).resolve().parent
        candidate = here
        for _ in range(5):  # walk up at most 5 levels
            candidate = candidate.parent
            probe = candidate / "config.yaml"
            if probe.exists():
                config_path = str(probe)
                break
        if config_path is None:
            raise FileNotFoundError(
                "config.yaml not found when walking up from src/. "
                "Pass config_path explicitly."
            )

    with open(config_path, "r", encoding="utf-8") as fh:
        cfg: Dict[str, Any] = yaml.safe_load(fh)

    return cfg


def setup_logging(cfg: Dict[str, Any]) -> logging.Logger:
    """Configure root logger from the ``logging`` section of the config.

    Parameters
    ----------
    cfg:
        Full config dict as returned by :func:`load_config`.

    Returns
    -------
    logging.Logger
        The root logger, already configured.
    """
    log_cfg = cfg["logging"]
    level_str: str = log_cfg["level"]
    log_file: str = log_cfg["log_file"]
    max_bytes: int = int(log_cfg["max_bytes"])
    backup_count: int = int(log_cfg["backup_count"])

    level = getattr(logging, level_str.upper(), logging.INFO)

    # Ensure logs directory exists
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    if not root_logger.handlers:
        # Rotating file handler
        fh = RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count
        )
        fh.setLevel(level)
        fh.setFormatter(formatter)
        root_logger.addHandler(fh)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)
        root_logger.addHandler(ch)

    return root_logger
