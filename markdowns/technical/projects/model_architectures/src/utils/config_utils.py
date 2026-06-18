"""YAML config loading and merging utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a nested dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    logger.debug("Loading config from %s", path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    logger.debug("Loaded config keys: %s", list(cfg.keys()))
    return cfg


def merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge *override* into *base* and return the merged result."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged


def get_section(cfg: dict[str, Any], section: str) -> dict[str, Any]:
    """Extract a top-level section from a config dict, raising if absent."""
    if section not in cfg:
        raise KeyError(f"Section '{section}' not found in config. Available: {list(cfg.keys())}")
    return cfg[section]
