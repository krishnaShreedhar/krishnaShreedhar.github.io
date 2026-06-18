"""
YAML-based configuration loader with dot-notation access and deep merge support.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, Optional

import yaml


class ConfigLoader:
    """
    Loads and merges YAML configuration files.

    Responsibilities:
    - Load a YAML file into an internal dict.
    - Deep-merge two ConfigLoader instances (algo values override global values).
    - Provide dot-notation access ("training.batch_size") via get() and __getitem__.

    No fallbacks: if a required key is missing and no default is given, KeyError is raised.
    """

    def __init__(self, config: Dict) -> None:
        self._config: Dict = copy.deepcopy(config)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_file(cls, path: str) -> "ConfigLoader":
        """Load configuration from a YAML file."""
        with open(path, "r") as fh:
            data = yaml.safe_load(fh)
        if not isinstance(data, dict):
            raise ValueError(f"Expected a YAML mapping at the top level in '{path}', got {type(data).__name__}.")
        return cls(data)

    @classmethod
    def merge(cls, global_path: str, algo_path: str) -> "ConfigLoader":
        """
        Deep-merge global config with algorithm-specific config.

        Algorithm values override global values at every nesting level.
        """
        global_cfg = cls.from_file(global_path)
        algo_cfg = cls.from_file(algo_path)
        merged = cls._deep_merge(global_cfg._config, algo_cfg._config)
        return cls(merged)

    # ------------------------------------------------------------------
    # Access helpers
    # ------------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a value using dot notation.

        Example: config.get("training.batch_size", 64)
        Returns *default* when the key path does not exist.
        """
        try:
            return self._resolve(key)
        except KeyError:
            return default

    def __getitem__(self, key: str) -> Any:
        """
        Retrieve a value using dot notation; raises KeyError if missing.

        Example: config["training.batch_size"]
        """
        return self._resolve(key)

    @property
    def raw(self) -> Dict:
        """Return a deep copy of the underlying configuration dict."""
        return copy.deepcopy(self._config)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve(self, key: str) -> Any:
        """Traverse nested dicts using dot-separated key path."""
        parts = key.split(".")
        node: Any = self._config
        for part in parts:
            if not isinstance(node, dict):
                raise KeyError(f"Key segment '{part}' cannot be resolved; parent is not a dict.")
            if part not in node:
                raise KeyError(f"Key '{key}' not found (missing segment '{part}').")
            node = node[part]
        return node

    @staticmethod
    def _deep_merge(base: Dict, override: Dict) -> Dict:
        """
        Recursively merge *override* into *base*.

        - If both values for a key are dicts, recurse.
        - Otherwise the override value wins.
        """
        result = copy.deepcopy(base)
        for k, v in override.items():
            if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                result[k] = ConfigLoader._deep_merge(result[k], v)
            else:
                result[k] = copy.deepcopy(v)
        return result
