"""
Configuration loader for the Ray Engineering project.

Reads the project-level ``config.yaml`` and returns the parsed dict.
All modules call this at startup so that no constant is ever hardcoded.
"""

import os
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str | None = None) -> dict[str, Any]:
    """
    Load and return the YAML configuration file.

    Parameters
    ----------
    config_path:
        Explicit path to ``config.yaml``.  When *None*, the function walks
        up from the caller's working directory until it finds a
        ``config.yaml`` file or reaches the filesystem root.

    Returns
    -------
    dict
        Parsed YAML content.

    Raises
    ------
    FileNotFoundError
        If no ``config.yaml`` can be located.
    yaml.YAMLError
        If the file exists but cannot be parsed.
    """
    if config_path is not None:
        path = Path(config_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Config file not found at: {path}")
        return _parse(path)

    # Auto-discovery: walk upward from cwd
    candidate = Path(os.getcwd()).resolve()
    while True:
        config_file = candidate / "config.yaml"
        if config_file.exists():
            return _parse(config_file)
        parent = candidate.parent
        if parent == candidate:
            break
        candidate = parent

    raise FileNotFoundError(
        "No config.yaml found in the current directory or any of its parents. "
        "Pass an explicit path via the config_path argument."
    )


def _parse(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a YAML mapping at the top level, got {type(data).__name__}"
        )
    return data
