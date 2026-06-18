"""
logger_factory.py
=================
Centralised logger factory using `logging.config.dictConfig`:

  - Single call configures all loggers in the process
  - Supports JSON or text formatters via config flag
  - RotatingFileHandler parameters from config.yaml
  - Demonstrates dictConfig schema for production use
  - Shows per-module logger hierarchy and effective level inheritance

Usage::
    factory = LoggerFactory(cfg)
    factory.configure()
    logger = factory.get_logger("my.module")
    logger.info("Ready")
"""

from __future__ import annotations

import json
import logging
import logging.config
import logging.handlers
import pathlib
from datetime import datetime, timezone
from typing import Any

import yaml

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
_CONFIG_PATH = _PROJECT_ROOT / "config.yaml"


# ---------------------------------------------------------------------------
# Re-usable JSON formatter (importable from this module)
# ---------------------------------------------------------------------------
class JSONFormatter(logging.Formatter):
    """Produce structured JSON log lines."""

    _SKIP: frozenset[str] = frozenset(
        ["args", "created", "exc_info", "exc_text", "filename",
         "levelno", "message", "msecs", "msg", "relativeCreated",
         "stack_info", "taskName"]
    )

    def format(self, record: logging.LogRecord) -> str:
        record.message = record.getMessage()
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.message,
            "module": record.module,
            "func": record.funcName,
            "line": record.lineno,
        }
        for k, v in record.__dict__.items():
            if k not in self._SKIP and not k.startswith("_") and k not in payload:
                payload[k] = v
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


# ---------------------------------------------------------------------------
# LoggerFactory
# ---------------------------------------------------------------------------
class LoggerFactory:
    """Configure the Python logging system from a config dict.

    Follows SOLID / Single-Responsibility: this class ONLY knows about
    logging configuration, not about application logic.
    """

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._log_cfg = cfg["logging"]
        self._log_file = _PROJECT_ROOT / self._log_cfg["log_file"]
        self._log_file.parent.mkdir(parents=True, exist_ok=True)
        self._configured = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def configure(self, use_json: bool = False) -> None:
        """Apply dictConfig to configure all handlers and formatters."""
        level: str = self._log_cfg["level"]
        log_path = str(self._log_file)

        formatters: dict[str, Any] = {
            "text": {
                "format": "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
                "datefmt": "%Y-%m-%dT%H:%M:%S",
            },
            "json": {
                "()": JSONFormatter,
            },
        }

        active_fmt = "json" if use_json else "text"

        handlers: dict[str, Any] = {
            "console": {
                "class": "logging.StreamHandler",
                "level": level,
                "formatter": "text",  # always text on console for readability
                "stream": "ext://sys.stdout",
            },
            "rotating_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "DEBUG",
                "formatter": active_fmt,
                "filename": log_path,
                "maxBytes": self._log_cfg["max_bytes"],
                "backupCount": self._log_cfg["backup_count"],
                "encoding": "utf-8",
            },
            "error_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "ERROR",
                "formatter": active_fmt,
                "filename": log_path.replace(".log", "_errors.log"),
                "maxBytes": self._log_cfg["max_bytes"],
                "backupCount": 2,
                "encoding": "utf-8",
            },
        }

        # Per-module logger overrides – suppress noisy third-party loggers
        loggers: dict[str, Any] = {
            "asyncio": {"level": "WARNING", "propagate": True},
            "urllib3": {"level": "WARNING", "propagate": True},
            "aiohttp": {"level": "WARNING", "propagate": True},
        }

        config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": formatters,
            "handlers": handlers,
            "loggers": loggers,
            "root": {
                "level": level,
                "handlers": ["console", "rotating_file", "error_file"],
            },
        }

        logging.config.dictConfig(config)
        self._configured = True

    def get_logger(self, name: str) -> logging.Logger:
        if not self._configured:
            self.configure()
        return logging.getLogger(name)

    def get_adapter(
        self, name: str, context: dict[str, Any]
    ) -> logging.LoggerAdapter:
        logger = self.get_logger(name)
        return logging.LoggerAdapter(logger, context)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
def demo_hierarchy(factory: LoggerFactory) -> None:
    root_logger = factory.get_logger("app")
    child_logger = factory.get_logger("app.service")
    grandchild_logger = factory.get_logger("app.service.db")

    root_logger.info("Root logger message")
    child_logger.info("Child logger inherits root handlers")
    grandchild_logger.debug(
        "Grandchild DEBUG (visible if level <= DEBUG)"
    )
    grandchild_logger.warning("Grandchild WARNING always visible")

    root_logger.info(
        "Effective levels: root=%s  child=%s  grandchild=%s",
        logging.getLevelName(root_logger.getEffectiveLevel()),
        logging.getLevelName(child_logger.getEffectiveLevel()),
        logging.getLevelName(grandchild_logger.getEffectiveLevel()),
    )


def demo_adapter_context(factory: LoggerFactory) -> None:
    adapter = factory.get_adapter(
        "app.service",
        {"request_id": "req-xyz-789", "user_id": "user-42", "env": "staging"},
    )
    adapter.info("Handling API request")
    adapter.warning("Slow query detected")
    adapter.error("Upstream service returned 503")


def demo_noise_suppression(factory: LoggerFactory) -> None:
    noisy = factory.get_logger("urllib3.connectionpool")
    # This should be suppressed to WARNING level by dictConfig override
    noisy.debug("This DEBUG message should NOT appear (suppressed to WARNING)")
    noisy.warning("This WARNING message SHOULD appear")


def main() -> None:
    with open(_CONFIG_PATH) as fh:
        cfg = yaml.safe_load(fh)

    factory = LoggerFactory(cfg)
    factory.configure(use_json=False)  # switch to True for JSON file output

    bootstrap_logger = factory.get_logger("logger_factory")
    bootstrap_logger.info("=== logger_factory demo start ===")
    bootstrap_logger.info("Log file: %s", _PROJECT_ROOT / cfg["logging"]["log_file"])

    demo_hierarchy(factory)
    demo_adapter_context(factory)
    demo_noise_suppression(factory)

    bootstrap_logger.info("=== logger_factory demo complete ===")


if __name__ == "__main__":
    main()
