"""
json_logger.py
==============
Demonstrates structured JSON logging:
  - JSONFormatter      : every log record is a JSON object
  - LoggerAdapter      : persistent context (request_id, user_id, service)
  - Log levels         : DEBUG, INFO, WARNING, ERROR, CRITICAL
  - Exception capture  : exc_info / stack_info included in JSON

All configuration from config.yaml; output to logs/python_engg.log.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import pathlib
import traceback
from datetime import datetime, timezone
from typing import Any, MutableMapping

import yaml

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
_CONFIG_PATH = _PROJECT_ROOT / "config.yaml"


# ---------------------------------------------------------------------------
# JSON Formatter
# ---------------------------------------------------------------------------
class JSONFormatter(logging.Formatter):
    """Emit each log record as a single-line JSON object.

    Fields included:
      timestamp, level, logger, message, module, funcName, lineno,
      plus any extra fields added via LoggerAdapter or logger.info(..., extra={...})
    """

    # Fields that are standard LogRecord attributes (not user extras)
    _STANDARD_ATTRS: frozenset[str] = frozenset(
        [
            "args", "created", "exc_info", "exc_text", "filename",
            "funcName", "id", "levelname", "levelno", "lineno",
            "message", "module", "msecs", "msg", "name", "pathname",
            "process", "processName", "relativeCreated", "stack_info",
            "thread", "threadName", "taskName",
        ]
    )

    def format(self, record: logging.LogRecord) -> str:
        record.message = record.getMessage()

        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.message,
            "module": record.module,
            "func": record.funcName,
            "line": record.lineno,
            "thread": record.threadName,
            "process": record.processName,
        }

        # Extra fields injected via LoggerAdapter or extra= kwarg
        for key, value in record.__dict__.items():
            if key not in self._STANDARD_ATTRS and not key.startswith("_"):
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        if record.stack_info:
            payload["stack_info"] = self.formatStack(record.stack_info)

        return json.dumps(payload, default=str)


# ---------------------------------------------------------------------------
# Logger factory with RotatingFileHandler + JSONFormatter
# ---------------------------------------------------------------------------
def build_json_logger(
    name: str,
    log_file: pathlib.Path,
    level: str,
    max_bytes: int,
    backup_count: int,
) -> logging.Logger:
    """Return a logger that writes JSON to both console and rotating file."""
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    logger.propagate = False  # don't bubble up to root

    json_fmt = JSONFormatter()

    # Console handler (plain text is more readable in dev; JSON in prod)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    )
    console_handler.setLevel(getattr(logging, level.upper()))

    # Rotating file handler writes JSON
    file_handler = logging.handlers.RotatingFileHandler(
        filename=str(log_file),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(json_fmt)
    file_handler.setLevel(logging.DEBUG)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


# ---------------------------------------------------------------------------
# Context-aware LoggerAdapter
# ---------------------------------------------------------------------------
class ContextAdapter(logging.LoggerAdapter):
    """Injects persistent context fields into every log record.

    Usage::
        logger = ContextAdapter(base_logger, {"request_id": "abc", "user_id": 42})
        logger.info("Processing request")
        # -> JSON contains request_id, user_id fields
    """

    def process(
        self, msg: str, kwargs: MutableMapping[str, Any]
    ) -> tuple[str, MutableMapping[str, Any]]:
        extra = kwargs.setdefault("extra", {})
        extra.update(self.extra)
        return msg, kwargs


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
def demo_log_levels(adapter: ContextAdapter) -> None:
    adapter.debug("Debug: low-level diagnostic information")
    adapter.info("Info: normal operational message")
    adapter.warning("Warning: unexpected but recoverable situation")
    adapter.error("Error: operation failed but service continues")
    adapter.critical("Critical: service integrity at risk")


def demo_exception_capture(adapter: ContextAdapter) -> None:
    try:
        result = 1 / 0
    except ZeroDivisionError:
        adapter.error("Caught division by zero", exc_info=True)

    try:
        items: list[int] = []
        _ = items[99]
    except IndexError:
        adapter.exception("Index out of range (exc_info auto-captured)")


def demo_dynamic_extra(logger: logging.Logger) -> None:
    """Log with per-call extra fields (no adapter needed)."""
    for job_id in range(1, 4):
        logger.info(
            "Processing job",
            extra={"job_id": job_id, "queue": "high-priority"},
        )


def demo_multiple_adapters(base_logger: logging.Logger) -> None:
    """Simulate two concurrent request contexts sharing the same underlying logger."""
    req_a = ContextAdapter(base_logger, {"request_id": "req-001", "user_id": "user-A"})
    req_b = ContextAdapter(base_logger, {"request_id": "req-002", "user_id": "user-B"})

    req_a.info("Authenticated successfully")
    req_b.info("Authenticated successfully")
    req_a.warning("Rate limit approaching")
    req_b.error("Payment gateway timeout")


def main() -> None:
    with open(_CONFIG_PATH) as fh:
        cfg = yaml.safe_load(fh)

    log_cfg = cfg["logging"]
    log_file = _PROJECT_ROOT / log_cfg["log_file"]

    base_logger = build_json_logger(
        name="json_logger",
        log_file=log_file,
        level=log_cfg["level"],
        max_bytes=log_cfg["max_bytes"],
        backup_count=log_cfg["backup_count"],
    )

    # Default adapter context simulating a web service
    adapter = ContextAdapter(
        base_logger,
        {"service": "python-engg-demo", "version": "0.1.0", "env": "development"},
    )

    adapter.info("=== json_logger demo start ===")
    demo_log_levels(adapter)
    demo_exception_capture(adapter)
    demo_dynamic_extra(base_logger)
    demo_multiple_adapters(base_logger)
    adapter.info("=== json_logger demo complete ===")


if __name__ == "__main__":
    main()
