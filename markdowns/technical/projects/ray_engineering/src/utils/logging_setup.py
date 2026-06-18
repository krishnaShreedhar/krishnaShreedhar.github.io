"""
Logging setup utility for the Ray Engineering project.

Provides a factory function to configure a JSON-formatted, rotating-file
logger that also writes to stdout.  All modules import this to obtain a
consistent, configurable logger.
"""

import json
import logging
import logging.handlers
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# JSON formatter
# ---------------------------------------------------------------------------

class JsonFormatter(logging.Formatter):
    """Emit each log record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        log_object: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
        }

        if record.exc_info:
            log_object["exception"] = self.formatException(record.exc_info)

        if record.stack_info:
            log_object["stack_info"] = self.formatStack(record.stack_info)

        # Attach any extra keys passed via the extra= kwarg
        standard_keys = {
            "name", "msg", "args", "levelname", "levelno", "pathname",
            "filename", "module", "exc_info", "exc_text", "stack_info",
            "lineno", "funcName", "created", "msecs", "relativeCreated",
            "thread", "threadName", "processName", "process", "taskName",
            "message",
        }
        for key, value in record.__dict__.items():
            if key not in standard_keys:
                log_object[key] = value

        return json.dumps(log_object)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def get_logger(name: str, config: dict) -> logging.Logger:
    """
    Build and return a named logger configured from *config*.

    Parameters
    ----------
    name:
        Logger name (typically ``__name__`` of the calling module).
    config:
        Full project config dict.  The ``logging`` sub-section is used:

        .. code-block:: yaml

            logging:
              level: INFO
              log_file: logs/ray_engineering.log
              max_bytes: 104857600
              backup_count: 5

    Returns
    -------
    logging.Logger
        Configured logger instance.  Repeated calls with the same *name*
        return the same underlying logger (Python stdlib behaviour).
    """
    log_cfg = config.get("logging", {})
    level_name: str = log_cfg.get("level", "INFO").upper()
    level: int = getattr(logging, level_name, logging.INFO)

    log_file: str = log_cfg.get("log_file", "logs/ray_engineering.log")
    max_bytes: int = int(log_cfg.get("max_bytes", 104_857_600))
    backup_count: int = int(log_cfg.get("backup_count", 5))

    # Ensure the log directory exists
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers when called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(level)

    formatter = JsonFormatter()

    # Rotating file handler
    file_handler = logging.handlers.RotatingFileHandler(
        filename=str(log_path),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    # Console (stdout) handler - human-readable plain text
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    plain_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    console_handler.setFormatter(plain_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Do not propagate to the root logger to avoid duplicate output
    logger.propagate = False

    logger.debug(
        "Logger initialised",
        extra={"log_level": level_name, "log_file": str(log_path)},
    )
    return logger
