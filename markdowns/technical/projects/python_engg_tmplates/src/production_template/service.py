"""
service.py
==========
Complete production-grade service template:
  - Graceful shutdown via SIGTERM/SIGINT handlers
  - Batch processing loop with explicit GC control
  - stream_jsonl generator for large file iteration
  - Integrates MetricsRegistry for request/error/latency tracking
  - Exception hierarchy usage throughout
  - All constants from config.yaml
  - Logs every significant lifecycle event

Design: The ``ProductionService`` class follows SOLID principles:
  - Single Responsibility  : lifecycle management + batch processing
  - Open/Closed            : subclass to customise ``_process_batch``
  - Dependency Inversion   : depends on abstract MetricsRegistry interface
"""

from __future__ import annotations

import gc
import json
import logging
import logging.config
import pathlib
import random
import signal
import sys
import time
from typing import Any, Generator, Optional

import yaml

from metrics import MetricsRegistry
from exception_hierarchy import (
    AppError,
    RetryableError,
    ValidationError,
    install_global_exception_handler,
)

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
_CONFIG_PATH = _PROJECT_ROOT / "config.yaml"


# ---------------------------------------------------------------------------
# Config & logging
# ---------------------------------------------------------------------------
def _load_config() -> dict:
    with open(_CONFIG_PATH) as fh:
        return yaml.safe_load(fh)


def _setup_logging(cfg: dict) -> logging.Logger:
    log_cfg = cfg["logging"]
    log_file = _PROJECT_ROOT / log_cfg["log_file"]
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "json_fmt": {
                    "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "json_fmt",
                    "stream": "ext://sys.stdout",
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "formatter": "json_fmt",
                    "filename": str(log_file),
                    "maxBytes": log_cfg["max_bytes"],
                    "backupCount": log_cfg["backup_count"],
                    "encoding": "utf-8",
                },
            },
            "root": {"level": log_cfg["level"], "handlers": ["console", "file"]},
        }
    )
    return logging.getLogger("service")


# ---------------------------------------------------------------------------
# JSONL streaming utility
# ---------------------------------------------------------------------------
def stream_jsonl(file_path: pathlib.Path) -> Generator[dict, None, None]:
    """Yield parsed JSON objects from a newline-delimited JSON file.

    Reads line-by-line to keep memory constant regardless of file size.
    Malformed lines are logged and skipped (no silent failures).
    """
    log = logging.getLogger("service.stream_jsonl")
    with open(file_path, "r", encoding="utf-8") as fh:
        for lineno, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                log.error("Malformed JSON at line %d: %s  raw=%r", lineno, exc, line[:120])


# ---------------------------------------------------------------------------
# Production Service
# ---------------------------------------------------------------------------
class ProductionService:
    """Batch-processing service with graceful shutdown and metrics.

    Lifecycle::
        service = ProductionService(cfg)
        service.start()         # install signal handlers, initialise resources
        service.run(records)    # process a dataset in batches
        service.stop()          # flush metrics, release resources
    """

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._svc_cfg = cfg["service"]
        self._log = logging.getLogger("service.ProductionService")

        self._name: str = self._svc_cfg["name"]
        self._batch_size: int = self._svc_cfg["batch_size"]
        self._gc_threshold: int = self._svc_cfg["gc_threshold"]

        metrics_cfg = cfg["metrics"]
        self._metrics = MetricsRegistry(
            maxlen=metrics_cfg["histogram_maxlen"],
            percentiles=metrics_cfg["percentiles"],
        )

        self._running: bool = False
        self._shutdown_requested: bool = False
        self._start_time: Optional[float] = None

        # Metrics
        self._m_processed = self._metrics.counter(
            "service.records.processed", "Total records processed"
        )
        self._m_errors = self._metrics.counter(
            "service.records.errors", "Total processing errors"
        )
        self._m_batches = self._metrics.counter(
            "service.batches.total", "Total batches processed"
        )
        self._m_latency = self._metrics.histogram(
            "service.batch_latency_ms", "Per-batch processing latency (ms)"
        )
        self._m_active = self._metrics.gauge(
            "service.active", "1 while service is running"
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Install signal handlers and mark service as running."""
        self._log.info("Service '%s' starting up", self._name)
        self._start_time = time.monotonic()
        self._running = True
        self._m_active.set(1.0)

        signal.signal(signal.SIGTERM, self._handle_shutdown_signal)
        signal.signal(signal.SIGINT, self._handle_shutdown_signal)

        # Tune GC: collect less frequently during heavy batch processing
        gc.set_threshold(self._gc_threshold, 10, 10)
        self._log.debug("GC threshold set to %d", self._gc_threshold)
        self._log.info("Service '%s' started (pid=%d)", self._name, __import__("os").getpid())

    def stop(self) -> None:
        """Flush metrics, restore GC, mark service stopped."""
        self._running = False
        self._m_active.set(0.0)
        gc.collect()  # final GC pass

        elapsed = time.monotonic() - (self._start_time or time.monotonic())
        self._log.info(
            "Service '%s' stopping after %.1f s", self._name, elapsed
        )
        self._metrics.log_report(self._log)
        self._log.info("Service '%s' stopped", self._name)

    def _handle_shutdown_signal(self, signum: int, frame: Any) -> None:
        sig_name = signal.Signals(signum).name
        self._log.warning(
            "Received signal %s (%d) – requesting graceful shutdown",
            sig_name, signum,
        )
        self._shutdown_requested = True

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------
    def _validate_record(self, record: dict) -> None:
        """Validate a single record; raise ValidationError on failure."""
        if "id" not in record:
            raise ValidationError("Record missing required field 'id'", field="id")
        if not isinstance(record.get("value"), (int, float)):
            raise ValidationError(
                "Field 'value' must be numeric",
                field="value",
                value=record.get("value"),
            )

    def _process_record(self, record: dict) -> dict:
        """Transform a single record (subclasses override for custom logic)."""
        self._validate_record(record)
        # Simulate processing (e.g., inference, enrichment)
        return {
            "id": record["id"],
            "value": record["value"],
            "processed_value": record["value"] * 2 + 1,
            "tag": record.get("tag", "untagged"),
        }

    def _process_batch(self, batch: list[dict]) -> list[dict]:
        """Process a list of records and return results."""
        results: list[dict] = []
        for record in batch:
            try:
                result = self._process_record(record)
                results.append(result)
                self._m_processed.increment()
            except ValidationError as exc:
                self._m_errors.increment()
                self._log.warning(
                    "Validation failed for record id=%s: %s",
                    record.get("id", "?"), exc.to_dict(),
                )
            except AppError as exc:
                self._m_errors.increment()
                self._log.error("AppError processing record: %s", exc.to_dict())
        return results

    def run(self, records: list[dict]) -> list[dict]:
        """Process *records* in batches until exhausted or shutdown requested."""
        self._log.info(
            "Processing %d records with batch_size=%d",
            len(records), self._batch_size,
        )

        all_results: list[dict] = []
        total = len(records)
        processed_count = 0

        for batch_idx in range(0, total, self._batch_size):
            if self._shutdown_requested:
                self._log.warning(
                    "Shutdown requested; stopping after %d/%d records",
                    processed_count, total,
                )
                break

            batch = records[batch_idx : batch_idx + self._batch_size]
            t0 = time.perf_counter()

            batch_results = self._process_batch(batch)
            all_results.extend(batch_results)

            elapsed_ms = (time.perf_counter() - t0) * 1000
            self._m_latency.observe(elapsed_ms)
            self._m_batches.increment()
            processed_count += len(batch)

            self._log.info(
                "Batch %d: size=%d  ok=%d  latency=%.2f ms  progress=%d/%d",
                batch_idx // self._batch_size + 1,
                len(batch),
                len(batch_results),
                elapsed_ms,
                processed_count,
                total,
            )

            # Periodic GC to release inter-batch garbage
            if processed_count % (self._batch_size * 10) == 0:
                collected = gc.collect()
                self._log.debug("GC collected %d objects", collected)

        self._log.info(
            "Run complete: %d/%d records produced results",
            len(all_results), total,
        )
        return all_results

    def run_from_jsonl(self, file_path: pathlib.Path) -> int:
        """Process records streamed from a JSONL file.

        Keeps only one batch in memory at a time.
        Returns total records processed.
        """
        self._log.info("Streaming records from %s", file_path)

        batch: list[dict] = []
        total_processed = 0

        for record in stream_jsonl(file_path):
            batch.append(record)
            if len(batch) >= self._batch_size:
                self._process_batch(batch)
                total_processed += len(batch)
                batch = []

                if self._shutdown_requested:
                    self._log.warning("Shutdown during stream; stopping early")
                    break

        # Process remaining records
        if batch:
            self._process_batch(batch)
            total_processed += len(batch)

        self._log.info("JSONL stream complete: %d records total", total_processed)
        return total_processed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    cfg = _load_config()
    logger = _setup_logging(cfg)
    install_global_exception_handler(logger)

    logger.info("=== Production Service Demo ===")
    logger.info("Config: %s", _CONFIG_PATH)

    service = ProductionService(cfg)
    service.start()

    # Build synthetic dataset
    rng = random.Random(42)
    records: list[dict] = []
    for i in range(600):
        record: dict = {"id": f"rec-{i:04d}", "tag": f"group-{i % 5}"}
        if rng.random() < 0.05:
            record["value"] = "bad"  # intentionally invalid – triggers ValidationError
        else:
            record["value"] = rng.uniform(0, 100)
        records.append(record)

    results = service.run(records)
    logger.info(
        "Service processed %d input records -> %d results",
        len(records), len(results),
    )

    # Demonstrate JSONL streaming
    tmp_path = _PROJECT_ROOT / "logs" / "_demo_stream.jsonl"
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_path, "w") as fh:
        for i in range(50):
            fh.write(json.dumps({"id": f"stream-{i}", "value": float(i)}) + "\n")

    streamed = service.run_from_jsonl(tmp_path)
    logger.info("Streamed %d records from JSONL", streamed)
    tmp_path.unlink(missing_ok=True)

    service.stop()
    logger.info("=== Production Service Demo complete ===")


if __name__ == "__main__":
    main()
