"""
metrics.py
==========
In-process metrics collection without external dependencies:
  - Counter    : monotonically increasing integer counters
  - Gauge      : current value snapshot (can go up or down)
  - Histogram  : record observations; compute percentiles on demand
  - MetricsRegistry : thread-safe registry of named metrics
  - report()   : formatted text summary for logging

All histogram config (maxlen, percentiles) from config.yaml.
"""

from __future__ import annotations

import logging
import logging.config
import math
import pathlib
import statistics
import threading
import time
from collections import deque
from typing import Optional

import yaml

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
                "fmt": {
                    "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "fmt",
                    "stream": "ext://sys.stdout",
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "formatter": "fmt",
                    "filename": str(log_file),
                    "maxBytes": log_cfg["max_bytes"],
                    "backupCount": log_cfg["backup_count"],
                    "encoding": "utf-8",
                },
            },
            "root": {"level": log_cfg["level"], "handlers": ["console", "file"]},
        }
    )
    return logging.getLogger("metrics")


# ---------------------------------------------------------------------------
# Metric types
# ---------------------------------------------------------------------------
class Counter:
    """Thread-safe monotonically increasing counter."""

    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description
        self._value: int = 0
        self._lock = threading.Lock()

    def increment(self, amount: int = 1) -> None:
        with self._lock:
            self._value += amount

    def value(self) -> int:
        with self._lock:
            return self._value

    def snapshot(self) -> dict:
        return {
            "type": "counter",
            "name": self.name,
            "value": self.value(),
        }


class Gauge:
    """Thread-safe gauge (can go up or down)."""

    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description
        self._value: float = 0.0
        self._lock = threading.Lock()

    def set(self, value: float) -> None:
        with self._lock:
            self._value = value

    def increment(self, amount: float = 1.0) -> None:
        with self._lock:
            self._value += amount

    def decrement(self, amount: float = 1.0) -> None:
        with self._lock:
            self._value -= amount

    def value(self) -> float:
        with self._lock:
            return self._value

    def snapshot(self) -> dict:
        return {
            "type": "gauge",
            "name": self.name,
            "value": self.value(),
        }


class Histogram:
    """Thread-safe fixed-window histogram with percentile computation.

    Stores the last *maxlen* observations in a circular deque.
    Percentiles are computed lazily on ``snapshot()``.
    """

    def __init__(
        self,
        name: str,
        maxlen: int,
        percentiles: list[float],
        description: str = "",
    ) -> None:
        self.name = name
        self.description = description
        self._maxlen = maxlen
        self._percentiles = sorted(percentiles)
        self._observations: deque[float] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._total_count: int = 0
        self._total_sum: float = 0.0

    def observe(self, value: float) -> None:
        with self._lock:
            self._observations.append(value)
            self._total_count += 1
            self._total_sum += value

    def _percentile(self, sorted_data: list[float], p: float) -> float:
        """Compute p-th percentile (0 < p <= 1) from sorted data."""
        if not sorted_data:
            return float("nan")
        n = len(sorted_data)
        idx = p * (n - 1)
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        return sorted_data[lo] + (idx - lo) * (sorted_data[hi] - sorted_data[lo])

    def snapshot(self) -> dict:
        with self._lock:
            data = sorted(self._observations)
            count = self._total_count
            total_sum = self._total_sum

        if not data:
            return {
                "type": "histogram",
                "name": self.name,
                "count": 0,
                "sum": 0.0,
                "mean": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
                "percentiles": {},
            }

        n = len(data)
        mean = total_sum / count
        variance = sum((x - mean) ** 2 for x in data) / n if n > 1 else 0.0

        return {
            "type": "histogram",
            "name": self.name,
            "count": count,
            "window_size": n,
            "sum": total_sum,
            "mean": mean,
            "std": math.sqrt(variance),
            "min": data[0],
            "max": data[-1],
            "percentiles": {
                f"p{int(p * 100)}": self._percentile(data, p)
                for p in self._percentiles
            },
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
class MetricsRegistry:
    """Central registry for all metrics in a service.

    Metrics are identified by name; a second call with the same name
    returns the existing metric (singleton per name).
    """

    def __init__(self, maxlen: int, percentiles: list[float]) -> None:
        self._maxlen = maxlen
        self._percentiles = percentiles
        self._counters: dict[str, Counter] = {}
        self._gauges: dict[str, Gauge] = {}
        self._histograms: dict[str, Histogram] = {}
        self._lock = threading.Lock()
        self._log = logging.getLogger("metrics.MetricsRegistry")

    def counter(self, name: str, description: str = "") -> Counter:
        with self._lock:
            if name not in self._counters:
                self._counters[name] = Counter(name, description)
                self._log.debug("Registered counter '%s'", name)
            return self._counters[name]

    def gauge(self, name: str, description: str = "") -> Gauge:
        with self._lock:
            if name not in self._gauges:
                self._gauges[name] = Gauge(name, description)
                self._log.debug("Registered gauge '%s'", name)
            return self._gauges[name]

    def histogram(self, name: str, description: str = "") -> Histogram:
        with self._lock:
            if name not in self._histograms:
                self._histograms[name] = Histogram(
                    name, self._maxlen, self._percentiles, description
                )
                self._log.debug("Registered histogram '%s'", name)
            return self._histograms[name]

    def report(self) -> list[dict]:
        """Return a list of metric snapshot dicts."""
        snapshots: list[dict] = []
        with self._lock:
            for m in self._counters.values():
                snapshots.append(m.snapshot())
            for m in self._gauges.values():
                snapshots.append(m.snapshot())
            for m in self._histograms.values():
                snapshots.append(m.snapshot())
        return snapshots

    def log_report(self, logger: logging.Logger) -> None:
        """Log a formatted report of all metrics."""
        logger.info("===== Metrics Report =====")
        for snap in self.report():
            metric_type = snap["type"]
            name = snap["name"]
            if metric_type == "counter":
                logger.info("  COUNTER  %-40s  value=%d", name, snap["value"])
            elif metric_type == "gauge":
                logger.info("  GAUGE    %-40s  value=%.4f", name, snap["value"])
            elif metric_type == "histogram":
                pcts = snap.get("percentiles", {})
                logger.info(
                    "  HIST     %-40s  count=%d  mean=%.3f  min=%.3f  max=%.3f  %s",
                    name,
                    snap["count"],
                    snap.get("mean", float("nan")),
                    snap.get("min", float("nan")),
                    snap.get("max", float("nan")),
                    "  ".join(f"{k}={v:.3f}" for k, v in pcts.items()),
                )
        logger.info("===== End Metrics Report =====")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
def demo_metrics(logger: logging.Logger, cfg: dict) -> None:
    logger.info("=== Metrics demo ===")

    metrics_cfg = cfg["metrics"]
    registry = MetricsRegistry(
        maxlen=metrics_cfg["histogram_maxlen"],
        percentiles=metrics_cfg["percentiles"],
    )

    # Counters
    request_counter = registry.counter("http.requests.total", "Total HTTP requests")
    error_counter = registry.counter("http.errors.total", "Total HTTP errors")

    # Gauge
    active_connections = registry.gauge("connections.active", "Active connections")

    # Histogram
    latency_hist = registry.histogram("http.request_latency_ms", "Request latency (ms)")

    import random

    # Simulate 200 requests
    for i in range(200):
        request_counter.increment()
        active_connections.set(random.randint(5, 50))

        latency = random.lognormvariate(3.5, 0.8)  # log-normal latency
        latency_hist.observe(latency)

        if random.random() < 0.05:  # 5% error rate
            error_counter.increment()

    # Same name -> same object
    assert registry.counter("http.requests.total") is request_counter

    registry.log_report(logger)


def main() -> None:
    cfg = _load_config()
    logger = _setup_logging(cfg)
    logger.info("Starting metrics demo  (config: %s)", _CONFIG_PATH)
    demo_metrics(logger, cfg)
    logger.info("metrics demo complete.")


if __name__ == "__main__":
    main()
