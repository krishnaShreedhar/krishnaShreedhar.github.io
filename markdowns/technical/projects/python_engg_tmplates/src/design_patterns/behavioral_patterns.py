"""
behavioral_patterns.py
======================
Demonstrates behavioral design patterns:
  - Observer / EventBus    : publish-subscribe with weak references
  - Strategy Pattern       : SamplingStrategy hierarchy (Random, Stratified)
  - Retry Decorator        : exponential backoff + optional jitter
  - Context Managers       : class-based, generator (contextlib), ExitStack,
                             asynccontextmanager

All constants from config.yaml; logs to logs/python_engg.log.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import logging
import logging.config
import math
import pathlib
import random
import time
import weakref
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Callable, Generator, Optional

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
    return logging.getLogger("behavioral_patterns")


# ===========================================================================
# 1. Observer / EventBus
# ===========================================================================
EventHandler = Callable[[str, Any], None]


class EventBus:
    """Simple publish-subscribe event bus.

    Subscribers register callables for named events.  When an event is
    published, all registered handlers for that event are called.
    Dead (garbage-collected) handlers are silently pruned.
    """

    def __init__(self) -> None:
        self._handlers: defaultdict[str, list[weakref.ref[Any]]] = defaultdict(list)
        self._log = logging.getLogger("behavioral_patterns.EventBus")

    def subscribe(self, event: str, handler: EventHandler) -> None:
        ref = weakref.WeakMethod(handler) if hasattr(handler, "__self__") else weakref.ref(handler)
        self._handlers[event].append(ref)
        self._log.debug("Subscribed %s to event '%s'", handler, event)

    def unsubscribe(self, event: str, handler: EventHandler) -> None:
        self._handlers[event] = [
            ref for ref in self._handlers[event]
            if ref() is not None and ref() is not handler
        ]

    def publish(self, event: str, data: Any = None) -> int:
        """Publish *event* and return the number of handlers notified."""
        active: list[weakref.ref[Any]] = []
        notified = 0
        for ref in self._handlers.get(event, []):
            handler = ref()
            if handler is not None:
                try:
                    handler(event, data)
                    notified += 1
                except Exception:
                    self._log.exception("Handler %s raised for event '%s'", handler, event)
                active.append(ref)
        self._handlers[event] = active
        self._log.debug("Published '%s'  notified=%d", event, notified)
        return notified


class MetricsCollector:
    """Sample observer that accumulates event counts."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._counts: defaultdict[str, int] = defaultdict(int)
        self._log = logging.getLogger(f"behavioral_patterns.{name}")

    def on_event(self, event: str, data: Any) -> None:
        self._counts[event] += 1
        self._log.debug("[%s] received event='%s'  data=%r", self.name, event, data)

    def report(self) -> dict[str, int]:
        return dict(self._counts)


class AuditLogger:
    """Observer that writes audit trail entries."""

    def __init__(self) -> None:
        self._log = logging.getLogger("behavioral_patterns.AuditLogger")
        self._entries: list[dict[str, Any]] = []

    def on_event(self, event: str, data: Any) -> None:
        entry = {"event": event, "data": data, "ts": time.time()}
        self._entries.append(entry)
        self._log.info("AUDIT | event=%s  data=%r", event, data)

    def entries(self) -> list[dict[str, Any]]:
        return list(self._entries)


def demo_event_bus(logger: logging.Logger) -> None:
    logger.info("=== Observer / EventBus ===")

    bus = EventBus()
    metrics = MetricsCollector("metrics")
    audit = AuditLogger()

    bus.subscribe("user.login", metrics.on_event)
    bus.subscribe("user.login", audit.on_event)
    bus.subscribe("order.placed", metrics.on_event)
    bus.subscribe("order.placed", audit.on_event)
    bus.subscribe("payment.failed", metrics.on_event)

    bus.publish("user.login", {"user_id": "u-001", "ip": "10.0.0.1"})
    bus.publish("order.placed", {"order_id": "ord-42", "total": 149.99})
    bus.publish("order.placed", {"order_id": "ord-43", "total": 29.50})
    bus.publish("payment.failed", {"order_id": "ord-43", "reason": "insufficient_funds"})
    bus.publish("unknown.event", {})  # no subscribers

    logger.info("Metrics report: %s", metrics.report())
    logger.info("Audit entries: %d", len(audit.entries()))


# ===========================================================================
# 2. Strategy Pattern – Sampling
# ===========================================================================
class SamplingStrategy(ABC):
    """Abstract base for data sampling strategies."""

    @abstractmethod
    def sample(self, dataset: list[dict], n: int) -> list[dict]:
        """Return *n* samples from *dataset*."""


class RandomSampling(SamplingStrategy):
    """Uniform random sampling without replacement."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)
        self._log = logging.getLogger("behavioral_patterns.RandomSampling")

    def sample(self, dataset: list[dict], n: int) -> list[dict]:
        n = min(n, len(dataset))
        result = self._rng.sample(dataset, n)
        self._log.debug("RandomSampling: selected %d from %d", n, len(dataset))
        return result


class StratifiedSampling(SamplingStrategy):
    """Stratified sampling preserving class distribution.

    Draws proportionally from each stratum (label bucket).
    """

    def __init__(self, label_key: str, seed: Optional[int] = None) -> None:
        self._label_key = label_key
        self._rng = random.Random(seed)
        self._log = logging.getLogger("behavioral_patterns.StratifiedSampling")

    def sample(self, dataset: list[dict], n: int) -> list[dict]:
        # Group by label
        strata: defaultdict[Any, list[dict]] = defaultdict(list)
        for item in dataset:
            strata[item[self._label_key]].append(item)

        total = len(dataset)
        result: list[dict] = []
        for label, group in strata.items():
            quota = max(1, round(n * len(group) / total))
            quota = min(quota, len(group))
            result.extend(self._rng.sample(group, quota))
            self._log.debug(
                "Stratum '%s': size=%d  quota=%d", label, len(group), quota
            )

        # Adjust to exactly n
        if len(result) > n:
            result = result[:n]
        self._log.debug("StratifiedSampling: selected %d from %d", len(result), total)
        return result


class DataSampler:
    """Context that delegates sampling to a SamplingStrategy."""

    def __init__(self, strategy: SamplingStrategy) -> None:
        self._strategy = strategy
        self._log = logging.getLogger("behavioral_patterns.DataSampler")

    def set_strategy(self, strategy: SamplingStrategy) -> None:
        self._log.info("Switching strategy to %s", type(strategy).__name__)
        self._strategy = strategy

    def draw(self, dataset: list[dict], n: int) -> list[dict]:
        self._log.info(
            "Drawing %d samples using %s", n, type(self._strategy).__name__
        )
        return self._strategy.sample(dataset, n)


def demo_strategy(logger: logging.Logger) -> None:
    logger.info("=== Strategy Pattern (Sampling) ===")

    # Build a synthetic dataset with 3 classes
    dataset = (
        [{"id": i, "label": "A", "value": i} for i in range(60)]
        + [{"id": i + 60, "label": "B", "value": i} for i in range(30)]
        + [{"id": i + 90, "label": "C", "value": i} for i in range(10)]
    )  # 100 items: A=60, B=30, C=10

    sampler = DataSampler(RandomSampling(seed=42))
    random_sample = sampler.draw(dataset, 20)
    label_counts = defaultdict(int)
    for item in random_sample:
        label_counts[item["label"]] += 1
    logger.info("Random sample label counts: %s", dict(label_counts))

    sampler.set_strategy(StratifiedSampling(label_key="label", seed=42))
    stratified_sample = sampler.draw(dataset, 20)
    label_counts = defaultdict(int)
    for item in stratified_sample:
        label_counts[item["label"]] += 1
    logger.info("Stratified sample label counts: %s", dict(label_counts))


# ===========================================================================
# 3. Retry Decorator with exponential backoff + jitter
# ===========================================================================
def retry(
    max_attempts: int,
    backoff_factor: float,
    jitter: bool,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable:
    """Decorator factory: retry *func* up to *max_attempts* times.

    Args:
        max_attempts: Maximum number of total attempts (1 = no retries).
        backoff_factor: Base wait seconds; actual wait = factor * 2^(attempt-1).
        jitter: If True, add uniform random [0, wait] to each delay.
        exceptions: Exception types that trigger a retry.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            log = logging.getLogger(f"retry.{func.__name__}")
            last_exc: Optional[Exception] = None
            for attempt in range(1, max_attempts + 1):
                try:
                    result = func(*args, **kwargs)
                    if attempt > 1:
                        log.info("%s succeeded on attempt %d", func.__name__, attempt)
                    return result
                except exceptions as exc:
                    last_exc = exc
                    if attempt == max_attempts:
                        log.error(
                            "%s failed after %d attempts: %s",
                            func.__name__, max_attempts, exc,
                        )
                        raise
                    wait = backoff_factor * (2 ** (attempt - 1))
                    if jitter:
                        wait += random.uniform(0, wait)
                    log.warning(
                        "%s attempt %d/%d failed (%s); retrying in %.3f s",
                        func.__name__, attempt, max_attempts, exc, wait,
                    )
                    time.sleep(wait)
            raise RuntimeError("Unreachable")

        return wrapper

    return decorator


def demo_retry(logger: logging.Logger, cfg: dict) -> None:
    logger.info("=== Retry Decorator ===")

    max_attempts: int = cfg["retry"]["max_attempts"]
    backoff_factor: float = cfg["retry"]["backoff_factor"]
    use_jitter: bool = cfg["retry"]["jitter"]

    call_count = 0

    @retry(
        max_attempts=max_attempts,
        backoff_factor=0.05,  # short waits for demo
        jitter=use_jitter,
        exceptions=(ValueError,),
    )
    def flaky_service(threshold: float) -> str:
        nonlocal call_count
        call_count += 1
        if random.random() < threshold:
            raise ValueError(f"Transient error on call #{call_count}")
        return f"success on call #{call_count}"

    # Should succeed within max_attempts
    call_count = 0
    try:
        result = flaky_service(threshold=0.7)
        logger.info("flaky_service result: %s", result)
    except ValueError:
        logger.warning("flaky_service exhausted all retries")

    # Should always fail (threshold=1.0)
    call_count = 0
    try:
        flaky_service(threshold=1.0)
    except ValueError as exc:
        logger.info("Expected exhaustion: %s", exc)


# ===========================================================================
# 4. Context Managers
# ===========================================================================
class ManagedResource:
    """Class-based context manager demonstrating __enter__ / __exit__."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._log = logging.getLogger("behavioral_patterns.ManagedResource")

    def __enter__(self) -> "ManagedResource":
        self._log.debug("Acquiring resource: %s", self.name)
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: Exception | None,
        exc_tb: Any,
    ) -> bool:
        if exc_type is not None:
            self._log.warning(
                "Resource %s released after exception: %s", self.name, exc_val
            )
        else:
            self._log.debug("Resource %s released cleanly", self.name)
        return False  # do not suppress exceptions


@contextlib.contextmanager
def timer(label: str) -> Generator[None, None, None]:
    """Generator-based context manager for timing code blocks."""
    log = logging.getLogger("behavioral_patterns.timer")
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = (time.perf_counter() - t0) * 1000
        log.info("[%s] elapsed: %.3f ms", label, elapsed)


@contextlib.asynccontextmanager
async def async_db_transaction(db_name: str):
    """Async context manager simulating a DB transaction."""
    log = logging.getLogger("behavioral_patterns.AsyncTransaction")
    log.debug("BEGIN TRANSACTION on %s", db_name)
    try:
        yield {"db": db_name, "transaction_id": "txn-001"}
        log.debug("COMMIT on %s", db_name)
    except Exception:
        log.error("ROLLBACK on %s", db_name)
        raise


def demo_context_managers(logger: logging.Logger) -> None:
    logger.info("=== Context Managers ===")

    # Class-based
    with ManagedResource("database-connection") as res:
        logger.info("Using resource: %s", res.name)
        time.sleep(0.01)

    # Generator-based timer
    with timer("prime_check"):
        count = sum(1 for n in range(10_000) if all(n % i != 0 for i in range(2, int(math.isqrt(n)) + 1) if i > 1) and n > 1)
    logger.info("Primes < 10_000: %d", count)

    # ExitStack – dynamic number of context managers
    resources = [f"resource-{i}" for i in range(4)]
    with contextlib.ExitStack() as stack:
        managed = [stack.enter_context(ManagedResource(r)) for r in resources]
        logger.info("Opened %d resources via ExitStack", len(managed))
    # All resources released here

    # Async context manager
    async def run_async_cm() -> None:
        async with async_db_transaction("users_db") as txn:
            logger.info("Inside async transaction: %s", txn)
            await asyncio.sleep(0.01)

    asyncio.run(run_async_cm())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    cfg = _load_config()
    logger = _setup_logging(cfg)

    logger.info("Starting behavioral_patterns  (config: %s)", _CONFIG_PATH)

    demo_event_bus(logger)
    demo_strategy(logger)
    demo_retry(logger, cfg)
    demo_context_managers(logger)

    logger.info("behavioral_patterns complete.")


if __name__ == "__main__":
    main()
