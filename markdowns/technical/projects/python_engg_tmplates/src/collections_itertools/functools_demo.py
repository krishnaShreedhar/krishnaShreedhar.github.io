"""
functools_demo.py
=================
Demonstrates Python's `functools` module:
  - reduce          : fold / aggregate
  - partial         : pre-fill function arguments
  - lru_cache       : memoization with cache_info statistics
  - cached_property : lazy one-time attribute computation
  - singledispatch  : function overloading by argument type
  - wraps           : proper decorator metadata forwarding
  - total_ordering  : fill in comparison methods

All constants from config.yaml; logs written to logs/python_engg.log.
"""

from __future__ import annotations

import functools
import logging
import logging.config
import math
import pathlib
import time
from typing import Any

import yaml

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
_CONFIG_PATH = _PROJECT_ROOT / "config.yaml"


# ---------------------------------------------------------------------------
# Config & logging
# ---------------------------------------------------------------------------
def _load_config() -> dict:
    with open(_CONFIG_PATH, "r") as fh:
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
                "detailed": {
                    "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                    "formatter": "detailed",
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": str(log_file),
                    "maxBytes": log_cfg["max_bytes"],
                    "backupCount": log_cfg["backup_count"],
                    "formatter": "detailed",
                    "encoding": "utf-8",
                },
            },
            "root": {"level": log_cfg["level"], "handlers": ["console", "file"]},
        }
    )
    return logging.getLogger("functools_demo")


# ---------------------------------------------------------------------------
# 1. functools.reduce
# ---------------------------------------------------------------------------
def demo_reduce(logger: logging.Logger) -> None:
    logger.info("=== functools.reduce ===")

    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    total = functools.reduce(lambda acc, x: acc + x, data)
    logger.info("Sum of 1..10 via reduce: %d", total)

    product = functools.reduce(lambda acc, x: acc * x, data)
    logger.info("Product of 1..10 via reduce: %d", product)

    # Flatten nested list via reduce
    nested = [[1, 2], [3, 4], [5, 6]]
    flat = functools.reduce(lambda acc, x: acc + x, nested, [])
    logger.info("Flatten %s -> %s", nested, flat)

    # Pipeline of functions via reduce
    pipeline = [
        lambda x: x * 2,
        lambda x: x + 10,
        lambda x: x ** 2,
    ]
    result = functools.reduce(lambda v, fn: fn(v), pipeline, 5)
    # (5*2 + 10)^2 = 400
    logger.info("Pipeline (5 -> *2 -> +10 -> ^2): %d  (expected 400)", result)


# ---------------------------------------------------------------------------
# 2. functools.partial
# ---------------------------------------------------------------------------
def _power(base: float, exponent: float) -> float:
    return base ** exponent


def demo_partial(logger: logging.Logger) -> None:
    logger.info("=== functools.partial ===")

    square = functools.partial(_power, exponent=2)
    cube = functools.partial(_power, exponent=3)
    sqrt = functools.partial(_power, exponent=0.5)

    for n in [2, 3, 4, 9, 16]:
        logger.info(
            "n=%d  square=%.0f  cube=%.0f  sqrt=%.3f",
            n, square(n), cube(n), sqrt(n),
        )

    # partial for filtering
    numbers = list(range(-5, 6))
    above_zero = list(filter(functools.partial(lambda threshold, x: x > threshold, 0), numbers))
    logger.info("Above zero: %s", above_zero)


# ---------------------------------------------------------------------------
# 3. lru_cache with cache_info
# ---------------------------------------------------------------------------
def _make_cached_fibonacci(maxsize: int):
    @functools.lru_cache(maxsize=maxsize)
    def fib(n: int) -> int:
        if n < 2:
            return n
        return fib(n - 1) + fib(n - 2)

    return fib


def demo_lru_cache(logger: logging.Logger, cfg: dict) -> None:
    logger.info("=== functools.lru_cache ===")

    maxsize: int = cfg["collections"]["lru_cache_size"]
    fib = _make_cached_fibonacci(maxsize)

    # First call – cold cache
    t0 = time.perf_counter()
    result = fib(35)
    cold_ms = (time.perf_counter() - t0) * 1000

    info = fib.cache_info()
    logger.info(
        "fib(35)=%d  cold=%.3f ms  hits=%d  misses=%d  currsize=%d",
        result, cold_ms, info.hits, info.misses, info.currsize,
    )

    # Second call – warm cache
    t0 = time.perf_counter()
    fib(35)
    warm_ms = (time.perf_counter() - t0) * 1000
    info = fib.cache_info()
    logger.info(
        "Second call: warm=%.4f ms  hits=%d  speedup=%.0fx",
        warm_ms, info.hits, cold_ms / max(warm_ms, 1e-9),
    )

    # Demonstrate cache clearing
    fib.cache_clear()
    logger.info("Cache cleared. cache_info: %s", fib.cache_info())


# ---------------------------------------------------------------------------
# 4. cached_property
# ---------------------------------------------------------------------------
class DataPipeline:
    """Simulates an expensive data-loading step via cached_property."""

    def __init__(self, source: list[int]) -> None:
        self._source = source
        self._logger = logging.getLogger("functools_demo.DataPipeline")

    @functools.cached_property
    def statistics(self) -> dict[str, float]:
        self._logger.debug("Computing statistics (expensive operation) ...")
        time.sleep(0.05)  # simulate I/O / computation
        n = len(self._source)
        mean = sum(self._source) / n
        variance = sum((x - mean) ** 2 for x in self._source) / n
        return {
            "n": float(n),
            "mean": mean,
            "std": math.sqrt(variance),
            "min": float(min(self._source)),
            "max": float(max(self._source)),
        }


def demo_cached_property(logger: logging.Logger) -> None:
    logger.info("=== functools.cached_property ===")

    pipeline = DataPipeline(list(range(1, 101)))

    t0 = time.perf_counter()
    stats = pipeline.statistics
    t1 = time.perf_counter()
    logger.info("First access: %.1f ms  stats=%s", (t1 - t0) * 1000, stats)

    t0 = time.perf_counter()
    _ = pipeline.statistics  # cached
    t1 = time.perf_counter()
    logger.info("Second access: %.4f ms  (cached, no recomputation)", (t1 - t0) * 1000)


# ---------------------------------------------------------------------------
# 5. singledispatch – polymorphic serialisation
# ---------------------------------------------------------------------------
@functools.singledispatch
def serialize(obj: Any) -> str:
    raise NotImplementedError(f"No serializer for type {type(obj).__name__}")


@serialize.register(int)
@serialize.register(float)
def _serialize_number(obj: int | float) -> str:
    return f"NUMBER:{obj}"


@serialize.register(str)
def _serialize_str(obj: str) -> str:
    return f"STRING:{obj!r}"


@serialize.register(list)
def _serialize_list(obj: list) -> str:
    return "LIST:[" + ", ".join(serialize(item) for item in obj) + "]"


@serialize.register(dict)
def _serialize_dict(obj: dict) -> str:
    pairs = ", ".join(f"{k!r}: {serialize(v)}" for k, v in obj.items())
    return f"DICT:{{{pairs}}}"


def demo_singledispatch(logger: logging.Logger) -> None:
    logger.info("=== functools.singledispatch ===")

    samples: list[Any] = [
        42,
        3.14,
        "hello",
        [1, "two", 3.0],
        {"a": 1, "b": [2, 3]},
    ]
    for obj in samples:
        result = serialize(obj)
        logger.info("  %-30r -> %s", obj, result)


# ---------------------------------------------------------------------------
# 6. wraps – preserving decorator metadata
# ---------------------------------------------------------------------------
def timing_decorator(func):
    """Measure and log function execution time."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        log = logging.getLogger("functools_demo.timer")
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = (time.perf_counter() - t0) * 1000
        log.debug("  %s took %.3f ms", func.__name__, elapsed)
        return result

    return wrapper


@timing_decorator
def expensive_sort(data: list[int]) -> list[int]:
    """Sort a list using Timsort."""
    return sorted(data)


def demo_wraps(logger: logging.Logger) -> None:
    logger.info("=== functools.wraps ===")

    import random
    data = random.sample(range(10_000), 1_000)
    sorted_data = expensive_sort(data)

    logger.info(
        "expensive_sort.__name__='%s'  __doc__='%s'  first_5=%s",
        expensive_sort.__name__,
        expensive_sort.__doc__,
        sorted_data[:5],
    )


# ---------------------------------------------------------------------------
# 7. total_ordering
# ---------------------------------------------------------------------------
@functools.total_ordering
class Version:
    """Semantic version with only __eq__ and __lt__ defined; rest auto-filled."""

    def __init__(self, major: int, minor: int, patch: int) -> None:
        self.major = major
        self.minor = minor
        self.patch = patch

    def _tuple(self) -> tuple[int, int, int]:
        return (self.major, self.minor, self.patch)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Version):
            return NotImplemented
        return self._tuple() == other._tuple()

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Version):
            return NotImplemented
        return self._tuple() < other._tuple()

    def __repr__(self) -> str:
        return f"Version({self.major}.{self.minor}.{self.patch})"


def demo_total_ordering(logger: logging.Logger) -> None:
    logger.info("=== functools.total_ordering ===")

    versions = [
        Version(2, 0, 0),
        Version(1, 9, 3),
        Version(1, 10, 0),
        Version(2, 0, 1),
        Version(1, 9, 3),
    ]
    sorted_versions = sorted(versions)
    logger.info("Sorted versions: %s", sorted_versions)
    logger.info("Max version: %s", max(versions))
    logger.info("1.9.3 == 1.9.3: %s", Version(1, 9, 3) == Version(1, 9, 3))
    logger.info("1.9.3 >= 1.10.0: %s", Version(1, 9, 3) >= Version(1, 10, 0))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    cfg = _load_config()
    logger = _setup_logging(cfg)

    logger.info("Starting functools_demo  (config: %s)", _CONFIG_PATH)

    demo_reduce(logger)
    demo_partial(logger)
    demo_lru_cache(logger, cfg)
    demo_cached_property(logger)
    demo_singledispatch(logger)
    demo_wraps(logger)
    demo_total_ordering(logger)

    logger.info("functools_demo complete.")


if __name__ == "__main__":
    main()
