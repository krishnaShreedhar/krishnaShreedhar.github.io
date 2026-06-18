"""
structural_patterns.py
======================
Demonstrates structural design patterns:
  - Adapter     : wrap incompatible interfaces
  - Decorator   : transparent capability layering
  - Facade      : simplified high-level API over a subsystem
  - Proxy       : lazy initialisation + access control + caching proxy
  - Composite   : tree of components (pipeline DAG)

All constants from config.yaml; logs to logs/python_engg.log.
"""

from __future__ import annotations

import logging
import logging.config
import pathlib
import time
from abc import ABC, abstractmethod
from typing import Any, Optional

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
    return logging.getLogger("structural_patterns")


# ===========================================================================
# 1. Adapter – unify two incompatible storage back-ends
# ===========================================================================
class ModernStorageInterface(ABC):
    """Target interface that application code depends on."""

    @abstractmethod
    def read(self, key: str) -> Optional[str]:
        ...

    @abstractmethod
    def write(self, key: str, value: str) -> None:
        ...

    @abstractmethod
    def delete(self, key: str) -> None:
        ...


class LegacyKeyValueStore:
    """Simulates a legacy key-value API with different method signatures."""

    def __init__(self) -> None:
        self._data: dict[str, str] = {}

    def fetch(self, identifier: str) -> str | None:
        return self._data.get(identifier)

    def store(self, identifier: str, payload: str) -> bool:
        self._data[identifier] = payload
        return True

    def remove(self, identifier: str) -> bool:
        return self._data.pop(identifier, None) is not None


class LegacyStoreAdapter(ModernStorageInterface):
    """Adapts LegacyKeyValueStore to ModernStorageInterface."""

    def __init__(self, legacy: LegacyKeyValueStore) -> None:
        self._legacy = legacy
        self._log = logging.getLogger("structural_patterns.LegacyStoreAdapter")

    def read(self, key: str) -> Optional[str]:
        value = self._legacy.fetch(key)
        self._log.debug("read(%s) -> %r", key, value)
        return value

    def write(self, key: str, value: str) -> None:
        ok = self._legacy.store(key, value)
        self._log.debug("write(%s) ok=%s", key, ok)

    def delete(self, key: str) -> None:
        ok = self._legacy.remove(key)
        self._log.debug("delete(%s) ok=%s", key, ok)


def demo_adapter(logger: logging.Logger) -> None:
    logger.info("=== Adapter pattern ===")

    legacy = LegacyKeyValueStore()
    storage: ModernStorageInterface = LegacyStoreAdapter(legacy)

    storage.write("config.debug", "true")
    storage.write("config.workers", "8")
    logger.info("read config.debug: %r", storage.read("config.debug"))
    logger.info("read config.workers: %r", storage.read("config.workers"))
    storage.delete("config.debug")
    logger.info("after delete, config.debug: %r", storage.read("config.debug"))


# ===========================================================================
# 2. Decorator pattern – add capabilities to an I/O reader
# ===========================================================================
class DataReader(ABC):
    @abstractmethod
    def read(self) -> list[dict]:
        ...


class CSVDataReader(DataReader):
    """Simulates reading CSV rows."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._log = logging.getLogger("structural_patterns.CSVDataReader")

    def read(self) -> list[dict]:
        self._log.debug("Reading CSV from %s", self._path)
        # Simulate CSV rows
        return [
            {"id": 1, "name": "Alice", "score": 92.5},
            {"id": 2, "name": "Bob", "score": None},
            {"id": 3, "name": "Carol", "score": 87.0},
            {"id": 4, "name": None, "score": 75.0},
        ]


class ValidatingReader(DataReader):
    """Decorator: drop rows with null values."""

    def __init__(self, wrapped: DataReader, required_fields: list[str]) -> None:
        self._wrapped = wrapped
        self._required = required_fields
        self._log = logging.getLogger("structural_patterns.ValidatingReader")

    def read(self) -> list[dict]:
        raw = self._wrapped.read()
        valid = [r for r in raw if all(r.get(f) is not None for f in self._required)]
        self._log.debug(
            "Validated: %d/%d rows passed", len(valid), len(raw)
        )
        return valid


class TimingReader(DataReader):
    """Decorator: measure and log read latency."""

    def __init__(self, wrapped: DataReader) -> None:
        self._wrapped = wrapped
        self._log = logging.getLogger("structural_patterns.TimingReader")

    def read(self) -> list[dict]:
        t0 = time.perf_counter()
        result = self._wrapped.read()
        elapsed = (time.perf_counter() - t0) * 1000
        self._log.info("Read %d rows in %.3f ms", len(result), elapsed)
        return result


def demo_decorator_pattern(logger: logging.Logger) -> None:
    logger.info("=== Decorator pattern (DataReader) ===")

    # Compose decorators: CSV -> validate -> time
    reader: DataReader = TimingReader(
        ValidatingReader(
            CSVDataReader("/data/students.csv"),
            required_fields=["id", "name", "score"],
        )
    )

    rows = reader.read()
    logger.info("Final rows: %s", rows)


# ===========================================================================
# 3. Facade – simplified ML pipeline API
# ===========================================================================
class _FeatureExtractor:
    """Internal subsystem component."""

    def extract(self, raw: list[dict]) -> list[list[float]]:
        return [[float(r.get("score", 0)), float(r.get("id", 0))] for r in raw]


class _ModelInference:
    """Internal subsystem component."""

    def predict(self, features: list[list[float]]) -> list[float]:
        return [sum(f) / len(f) for f in features]


class _ResultFormatter:
    """Internal subsystem component."""

    def format(self, records: list[dict], predictions: list[float]) -> list[dict]:
        return [
            {**r, "prediction": p}
            for r, p in zip(records, predictions)
        ]


class MLPipelineFacade:
    """High-level facade hiding the subsystem complexity."""

    def __init__(self) -> None:
        self._extractor = _FeatureExtractor()
        self._model = _ModelInference()
        self._formatter = _ResultFormatter()
        self._log = logging.getLogger("structural_patterns.MLPipelineFacade")

    def run(self, raw_records: list[dict]) -> list[dict]:
        self._log.info("Facade: running pipeline on %d records", len(raw_records))
        features = self._extractor.extract(raw_records)
        predictions = self._model.predict(features)
        result = self._formatter.format(raw_records, predictions)
        self._log.info("Facade: pipeline complete")
        return result


def demo_facade(logger: logging.Logger) -> None:
    logger.info("=== Facade pattern (MLPipeline) ===")

    records = [
        {"id": 1, "name": "Alice", "score": 92.5},
        {"id": 2, "name": "Bob", "score": 78.0},
        {"id": 3, "name": "Carol", "score": 88.5},
    ]

    pipeline = MLPipelineFacade()
    output = pipeline.run(records)
    for row in output:
        logger.info("  %s -> prediction=%.2f", row["name"], row["prediction"])


# ===========================================================================
# 4. Proxy – lazy init + caching proxy
# ===========================================================================
class ExpensiveService:
    """Simulates a resource-heavy object (e.g., a large ML model)."""

    def __init__(self) -> None:
        logging.getLogger("structural_patterns.ExpensiveService").info(
            "ExpensiveService initialising (takes time) ..."
        )
        time.sleep(0.05)  # simulate slow init
        self._data = list(range(1000))

    def compute(self, n: int) -> int:
        return sum(self._data[:n])


class CachingProxy:
    """Proxy that lazily creates the real service and caches results."""

    def __init__(self) -> None:
        self._real: Optional[ExpensiveService] = None
        self._cache: dict[int, int] = {}
        self._log = logging.getLogger("structural_patterns.CachingProxy")

    def compute(self, n: int) -> int:
        if n in self._cache:
            self._log.debug("Cache hit for n=%d", n)
            return self._cache[n]

        # Lazy init
        if self._real is None:
            self._log.info("Proxy: creating real service (lazy init)")
            self._real = ExpensiveService()

        result = self._real.compute(n)
        self._cache[n] = result
        self._log.debug("Computed and cached n=%d -> %d", n, result)
        return result


def demo_proxy(logger: logging.Logger) -> None:
    logger.info("=== Proxy pattern (lazy init + caching) ===")

    proxy = CachingProxy()

    for n in [10, 50, 10, 100, 50]:  # 10 and 50 hit cache on second call
        t0 = time.perf_counter()
        result = proxy.compute(n)
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info("compute(%d) = %d  (%.3f ms)", n, result, elapsed)


# ===========================================================================
# 5. Composite – pipeline DAG
# ===========================================================================
class PipelineComponent(ABC):
    """Component: leaf or composite node in a processing pipeline."""

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def execute(self, data: Any) -> Any:
        ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}('{self.name}')"


class TransformStep(PipelineComponent):
    """Leaf: applies a single transformation function."""

    def __init__(self, name: str, fn: Any) -> None:
        super().__init__(name)
        self._fn = fn
        self._log = logging.getLogger(f"structural_patterns.{name}")

    def execute(self, data: Any) -> Any:
        result = self._fn(data)
        self._log.debug("%s: %r -> %r", self.name, data, result)
        return result


class SequentialPipeline(PipelineComponent):
    """Composite: runs children sequentially, passing output to next input."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._steps: list[PipelineComponent] = []
        self._log = logging.getLogger(f"structural_patterns.{name}")

    def add(self, step: PipelineComponent) -> "SequentialPipeline":
        self._steps.append(step)
        return self

    def execute(self, data: Any) -> Any:
        self._log.info("Pipeline '%s' starting with %d steps", self.name, len(self._steps))
        result = data
        for step in self._steps:
            result = step.execute(result)
        self._log.info("Pipeline '%s' done -> %r", self.name, result)
        return result


def demo_composite(logger: logging.Logger) -> None:
    logger.info("=== Composite pattern (Pipeline DAG) ===")

    pipeline = (
        SequentialPipeline("data-prep")
        .add(TransformStep("normalise", lambda x: [v / max(x) for v in x]))
        .add(TransformStep("clip", lambda x: [min(1.0, max(0.0, v)) for v in x]))
        .add(TransformStep("round", lambda x: [round(v, 3) for v in x]))
    )

    raw = [0.5, 3.2, 1.1, 0.0, 4.8, 2.7]
    output = pipeline.execute(raw)
    logger.info("Input:  %s", raw)
    logger.info("Output: %s", output)

    # Nested pipelines (composite of composites)
    inner1 = SequentialPipeline("inner-1").add(
        TransformStep("double", lambda x: [v * 2 for v in x])
    )
    inner2 = SequentialPipeline("inner-2").add(
        TransformStep("negate", lambda x: [-v for v in x])
    )
    outer = (
        SequentialPipeline("outer")
        .add(inner1)
        .add(inner2)
    )
    result = outer.execute([1.0, 2.0, 3.0])
    logger.info("Nested pipeline output: %s", result)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    cfg = _load_config()
    logger = _setup_logging(cfg)

    logger.info("Starting structural_patterns  (config: %s)", _CONFIG_PATH)

    demo_adapter(logger)
    demo_decorator_pattern(logger)
    demo_facade(logger)
    demo_proxy(logger)
    demo_composite(logger)

    logger.info("structural_patterns complete.")


if __name__ == "__main__":
    main()
