"""
creational_patterns.py
======================
Demonstrates creational design patterns:
  - Singleton      : metaclass-based, thread-safe (double-checked locking)
  - Builder        : fluent QueryBuilder interface
  - TTL Cache      : thread-safe OrderedDict with expiry

All constants from config.yaml; logs to logs/python_engg.log.
"""

from __future__ import annotations

import logging
import logging.config
import pathlib
import threading
import time
from collections import OrderedDict
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
    return logging.getLogger("creational_patterns")


# ===========================================================================
# 1. Singleton – metaclass + thread-safe double-checked locking
# ===========================================================================
class SingletonMeta(type):
    """Thread-safe Singleton metaclass.

    Uses a class-level lock to guard the first construction.  After the
    instance is created, subsequent calls skip the lock entirely (the
    double-checked locking pattern).
    """

    _instances: dict[type, Any] = {}
    _lock: threading.Lock = threading.Lock()

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        # First check without lock (fast path for already-created singletons)
        if cls not in cls._instances:
            with cls._lock:
                # Second check inside lock (guard against race on first creation)
                if cls not in cls._instances:
                    instance = super().__call__(*args, **kwargs)
                    cls._instances[cls] = instance
        return cls._instances[cls]


class ConfigurationManager(metaclass=SingletonMeta):
    """Singleton that holds application-wide configuration.

    Only one instance exists per process, regardless of how many times
    the class is instantiated.
    """

    def __init__(self, data: dict | None = None) -> None:
        # __init__ may be called multiple times on the *same* instance when
        # the Singleton pattern is used; guard with an initialised flag.
        if not hasattr(self, "_initialised"):
            self._data: dict = data or {}
            self._initialised = True
            logging.getLogger("creational_patterns.Singleton").debug(
                "ConfigurationManager initialised (id=%d)", id(self)
            )

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def __repr__(self) -> str:
        return f"ConfigurationManager(id={id(self)}, keys={list(self._data.keys())})"


def demo_singleton(logger: logging.Logger) -> None:
    logger.info("=== Singleton pattern ===")

    cfg1 = ConfigurationManager({"env": "prod", "debug": False})
    cfg2 = ConfigurationManager({"this_is_ignored": True})  # same instance

    logger.info("cfg1 is cfg2: %s", cfg1 is cfg2)
    logger.info("cfg1 id=%d  cfg2 id=%d", id(cfg1), id(cfg2))
    logger.info("cfg2.get('env')=%r", cfg2.get("env"))

    # Thread-safety test: 50 threads all call the constructor simultaneously
    instances: list[ConfigurationManager] = []
    lock = threading.Lock()

    def get_instance() -> None:
        inst = ConfigurationManager()
        with lock:
            instances.append(inst)

    threads = [threading.Thread(target=get_instance) for _ in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    unique_ids = {id(i) for i in instances}
    logger.info(
        "50 threads -> unique instance IDs: %d  (should be 1)", len(unique_ids)
    )


# ===========================================================================
# 2. Builder – fluent QueryBuilder
# ===========================================================================
class QueryBuilder:
    """Build SQL SELECT queries via a fluent interface.

    Each method returns *self* so calls can be chained.  The final
    ``build()`` call returns the SQL string.
    """

    def __init__(self) -> None:
        self._table: str = ""
        self._columns: list[str] = []
        self._conditions: list[str] = []
        self._order_by: list[str] = []
        self._limit: Optional[int] = None
        self._offset: Optional[int] = None
        self._joins: list[str] = []

    def from_table(self, table: str) -> "QueryBuilder":
        self._table = table
        return self

    def select(self, *columns: str) -> "QueryBuilder":
        self._columns.extend(columns)
        return self

    def where(self, condition: str) -> "QueryBuilder":
        self._conditions.append(condition)
        return self

    def join(self, table: str, on: str, join_type: str = "INNER") -> "QueryBuilder":
        self._joins.append(f"{join_type} JOIN {table} ON {on}")
        return self

    def order_by(self, *columns: str) -> "QueryBuilder":
        self._order_by.extend(columns)
        return self

    def limit(self, n: int) -> "QueryBuilder":
        self._limit = n
        return self

    def offset(self, n: int) -> "QueryBuilder":
        self._offset = n
        return self

    def build(self) -> str:
        if not self._table:
            raise ValueError("Table must be specified via .from_table()")

        cols = ", ".join(self._columns) if self._columns else "*"
        sql = f"SELECT {cols} FROM {self._table}"

        for join in self._joins:
            sql += f"\n  {join}"

        if self._conditions:
            sql += "\n  WHERE " + " AND ".join(self._conditions)

        if self._order_by:
            sql += "\n  ORDER BY " + ", ".join(self._order_by)

        if self._limit is not None:
            sql += f"\n  LIMIT {self._limit}"

        if self._offset is not None:
            sql += f"\n  OFFSET {self._offset}"

        return sql


def demo_builder(logger: logging.Logger) -> None:
    logger.info("=== Builder pattern (QueryBuilder) ===")

    # Simple query
    simple = (
        QueryBuilder()
        .from_table("users")
        .select("id", "name", "email")
        .where("active = TRUE")
        .order_by("name ASC")
        .limit(10)
        .build()
    )
    logger.info("Simple query:\n%s", simple)

    # Complex query with join
    complex_query = (
        QueryBuilder()
        .from_table("orders o")
        .select("o.id", "o.total", "u.name", "u.email")
        .join("users u", "o.user_id = u.id")
        .join("products p", "o.product_id = p.id", join_type="LEFT")
        .where("o.status = 'shipped'")
        .where("o.total > 100")
        .order_by("o.created_at DESC")
        .limit(50)
        .offset(100)
        .build()
    )
    logger.info("Complex query:\n%s", complex_query)


# ===========================================================================
# 3. TTL Cache – thread-safe, OrderedDict-backed
# ===========================================================================
class TTLCache:
    """Thread-safe in-memory cache with per-entry TTL.

    Eviction is lazy (on access) plus eager (on ``put`` when at capacity).
    Uses ``OrderedDict`` for O(1) LRU-style eviction.
    """

    def __init__(self, max_size: int, ttl_seconds: float) -> None:
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._store: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._lock = threading.Lock()
        self._log = logging.getLogger("creational_patterns.TTLCache")
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._store:
                self._misses += 1
                return None
            value, expires_at = self._store[key]
            if time.monotonic() > expires_at:
                del self._store[key]
                self._misses += 1
                self._log.debug("Cache miss (expired): %s", key)
                return None
            # Move to end (most recently used)
            self._store.move_to_end(key)
            self._hits += 1
            self._log.debug("Cache hit: %s", key)
            return value

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = (value, time.monotonic() + self._ttl)
            # Evict oldest entry if over capacity
            while len(self._store) > self._max_size:
                evicted_key, _ = self._store.popitem(last=False)
                self._log.debug("Evicted (capacity): %s", evicted_key)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "size": len(self._store),
            }

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


def demo_ttl_cache(logger: logging.Logger, cfg: dict) -> None:
    logger.info("=== TTL Cache ===")

    max_size: int = cfg["cache"]["max_size"]
    ttl: float = cfg["cache"]["ttl_seconds"]

    cache = TTLCache(max_size=max_size, ttl_seconds=ttl)

    # Fill cache
    for i in range(10):
        cache.put(f"key-{i}", f"value-{i}")

    logger.info("Cache size after 10 puts: %d", len(cache))

    # Hits
    for i in range(5):
        val = cache.get(f"key-{i}")
        logger.debug("get(key-%d) = %r", i, val)

    # Miss
    miss = cache.get("nonexistent")
    logger.info("get(nonexistent) = %r", miss)

    # Demonstrate TTL expiry with a short-lived cache
    short_cache = TTLCache(max_size=10, ttl_seconds=0.05)
    short_cache.put("ephemeral", "I will expire")
    logger.info("Before TTL: %r", short_cache.get("ephemeral"))
    time.sleep(0.06)
    logger.info("After TTL: %r", short_cache.get("ephemeral"))

    # Capacity eviction
    tiny_cache = TTLCache(max_size=3, ttl_seconds=60)
    for i in range(6):
        tiny_cache.put(f"item-{i}", i)
    logger.info("Tiny cache (max=3) size after 6 puts: %d", len(tiny_cache))

    logger.info("Cache stats: %s", cache.stats())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    cfg = _load_config()
    logger = _setup_logging(cfg)

    logger.info("Starting creational_patterns  (config: %s)", _CONFIG_PATH)

    demo_singleton(logger)
    demo_builder(logger)
    demo_ttl_cache(logger, cfg)

    logger.info("creational_patterns complete.")


if __name__ == "__main__":
    main()
