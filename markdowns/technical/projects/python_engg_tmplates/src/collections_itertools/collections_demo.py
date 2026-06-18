"""
collections_demo.py
===================
Demonstrates Python's `collections` module:
  - defaultdict  : graph adjacency, word count, nested structures
  - Counter      : most_common, arithmetic, set operations
  - deque        : bounded sliding-window buffer
  - namedtuple   : lightweight immutable record
  - NamedTuple   : typed variant with defaults
  - ChainMap     : layered configuration (env > file > defaults)

Configuration is read from config.yaml; all output goes through the
project logging infrastructure.
"""

from __future__ import annotations

import logging
import logging.config
import os
import pathlib
import sys
from collections import ChainMap, Counter, defaultdict, deque, namedtuple
from typing import NamedTuple

import yaml

# ---------------------------------------------------------------------------
# Bootstrap: locate config.yaml relative to this file
# ---------------------------------------------------------------------------
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
_CONFIG_PATH = _PROJECT_ROOT / "config.yaml"


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
                    "format": (
                        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
                    )
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "detailed",
                    "stream": "ext://sys.stdout",
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "formatter": "detailed",
                    "filename": str(log_file),
                    "maxBytes": log_cfg["max_bytes"],
                    "backupCount": log_cfg["backup_count"],
                    "encoding": "utf-8",
                },
            },
            "root": {
                "level": log_cfg["level"],
                "handlers": ["console", "file"],
            },
        }
    )
    return logging.getLogger("collections_demo")


# ---------------------------------------------------------------------------
# 1. defaultdict – graph adjacency list
# ---------------------------------------------------------------------------
def demo_defaultdict_graph(logger: logging.Logger) -> None:
    logger.info("=== defaultdict: directed graph ===")

    graph: defaultdict[str, list[str]] = defaultdict(list)
    edges = [
        ("A", "B"), ("A", "C"), ("B", "D"),
        ("C", "D"), ("D", "E"), ("B", "E"),
    ]
    for src, dst in edges:
        graph[src].append(dst)

    for node, neighbours in sorted(graph.items()):
        logger.debug("  %s -> %s", node, neighbours)

    logger.info("Graph nodes: %s", sorted(graph.keys()))
    logger.info("Neighbours of A: %s", graph["A"])

    # BFS
    visited: list[str] = []
    queue = deque(["A"])
    seen: set[str] = {"A"}
    while queue:
        node = queue.popleft()
        visited.append(node)
        for nb in graph[node]:
            if nb not in seen:
                seen.add(nb)
                queue.append(nb)
    logger.info("BFS order from A: %s", visited)


# ---------------------------------------------------------------------------
# 2. defaultdict – word frequency (nested)
# ---------------------------------------------------------------------------
def demo_defaultdict_nested(logger: logging.Logger) -> None:
    logger.info("=== defaultdict: nested word frequency ===")

    corpus = [
        ("chapter_1", "the quick brown fox jumps over the lazy dog"),
        ("chapter_2", "the fox and the hound are friends"),
        ("chapter_3", "quick brown foxes are fast"),
    ]

    # doc -> word -> count  (nested defaultdict)
    doc_word: defaultdict[str, defaultdict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for doc, text in corpus:
        for word in text.split():
            doc_word[doc][word] += 1

    for doc, counts in doc_word.items():
        top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:3]
        logger.info("  %s top-3: %s", doc, top)


# ---------------------------------------------------------------------------
# 3. Counter
# ---------------------------------------------------------------------------
def demo_counter(logger: logging.Logger) -> None:
    logger.info("=== Counter operations ===")

    text = (
        "to be or not to be that is the question "
        "whether tis nobler in the mind to suffer"
    )
    c1 = Counter(text.split())
    logger.info("most_common(5): %s", c1.most_common(5))
    logger.info("total tokens: %d", c1.total())

    # Arithmetic
    c2 = Counter({"to": 10, "be": 5, "not": 2})
    combined = c1 + c2
    logger.info("After adding c2, 'to' count: %d", combined["to"])

    diff = c1 - c2
    logger.info("After subtracting c2, 'to' count: %d", diff["to"])

    # Set ops
    intersection = c1 & c2
    logger.info("Intersection (min counts): %s", dict(intersection))

    union = c1 | c2
    logger.info("'to' in union: %d", union["to"])


# ---------------------------------------------------------------------------
# 4. deque – bounded sliding-window
# ---------------------------------------------------------------------------
def demo_deque_sliding_window(
    logger: logging.Logger, cfg: dict
) -> None:
    logger.info("=== deque: sliding window ===")

    window_size: int = cfg["collections"]["sliding_window_size"]
    maxlen: int = cfg["collections"]["deque_maxlen"]

    window: deque[float] = deque(maxlen=window_size)
    data_stream = [1.0, 3.0, 2.0, 5.0, 4.0, 6.0, 8.0, 7.0, 9.0, 10.0]

    averages: list[float] = []
    for value in data_stream:
        window.append(value)
        if len(window) == window_size:
            avg = sum(window) / window_size
            averages.append(avg)
            logger.debug("  window=%s  avg=%.2f", list(window), avg)

    logger.info("Sliding averages (window=%d): %s", window_size, averages)

    # Demonstrate maxlen behaviour with a large bounded deque
    bounded: deque[int] = deque(maxlen=maxlen)
    for i in range(maxlen + 500):
        bounded.append(i)
    logger.info(
        "Bounded deque (maxlen=%d) length after %d appends: %d",
        maxlen, maxlen + 500, len(bounded),
    )
    logger.info("First element: %d  Last element: %d", bounded[0], bounded[-1])


# ---------------------------------------------------------------------------
# 5. namedtuple & typed NamedTuple
# ---------------------------------------------------------------------------
Point2D = namedtuple("Point2D", ["x", "y"])


class ModelMetrics(NamedTuple):
    """Typed NamedTuple for ML experiment tracking."""

    experiment_id: str
    accuracy: float
    loss: float
    epochs: int
    learning_rate: float = 1e-3  # default


def demo_namedtuple(logger: logging.Logger) -> None:
    logger.info("=== namedtuple / NamedTuple ===")

    p = Point2D(3.0, 4.0)
    logger.info("Point2D: x=%.1f  y=%.1f  magnitude=%.3f", p.x, p.y, (p.x**2 + p.y**2) ** 0.5)
    logger.debug("As dict: %s", p._asdict())

    m = ModelMetrics("exp-42", accuracy=0.9823, loss=0.0412, epochs=50)
    logger.info(
        "ModelMetrics: id=%s  acc=%.4f  loss=%.4f  lr=%g",
        m.experiment_id, m.accuracy, m.loss, m.learning_rate,
    )

    # _replace creates a new instance (immutable)
    m2 = m._replace(accuracy=0.9901, epochs=100)
    logger.info("After _replace: acc=%.4f  epochs=%d", m2.accuracy, m2.epochs)

    # NamedTuples support unpacking
    eid, acc, loss, ep, lr = m
    logger.debug("Unpacked: id=%s  acc=%.4f", eid, acc)


# ---------------------------------------------------------------------------
# 6. ChainMap – layered config (env overrides file overrides defaults)
# ---------------------------------------------------------------------------
def demo_chainmap(logger: logging.Logger) -> None:
    logger.info("=== ChainMap: layered configuration ===")

    defaults = {
        "debug": False,
        "workers": 2,
        "timeout": 30,
        "host": "localhost",
        "port": 8080,
    }
    file_config = {
        "workers": 8,
        "timeout": 60,
        "host": "0.0.0.0",
    }
    env_config = {
        "debug": True,
        "port": 9090,
    }

    # Higher-priority maps come first
    merged = ChainMap(env_config, file_config, defaults)

    for key in sorted(merged):
        logger.info("  %-10s = %s", key, merged[key])

    # Add a new override layer at runtime
    runtime = {"workers": 16}
    merged = merged.new_child(runtime)
    logger.info("After runtime override, workers=%d", merged["workers"])

    # Inspect which map each key resolves from
    for key in ["workers", "debug", "host"]:
        for idx, m in enumerate(merged.maps):
            if key in m:
                logger.debug("  '%s' resolved from map[%d]: %s", key, idx, m)
                break


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    cfg = _load_config()
    logger = _setup_logging(cfg)

    logger.info("Starting collections_demo  (config: %s)", _CONFIG_PATH)

    demo_defaultdict_graph(logger)
    demo_defaultdict_nested(logger)
    demo_counter(logger)
    demo_deque_sliding_window(logger, cfg)
    demo_namedtuple(logger)
    demo_chainmap(logger)

    logger.info("collections_demo complete.")


if __name__ == "__main__":
    main()
