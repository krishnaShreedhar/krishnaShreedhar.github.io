"""
itertools_demo.py
=================
Demonstrates Python's `itertools` module:
  - combinations / permutations / combinations_with_replacement
  - product
  - groupby        : group sorted iterable by key
  - chain / chain.from_iterable
  - accumulate     : running totals, custom operators
  - islice / takewhile / dropwhile
  - batched        : chunk an iterable (Python 3.12+, backfill for 3.11)
  - compress / filterfalse
  - cycle / repeat / count

Also covers generator patterns:
  - batch_generator      : memory-efficient chunking
  - sliding_window       : via deque
  - memory-mapped file iteration

Configuration is read from config.yaml.
"""

from __future__ import annotations

import itertools
import logging
import logging.config
import mmap
import operator
import pathlib
import tempfile
from collections import deque
from typing import Generator, Iterable, Iterator, TypeVar

import yaml

T = TypeVar("T")

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
_CONFIG_PATH = _PROJECT_ROOT / "config.yaml"


# ---------------------------------------------------------------------------
# Config & logging bootstrap
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
    return logging.getLogger("itertools_demo")


# ---------------------------------------------------------------------------
# Backfill itertools.batched for Python < 3.12
# ---------------------------------------------------------------------------
try:
    from itertools import batched  # type: ignore[attr-defined]
except ImportError:

    def batched(iterable: Iterable[T], n: int) -> Iterator[tuple[T, ...]]:  # type: ignore[misc]
        """Chunk *iterable* into tuples of length *n* (last chunk may be shorter)."""
        if n < 1:
            raise ValueError("n must be >= 1")
        it = iter(iterable)
        while chunk := tuple(itertools.islice(it, n)):
            yield chunk


# ---------------------------------------------------------------------------
# Generator utilities
# ---------------------------------------------------------------------------
def batch_generator(iterable: Iterable[T], batch_size: int) -> Generator[list[T], None, None]:
    """Yield successive lists of *batch_size* items from *iterable*.

    Memory-efficient: pulls one item at a time from the source iterator.
    """
    batch: list[T] = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def sliding_window(iterable: Iterable[T], window_size: int) -> Generator[tuple[T, ...], None, None]:
    """Yield overlapping windows of *window_size* elements."""
    it = iter(iterable)
    window: deque[T] = deque(itertools.islice(it, window_size), maxlen=window_size)
    if len(window) == window_size:
        yield tuple(window)
    for item in it:
        window.append(item)
        yield tuple(window)


def mmap_line_generator(file_path: str) -> Generator[str, None, None]:
    """Iterate over lines of a (potentially large) file via mmap.

    The OS page-caches only touched pages, so peak RSS stays low.
    """
    with open(file_path, "rb") as fh:
        with mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            for line in iter(mm.readline, b""):
                yield line.decode("utf-8", errors="replace").rstrip("\n")


# ---------------------------------------------------------------------------
# Demo functions
# ---------------------------------------------------------------------------
def demo_combinations_permutations(logger: logging.Logger) -> None:
    logger.info("=== combinations / permutations / product ===")

    items = ["A", "B", "C", "D"]

    combs = list(itertools.combinations(items, 2))
    logger.info("C(4,2): %s  (count=%d)", combs, len(combs))

    perms = list(itertools.permutations(items, 2))
    logger.info("P(4,2) count: %d  first-5=%s", len(perms), perms[:5])

    combs_r = list(itertools.combinations_with_replacement(["x", "y", "z"], 2))
    logger.info("C_r({x,y,z}, 2): %s", combs_r)

    # Cartesian product – hyperparameter grid
    lrs = [1e-3, 1e-4]
    batch_sizes = [32, 64, 128]
    grid = list(itertools.product(lrs, batch_sizes))
    logger.info("Hyperparameter grid (lr x batch_size): %s", grid)


def demo_groupby(logger: logging.Logger) -> None:
    logger.info("=== groupby ===")

    records = [
        {"name": "Alice", "dept": "eng"},
        {"name": "Bob", "dept": "eng"},
        {"name": "Carol", "dept": "hr"},
        {"name": "Dave", "dept": "hr"},
        {"name": "Eve", "dept": "eng"},
        {"name": "Frank", "dept": "finance"},
    ]
    # groupby requires the iterable to be sorted by the grouping key
    records_sorted = sorted(records, key=lambda r: r["dept"])

    for dept, members in itertools.groupby(records_sorted, key=lambda r: r["dept"]):
        names = [m["name"] for m in members]
        logger.info("  dept=%-10s  members=%s", dept, names)


def demo_chain_accumulate(logger: logging.Logger) -> None:
    logger.info("=== chain / accumulate ===")

    lists = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    flat = list(itertools.chain.from_iterable(lists))
    logger.info("chain.from_iterable: %s", flat)

    # Running sum
    running_sum = list(itertools.accumulate(flat))
    logger.info("Running sum: %s", running_sum)

    # Running product
    running_prod = list(itertools.accumulate(flat, operator.mul))
    logger.info("Running product (first 5): %s", running_prod[:5])

    # Cumulative max
    data = [3, 1, 4, 1, 5, 9, 2, 6]
    cum_max = list(itertools.accumulate(data, max))
    logger.info("Cumulative max of %s: %s", data, cum_max)


def demo_islice_takewhile_dropwhile(logger: logging.Logger) -> None:
    logger.info("=== islice / takewhile / dropwhile ===")

    naturals = itertools.count(1)

    # islice: first 10 naturals
    first_10 = list(itertools.islice(naturals, 10))
    logger.info("First 10 naturals: %s", first_10)

    # takewhile: values < 6
    data = [1, 2, 3, 4, 5, 3, 2, 1]
    taken = list(itertools.takewhile(lambda x: x < 6, data))
    logger.info("takewhile(x<6): %s", taken)

    # dropwhile: skip initial ascending run
    dropped = list(itertools.dropwhile(lambda x: x < 4, data))
    logger.info("dropwhile(x<4): %s", dropped)


def demo_compress_filterfalse(logger: logging.Logger) -> None:
    logger.info("=== compress / filterfalse ===")

    data = ["apple", "banana", "cherry", "date", "elderberry"]
    selectors = [1, 0, 1, 0, 1]
    selected = list(itertools.compress(data, selectors))
    logger.info("compress: %s", selected)

    evens = list(itertools.filterfalse(lambda x: x % 2, range(10)))
    logger.info("filterfalse(odd): %s", evens)


def demo_batched(logger: logging.Logger, cfg: dict) -> None:
    logger.info("=== batched (chunking) ===")

    batch_size: int = cfg["collections"]["batch_size"]
    dataset = list(range(1, 251))  # 250 items

    chunks = list(batched(dataset, batch_size))
    logger.info(
        "Batched %d items into %d chunks of size <= %d",
        len(dataset), len(chunks), batch_size,
    )
    logger.info("First chunk: %s...", list(chunks[0])[:5])
    logger.info("Last chunk size: %d", len(chunks[-1]))

    # batch_generator as a lazy alternative
    lazy_chunks = list(batch_generator(iter(dataset), batch_size))
    assert len(lazy_chunks) == len(chunks), "Mismatch in chunk count"
    logger.info("batch_generator produced same %d chunks", len(lazy_chunks))


def demo_sliding_window(logger: logging.Logger, cfg: dict) -> None:
    logger.info("=== sliding_window generator ===")

    window_size: int = cfg["collections"]["sliding_window_size"]
    series = [2, 4, 6, 8, 10, 12, 14, 16]

    windows = list(sliding_window(series, window_size))
    logger.info(
        "sliding_window(size=%d) over %s:", window_size, series
    )
    for w in windows:
        avg = sum(w) / len(w)
        logger.debug("  window=%s  avg=%.2f", w, avg)
    logger.info("Total windows produced: %d", len(windows))


def demo_mmap_generator(logger: logging.Logger) -> None:
    logger.info("=== mmap line generator ===")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as tmp:
        for i in range(1, 6):
            tmp.write(f"line {i}: the quick brown fox\n")
        tmp_path = tmp.name

    lines = list(mmap_line_generator(tmp_path))
    logger.info("mmap read %d lines from temp file", len(lines))
    for line in lines:
        logger.debug("  %s", line)

    pathlib.Path(tmp_path).unlink(missing_ok=True)


def demo_cycle_repeat(logger: logging.Logger) -> None:
    logger.info("=== cycle / repeat ===")

    colours = ["red", "green", "blue"]
    cycled = list(itertools.islice(itertools.cycle(colours), 9))
    logger.info("cycle(colours) x9: %s", cycled)

    repeated = list(itertools.repeat("NA", 5))
    logger.info("repeat('NA', 5): %s", repeated)

    # starmap with repeat for broadcasting
    doubled = list(itertools.starmap(operator.mul, itertools.zip_longest(range(1, 6), [], fillvalue=2)))
    logger.info("starmap mul x2: %s", doubled)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    cfg = _load_config()
    logger = _setup_logging(cfg)

    logger.info("Starting itertools_demo  (config: %s)", _CONFIG_PATH)

    demo_combinations_permutations(logger)
    demo_groupby(logger)
    demo_chain_accumulate(logger)
    demo_islice_takewhile_dropwhile(logger)
    demo_compress_filterfalse(logger)
    demo_batched(logger, cfg)
    demo_sliding_window(logger, cfg)
    demo_mmap_generator(logger)
    demo_cycle_repeat(logger)

    logger.info("itertools_demo complete.")


if __name__ == "__main__":
    main()
