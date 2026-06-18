"""
multiprocessing_patterns.py
============================
Demonstrates Python multiprocessing for CPU-bound work:
  - ProcessPoolExecutor       : map / submit with futures
  - Pool.imap_unordered       : streaming results
  - SharedMemory              : zero-copy data sharing between processes
  - Manager (dict/list/Queue) : coordinated shared state
  - Pool initializer          : load a "model" once per worker process
  - Benchmark vs threading for CPU-bound vs I/O-bound tasks

All constants from config.yaml; logs to logs/python_engg.log.
"""

from __future__ import annotations

import logging
import logging.config
import math
import multiprocessing
import pathlib
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager, Pool
from multiprocessing.shared_memory import SharedMemory
from typing import Any

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
                    "format": (
                        "%(asctime)s | %(levelname)-8s | %(name)s"
                        " | pid=%(process)d | %(message)s"
                    )
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
    return logging.getLogger("multiprocessing_patterns")


# ---------------------------------------------------------------------------
# CPU-bound work functions (top-level for pickling)
# ---------------------------------------------------------------------------
def _is_prime(n: int) -> bool:
    """Deterministic primality test."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(math.isqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True


def _count_primes_in_range(args: tuple[int, int]) -> int:
    """Count primes in [start, end)."""
    start, end = args
    return sum(1 for n in range(start, end) if _is_prime(n))


def _compute_chunk_stats(chunk: list[float]) -> dict[str, float]:
    """Compute mean and std for a data chunk."""
    n = len(chunk)
    mean = sum(chunk) / n
    variance = sum((x - mean) ** 2 for x in chunk) / n
    return {"n": n, "mean": mean, "std": math.sqrt(variance)}


# ---------------------------------------------------------------------------
# Pool initializer pattern (simulates model loading once per worker)
# ---------------------------------------------------------------------------
_worker_model: dict[str, Any] = {}


def _init_worker(model_config: dict) -> None:
    """Called once per worker process to initialise expensive resources."""
    import os
    _worker_model["weights"] = list(range(model_config["size"]))
    _worker_model["name"] = model_config["name"]
    logging.basicConfig(level=logging.WARNING)  # minimal logging in workers
    logging.getLogger().debug(
        "Worker PID=%d initialised model '%s'", os.getpid(), _worker_model["name"]
    )


def _predict(input_val: float) -> float:
    """Use pre-loaded model to make a prediction."""
    # Simulate inference with the pre-loaded weights
    weights = _worker_model["weights"]
    return sum(w * input_val for w in weights[:5]) / 5.0


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------
def demo_process_pool_executor(logger: logging.Logger, cfg: dict) -> None:
    logger.info("=== ProcessPoolExecutor: prime counting ===")

    num_processes: int = cfg["concurrency"]["process_pool_size"]
    limit = 100_000
    chunk_size = limit // num_processes
    ranges = [
        (i * chunk_size, min((i + 1) * chunk_size, limit))
        for i in range(num_processes)
    ]

    t0 = time.perf_counter()
    total_primes = 0

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = {
            executor.submit(_count_primes_in_range, r): r for r in ranges
        }
        for future in as_completed(futures):
            rng = futures[future]
            count = future.result()
            total_primes += count
            logger.debug("  range=%s  primes=%d", rng, count)

    elapsed = time.perf_counter() - t0
    logger.info(
        "Primes up to %d: %d  (found in %.3f s with %d processes)",
        limit, total_primes, elapsed, num_processes,
    )


def demo_pool_imap_unordered(logger: logging.Logger, cfg: dict) -> None:
    logger.info("=== Pool.imap_unordered: streaming results ===")

    import random
    num_processes: int = cfg["concurrency"]["process_pool_size"]
    chunks = [
        [random.gauss(0, 1) for _ in range(500)]
        for _ in range(20)
    ]

    t0 = time.perf_counter()
    all_means: list[float] = []

    with Pool(processes=num_processes) as pool:
        for stats in pool.imap_unordered(_compute_chunk_stats, chunks, chunksize=4):
            all_means.append(stats["mean"])
            logger.debug("  chunk mean=%.4f  std=%.4f", stats["mean"], stats["std"])

    grand_mean = sum(all_means) / len(all_means)
    logger.info(
        "imap_unordered: %d chunks processed in %.3f s  grand_mean=%.4f",
        len(chunks), time.perf_counter() - t0, grand_mean,
    )


def demo_shared_memory(logger: logging.Logger) -> None:
    logger.info("=== SharedMemory: zero-copy IPC ===")
    import array

    # Build a 1000-element integer array in shared memory
    num_items = 1_000
    item_size = 4  # 4 bytes per int32
    shm = SharedMemory(create=True, size=num_items * item_size)

    # Write into shared memory via a memoryview-backed array
    buf = array.array("i", range(num_items))
    shm.buf[: num_items * item_size] = buf.tobytes()

    def sum_segment(args: tuple[str, int, int, int]) -> int:
        shm_name, start, end, item_sz = args
        seg = SharedMemory(name=shm_name)
        data = array.array("i")
        data.frombytes(bytes(seg.buf[start * item_sz : end * item_sz]))
        seg.close()
        return sum(data)

    mid = num_items // 2
    with Pool(processes=2) as pool:
        results = pool.map(
            sum_segment,
            [
                (shm.name, 0, mid, item_size),
                (shm.name, mid, num_items, item_size),
            ],
        )

    total = sum(results)
    expected = sum(range(num_items))
    shm.close()
    shm.unlink()

    logger.info(
        "SharedMemory sum: computed=%d  expected=%d  correct=%s",
        total, expected, total == expected,
    )


def demo_manager(logger: logging.Logger, cfg: dict) -> None:
    logger.info("=== Manager: coordinated shared state ===")

    num_processes: int = cfg["concurrency"]["process_pool_size"]

    def worker_fn(args: tuple[Any, int, int]) -> None:
        shared_dict, shared_list, worker_id = args
        for i in range(5):
            shared_dict[f"worker-{worker_id}-{i}"] = worker_id * 10 + i
            shared_list.append(worker_id)

    with Manager() as manager:
        shared_dict = manager.dict()
        shared_list = manager.list()

        with Pool(processes=num_processes) as pool:
            pool.map(worker_fn, [(shared_dict, shared_list, i) for i in range(num_processes)])

        logger.info(
            "Manager dict size=%d  list length=%d",
            len(shared_dict), len(shared_list),
        )
        logger.debug("Dict sample: %s", dict(list(shared_dict.items())[:5]))


def demo_pool_initializer(logger: logging.Logger, cfg: dict) -> None:
    logger.info("=== Pool initializer: load model once per worker ===")

    model_config = {"name": "tiny-mlp", "size": 128}
    inputs = [float(i) for i in range(20)]
    num_processes: int = cfg["concurrency"]["process_pool_size"]

    t0 = time.perf_counter()
    with Pool(
        processes=num_processes,
        initializer=_init_worker,
        initargs=(model_config,),
    ) as pool:
        predictions = pool.map(_predict, inputs)

    logger.info(
        "Pool initializer: %d predictions in %.3f s  first_5=%s",
        len(predictions), time.perf_counter() - t0, predictions[:5],
    )


def demo_benchmark(logger: logging.Logger, cfg: dict) -> None:
    """Compare sequential vs multiprocessing for CPU-bound prime counting."""
    logger.info("=== Benchmark: sequential vs multiprocessing ===")

    limit = 50_000
    ranges = [(0, limit)]

    # Sequential
    t0 = time.perf_counter()
    seq_count = _count_primes_in_range(ranges[0])
    seq_time = time.perf_counter() - t0

    # Multiprocessing
    num_processes: int = cfg["concurrency"]["process_pool_size"]
    chunk = limit // num_processes
    sub_ranges = [(i * chunk, min((i + 1) * chunk, limit)) for i in range(num_processes)]

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        total = sum(executor.map(_count_primes_in_range, sub_ranges))
    mp_time = time.perf_counter() - t0

    assert total == seq_count
    logger.info(
        "Sequential: %.3f s  Multiprocessing (%d proc): %.3f s  speedup=%.2fx",
        seq_time, num_processes, mp_time, seq_time / mp_time,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    multiprocessing.set_start_method("spawn", force=True)

    cfg = _load_config()
    logger = _setup_logging(cfg)

    logger.info("Starting multiprocessing_patterns  (config: %s)", _CONFIG_PATH)

    demo_process_pool_executor(logger, cfg)
    demo_pool_imap_unordered(logger, cfg)
    demo_shared_memory(logger)
    demo_manager(logger, cfg)
    demo_pool_initializer(logger, cfg)
    demo_benchmark(logger, cfg)

    logger.info("multiprocessing_patterns complete.")


if __name__ == "__main__":
    main()
