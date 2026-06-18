"""
threading_patterns.py
=====================
Demonstrates Python threading primitives and patterns:
  - ThreadPoolExecutor with as_completed
  - Thread-safe Queue (producer/consumer)
  - Lock        : mutual exclusion
  - Semaphore   : bounded concurrency
  - Event       : one-shot signalling
  - Barrier     : synchronise N threads at a checkpoint
  - Thread-local storage
  - Benchmark vs sequential for I/O-bound work

All constants from config.yaml; logs to logs/python_engg.log.
"""

from __future__ import annotations

import logging
import logging.config
import pathlib
import queue
import random
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
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
                    "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(threadName)s | %(message)s"
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
    return logging.getLogger("threading_patterns")


# ---------------------------------------------------------------------------
# 1. ThreadPoolExecutor + as_completed
# ---------------------------------------------------------------------------
def _simulate_io_task(task_id: int, sleep_s: float) -> dict[str, Any]:
    """Simulate an I/O-bound operation (e.g., HTTP request, DB query)."""
    time.sleep(sleep_s)
    return {"task_id": task_id, "latency_ms": sleep_s * 1000, "result": task_id ** 2}


def demo_thread_pool(logger: logging.Logger, cfg: dict) -> None:
    logger.info("=== ThreadPoolExecutor + as_completed ===")

    pool_size: int = cfg["concurrency"]["thread_pool_size"]
    num_tasks = 20
    latencies = [random.uniform(0.01, 0.15) for _ in range(num_tasks)]

    t0 = time.perf_counter()
    completed_order: list[int] = []

    with ThreadPoolExecutor(max_workers=pool_size, thread_name_prefix="io-worker") as executor:
        future_to_id: dict[Future, int] = {
            executor.submit(_simulate_io_task, i, latencies[i]): i
            for i in range(num_tasks)
        }

        for future in as_completed(future_to_id):
            task_id = future_to_id[future]
            result = future.result()  # raises if the task raised
            completed_order.append(task_id)
            logger.debug("  Task %d done: %s", task_id, result)

    elapsed = time.perf_counter() - t0
    sequential_time = sum(latencies)
    logger.info(
        "Finished %d tasks in %.3f s (sequential would take %.3f s, speedup=%.1fx)",
        num_tasks, elapsed, sequential_time, sequential_time / elapsed,
    )
    logger.info("Completion order (first 10): %s", completed_order[:10])


# ---------------------------------------------------------------------------
# 2. Thread-safe Queue – producer / consumer
# ---------------------------------------------------------------------------
def demo_thread_queue(logger: logging.Logger) -> None:
    logger.info("=== Thread-safe Queue (producer/consumer) ===")

    work_queue: queue.Queue[int | None] = queue.Queue(maxsize=10)
    results: list[int] = []
    lock = threading.Lock()
    num_items = 30

    def producer() -> None:
        for i in range(num_items):
            work_queue.put(i)
            logger.debug("Produced %d", i)
        work_queue.put(None)  # sentinel

    def consumer() -> None:
        while True:
            item = work_queue.get()
            if item is None:
                work_queue.task_done()
                break
            time.sleep(0.005)
            with lock:
                results.append(item * item)
            work_queue.task_done()

    prod = threading.Thread(target=producer, name="producer")
    cons = threading.Thread(target=consumer, name="consumer")

    t0 = time.perf_counter()
    prod.start()
    cons.start()
    prod.join()
    cons.join()

    logger.info(
        "Processed %d items in %.3f s  sum_of_squares=%d",
        len(results), time.perf_counter() - t0, sum(results),
    )


# ---------------------------------------------------------------------------
# 3. Lock – thread-safe counter
# ---------------------------------------------------------------------------
class ThreadSafeCounter:
    def __init__(self) -> None:
        self._value = 0
        self._lock = threading.Lock()

    def increment(self) -> None:
        with self._lock:
            self._value += 1

    def get(self) -> int:
        with self._lock:
            return self._value


def demo_lock(logger: logging.Logger) -> None:
    logger.info("=== Lock: thread-safe counter ===")

    counter = ThreadSafeCounter()
    num_threads = 50
    increments_per_thread = 200

    def worker() -> None:
        for _ in range(increments_per_thread):
            counter.increment()

    threads = [threading.Thread(target=worker) for _ in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    expected = num_threads * increments_per_thread
    actual = counter.get()
    logger.info(
        "Counter expected=%d  actual=%d  correct=%s",
        expected, actual, expected == actual,
    )


# ---------------------------------------------------------------------------
# 4. Semaphore – bounded concurrency
# ---------------------------------------------------------------------------
def demo_semaphore(logger: logging.Logger, cfg: dict) -> None:
    logger.info("=== Semaphore: bounded concurrency ===")

    limit: int = cfg["concurrency"]["semaphore_limit"]
    sem = threading.Semaphore(limit)
    active_count = 0
    max_active_seen = 0
    lock = threading.Lock()

    def bounded_task(task_id: int) -> None:
        nonlocal active_count, max_active_seen
        with sem:
            with lock:
                active_count += 1
                max_active_seen = max(max_active_seen, active_count)
            logger.debug("Task %d running (active=%d)", task_id, active_count)
            time.sleep(0.03)
            with lock:
                active_count -= 1

    threads = [threading.Thread(target=bounded_task, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    logger.info(
        "Semaphore limit=%d  max concurrent observed=%d",
        limit, max_active_seen,
    )
    assert max_active_seen <= limit, "Semaphore violated!"


# ---------------------------------------------------------------------------
# 5. Event – one-shot signal
# ---------------------------------------------------------------------------
def demo_event(logger: logging.Logger) -> None:
    logger.info("=== Event: one-shot signal ===")

    ready_event = threading.Event()
    results: list[str] = []

    def data_loader() -> None:
        logger.debug("Loader: fetching data ...")
        time.sleep(0.05)
        results.append("data loaded")
        ready_event.set()
        logger.debug("Loader: event set")

    def data_processor() -> None:
        logger.debug("Processor: waiting for data ...")
        ready_event.wait(timeout=5.0)
        results.append("data processed")
        logger.debug("Processor: processing complete")

    loader = threading.Thread(target=data_loader, name="loader")
    processor = threading.Thread(target=data_processor, name="processor")

    loader.start()
    processor.start()
    loader.join()
    processor.join()

    logger.info("Event demo results: %s", results)


# ---------------------------------------------------------------------------
# 6. Barrier – synchronise N threads at a checkpoint
# ---------------------------------------------------------------------------
def demo_barrier(logger: logging.Logger) -> None:
    logger.info("=== Barrier: phase synchronisation ===")

    num_workers = 4
    barrier = threading.Barrier(num_workers)
    phase_log: list[str] = []
    lock = threading.Lock()

    def phase_worker(worker_id: int) -> None:
        # Phase 1
        time.sleep(random.uniform(0.01, 0.05))
        with lock:
            phase_log.append(f"w{worker_id}:phase1")
        barrier.wait()  # all must reach here before any proceeds

        # Phase 2
        time.sleep(random.uniform(0.01, 0.03))
        with lock:
            phase_log.append(f"w{worker_id}:phase2")
        barrier.wait()

        with lock:
            phase_log.append(f"w{worker_id}:done")

    threads = [
        threading.Thread(target=phase_worker, args=(i,), name=f"phaser-{i}")
        for i in range(num_workers)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    logger.info("Barrier execution log: %s", phase_log)


# ---------------------------------------------------------------------------
# 7. Thread-local storage
# ---------------------------------------------------------------------------
_thread_local = threading.local()


def demo_thread_local(logger: logging.Logger) -> None:
    logger.info("=== Thread-local storage ===")

    def worker(worker_id: int) -> None:
        _thread_local.worker_id = worker_id
        _thread_local.request_count = 0
        for _ in range(3):
            _thread_local.request_count += 1
            time.sleep(0.005)
        logger.debug(
            "Worker %d: request_count=%d",
            _thread_local.worker_id,
            _thread_local.request_count,
        )

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    logger.info("Thread-local demo complete (each thread has isolated state)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    cfg = _load_config()
    logger = _setup_logging(cfg)

    logger.info("Starting threading_patterns  (config: %s)", _CONFIG_PATH)

    demo_thread_pool(logger, cfg)
    demo_thread_queue(logger)
    demo_lock(logger)
    demo_semaphore(logger, cfg)
    demo_event(logger)
    demo_barrier(logger)
    demo_thread_local(logger)

    logger.info("threading_patterns complete.")


if __name__ == "__main__":
    main()
