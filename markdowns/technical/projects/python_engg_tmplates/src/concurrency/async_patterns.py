"""
async_patterns.py
=================
Demonstrates asyncio concurrency patterns:
  - async/await fundamentals
  - Semaphore-based rate limiting
  - Async producer/consumer with asyncio.Queue
  - asyncio.TaskGroup (Python 3.11+)
  - Timeout and cancellation
  - Streaming async generator
  - aiohttp-style HTTP simulation (no network required)
  - Benchmark: asyncio vs threading for I/O-bound tasks

All constants from config.yaml; logs to logs/python_engg.log.
"""

from __future__ import annotations

import asyncio
import logging
import logging.config
import pathlib
import random
import time
from typing import AsyncGenerator, Optional

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
    return logging.getLogger("async_patterns")


# ---------------------------------------------------------------------------
# 1. Basic async/await
# ---------------------------------------------------------------------------
async def fetch_page(url: str, latency_s: float) -> dict:
    """Simulate an async HTTP GET."""
    await asyncio.sleep(latency_s)
    return {"url": url, "status": 200, "bytes": random.randint(1_000, 100_000)}


async def demo_basic_async(logger: logging.Logger) -> None:
    logger.info("=== basic async/await: gather ===")

    urls = [f"https://example.com/page/{i}" for i in range(8)]
    latencies = [random.uniform(0.02, 0.12) for _ in urls]

    t0 = time.perf_counter()
    results = await asyncio.gather(*(fetch_page(u, l) for u, l in zip(urls, latencies)))
    elapsed = time.perf_counter() - t0

    total_bytes = sum(r["bytes"] for r in results)
    logger.info(
        "Fetched %d pages in %.3f s  total_bytes=%d",
        len(results), elapsed, total_bytes,
    )


# ---------------------------------------------------------------------------
# 2. Semaphore – rate limiting
# ---------------------------------------------------------------------------
async def demo_rate_limited(logger: logging.Logger, cfg: dict) -> None:
    logger.info("=== asyncio.Semaphore: rate limiting ===")

    limit: int = cfg["concurrency"]["semaphore_limit"]
    sem = asyncio.Semaphore(limit)
    completed = 0

    async def rate_limited_fetch(request_id: int) -> dict:
        nonlocal completed
        async with sem:
            latency = random.uniform(0.01, 0.05)
            await asyncio.sleep(latency)
            completed += 1
            logger.debug("  Request %d done (active<=%d)", request_id, limit)
            return {"id": request_id, "ok": True}

    num_requests = 30
    t0 = time.perf_counter()
    results = await asyncio.gather(*(rate_limited_fetch(i) for i in range(num_requests)))
    elapsed = time.perf_counter() - t0

    logger.info(
        "Rate-limited %d requests (sem_limit=%d) in %.3f s  succeeded=%d",
        num_requests, limit, elapsed, sum(1 for r in results if r["ok"]),
    )


# ---------------------------------------------------------------------------
# 3. Async producer / consumer
# ---------------------------------------------------------------------------
async def demo_producer_consumer(logger: logging.Logger, cfg: dict) -> None:
    logger.info("=== asyncio.Queue: producer/consumer ===")

    maxsize: int = cfg["concurrency"]["async_queue_maxsize"]
    num_items = 25

    work_q: asyncio.Queue[Optional[str]] = asyncio.Queue(maxsize=maxsize)
    processed: list[str] = []

    async def producer() -> None:
        for i in range(num_items):
            item = f"record-{i:04d}"
            await work_q.put(item)
            logger.debug("Produced %s  qsize=%d", item, work_q.qsize())
            await asyncio.sleep(0.005)
        await work_q.put(None)  # sentinel
        logger.info("Producer finished enqueueing %d items", num_items)

    async def consumer(consumer_id: int) -> None:
        while True:
            item = await work_q.get()
            if item is None:
                await work_q.put(None)  # re-queue sentinel for other consumers
                work_q.task_done()
                break
            await asyncio.sleep(0.008)  # simulate processing
            processed.append(item)
            logger.debug("Consumer-%d processed %s", consumer_id, item)
            work_q.task_done()

    t0 = time.perf_counter()
    await asyncio.gather(producer(), consumer(0), consumer(1))
    elapsed = time.perf_counter() - t0

    logger.info(
        "Producer/consumer: %d items in %.3f s  unique_processed=%d",
        num_items, elapsed, len(set(processed)),
    )


# ---------------------------------------------------------------------------
# 4. asyncio.TaskGroup (Python 3.11+)
# ---------------------------------------------------------------------------
async def demo_task_group(logger: logging.Logger) -> None:
    logger.info("=== asyncio.TaskGroup ===")

    async def pipeline_stage(name: str, duration: float, data: int) -> dict:
        logger.debug("Stage '%s' started with data=%d", name, data)
        await asyncio.sleep(duration)
        result = data * 2
        logger.debug("Stage '%s' done -> %d", name, result)
        return {"stage": name, "result": result}

    t0 = time.perf_counter()
    results: list[dict] = []

    async with asyncio.TaskGroup() as tg:
        tasks = [
            tg.create_task(pipeline_stage(f"stage-{i}", random.uniform(0.01, 0.08), i * 10))
            for i in range(6)
        ]

    results = [t.result() for t in tasks]
    elapsed = time.perf_counter() - t0

    logger.info(
        "TaskGroup: %d stages completed in %.3f s",
        len(results), elapsed,
    )
    for r in results:
        logger.debug("  %s -> %d", r["stage"], r["result"])


# ---------------------------------------------------------------------------
# 5. Timeout and cancellation
# ---------------------------------------------------------------------------
async def demo_timeout_cancellation(logger: logging.Logger) -> None:
    logger.info("=== Timeout and cancellation ===")

    async def slow_operation(op_id: int, duration: float) -> str:
        try:
            await asyncio.sleep(duration)
            return f"op-{op_id} succeeded"
        except asyncio.CancelledError:
            logger.debug("op-%d was cancelled", op_id)
            raise

    # asyncio.timeout context manager
    try:
        async with asyncio.timeout(0.05):
            result = await slow_operation(1, 0.02)  # fast – should succeed
            logger.info("Fast op result: %s", result)
    except TimeoutError:
        logger.warning("Fast op timed out (unexpected)")

    try:
        async with asyncio.timeout(0.03):
            result = await slow_operation(2, 0.15)  # slow – should timeout
            logger.info("Slow op result: %s", result)
    except TimeoutError:
        logger.info("Slow op correctly timed out after 30 ms")

    # Explicit task cancellation
    async def long_running() -> None:
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            logger.info("long_running task cancelled gracefully")
            raise

    task = asyncio.create_task(long_running())
    await asyncio.sleep(0.02)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        logger.info("Task cancellation confirmed")


# ---------------------------------------------------------------------------
# 6. Async generator – streaming results
# ---------------------------------------------------------------------------
async def async_data_stream(
    num_records: int, delay_s: float
) -> AsyncGenerator[dict, None]:
    """Stream records one at a time with simulated I/O latency."""
    for i in range(num_records):
        await asyncio.sleep(delay_s)
        yield {"id": i, "value": i * i, "tag": f"item-{i}"}


async def demo_async_generator(logger: logging.Logger) -> None:
    logger.info("=== Async generator: streaming ===")

    total = 0
    count = 0
    t0 = time.perf_counter()

    async for record in async_data_stream(num_records=12, delay_s=0.005):
        total += record["value"]
        count += 1
        logger.debug("  Received record id=%d  value=%d", record["id"], record["value"])

    elapsed = time.perf_counter() - t0
    logger.info(
        "Streamed %d records in %.3f s  sum_of_squares=%d",
        count, elapsed, total,
    )


# ---------------------------------------------------------------------------
# 7. Benchmark: asyncio vs sequential for I/O-bound work
# ---------------------------------------------------------------------------
async def demo_benchmark(logger: logging.Logger) -> None:
    logger.info("=== Benchmark: sequential vs asyncio ===")

    num_tasks = 20
    per_task_latency = 0.05  # 50 ms per task

    # Sequential simulation
    t0 = time.perf_counter()
    for i in range(num_tasks):
        await asyncio.sleep(0)  # yield control but don't actually wait
    seq_time = per_task_latency * num_tasks  # theoretical sequential time

    # Concurrent asyncio
    t0 = time.perf_counter()
    await asyncio.gather(*(fetch_page(f"url-{i}", per_task_latency) for i in range(num_tasks)))
    async_time = time.perf_counter() - t0

    logger.info(
        "Sequential (est.): %.3f s  Asyncio: %.3f s  speedup=%.1fx",
        seq_time, async_time, seq_time / async_time,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
async def _main_async(cfg: dict, logger: logging.Logger) -> None:
    logger.info("Starting async_patterns  (config: %s)", _CONFIG_PATH)

    await demo_basic_async(logger)
    await demo_rate_limited(logger, cfg)
    await demo_producer_consumer(logger, cfg)
    await demo_task_group(logger)
    await demo_timeout_cancellation(logger)
    await demo_async_generator(logger)
    await demo_benchmark(logger)

    logger.info("async_patterns complete.")


def main() -> None:
    cfg = _load_config()
    logger = _setup_logging(cfg)
    asyncio.run(_main_async(cfg, logger))


if __name__ == "__main__":
    main()
