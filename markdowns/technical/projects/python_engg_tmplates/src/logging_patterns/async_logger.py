"""
async_logger.py
===============
Demonstrates non-blocking logging using `QueueHandler` + `QueueListener`:

  - QueueHandler      : worker threads / async tasks push records onto a queue
                        and return immediately (no I/O on the hot path)
  - QueueListener     : dedicated background thread drains the queue and
                        dispatches to real handlers (file, console)
  - AsyncLoggerBridge : thin wrapper so asyncio coroutines can call
                        logger.info() without blocking the event loop

Architecture::

    [app thread / coroutine]
          |  (non-blocking put)
          v
    [ logging.Queue ]
          |  (background thread)
          v
    [ QueueListener ]
          |
    [ RotatingFileHandler + StreamHandler ]

All configuration from config.yaml.
"""

from __future__ import annotations

import asyncio
import logging
import logging.handlers
import pathlib
import queue
import time
from typing import Optional

import yaml

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
_CONFIG_PATH = _PROJECT_ROOT / "config.yaml"


# ---------------------------------------------------------------------------
# Build the async logging infrastructure
# ---------------------------------------------------------------------------
class AsyncLoggingSetup:
    """Encapsulates QueueHandler + QueueListener lifecycle.

    Usage::
        setup = AsyncLoggingSetup(cfg)
        setup.start()
        logger = setup.get_logger("my.module")
        ...
        setup.stop()
    """

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._log_cfg = cfg["logging"]
        self._log_file = _PROJECT_ROOT / self._log_cfg["log_file"]
        self._log_file.parent.mkdir(parents=True, exist_ok=True)

        # Queue(-1) = unbounded; mirrors config queue_size setting
        queue_size: int = self._log_cfg["queue_size"]
        self._queue: queue.Queue = queue.Queue(maxsize=max(queue_size, 0) if queue_size > 0 else 0)

        self._listener: Optional[logging.handlers.QueueListener] = None
        self._level = getattr(logging, self._log_cfg["level"].upper())

    def _build_real_handlers(self) -> list[logging.Handler]:
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        )
        file_handler = logging.handlers.RotatingFileHandler(
            filename=str(self._log_file),
            maxBytes=self._log_cfg["max_bytes"],
            backupCount=self._log_cfg["backup_count"],
            encoding="utf-8",
        )
        file_handler.setFormatter(fmt)
        file_handler.setLevel(logging.DEBUG)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(fmt)
        console_handler.setLevel(self._level)

        return [file_handler, console_handler]

    def start(self) -> None:
        """Install QueueHandler on root logger and start QueueListener."""
        real_handlers = self._build_real_handlers()
        self._listener = logging.handlers.QueueListener(
            self._queue,
            *real_handlers,
            respect_handler_level=True,
        )
        self._listener.start()

        queue_handler = logging.handlers.QueueHandler(self._queue)
        root = logging.getLogger()
        root.setLevel(logging.DEBUG)
        root.handlers.clear()
        root.addHandler(queue_handler)

    def stop(self) -> None:
        """Flush and stop the background listener thread."""
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    def get_logger(self, name: str) -> logging.Logger:
        return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Async producer/consumer simulation
# ---------------------------------------------------------------------------
async def simulate_web_request(
    logger: logging.Logger,
    request_id: str,
    latency_ms: float,
) -> dict:
    """Simulate an async I/O operation that logs without blocking."""
    logger.info("Request %s started", request_id)
    await asyncio.sleep(latency_ms / 1000)
    logger.info("Request %s completed in %.0f ms", request_id, latency_ms)
    return {"request_id": request_id, "latency_ms": latency_ms, "status": "ok"}


async def async_batch_processor(
    logger: logging.Logger,
    items: list[str],
    concurrency: int,
) -> list[dict]:
    """Process items concurrently with a semaphore limit."""
    sem = asyncio.Semaphore(concurrency)
    results: list[dict] = []

    async def process_one(item: str, idx: int) -> dict:
        async with sem:
            logger.debug("Processing item[%d]=%s", idx, item)
            await asyncio.sleep(0.01)  # simulate async I/O
            return {"item": item, "idx": idx, "ok": True}

    tasks = [process_one(item, i) for i, item in enumerate(items)]
    results = await asyncio.gather(*tasks)
    return list(results)


async def async_producer_consumer(
    logger: logging.Logger,
    queue_maxsize: int,
    num_items: int,
) -> None:
    """Classic async producer/consumer with asyncio.Queue."""
    work_queue: asyncio.Queue[Optional[str]] = asyncio.Queue(maxsize=queue_maxsize)

    async def producer() -> None:
        for i in range(num_items):
            item = f"item-{i:03d}"
            await work_queue.put(item)
            logger.debug("Produced %s", item)
        # Sentinel to signal consumer to stop
        await work_queue.put(None)
        logger.info("Producer done")

    async def consumer() -> None:
        consumed = 0
        while True:
            item = await work_queue.get()
            if item is None:
                work_queue.task_done()
                break
            logger.debug("Consumed %s", item)
            consumed += 1
            await asyncio.sleep(0.002)  # simulate processing
            work_queue.task_done()
        logger.info("Consumer done. Consumed %d items", consumed)

    await asyncio.gather(producer(), consumer())


async def main_async(cfg: dict) -> None:
    setup = AsyncLoggingSetup(cfg)
    setup.start()

    logger = setup.get_logger("async_logger")
    logger.info("=== async_logger demo start ===")

    # 1. Concurrent web requests
    logger.info("--- Concurrent web request simulation ---")
    import random
    requests = [
        simulate_web_request(logger, f"req-{i:03d}", random.uniform(10, 100))
        for i in range(8)
    ]
    results = await asyncio.gather(*requests)
    logger.info("All requests done. Count=%d", len(results))

    # 2. Batch processor with semaphore
    logger.info("--- Semaphore-limited batch processor ---")
    items = [f"doc-{i}" for i in range(20)]
    concurrency: int = cfg["concurrency"]["semaphore_limit"]
    processed = await async_batch_processor(logger, items, concurrency)
    logger.info("Processed %d items", len(processed))

    # 3. Producer / consumer
    logger.info("--- async producer/consumer ---")
    await async_producer_consumer(
        logger,
        queue_maxsize=cfg["concurrency"]["async_queue_maxsize"],
        num_items=15,
    )

    logger.info("=== async_logger demo complete ===")

    # Give the queue listener time to flush before stopping
    await asyncio.sleep(0.1)
    setup.stop()


def main() -> None:
    with open(_CONFIG_PATH) as fh:
        cfg = yaml.safe_load(fh)

    asyncio.run(main_async(cfg))


if __name__ == "__main__":
    main()
