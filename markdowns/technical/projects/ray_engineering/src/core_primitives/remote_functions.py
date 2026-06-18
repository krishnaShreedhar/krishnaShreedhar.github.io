"""
Ray Core Primitives – Remote Functions (Tasks).

Demonstrates:
  - @ray.remote task declaration
  - ray.get()  – blocking result retrieval
  - ray.wait() – non-blocking partial result collection
  - Task chaining / DAG composition
  - Anti-pattern: calling ray.get() inside a tight loop (serialises work)
  - Correct pattern: collect all futures first, then ray.get() once

All constants are read from config.yaml.  No values are hardcoded.
"""

import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import ray

# ---------------------------------------------------------------------------
# Bootstrap: config + logging (before Ray is initialised)
# ---------------------------------------------------------------------------
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root src/

from utils.config_loader import load_config
from utils.logging_setup import get_logger

_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config.yaml"
config: dict[str, Any] = load_config(str(_CONFIG_PATH))
logger = get_logger(__name__, config)

_CORE_CFG = config["core_primitives"]
_RAY_CFG = config["ray"]["init"]

BATCH_SIZE: int = _CORE_CFG["batch_size"]
NUM_WORKERS: int = _CORE_CFG["num_workers"]
CHUNK_SIZE: int = _CORE_CFG["chunk_size"]


# ===========================================================================
# Ray remote task definitions
# ===========================================================================

@ray.remote
def compute_square(x: float) -> float:
    """Return x² – a trivial stateless remote task."""
    return x * x


@ray.remote
def compute_stats(chunk: list[float]) -> dict[str, float]:
    """Compute descriptive statistics for a list of numbers.

    This task is intentionally compute-heavy enough to be worth distributing.
    """
    arr = np.array(chunk, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "sum": float(arr.sum()),
        "count": int(len(arr)),
    }


@ray.remote
def aggregate_stats(stats_list: list[dict[str, float]]) -> dict[str, float]:
    """Reduce partial statistics from multiple workers into a global result.

    Demonstrates task chaining: this task depends on the outputs of
    ``compute_stats`` tasks.
    """
    total_count = sum(s["count"] for s in stats_list)
    total_sum = sum(s["sum"] for s in stats_list)
    global_mean = total_sum / total_count

    # Weighted variance aggregation (parallel / Chan's algorithm)
    combined_variance = 0.0
    for s in stats_list:
        n = s["count"]
        delta = s["mean"] - global_mean
        combined_variance += n * (s["std"] ** 2 + delta ** 2)
    global_std = math.sqrt(combined_variance / total_count)

    return {
        "global_mean": global_mean,
        "global_std": global_std,
        "global_min": min(s["min"] for s in stats_list),
        "global_max": max(s["max"] for s in stats_list),
        "total_count": total_count,
    }


@ray.remote(num_cpus=1)
def simulate_io_task(task_id: int, sleep_secs: float) -> dict[str, Any]:
    """Simulate an I/O-bound task (e.g., fetching data from a remote store)."""
    time.sleep(sleep_secs)
    return {"task_id": task_id, "elapsed_secs": sleep_secs, "status": "done"}


# ===========================================================================
# Demonstration runners
# ===========================================================================

def demo_basic_tasks() -> None:
    """Submit a batch of square-computation tasks and retrieve results."""
    logger.info(
        "Starting basic remote task demo",
        extra={"batch_size": BATCH_SIZE},
    )

    data = list(range(BATCH_SIZE))

    # Good pattern: submit all futures first
    futures = [compute_square.remote(x) for x in data]
    logger.debug("All futures submitted", extra={"num_futures": len(futures)})

    results = ray.get(futures)  # single blocking call
    logger.info(
        "Basic task results sample",
        extra={
            "first_five": results[:5],
            "last_five": results[-5:],
            "total_tasks": len(results),
        },
    )


def demo_anti_pattern() -> None:
    """
    ANTI-PATTERN: calling ray.get() inside the submission loop.

    This serialises execution because each ray.get() blocks until one
    future is done before the next task is even submitted.  The correct
    pattern is shown in demo_basic_tasks().
    """
    logger.warning(
        "Running anti-pattern demo: ray.get() inside loop – this is SLOW",
    )
    data = list(range(20))  # small to keep the demo fast
    results = []
    for x in data:
        # BAD: blocks here, no parallelism
        result = ray.get(compute_square.remote(x))
        results.append(result)

    logger.warning(
        "Anti-pattern finished (serially). Compare wall-time with correct pattern.",
        extra={"num_results": len(results)},
    )


def demo_ray_wait() -> None:
    """
    Use ray.wait() to process tasks as they complete (streaming pattern).

    Useful when tasks have heterogeneous runtimes and you want to act on
    early finishers without waiting for the slowest task.
    """
    import random

    logger.info("Starting ray.wait() streaming demo")
    sleep_times = [round(random.uniform(0.1, 0.5), 2) for _ in range(8)]
    futures = [simulate_io_task.remote(i, t) for i, t in enumerate(sleep_times)]

    remaining = futures.copy()
    completed_results = []

    while remaining:
        # Wait for at least one future to be ready (timeout=1 s)
        ready, remaining = ray.wait(remaining, num_returns=1, timeout=1.0)
        if ready:
            result = ray.get(ready[0])
            completed_results.append(result)
            logger.debug(
                "Task completed via ray.wait()",
                extra={"result": result, "still_pending": len(remaining)},
            )

    logger.info(
        "All tasks collected via streaming ray.wait()",
        extra={"total_tasks": len(completed_results)},
    )


def demo_task_dag() -> None:
    """
    Task DAG: split data into chunks → parallel stats → single aggregation.

    Shows how Ray can express a MapReduce-style computation using only
    plain Python function composition.
    """
    logger.info(
        "Starting task DAG demo (MapReduce-style stats)",
        extra={"batch_size": BATCH_SIZE, "chunk_size": CHUNK_SIZE},
    )

    rng = np.random.default_rng(seed=42)
    data = rng.normal(loc=5.0, scale=2.0, size=BATCH_SIZE).tolist()

    # Map: split into chunks, submit in parallel
    chunks = [
        data[i : i + CHUNK_SIZE] for i in range(0, len(data), CHUNK_SIZE)
    ]
    logger.debug("Data chunked", extra={"num_chunks": len(chunks)})

    partial_futures = [compute_stats.remote(chunk) for chunk in chunks]

    # Reduce: pass futures directly as arguments – Ray resolves them lazily
    agg_future = aggregate_stats.remote(ray.get(partial_futures))
    global_result = ray.get(agg_future)

    logger.info(
        "Task DAG aggregation complete",
        extra={"global_stats": global_result},
    )


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    logger.info("Initialising Ray", extra={"ray_config": _RAY_CFG})
    ray.init(
        num_cpus=_RAY_CFG["num_cpus"],
        num_gpus=_RAY_CFG["num_gpus"],
        object_store_memory=_RAY_CFG["object_store_memory"],
        ignore_reinit_error=True,
    )
    logger.info(
        "Ray initialised",
        extra={"resources": ray.cluster_resources()},
    )

    try:
        demo_basic_tasks()
        demo_anti_pattern()
        demo_ray_wait()
        demo_task_dag()
    finally:
        logger.info("Shutting down Ray")
        ray.shutdown()


if __name__ == "__main__":
    main()
