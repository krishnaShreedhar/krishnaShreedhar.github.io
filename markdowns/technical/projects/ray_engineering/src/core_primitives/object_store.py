"""
Ray Core Primitives – Object Store.

Demonstrates:
  - ray.put() to place large objects in the shared plasma object store
  - Passing object references instead of raw data (zero-copy reads)
  - Reference counting and lifetime management
  - Anti-pattern: serialising large objects through task arguments without
    ray.put() (causes repeated serialisation over the network)

All constants are read from config.yaml.  No values are hardcoded.
"""

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import ray

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.config_loader import load_config
from utils.logging_setup import get_logger

_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config.yaml"
config: dict[str, Any] = load_config(str(_CONFIG_PATH))
logger = get_logger(__name__, config)

_CORE_CFG = config["core_primitives"]
_RAY_CFG = config["ray"]["init"]

NUM_WORKERS: int = _CORE_CFG["num_workers"]
BATCH_SIZE: int = _CORE_CFG["batch_size"]


# ===========================================================================
# Remote tasks that operate on object references
# ===========================================================================

@ray.remote
def compute_dot_product(
    matrix_ref: ray.ObjectRef, vector_ref: ray.ObjectRef
) -> float:
    """
    Compute sum(matrix @ vector) using object references.

    Ray deserialises the objects in the worker's memory space using a
    zero-copy shared-memory path (for NumPy arrays backed by plasma).
    The objects are NOT re-serialised when passed as ObjectRef args.
    """
    matrix: np.ndarray = matrix_ref  # automatically dereferenced by Ray
    vector: np.ndarray = vector_ref
    result = float(np.sum(matrix @ vector))
    return result


@ray.remote
def compute_row_norms(matrix_ref: ray.ObjectRef) -> np.ndarray:
    """Compute L2 norm of each row in a matrix stored in the object store."""
    matrix: np.ndarray = matrix_ref
    return np.linalg.norm(matrix, axis=1)


@ray.remote
def slow_task_with_shared_data(
    data_ref: ray.ObjectRef, task_id: int
) -> dict[str, Any]:
    """
    Simulate a worker that reads a large shared dataset from the object store.

    All workers share the same plasma buffer – the data is NOT copied per
    worker.  This is the key advantage of ray.put().
    """
    data: np.ndarray = data_ref
    # Simulate processing
    partial_sum = float(data[task_id * 10 : task_id * 10 + 10].sum())
    return {
        "task_id": task_id,
        "partial_sum": partial_sum,
        "data_shape": list(data.shape),
    }


# ===========================================================================
# Anti-pattern task (do NOT do this for large data)
# ===========================================================================

@ray.remote
def anti_pattern_task(data: np.ndarray, task_id: int) -> float:
    """
    ANTI-PATTERN: accepts the raw numpy array as an argument.

    When called by N tasks, the array is serialised N times through the
    Ray object store rather than being stored once and referenced N times.
    For large arrays this is both slow and memory-wasteful.
    """
    return float(data.sum())


# ===========================================================================
# Demonstrations
# ===========================================================================

def demo_ray_put_basic() -> None:
    """Show how ray.put() stores an object and returns a reference."""
    logger.info("Starting ray.put() basic demo")

    data = np.arange(1000, dtype=np.float64)
    ref = ray.put(data)

    logger.info(
        "Object placed in object store",
        extra={"type": type(ref).__name__, "ref": str(ref)},
    )

    # Retrieve via ray.get()
    retrieved = ray.get(ref)
    logger.debug(
        "Object retrieved from store",
        extra={
            "shape": list(retrieved.shape),
            "sum": float(retrieved.sum()),
            "allclose": bool(np.allclose(data, retrieved)),
        },
    )
    logger.info("ray.put() basic demo complete")


def demo_shared_large_object() -> None:
    """
    Correct pattern: store a large array once, fan out references.

    All NUM_WORKERS tasks read from the SAME plasma buffer without copying.
    """
    logger.info(
        "Starting shared large object demo",
        extra={"num_workers": NUM_WORKERS, "array_size": BATCH_SIZE},
    )

    rng = np.random.default_rng(seed=0)
    large_data = rng.uniform(size=BATCH_SIZE).astype(np.float32)

    # Put once → reference reused across all tasks
    data_ref = ray.put(large_data)
    logger.debug(
        "Large array placed in object store",
        extra={"nbytes": large_data.nbytes, "ref": str(data_ref)},
    )

    futures = [
        slow_task_with_shared_data.remote(data_ref, task_id)
        for task_id in range(NUM_WORKERS)
    ]
    results = ray.get(futures)

    logger.info(
        "Shared object fan-out complete",
        extra={"results": results},
    )


def demo_matrix_ops_via_refs() -> None:
    """
    Zero-copy matrix operations using object store references.

    Demonstrates passing ObjectRef as task arguments for composed
    compute kernels (dot product + row norms) on the same data.
    """
    logger.info("Starting matrix ops via object references demo")

    rng = np.random.default_rng(seed=7)
    rows, cols = 50, 50
    matrix = rng.standard_normal((rows, cols)).astype(np.float32)
    vector = rng.standard_normal(cols).astype(np.float32)

    matrix_ref = ray.put(matrix)
    vector_ref = ray.put(vector)

    logger.debug(
        "Matrix and vector placed in object store",
        extra={"matrix_shape": [rows, cols], "vector_shape": [cols]},
    )

    dot_future = compute_dot_product.remote(matrix_ref, vector_ref)
    norms_future = compute_row_norms.remote(matrix_ref)

    dot_result = ray.get(dot_future)
    norms = ray.get(norms_future)

    logger.info(
        "Matrix ops complete",
        extra={
            "dot_product_sum": round(dot_result, 6),
            "row_norms_mean": round(float(norms.mean()), 6),
            "row_norms_std": round(float(norms.std()), 6),
        },
    )


def demo_anti_pattern_vs_correct() -> None:
    """
    Compare the anti-pattern (pass raw data) vs. correct pattern (ray.put).

    Measures and logs wall-time for both approaches on the same workload.
    """
    logger.warning(
        "Starting anti-pattern vs. correct pattern comparison"
    )

    rng = np.random.default_rng(seed=1)
    large_array = rng.uniform(size=BATCH_SIZE * 10).astype(np.float64)
    n_tasks = NUM_WORKERS

    # ---- ANTI-PATTERN ----
    logger.warning(
        "Anti-pattern: passing raw array to each task (causes N serialisations)"
    )
    t0 = time.perf_counter()
    bad_futures = [anti_pattern_task.remote(large_array, i) for i in range(n_tasks)]
    ray.get(bad_futures)
    anti_elapsed = time.perf_counter() - t0
    logger.warning(
        "Anti-pattern elapsed",
        extra={"elapsed_s": round(anti_elapsed, 4)},
    )

    # ---- CORRECT PATTERN ----
    logger.info(
        "Correct pattern: ray.put() once, pass reference to each task"
    )
    t0 = time.perf_counter()
    data_ref = ray.put(large_array)
    good_futures = [
        slow_task_with_shared_data.remote(data_ref, i) for i in range(n_tasks)
    ]
    ray.get(good_futures)
    correct_elapsed = time.perf_counter() - t0
    logger.info(
        "Correct pattern elapsed",
        extra={"elapsed_s": round(correct_elapsed, 4)},
    )

    speedup = anti_elapsed / correct_elapsed if correct_elapsed > 0 else float("inf")
    logger.info(
        "Speedup of correct over anti-pattern",
        extra={"speedup_x": round(speedup, 2)},
    )


def demo_object_lifecycle() -> None:
    """
    Demonstrate reference counting in the object store.

    When no Python ObjectRef points to a stored object, Ray garbage-collects
    it from the plasma store.  We show this by del-ing the reference.
    """
    logger.info("Starting object lifecycle demo")

    data = np.ones(100, dtype=np.float32)
    ref = ray.put(data)
    logger.debug("Object stored", extra={"ref": str(ref)})

    # Can still retrieve
    retrieved = ray.get(ref)
    logger.info(
        "Object retrieved before deletion",
        extra={"sum": float(retrieved.sum())},
    )

    # Delete the Python-side reference
    del ref
    logger.info(
        "ObjectRef deleted from Python scope. "
        "Ray will evict the plasma object when memory pressure demands it."
    )


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    logger.info("Initialising Ray for object store demo", extra={"ray_config": _RAY_CFG})
    ray.init(
        num_cpus=_RAY_CFG["num_cpus"],
        num_gpus=_RAY_CFG["num_gpus"],
        object_store_memory=_RAY_CFG["object_store_memory"],
        ignore_reinit_error=True,
    )
    logger.info("Ray initialised", extra={"resources": ray.cluster_resources()})

    try:
        demo_ray_put_basic()
        demo_shared_large_object()
        demo_matrix_ops_via_refs()
        demo_anti_pattern_vs_correct()
        demo_object_lifecycle()
    finally:
        logger.info("Shutting down Ray")
        ray.shutdown()


if __name__ == "__main__":
    main()
