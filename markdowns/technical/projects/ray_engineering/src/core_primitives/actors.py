"""
Ray Core Primitives – Actors.

Demonstrates:
  - Stateful @ray.remote class (actor)
  - ParameterServer pattern for distributed gradient averaging
  - ray.util.ActorPool for load-balanced worker pools
  - Anti-pattern: calling actor methods synchronously inside a loop

All constants are read from config.yaml.  No values are hardcoded.
"""

import time
from pathlib import Path
from typing import Any

import numpy as np
import ray
from ray.util import ActorPool

import sys

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
# Actor definitions
# ===========================================================================

@ray.remote
class ParameterServer:
    """
    Central parameter store for distributed training.

    Workers push gradients; the server averages them and returns updated
    parameters.  This is the classic PS-Worker pattern used in early
    distributed deep-learning frameworks.
    """

    def __init__(self, num_params: int, learning_rate: float) -> None:
        self._params = np.zeros(num_params, dtype=np.float32)
        self._lr = learning_rate
        self._update_count = 0
        # Actor-local logger (Ray worker process)
        import logging
        self._log = logging.getLogger(f"ParameterServer")

    def get_params(self) -> np.ndarray:
        """Return the current parameter vector."""
        return self._params.copy()

    def apply_gradients(self, *gradients: np.ndarray) -> np.ndarray:
        """
        Average the incoming gradients and apply an SGD step.

        Parameters
        ----------
        *gradients:
            One gradient array per worker.  All arrays must have the same
            shape as the parameter vector.

        Returns
        -------
        np.ndarray
            Updated parameter vector after the SGD step.
        """
        avg_grad = np.mean(np.stack(gradients), axis=0)
        self._params -= self._lr * avg_grad
        self._update_count += 1
        self._log.debug(
            "Gradient applied | update=%d | grad_norm=%.4f",
            self._update_count,
            float(np.linalg.norm(avg_grad)),
        )
        return self._params.copy()

    def get_update_count(self) -> int:
        """Return total number of gradient updates applied."""
        return self._update_count

    def reset(self) -> None:
        """Reset parameters to zero."""
        self._params[:] = 0.0
        self._update_count = 0


@ray.remote
class Worker:
    """
    Simulated training worker.

    Pulls parameters from the ParameterServer, computes a synthetic
    gradient (random noise decaying with iteration), and pushes it back.
    """

    def __init__(self, worker_id: int, num_params: int) -> None:
        self._id = worker_id
        self._num_params = num_params
        self._rng = np.random.default_rng(seed=worker_id)

    def compute_gradient(
        self, params: np.ndarray, iteration: int
    ) -> np.ndarray:
        """
        Compute a synthetic gradient.

        In a real trainer this would perform a forward + backward pass.
        Here we return scaled random noise that decays over iterations.
        """
        scale = 1.0 / (1.0 + iteration * 0.1)
        gradient = self._rng.normal(loc=0.0, scale=scale, size=self._num_params).astype(
            np.float32
        )
        return gradient

    def get_id(self) -> int:
        return self._id


@ray.remote
class StatefulCounter:
    """
    Simple stateful counter – the canonical actor tutorial example.

    Shows that actor state persists across method calls, which plain remote
    functions cannot achieve.
    """

    def __init__(self) -> None:
        self._count = 0

    def increment(self) -> int:
        self._count += 1
        return self._count

    def decrement(self) -> int:
        self._count -= 1
        return self._count

    def value(self) -> int:
        return self._count

    def reset(self) -> None:
        self._count = 0


@ray.remote
class DataProcessor:
    """Worker used by ActorPool to process data items."""

    def __init__(self, processor_id: int) -> None:
        self._id = processor_id
        self._processed = 0

    def process(self, item: float) -> dict[str, Any]:
        """Square the item and track state."""
        result = item ** 2
        self._processed += 1
        return {
            "processor_id": self._id,
            "input": item,
            "output": result,
            "total_processed": self._processed,
        }

    def total_processed(self) -> int:
        return self._processed


# ===========================================================================
# Demonstration runners
# ===========================================================================

def demo_stateful_counter() -> None:
    """Show that actor state is preserved across remote calls."""
    logger.info("Starting stateful counter demo")

    counter = StatefulCounter.remote()

    futures = [counter.increment.remote() for _ in range(10)]
    values = ray.get(futures)
    logger.info("Counter after 10 increments", extra={"values": values})

    current = ray.get(counter.value.remote())
    logger.info("Final counter value", extra={"value": current})

    ray.get(counter.decrement.remote())
    ray.get(counter.decrement.remote())
    after_dec = ray.get(counter.value.remote())
    logger.info("Counter after 2 decrements", extra={"value": after_dec})


def demo_parameter_server(num_iterations: int = 5, num_params: int = 64) -> None:
    """
    ParameterServer + Worker actor pattern.

    Shows the standard distributed training loop:
      for each iteration:
        1. pull params from PS
        2. workers compute gradients in parallel
        3. PS averages & applies them
    """
    learning_rate = config["distributed_training"]["learning_rate"]

    logger.info(
        "Starting ParameterServer demo",
        extra={
            "num_workers": NUM_WORKERS,
            "num_params": num_params,
            "num_iterations": num_iterations,
            "learning_rate": learning_rate,
        },
    )

    ps = ParameterServer.remote(num_params=num_params, learning_rate=learning_rate)
    workers = [Worker.remote(worker_id=i, num_params=num_params) for i in range(NUM_WORKERS)]

    for iteration in range(num_iterations):
        # 1. Pull current parameters (single call, fast)
        params = ray.get(ps.get_params.remote())

        # 2. All workers compute gradients in parallel
        grad_futures = [
            w.compute_gradient.remote(params, iteration) for w in workers
        ]
        gradients = ray.get(grad_futures)

        # 3. PS applies the averaged gradient
        new_params = ray.get(ps.apply_gradients.remote(*gradients))

        param_norm = float(np.linalg.norm(new_params))
        logger.info(
            "PS training iteration complete",
            extra={
                "iteration": iteration + 1,
                "param_norm": round(param_norm, 6),
                "total_updates": ray.get(ps.get_update_count.remote()),
            },
        )

    logger.info("ParameterServer demo complete")


def demo_actor_pool() -> None:
    """
    Use ray.util.ActorPool for load-balanced task dispatching.

    ActorPool maintains a set of actors and submits work to whichever
    actor is currently idle – similar to a thread pool executor.
    """
    logger.info(
        "Starting ActorPool demo",
        extra={"pool_size": NUM_WORKERS, "num_items": BATCH_SIZE},
    )

    processors = [DataProcessor.remote(i) for i in range(NUM_WORKERS)]
    pool = ActorPool(processors)

    data = list(range(BATCH_SIZE))

    # Submit all items; pool round-robins across actors
    results = list(
        pool.map(lambda actor, item: actor.process.remote(item), data)
    )

    logger.info(
        "ActorPool processing complete",
        extra={
            "num_results": len(results),
            "sample_results": results[:3],
        },
    )

    # Verify stateful tracking across actor calls
    total_processed = sum(
        ray.get(p.total_processed.remote()) for p in processors
    )
    logger.info(
        "ActorPool total processed (sum across actors)",
        extra={"total_processed": total_processed},
    )


def demo_anti_pattern_actor() -> None:
    """
    ANTI-PATTERN: calling ray.get() after every single actor method.

    Each ray.get() introduces a round-trip to the actor's remote process.
    When you need to sequence N operations, the overhead compounds.
    Correct pattern: pipeline futures or batch calls where possible.
    """
    logger.warning("Running actor anti-pattern demo")

    counter = StatefulCounter.remote()
    n = 20
    for _ in range(n):
        # BAD: blocks after every single increment
        ray.get(counter.increment.remote())

    final = ray.get(counter.value.remote())
    logger.warning(
        "Anti-pattern counter done (serially). Prefer batching futures.",
        extra={"final_value": final},
    )

    # CORRECT pattern for the same operation:
    logger.info("Correct pattern: batch futures then single ray.get()")
    ray.get(counter.reset.remote())
    futures = [counter.increment.remote() for _ in range(n)]
    values = ray.get(futures)  # one network round-trip
    logger.info(
        "Correct pattern done",
        extra={"final_value": values[-1]},
    )


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    logger.info("Initialising Ray for actors demo", extra={"ray_config": _RAY_CFG})
    ray.init(
        num_cpus=_RAY_CFG["num_cpus"],
        num_gpus=_RAY_CFG["num_gpus"],
        object_store_memory=_RAY_CFG["object_store_memory"],
        ignore_reinit_error=True,
    )
    logger.info("Ray initialised", extra={"resources": ray.cluster_resources()})

    try:
        demo_stateful_counter()
        demo_parameter_server()
        demo_actor_pool()
        demo_anti_pattern_actor()
    finally:
        logger.info("Shutting down Ray")
        ray.shutdown()


if __name__ == "__main__":
    main()
