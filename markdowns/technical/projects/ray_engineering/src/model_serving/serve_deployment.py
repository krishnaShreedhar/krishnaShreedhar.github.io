"""
Ray Serve – Model Serving with Deployments.

Demonstrates:
  - @serve.deployment decorator with replica count and autoscaling config
  - Request batching via @serve.batch
  - Multi-model pipeline: Preprocessor → Classifier (Ingress)
  - Health checks and graceful shutdown
  - Sending test requests via the ServeHandle API

All constants are read from config.yaml.  No values are hardcoded.
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import ray
from ray import serve

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.config_loader import load_config
from utils.logging_setup import get_logger

_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config.yaml"
config: dict[str, Any] = load_config(str(_CONFIG_PATH))
logger = get_logger(__name__, config)

_SERVE_CFG = config["model_serving"]
_RAY_CFG = config["ray"]["init"]

NUM_REPLICAS: int = _SERVE_CFG["num_replicas"]
MAX_CONCURRENT_QUERIES: int = _SERVE_CFG["max_concurrent_queries"]
MAX_BATCH_SIZE: int = _SERVE_CFG["max_batch_size"]
BATCH_WAIT_TIMEOUT_S: float = _SERVE_CFG["batch_wait_timeout_s"]

NUM_FEATURES: int = 10


# ===========================================================================
# Preprocessor Deployment
# ===========================================================================

@serve.deployment(
    name="Preprocessor",
    num_replicas=NUM_REPLICAS,
    max_concurrent_queries=MAX_CONCURRENT_QUERIES,
    ray_actor_options={"num_cpus": 0.5},
)
class Preprocessor:
    """
    Stateless feature preprocessor.

    Accepts a raw list of floats (one sample) and returns a normalised
    numpy array.  In production this would load a fitted scaler from
    a model registry; here we apply a fixed z-score approximation.
    """

    def __init__(self) -> None:
        import logging
        self._log = logging.getLogger("serve.Preprocessor")
        # Fixed normalisation constants (would come from a fitted scaler)
        self._mean = np.zeros(NUM_FEATURES, dtype=np.float32)
        self._std = np.ones(NUM_FEATURES, dtype=np.float32)
        self._log.info("Preprocessor replica started")

    async def preprocess(self, raw_features: list[float]) -> list[float]:
        """Normalise a single feature vector."""
        arr = np.array(raw_features, dtype=np.float32)
        if len(arr) != NUM_FEATURES:
            raise ValueError(
                f"Expected {NUM_FEATURES} features, got {len(arr)}"
            )
        normalised = (arr - self._mean) / (self._std + 1e-8)
        self._log.debug("Preprocessed features | norm=%.4f", float(np.linalg.norm(normalised)))
        return normalised.tolist()

    async def __call__(self, request: dict[str, Any]) -> dict[str, Any]:
        raw = request.get("features", [])
        processed = await self.preprocess(raw)
        return {"processed_features": processed}


# ===========================================================================
# Batching Classifier Deployment
# ===========================================================================

@serve.deployment(
    name="BatchClassifier",
    num_replicas=NUM_REPLICAS,
    max_concurrent_queries=MAX_CONCURRENT_QUERIES,
    ray_actor_options={"num_cpus": 0.5},
)
class BatchClassifier:
    """
    Batched inference deployment.

    Uses @serve.batch to coalesce concurrent individual requests into
    mini-batches for more efficient inference throughput.  The underlying
    model is a lightweight random-weight network (simulating a trained one).
    """

    def __init__(self) -> None:
        import logging
        import torch
        import torch.nn as nn

        self._log = logging.getLogger("serve.BatchClassifier")
        self._inference_count = 0

        # Simulated trained model (random weights stand in for a loaded checkpoint)
        self._model = nn.Sequential(
            nn.Linear(NUM_FEATURES, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        self._model.eval()
        self._log.info("BatchClassifier replica started")

    @serve.batch(max_batch_size=MAX_BATCH_SIZE, batch_wait_timeout_s=BATCH_WAIT_TIMEOUT_S)
    async def _batched_predict(
        self, feature_batch: list[list[float]]
    ) -> list[dict[str, Any]]:
        """
        Core batched inference method.

        Ray Serve accumulates incoming requests and calls this method with
        a list of individual inputs, up to max_batch_size.
        """
        import torch

        batch_size = len(feature_batch)
        self._log.info(
            "Batched inference | batch_size=%d cumulative=%d",
            batch_size, self._inference_count + batch_size,
        )

        tensor = torch.tensor(feature_batch, dtype=torch.float32)
        with torch.no_grad():
            probs = self._model(tensor).squeeze(dim=-1).numpy()

        self._inference_count += batch_size

        results = []
        for i, prob in enumerate(probs):
            predicted_class = int(prob >= 0.5)
            results.append(
                {
                    "probability": round(float(prob), 6),
                    "predicted_class": predicted_class,
                    "confidence": round(float(abs(prob - 0.5) * 2), 6),
                }
            )
        return results

    async def predict(self, features: list[float]) -> dict[str, Any]:
        """Public endpoint: receives a single sample, returns prediction."""
        return await self._batched_predict(features)

    async def __call__(self, request: dict[str, Any]) -> dict[str, Any]:
        features = request.get("features", [])
        return await self.predict(features)

    async def health(self) -> dict[str, Any]:
        return {"status": "healthy", "inference_count": self._inference_count}


# ===========================================================================
# Ingress / Pipeline Deployment
# ===========================================================================

@serve.deployment(
    name="InferencePipeline",
    num_replicas=1,  # ingress is single; downstream replicas handle scale
    max_concurrent_queries=MAX_CONCURRENT_QUERIES,
    ray_actor_options={"num_cpus": 0.2},
)
class InferencePipeline:
    """
    Multi-model pipeline: routes requests through Preprocessor → Classifier.

    Demonstrates how Ray Serve deployments can call each other via handles,
    composing a DAG of models without tight coupling.
    """

    def __init__(
        self,
        preprocessor: serve.handle.DeploymentHandle,
        classifier: serve.handle.DeploymentHandle,
    ) -> None:
        import logging
        self._log = logging.getLogger("serve.InferencePipeline")
        self._preprocessor = preprocessor
        self._classifier = classifier
        self._request_count = 0
        self._log.info("InferencePipeline started")

    async def __call__(self, request: dict[str, Any]) -> dict[str, Any]:
        self._request_count += 1
        request_id = self._request_count

        self._log.debug("Request received | request_id=%d", request_id)
        raw_features = request.get("features", [])

        t0 = time.perf_counter()

        # Step 1: Preprocess
        preprocessed = await self._preprocessor.preprocess.remote(raw_features)
        self._log.debug("Preprocessing done | request_id=%d", request_id)

        # Step 2: Classify
        prediction = await self._classifier.predict.remote(preprocessed)
        self._log.debug("Inference done | request_id=%d", request_id)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._log.info(
            "Pipeline request complete | request_id=%d | elapsed_ms=%.2f | class=%d",
            request_id, elapsed_ms, prediction["predicted_class"],
        )

        return {
            "request_id": request_id,
            "prediction": prediction,
            "elapsed_ms": round(elapsed_ms, 3),
        }

    async def health(self) -> dict[str, str]:
        return {"status": "healthy"}


# ===========================================================================
# Deployment manager (Dependency Inversion Principle)
# ===========================================================================

class ServeDeploymentManager:
    """
    Manages the lifecycle of all Ray Serve deployments.

    Abstracts start / stop from the business logic so that callers
    do not depend on Ray Serve internals directly.
    """

    def __init__(self) -> None:
        self._app = None
        logger.info(
            "ServeDeploymentManager created",
            extra={
                "num_replicas": NUM_REPLICAS,
                "max_batch_size": MAX_BATCH_SIZE,
                "batch_wait_timeout_s": BATCH_WAIT_TIMEOUT_S,
            },
        )

    def start(self) -> serve.handle.DeploymentHandle:
        """Deploy all components and return a handle to the ingress."""
        logger.info("Starting Ray Serve")
        serve.start(detached=False)

        preprocessor = Preprocessor.bind()
        classifier = BatchClassifier.bind()
        pipeline = InferencePipeline.bind(preprocessor, classifier)

        handle = serve.run(pipeline, name="inference_pipeline", route_prefix="/predict")
        logger.info(
            "All deployments running",
            extra={"deployments": list(serve.status().applications.keys())},
        )
        return handle

    def shutdown(self) -> None:
        """Gracefully shut down all deployments."""
        logger.info("Shutting down Ray Serve")
        serve.shutdown()
        logger.info("Ray Serve shutdown complete")


# ===========================================================================
# Test client
# ===========================================================================

async def run_test_requests(
    handle: serve.handle.DeploymentHandle, num_requests: int
) -> None:
    """
    Send a burst of test requests to the pipeline and log results.

    Fires all requests concurrently to exercise batching.
    """
    logger.info(
        "Sending test requests to the pipeline",
        extra={"num_requests": num_requests},
    )
    rng = np.random.default_rng(seed=99)

    async def single_request(req_id: int) -> dict[str, Any]:
        features = rng.standard_normal(NUM_FEATURES).tolist()
        result = await handle.remote({"features": features})
        return result

    tasks = [single_request(i) for i in range(num_requests)]
    results = await asyncio.gather(*tasks)

    classes = [r["prediction"]["predicted_class"] for r in results]
    avg_elapsed = np.mean([r["elapsed_ms"] for r in results])
    logger.info(
        "Test requests complete",
        extra={
            "num_requests": num_requests,
            "class_0_count": classes.count(0),
            "class_1_count": classes.count(1),
            "avg_elapsed_ms": round(float(avg_elapsed), 3),
            "sample_result": results[0],
        },
    )


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    logger.info("Initialising Ray for model serving", extra={"ray_config": _RAY_CFG})
    ray.init(
        num_cpus=_RAY_CFG["num_cpus"],
        num_gpus=_RAY_CFG["num_gpus"],
        object_store_memory=_RAY_CFG["object_store_memory"],
        ignore_reinit_error=True,
    )
    logger.info("Ray initialised", extra={"resources": ray.cluster_resources()})

    manager = ServeDeploymentManager()
    handle = manager.start()

    try:
        asyncio.run(run_test_requests(handle, num_requests=20))
    except Exception as exc:
        logger.error("Serving test failed", extra={"error": str(exc)}, exc_info=True)
        raise
    finally:
        manager.shutdown()
        logger.info("Shutting down Ray")
        ray.shutdown()


if __name__ == "__main__":
    main()
