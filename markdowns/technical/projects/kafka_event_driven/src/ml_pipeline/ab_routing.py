"""
ab_routing.py — Hash-based A/B model routing for online inference.

Concept
-------
A/B routing assigns each user deterministically to a model variant based on
a hash of their user_id.  This guarantees:
  * Consistency: the same user always gets the same model.
  * Configurable split: controlled by ``model_v2_traffic_pct`` in config.yaml.
  * No sticky sessions required: the routing decision is stateless.

Hash function
-------------
  bucket = hash(user_id.encode()) % 100
  if bucket < model_v2_traffic_pct → model_v2
  else                             → model_v1

Components
----------
ABRouter       : Stateless routing logic.
ModelRegistry  : Holds both model instances, delegates to ABRouter.
"""

from __future__ import annotations

import hashlib
import json
import logging
import logging.handlers
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from ml_pipeline.model_inference import MockModel

# ---------------------------------------------------------------------------
# Logging bootstrap
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    config_path = Path(__file__).resolve().parents[2] / "config.yaml"
    with open(config_path, "r") as fh:
        return yaml.safe_load(fh)


def _build_logger(name: str, cfg: dict) -> logging.Logger:
    log_cfg = cfg["logging"]
    log_file = Path(__file__).resolve().parents[2] / log_cfg["log_file"]
    log_file.parent.mkdir(parents=True, exist_ok=True)

    class _JSONFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            payload = {
                "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
            if record.exc_info:
                payload["exc_info"] = self.formatException(record.exc_info)
            return json.dumps(payload)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    level = getattr(logging, log_cfg["level"].upper(), logging.INFO)
    logger.setLevel(level)

    fh = logging.handlers.RotatingFileHandler(
        filename=str(log_file),
        maxBytes=log_cfg["max_bytes"],
        backupCount=log_cfg["backup_count"],
    )
    fh.setFormatter(_JSONFormatter())
    fh.setLevel(level)

    sh = logging.StreamHandler()
    sh.setFormatter(_JSONFormatter())
    sh.setLevel(level)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


_CONFIG = _load_config()
_logger = _build_logger("ml_pipeline.ab_routing", _CONFIG)


# ---------------------------------------------------------------------------
# ABRouter
# ---------------------------------------------------------------------------

class ABRouter:
    """
    Deterministic A/B router using a hash-based bucket assignment.

    Parameters
    ----------
    model_v2_traffic_pct : Percentage of traffic (0–100) routed to model_v2.
                           Sourced from config.yaml: ``ml_pipeline.model_v2_traffic_pct``.
    """

    MODEL_V1 = "v1"
    MODEL_V2 = "v2"

    def __init__(self, model_v2_traffic_pct: int = 20) -> None:
        if not 0 <= model_v2_traffic_pct <= 100:
            raise ValueError(
                f"model_v2_traffic_pct must be in [0, 100], got {model_v2_traffic_pct}"
            )
        self._v2_pct = model_v2_traffic_pct
        self._routing_counts: Counter = Counter()
        _logger.info(
            f"ABRouter initialised: model_v2_traffic_pct={model_v2_traffic_pct}%"
        )

    def route(self, user_id: str) -> str:
        """
        Return ``"v1"`` or ``"v2"`` for the given *user_id*.

        The assignment is deterministic: calling this function multiple times
        with the same *user_id* always returns the same result.

        Implementation uses MD5 for speed (not cryptographic security).
        Any 128-bit hash would work equally well here.
        """
        hash_bytes = hashlib.md5(user_id.encode("utf-8")).digest()
        # Take the first 4 bytes as a big-endian unsigned integer
        bucket = int.from_bytes(hash_bytes[:4], "big") % 100

        model_id = self.MODEL_V2 if bucket < self._v2_pct else self.MODEL_V1
        self._routing_counts[model_id] += 1

        _logger.debug(
            f"ABRouter.route: user_id={user_id!r}, bucket={bucket}, "
            f"model={model_id!r}"
        )
        return model_id

    def routing_distribution(self) -> Dict[str, Any]:
        """Return the observed routing distribution and configured split."""
        total = sum(self._routing_counts.values())
        result: Dict[str, Any] = {
            "configured_v2_pct": self._v2_pct,
            "total_routed": total,
            "v1_count": self._routing_counts.get(self.MODEL_V1, 0),
            "v2_count": self._routing_counts.get(self.MODEL_V2, 0),
        }
        if total > 0:
            result["observed_v2_pct"] = round(
                100.0 * result["v2_count"] / total, 2
            )
        _logger.info(f"ABRouter.routing_distribution: {result}")
        return result

    def explain(self, user_id: str) -> Dict[str, Any]:
        """Return a human-readable routing explanation for *user_id*."""
        hash_bytes = hashlib.md5(user_id.encode("utf-8")).digest()
        bucket = int.from_bytes(hash_bytes[:4], "big") % 100
        model_id = self.MODEL_V2 if bucket < self._v2_pct else self.MODEL_V1
        return {
            "user_id": user_id,
            "bucket": bucket,
            "threshold": self._v2_pct,
            "model": model_id,
            "reason": (
                f"bucket {bucket} < {self._v2_pct} → v2"
                if model_id == self.MODEL_V2
                else f"bucket {bucket} >= {self._v2_pct} → v1"
            ),
        }


# ---------------------------------------------------------------------------
# ModelRegistry
# ---------------------------------------------------------------------------

class ModelRegistry:
    """
    Holds model instances and routes inference requests via ``ABRouter``.

    Parameters
    ----------
    router : ``ABRouter`` instance.
    model_v1 : Model for the control group.
    model_v2 : Model for the treatment group.
    """

    def __init__(
        self,
        router: ABRouter,
        model_v1: MockModel,
        model_v2: MockModel,
    ) -> None:
        self._router = router
        self._models: Dict[str, MockModel] = {
            ABRouter.MODEL_V1: model_v1,
            ABRouter.MODEL_V2: model_v2,
        }
        self._inference_counts: Counter = Counter()
        _logger.info(
            f"ModelRegistry initialised with models: "
            f"{list(self._models.keys())}"
        )

    def predict(
        self,
        user_id: str,
        features: Dict[str, Any],
    ) -> Tuple[str, float]:
        """
        Route *user_id* to the appropriate model and return a prediction.

        Returns
        -------
        (model_id, score) tuple.
        """
        model_id = self._router.route(user_id)
        model = self._models[model_id]
        score = model.predict(features)
        self._inference_counts[model_id] += 1

        _logger.info(
            f"ModelRegistry.predict: user_id={user_id!r}, "
            f"model={model_id!r}, score={score:.4f}"
        )
        return model_id, score

    def inference_stats(self) -> Dict[str, Any]:
        """Return inference count per model variant."""
        return {
            "total": sum(self._inference_counts.values()),
            "per_model": dict(self._inference_counts),
            "routing_distribution": self._router.routing_distribution(),
        }

    def get_model(self, model_id: str) -> MockModel:
        """Return the ``MockModel`` instance for *model_id*."""
        if model_id not in self._models:
            raise KeyError(f"Unknown model_id: {model_id!r}")
        return self._models[model_id]


# ---------------------------------------------------------------------------
# Demo / main
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Demonstrate A/B routing with 20 users:
      * Show which model each user is routed to.
      * Verify the observed distribution matches the configured split.
    """
    _logger.info("=== ABRouting demo start ===")

    cfg = _CONFIG.get("ml_pipeline", {})
    v2_pct: int = int(cfg.get("model_v2_traffic_pct", 20))

    router = ABRouter(model_v2_traffic_pct=v2_pct)
    model_v1 = MockModel(model_id="v1", seed=42)
    model_v2 = MockModel(model_id="v2", seed=99)
    registry = ModelRegistry(router=router, model_v1=model_v1, model_v2=model_v2)

    # Simulate 20 users making predictions
    user_ids = [f"user-{i:03d}" for i in range(1, 21)]
    sample_features = {
        "click_count": 5,
        "purchase_count": 1,
        "session_count": 3,
        "last_event_ts": None,
    }

    _logger.info("--- Routing decisions for 20 users ---")
    for user_id in user_ids:
        explanation = router.explain(user_id)
        model_id, score = registry.predict(user_id, sample_features)
        _logger.info(
            f"  {user_id}: bucket={explanation['bucket']:3d}, "
            f"model={model_id}, score={score:.4f}, "
            f"reason={explanation['reason']!r}"
        )

    # Print routing distribution
    stats = registry.inference_stats()
    _logger.info(f"Inference stats: {stats}")
    distribution = stats["routing_distribution"]
    _logger.info(
        f"Routing distribution: "
        f"v1={distribution['v1_count']}/20 "
        f"({100 - distribution.get('observed_v2_pct', 0):.1f}%), "
        f"v2={distribution['v2_count']}/20 "
        f"({distribution.get('observed_v2_pct', 0):.1f}%)"
    )

    # Verify determinism: same user always gets same model
    for user_id in user_ids[:5]:
        m1 = router.route(user_id)
        m2 = router.route(user_id)
        assert m1 == m2, f"Non-deterministic routing for {user_id!r}"
    _logger.info("Determinism check passed for first 5 users")

    _logger.info("=== ABRouting demo complete ===")


if __name__ == "__main__":
    main()
