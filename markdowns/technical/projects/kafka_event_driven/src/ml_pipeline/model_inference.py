"""
model_inference.py — Online model inference pipeline over Kafka events.

Architecture
------------
  user_events topic
       │
       ▼
  InferencePipeline (consumer)
       │ extract features per event
       │ call MockModel.predict(features)
       ▼
  predictions topic (producer)
       │
       ▼
  MonitoringPipeline / downstream consumers

MockModel
---------
A deterministic scoring function that combines:
  * A fixed random offset (seeded per model version) for baseline variation.
  * The weighted sum of selected feature values.

This gives reproducible, non-trivial outputs without requiring a trained model.

InferencePipeline
-----------------
Consumes user_events, looks up the FeatureStore to enrich with per-user
aggregates, runs the model, then produces the prediction to the predictions
topic.  Every prediction is logged with user_id, features, and score.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from kafka_core.mock_kafka import MockKafkaBroker
from kafka_core.producer import MockKafkaProducer
from kafka_core.consumer import MockKafkaConsumer
from ml_pipeline.feature_pipeline import FeatureStore

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
_logger = _build_logger("ml_pipeline.model_inference", _CONFIG)


# ---------------------------------------------------------------------------
# MockModel
# ---------------------------------------------------------------------------

@dataclass
class MockModel:
    """
    Deterministic scoring model for demonstration purposes.

    The score is computed as:
      score = sigmoid(bias + w_click * click_count
                           + w_purchase * purchase_count * 3
                           + w_session * session_count
                           + w_recency * recency_factor)

    where recency_factor = 1 / (1 + seconds_since_last_event / 3600).

    Parameters
    ----------
    model_id : Identifier for logging (e.g. ``"v1"`` or ``"v2"``).
    seed     : Random seed used to initialise the weight offsets.
    """

    model_id: str
    seed: int = 42

    def __post_init__(self) -> None:
        rng = random.Random(self.seed)
        # Weights are deterministic but vary per model version
        self._bias = rng.uniform(-0.5, 0.5)
        self._w_click = rng.uniform(0.05, 0.15)
        self._w_purchase = rng.uniform(0.2, 0.4)
        self._w_session = rng.uniform(0.03, 0.1)
        self._w_recency = rng.uniform(0.1, 0.3)
        _logger.info(
            f"MockModel[{self.model_id}] initialised: "
            f"bias={self._bias:.4f}, "
            f"w_click={self._w_click:.4f}, "
            f"w_purchase={self._w_purchase:.4f}, "
            f"w_session={self._w_session:.4f}, "
            f"w_recency={self._w_recency:.4f}"
        )

    def predict(self, features: Dict[str, Any]) -> float:
        """
        Score a feature vector and return a float in [0, 1].

        Parameters
        ----------
        features : Dict containing keys: click_count, purchase_count,
                   session_count, last_event_ts.

        Returns
        -------
        Float in [0, 1] representing propensity to convert.
        """
        click_count = float(features.get("click_count", 0))
        purchase_count = float(features.get("purchase_count", 0))
        session_count = float(features.get("session_count", 0))
        last_event_ts = features.get("last_event_ts")

        if last_event_ts:
            seconds_since = max(0.0, time.time() - float(last_event_ts))
            recency_factor = 1.0 / (1.0 + seconds_since / 3600.0)
        else:
            recency_factor = 0.0

        linear = (
            self._bias
            + self._w_click * click_count
            + self._w_purchase * purchase_count * 3.0
            + self._w_session * session_count
            + self._w_recency * recency_factor
        )

        score = self._sigmoid(linear)
        _logger.debug(
            f"MockModel[{self.model_id}].predict: "
            f"click_count={click_count}, purchase_count={purchase_count}, "
            f"session_count={session_count}, recency_factor={recency_factor:.4f}, "
            f"linear={linear:.4f}, score={score:.4f}"
        )
        return score

    @staticmethod
    def _sigmoid(x: float) -> float:
        import math
        return 1.0 / (1.0 + math.exp(-x))


# ---------------------------------------------------------------------------
# InferencePipeline
# ---------------------------------------------------------------------------

class InferencePipeline:
    """
    Consume user_events → extract features → predict → produce to predictions topic.

    Parameters
    ----------
    broker           : Shared ``MockKafkaBroker`` instance.
    model            : ``MockModel`` instance for scoring.
    feature_store    : ``FeatureStore`` to read per-user features.
    group_id         : Kafka consumer group ID.
    source_topic     : Topic to consume events from (default: ``user_events``).
    output_topic     : Topic to produce predictions to (default: ``predictions``).
    """

    def __init__(
        self,
        broker: MockKafkaBroker,
        model: MockModel,
        feature_store: FeatureStore,
        group_id: str = "inference-pipeline",
        source_topic: str = "user_events",
        output_topic: str = "predictions",
    ) -> None:
        self._broker = broker
        self._model = model
        self._feature_store = feature_store
        self._source_topic = source_topic
        self._output_topic = output_topic

        self._consumer = MockKafkaConsumer(
            broker=broker,
            group_id=group_id,
            dlq_topic="dlq",
        )
        self._consumer.subscribe([source_topic])
        self._producer = MockKafkaProducer(broker=broker)

        # Ensure output topic exists
        if not broker.topic_exists(output_topic):
            broker.create_topic(output_topic, num_partitions=4, replication_factor=1)

        self._inference_count = 0
        _logger.info(
            f"InferencePipeline initialised: "
            f"model={model.model_id!r}, "
            f"source={source_topic!r}, "
            f"output={output_topic!r}"
        )

    def run(self, num_events: int, poll_timeout: float = 0.05) -> List[Dict[str, Any]]:
        """
        Process up to *num_events* events and return a list of prediction records.

        Each record contains: user_id, features, score, model_id, timestamp.
        """
        _logger.info(
            f"InferencePipeline.run: target_events={num_events}, "
            f"model={self._model.model_id!r}"
        )
        predictions: List[Dict[str, Any]] = []

        for _ in range(num_events):
            msg = self._consumer.poll(timeout=poll_timeout)
            if msg is None:
                break

            try:
                payload = json.loads(msg.value.decode("utf-8"))
                user_id = payload.get("user_id") or (
                    msg.key.decode("utf-8") if msg.key else "unknown"
                )

                # Update feature store with this event, then retrieve features
                self._feature_store.update(user_id=user_id, event=payload)
                features = self._feature_store.get_feature_dict(user_id) or {}

                score = self._model.predict(features)
                self._inference_count += 1

                prediction_record = {
                    "user_id": user_id,
                    "model_id": self._model.model_id,
                    "score": round(score, 6),
                    "features": features,
                    "source_event_type": payload.get("event_type"),
                    "timestamp": time.time(),
                }

                self._producer.produce(
                    topic=self._output_topic,
                    key=user_id,
                    value=prediction_record,
                )
                predictions.append(prediction_record)

                _logger.info(
                    f"Prediction: user_id={user_id!r}, "
                    f"model={self._model.model_id!r}, "
                    f"score={score:.4f}, "
                    f"click_count={features.get('click_count', 0)}, "
                    f"purchase_count={features.get('purchase_count', 0)}"
                )

                self._consumer.commit()
            except Exception as exc:
                _logger.error(
                    f"InferencePipeline error: {exc!r} — routing to DLQ"
                )
                self._consumer.route_to_dlq(msg, exc)
                self._consumer.commit()

        self._producer.flush()
        _logger.info(
            f"InferencePipeline.run complete: "
            f"total_inferences={self._inference_count}, "
            f"this_run={len(predictions)}"
        )
        return predictions

    def close(self) -> None:
        self._consumer.close()
        _logger.info(
            f"InferencePipeline closed: total_inferences={self._inference_count}"
        )

    @property
    def inference_count(self) -> int:
        return self._inference_count


# ---------------------------------------------------------------------------
# Demo / main
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Demonstrate InferencePipeline: produce events, run inference, show scores.
    """
    from ml_pipeline.feature_pipeline import FeatureStore, generate_user_events

    _logger.info("=== InferencePipeline demo start ===")

    broker = MockKafkaBroker()
    broker.create_topic("user_events", num_partitions=4, replication_factor=1)
    broker.create_topic("predictions", num_partitions=4, replication_factor=1)
    broker.create_topic("dlq", num_partitions=1, replication_factor=1)

    generate_user_events(broker, "user_events", num_events=30, num_users=8)

    feature_store = FeatureStore(feature_window_s=86_400)
    model = MockModel(model_id="v1", seed=42)
    pipeline = InferencePipeline(
        broker=broker,
        model=model,
        feature_store=feature_store,
        group_id="inference-demo",
    )

    predictions = pipeline.run(num_events=30)

    _logger.info(f"Generated {len(predictions)} predictions")
    scores = [p["score"] for p in predictions]
    _logger.info(
        f"Score distribution: min={min(scores):.4f}, "
        f"max={max(scores):.4f}, "
        f"mean={sum(scores)/len(scores):.4f}"
    )

    pipeline.close()
    _logger.info("=== InferencePipeline demo complete ===")


if __name__ == "__main__":
    main()
