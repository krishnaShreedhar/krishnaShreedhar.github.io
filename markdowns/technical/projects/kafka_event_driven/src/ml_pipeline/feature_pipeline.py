"""
feature_pipeline.py — Real-time feature computation from Kafka events.

Architecture
------------
  user_events topic
       │
       ▼
  FeaturePipeline (consumer)
       │ update per-user rolling aggregates
       ▼
  FeatureStore (in-memory, keyed by user_id)
       │
       ▼
  downstream: InferencePipeline reads features at inference time

Features computed per user
--------------------------
  click_count       : Total click events seen in the feature window.
  session_count     : Number of distinct sessions (inferred from timestamps).
  purchase_count    : Total purchase events.
  last_event_ts     : Unix timestamp of the most recent event.
  avg_session_gap_s : Average time between consecutive events (recency proxy).

The feature window (feature_window_s from config) is used to discard stale
events: any event older than now - feature_window_s is ignored.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import random
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from kafka_core.mock_kafka import MockKafkaBroker
from kafka_core.producer import MockKafkaProducer
from kafka_core.consumer import MockKafkaConsumer

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
_logger = _build_logger("ml_pipeline.feature_pipeline", _CONFIG)


# ---------------------------------------------------------------------------
# FeatureStore
# ---------------------------------------------------------------------------

@dataclass
class UserFeatures:
    """Per-user feature vector."""
    user_id: str
    click_count: int = 0
    session_count: int = 0
    purchase_count: int = 0
    last_event_ts: Optional[float] = None
    event_timestamps: List[float] = field(default_factory=list)

    @property
    def avg_session_gap_s(self) -> float:
        """Average time between consecutive events (seconds)."""
        if len(self.event_timestamps) < 2:
            return 0.0
        gaps = [
            self.event_timestamps[i] - self.event_timestamps[i - 1]
            for i in range(1, len(self.event_timestamps))
        ]
        return sum(gaps) / len(gaps)

    def to_feature_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "click_count": self.click_count,
            "session_count": self.session_count,
            "purchase_count": self.purchase_count,
            "last_event_ts": self.last_event_ts,
            "avg_session_gap_s": round(self.avg_session_gap_s, 3),
        }


class FeatureStore:
    """
    In-memory feature store keyed by user_id.

    Provides thread-safe read/write access to per-user feature vectors.
    In production this would be backed by Redis or a similar low-latency store.

    Parameters
    ----------
    feature_window_s : Events older than now - feature_window_s are excluded
                       from feature computation (rolling window).
    """

    def __init__(self, feature_window_s: float = 86_400.0) -> None:
        self._store: Dict[str, UserFeatures] = {}
        self._feature_window_s = feature_window_s
        self._update_count = 0
        _logger.info(
            f"FeatureStore initialised: feature_window_s={feature_window_s}"
        )

    def update(self, user_id: str, event: Dict[str, Any]) -> UserFeatures:
        """
        Update features for *user_id* based on *event*.

        Parameters
        ----------
        user_id : The user this event belongs to.
        event   : Dict with at least ``event_type`` and optionally ``timestamp``.

        Returns
        -------
        Updated ``UserFeatures`` for the user.
        """
        if user_id not in self._store:
            self._store[user_id] = UserFeatures(user_id=user_id)

        features = self._store[user_id]
        event_ts = float(event.get("timestamp", time.time()))
        event_type = event.get("event_type", "unknown")
        cutoff = time.time() - self._feature_window_s

        # Prune old timestamps
        features.event_timestamps = [
            ts for ts in features.event_timestamps if ts >= cutoff
        ]

        # Record this event's timestamp
        features.event_timestamps.append(event_ts)
        features.last_event_ts = event_ts

        # Update counters
        if event_type == "click":
            features.click_count += 1
        elif event_type == "purchase":
            features.purchase_count += 1
        elif event_type == "session_start":
            features.session_count += 1

        self._update_count += 1
        _logger.debug(
            f"FeatureStore.update: user_id={user_id!r}, "
            f"event_type={event_type!r}, "
            f"click_count={features.click_count}, "
            f"purchase_count={features.purchase_count}"
        )
        return features

    def get(self, user_id: str) -> Optional[UserFeatures]:
        """Return features for *user_id*, or ``None`` if not seen before."""
        features = self._store.get(user_id)
        _logger.debug(
            f"FeatureStore.get: user_id={user_id!r}, "
            f"found={'yes' if features else 'no'}"
        )
        return features

    def get_feature_dict(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Return serialised feature dict or None."""
        features = self.get(user_id)
        return features.to_feature_dict() if features else None

    def snapshot(self) -> Dict[str, Dict[str, Any]]:
        """Return a serialised snapshot of all user features."""
        return {uid: f.to_feature_dict() for uid, f in self._store.items()}

    @property
    def user_count(self) -> int:
        return len(self._store)

    @property
    def update_count(self) -> int:
        return self._update_count


# ---------------------------------------------------------------------------
# FeaturePipeline
# ---------------------------------------------------------------------------

class FeaturePipeline:
    """
    Consumes ``user_events`` from Kafka and maintains the ``FeatureStore``.

    Parameters
    ----------
    broker         : Shared ``MockKafkaBroker`` instance.
    feature_store  : ``FeatureStore`` to update.
    group_id       : Consumer group ID.
    topic          : Source topic (default: ``user_events``).
    """

    def __init__(
        self,
        broker: MockKafkaBroker,
        feature_store: FeatureStore,
        group_id: str = "feature-pipeline",
        topic: str = "user_events",
    ) -> None:
        self._broker = broker
        self._feature_store = feature_store
        self._topic = topic
        self._consumer = MockKafkaConsumer(
            broker=broker,
            group_id=group_id,
            dlq_topic="dlq",
        )
        self._consumer.subscribe([topic])
        self._processed_count = 0
        _logger.info(
            f"FeaturePipeline initialised: topic={topic!r}, group_id={group_id!r}"
        )

    def process_events(self, num_events: int, poll_timeout: float = 0.05) -> int:
        """
        Poll and process up to *num_events* events.

        Returns
        -------
        Number of events actually processed.
        """
        _logger.info(
            f"FeaturePipeline.process_events: target={num_events}"
        )
        processed = 0

        while processed < num_events:
            msg = self._consumer.poll(timeout=poll_timeout)
            if msg is None:
                _logger.debug("FeaturePipeline: no messages, stopping poll loop")
                break

            try:
                payload = json.loads(msg.value.decode("utf-8"))
                user_id = payload.get("user_id") or (
                    msg.key.decode("utf-8") if msg.key else "unknown"
                )
                self._feature_store.update(user_id=user_id, event=payload)
                self._consumer.commit()
                processed += 1
                self._processed_count += 1
                _logger.debug(
                    f"FeaturePipeline processed event {processed}: "
                    f"user_id={user_id!r}, "
                    f"event_type={payload.get('event_type')!r}"
                )
            except Exception as exc:
                _logger.error(
                    f"FeaturePipeline processing error: {exc!r}, "
                    f"routing to DLQ"
                )
                self._consumer.route_to_dlq(msg, exc)
                self._consumer.commit()

        _logger.info(
            f"FeaturePipeline.process_events complete: "
            f"processed={processed}, "
            f"feature_store_users={self._feature_store.user_count}"
        )
        return processed

    def close(self) -> None:
        self._consumer.close()
        _logger.info(
            f"FeaturePipeline closed: total_processed={self._processed_count}"
        )

    @property
    def processed_count(self) -> int:
        return self._processed_count


# ---------------------------------------------------------------------------
# Event generator for demo
# ---------------------------------------------------------------------------

def generate_user_events(
    broker: MockKafkaBroker,
    topic: str,
    num_events: int,
    num_users: int = 10,
) -> None:
    """Produce synthetic user events to *topic* on *broker*."""
    producer = MockKafkaProducer(broker=broker)
    event_types = ["click", "click", "click", "purchase", "session_start", "view"]

    rng = random.Random(42)
    base_ts = time.time() - 3600  # events from the past hour

    for i in range(num_events):
        user_id = f"user-{rng.randint(1, num_users):03d}"
        event_type = rng.choice(event_types)
        event = {
            "user_id": user_id,
            "event_type": event_type,
            "timestamp": base_ts + i * 10,  # 10s between events
            "page": f"/page-{rng.randint(1, 20)}",
            "session_id": str(uuid.uuid4())[:8],
        }
        producer.produce(topic=topic, key=user_id, value=event)

    producer.flush()
    _logger.info(
        f"Generated {num_events} synthetic events for {num_users} users on topic={topic!r}"
    )


# ---------------------------------------------------------------------------
# Demo / main
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Demonstrate FeaturePipeline: generate events, process them, print feature store.
    """
    _logger.info("=== FeaturePipeline demo start ===")

    cfg = _CONFIG.get("ml_pipeline", {})
    feature_window_s = float(cfg.get("feature_window_s", 86_400))

    broker = MockKafkaBroker()
    broker.create_topic("user_events", num_partitions=4, replication_factor=1)
    broker.create_topic("dlq", num_partitions=1, replication_factor=1)

    # Generate 50 events across 10 users
    generate_user_events(broker, "user_events", num_events=50, num_users=10)

    feature_store = FeatureStore(feature_window_s=feature_window_s)
    pipeline = FeaturePipeline(
        broker=broker,
        feature_store=feature_store,
        group_id="feature-pipeline-demo",
        topic="user_events",
    )

    processed = pipeline.process_events(num_events=50)
    _logger.info(f"Processed {processed} events")

    # Print feature store snapshot
    snapshot = feature_store.snapshot()
    _logger.info(f"Feature store snapshot ({feature_store.user_count} users):")
    for user_id, features in sorted(snapshot.items()):
        _logger.info(f"  {user_id}: {features}")

    pipeline.close()
    _logger.info("=== FeaturePipeline demo complete ===")


if __name__ == "__main__":
    main()
