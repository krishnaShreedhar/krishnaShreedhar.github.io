"""
event_processor.py — Running statistics (Welford) and consumer lag tracking.

Components
----------
WelfordOnlineStats
    Computes running mean and variance in a single pass using Welford's
    numerically stable online algorithm.  No need to store all values.

    Algorithm (Knuth / Welford):
      n += 1
      delta = x - mean
      mean += delta / n
      delta2 = x - mean
      M2 += delta * delta2
      variance = M2 / (n - 1)   [sample variance, n >= 2]

LagTracker
    Tracks the gap between the producer's latest offset (high-water mark)
    and the consumer's committed offset (consumer lag).  Emits an alert
    log when lag exceeds the configured threshold.

Both classes are designed to be composable:
  for event in stream:
      stats.update(event["value"])
      lag_tracker.record(producer_offset, consumer_offset)
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
_logger = _build_logger("streaming_analytics.event_processor", _CONFIG)


# ---------------------------------------------------------------------------
# WelfordOnlineStats
# ---------------------------------------------------------------------------

class WelfordOnlineStats:
    """
    Numerically stable online computation of mean and variance.

    Uses Welford's single-pass algorithm, which avoids catastrophic
    cancellation errors present in the naive two-pass approach.

    Reference: Knuth, TAOCP Vol 2, §4.2.2, Eqs. (15) and (16)

    Parameters
    ----------
    name : Human-readable label for logging (e.g. ``"score"`` or ``"latency_ms"``).
    """

    def __init__(self, name: str = "metric") -> None:
        self._name = name
        self._n: int = 0
        self._mean: float = 0.0
        self._M2: float = 0.0       # Sum of squared deviations from current mean
        self._min: float = float("inf")
        self._max: float = float("-inf")

        _logger.info(f"WelfordOnlineStats[{name}] initialised")

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, value: float) -> None:
        """
        Incorporate a new *value* into the running statistics.

        This is O(1) time and O(1) space.
        """
        self._n += 1
        delta = value - self._mean
        self._mean += delta / self._n
        delta2 = value - self._mean
        self._M2 += delta * delta2
        self._min = min(self._min, value)
        self._max = max(self._max, value)

        _logger.debug(
            f"WelfordOnlineStats[{self._name}] update #{self._n}: "
            f"value={value:.4f}, mean={self._mean:.4f}, "
            f"variance={self.variance:.4f}"
        )

    def update_batch(self, values: List[float]) -> None:
        """Bulk update from a list of values."""
        for v in values:
            self.update(v)
        _logger.info(
            f"WelfordOnlineStats[{self._name}] batch update: "
            f"added {len(values)} values, n={self._n}, mean={self._mean:.4f}"
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n(self) -> int:
        """Number of values seen so far."""
        return self._n

    @property
    def mean(self) -> float:
        """Running mean."""
        return self._mean

    @property
    def variance(self) -> float:
        """
        Sample variance (denominator = n-1).

        Returns 0.0 if fewer than 2 values have been seen.
        """
        if self._n < 2:
            return 0.0
        return self._M2 / (self._n - 1)

    @property
    def std(self) -> float:
        """Sample standard deviation."""
        return math.sqrt(self.variance)

    @property
    def min_value(self) -> float:
        return self._min

    @property
    def max_value(self) -> float:
        return self._max

    def snapshot(self) -> Dict[str, Any]:
        """Return a dict snapshot of all current statistics."""
        snap = {
            "name": self._name,
            "n": self._n,
            "mean": round(self._mean, 6),
            "variance": round(self.variance, 6),
            "std": round(self.std, 6),
            "min": round(self._min, 6) if self._n > 0 else None,
            "max": round(self._max, 6) if self._n > 0 else None,
        }
        _logger.info(f"WelfordOnlineStats[{self._name}] snapshot: {snap}")
        return snap

    def reset(self) -> None:
        """Reset all state (start fresh)."""
        self._n = 0
        self._mean = 0.0
        self._M2 = 0.0
        self._min = float("inf")
        self._max = float("-inf")
        _logger.info(f"WelfordOnlineStats[{self._name}] reset")


# ---------------------------------------------------------------------------
# LagTracker
# ---------------------------------------------------------------------------

@dataclass
class LagRecord:
    """Single lag observation."""
    timestamp: float
    topic: str
    partition: int
    producer_offset: int  # high-water mark
    consumer_offset: int  # committed offset
    lag: int              # producer_offset - consumer_offset


class LagTracker:
    """
    Tracks consumer lag per (topic, partition) over time.

    Records the difference between the producer's current high-water mark
    and the consumer group's committed offset.  Emits WARNING logs when
    lag exceeds the configured threshold.

    Parameters
    ----------
    broker              : Shared ``MockKafkaBroker`` for reading offsets.
    group_id            : Consumer group to track.
    alert_threshold     : Lag count above which an alert is logged.
    history_max_records : Maximum number of lag records to retain in memory.
    """

    def __init__(
        self,
        broker: MockKafkaBroker,
        group_id: str,
        alert_threshold: int = 1000,
        history_max_records: int = 10_000,
    ) -> None:
        self._broker = broker
        self._group_id = group_id
        self._alert_threshold = alert_threshold
        self._history_max = history_max_records
        self._records: List[LagRecord] = []
        self._alert_count = 0

        _logger.info(
            f"LagTracker initialised: group_id={group_id!r}, "
            f"alert_threshold={alert_threshold}"
        )

    def check(self, topics: List[str]) -> Dict[str, Dict[int, int]]:
        """
        Poll the broker for current lag on *topics* and record observations.

        Returns
        -------
        Dict of ``{topic: {partition: lag}}`` for all topics.
        """
        result: Dict[str, Dict[int, int]] = {}
        now = time.time()

        for topic in topics:
            if not self._broker.topic_exists(topic):
                _logger.warning(f"LagTracker: topic {topic!r} does not exist, skipping")
                continue

            lag_map = self._broker.get_lag(self._group_id, topic)
            result[topic] = lag_map
            cfg = self._broker.get_topic_config(topic)

            for partition in range(cfg.num_partitions):
                lag = lag_map.get(partition, 0)
                hw = self._broker.log_end_offset(topic, partition)
                committed = hw - lag

                record = LagRecord(
                    timestamp=now,
                    topic=topic,
                    partition=partition,
                    producer_offset=hw,
                    consumer_offset=committed,
                    lag=lag,
                )
                self._records.append(record)

                if lag > self._alert_threshold:
                    self._alert_count += 1
                    _logger.warning(
                        f"LAG ALERT: topic={topic!r}, partition={partition}, "
                        f"lag={lag}, threshold={self._alert_threshold}, "
                        f"group={self._group_id!r}"
                    )
                else:
                    _logger.debug(
                        f"LagTracker: topic={topic!r}, partition={partition}, "
                        f"lag={lag} (ok)"
                    )

            # Trim history
            if len(self._records) > self._history_max:
                self._records = self._records[-self._history_max :]

        return result

    def lag_stats(self) -> Dict[str, Any]:
        """Return summary statistics across all lag records."""
        if not self._records:
            return {"total_records": 0, "alert_count": self._alert_count}

        lags = [r.lag for r in self._records]
        stats = WelfordOnlineStats("lag")
        stats.update_batch(lags)
        snap = stats.snapshot()
        snap["alert_count"] = self._alert_count
        snap["latest_lag_by_partition"] = self._latest_lags()
        _logger.info(f"LagTracker.lag_stats: {snap}")
        return snap

    def _latest_lags(self) -> Dict[str, Dict[int, int]]:
        """Return the most recent lag per (topic, partition)."""
        latest: Dict[str, Dict[int, int]] = {}
        for record in reversed(self._records):
            if record.topic not in latest:
                latest[record.topic] = {}
            if record.partition not in latest[record.topic]:
                latest[record.topic][record.partition] = record.lag
        return latest

    @property
    def alert_count(self) -> int:
        return self._alert_count

    @property
    def records(self) -> List[LagRecord]:
        return list(self._records)


# ---------------------------------------------------------------------------
# Demo / main
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Demonstrate:
      1. WelfordOnlineStats: process a stream of synthetic values and show
         running statistics updating after each batch.
      2. LagTracker: simulate growing lag, check alerts, consume to reduce lag.
    """
    from kafka_core.mock_kafka import MockKafkaBroker
    from kafka_core.producer import MockKafkaProducer
    from kafka_core.consumer import MockKafkaConsumer

    _logger.info("=== EventProcessor demo start ===")

    cfg = _CONFIG.get("streaming", {})
    num_events: int = int(cfg.get("num_events_to_generate", 200))
    lag_alert_threshold: int = int(cfg.get("lag_alert_threshold", 1000))

    # -------------------------------------------------------------------
    # Part 1: WelfordOnlineStats
    # -------------------------------------------------------------------
    _logger.info("--- WelfordOnlineStats demo ---")

    rng = random.Random(42)
    score_stats = WelfordOnlineStats("prediction_score")
    latency_stats = WelfordOnlineStats("processing_latency_ms")

    _logger.info("Processing stream of values in batches of 50:")
    for batch_num in range(4):
        batch_scores = [rng.gauss(0.45, 0.12) for _ in range(50)]
        batch_latencies = [rng.expovariate(1 / 20.0) for _ in range(50)]

        score_stats.update_batch(batch_scores)
        latency_stats.update_batch(batch_latencies)

        _logger.info(
            f"Batch {batch_num + 1}: "
            f"score_mean={score_stats.mean:.4f} ± {score_stats.std:.4f}, "
            f"latency_mean={latency_stats.mean:.2f}ms ± {latency_stats.std:.2f}ms"
        )

    final_score_snap = score_stats.snapshot()
    final_latency_snap = latency_stats.snapshot()
    _logger.info(f"Final score stats: {final_score_snap}")
    _logger.info(f"Final latency stats: {final_latency_snap}")

    # Verify Welford converges close to true distribution parameters
    assert abs(score_stats.mean - 0.45) < 0.05, (
        f"Score mean {score_stats.mean:.4f} deviates too much from 0.45"
    )
    _logger.info("Welford convergence check passed")

    # -------------------------------------------------------------------
    # Part 2: LagTracker
    # -------------------------------------------------------------------
    _logger.info("--- LagTracker demo ---")

    broker = MockKafkaBroker()
    broker.create_topic("user_events", num_partitions=4, replication_factor=1)
    broker.create_topic("dlq", num_partitions=1, replication_factor=1)

    group_id = "lag-tracker-demo"
    tracker = LagTracker(
        broker=broker,
        group_id=group_id,
        alert_threshold=lag_alert_threshold,
    )

    # Produce a batch of messages to create lag
    producer = MockKafkaProducer(broker=broker)
    for i in range(num_events):
        payload = {
            "user_id": f"user-{i % 20:03d}",
            "value": round(rng.uniform(1.0, 100.0), 2),
            "timestamp": time.time(),
        }
        producer.produce("user_events", key=payload["user_id"], value=payload)
    producer.flush()
    _logger.info(f"Produced {num_events} messages to user_events")

    # Check lag before consuming (consumer group has never committed)
    lag_before = tracker.check(["user_events"])
    total_lag_before = sum(
        sum(parts.values()) for parts in lag_before.values()
    )
    _logger.info(
        f"Lag before consuming: {lag_before}, total={total_lag_before}"
    )

    # Consume half the messages
    consumer = MockKafkaConsumer(broker=broker, group_id=group_id)
    consumer.subscribe(["user_events"])
    consumed = 0
    target = num_events // 2
    while consumed < target:
        msg = consumer.poll(timeout=0.01)
        if msg is None:
            break
        consumer.commit()
        consumed += 1
    _logger.info(f"Consumed {consumed} messages")

    # Check lag after consuming half
    lag_after = tracker.check(["user_events"])
    total_lag_after = sum(
        sum(parts.values()) for parts in lag_after.values()
    )
    _logger.info(
        f"Lag after consuming half: {lag_after}, total={total_lag_after}"
    )
    assert total_lag_after < total_lag_before, "Lag should decrease after consuming"

    # Print final lag stats
    lag_summary = tracker.lag_stats()
    _logger.info(f"Lag tracker summary: {lag_summary}")

    consumer.close()
    _logger.info("=== EventProcessor demo complete ===")


if __name__ == "__main__":
    main()
