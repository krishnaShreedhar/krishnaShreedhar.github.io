"""
consumer.py — MockKafkaConsumer with manual commit and DLQ routing.

Responsibilities
----------------
* Subscribe to one or more topics.
* ``poll()`` returns the next available MockMessage across all assigned
  partitions (round-robin scan), or ``None`` on timeout.
* Manual offset commit: the caller commits only after successful processing.
* On processing failure the message is forwarded to the DLQ topic and the
  original offset is still committed to prevent infinite retry loops on the
  main topic (the DLQ handler owns retry logic).
* Tracks consumed message count and logs at configurable intervals.

Design follows the standard at-least-once delivery pattern:
  1. poll()       → receive message
  2. process      → business logic
  3. commit()     → advance consumer group offset
  If step 2 throws, the message is sent to DLQ; offset is committed anyway
  so the consumer does not re-read the same poisoned message.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml

from kafka_core.mock_kafka import MockKafkaBroker, MockMessage

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
_logger = _build_logger("kafka_core.consumer", _CONFIG)

# ---------------------------------------------------------------------------
# MockKafkaConsumer
# ---------------------------------------------------------------------------

class MockKafkaConsumer:
    """
    Simulates a Kafka consumer with manual offset commits and DLQ support.

    Parameters
    ----------
    broker      : Shared ``MockKafkaBroker`` instance.
    group_id    : Consumer group identifier.  Offsets are tracked per group.
    dlq_topic   : Topic name for dead-letter messages (default: ``"dlq"``).
    config      : Consumer config section from config.yaml.

    Usage pattern
    -------------
    consumer = MockKafkaConsumer(broker, group_id="my-group")
    consumer.subscribe(["user_events"])
    while running:
        msg = consumer.poll(timeout=1.0)
        if msg is None:
            continue
        try:
            process(msg)
            consumer.commit()
        except ProcessingError as exc:
            consumer.route_to_dlq(msg, exc)
            consumer.commit()   # commit anyway to skip the bad message
    consumer.close()
    """

    def __init__(
        self,
        broker: MockKafkaBroker,
        group_id: str,
        dlq_topic: str = "dlq",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._broker = broker
        self._group_id = group_id
        self._dlq_topic = dlq_topic
        self._config = config or _CONFIG.get("consumer", {})

        # Subscribed topics
        self._subscribed_topics: List[str] = []

        # Current read positions per (topic, partition) — NOT yet committed.
        # Updated on each successful poll().
        self._positions: Dict[tuple, int] = {}

        # Last returned message (needed for single-call commit())
        self._last_message: Optional[MockMessage] = None

        # Round-robin cursor: (topic_index, partition_index)
        self._rr_topic_idx: int = 0
        self._rr_part_idx: int = 0

        self._closed: bool = False
        self._consumed_count: int = 0

        _logger.info(
            f"MockKafkaConsumer initialised: group_id={group_id!r}, "
            f"dlq_topic={dlq_topic!r}"
        )

    # ------------------------------------------------------------------
    # Subscription
    # ------------------------------------------------------------------

    def subscribe(self, topics: List[str]) -> None:
        """
        Subscribe to a list of topic names.

        Initialises read positions from the committed offsets for this
        consumer group, or from 0 (earliest) if none exist.
        """
        if self._closed:
            raise RuntimeError("Cannot subscribe on a closed consumer")

        self._subscribed_topics = list(topics)
        self._positions.clear()

        for topic in topics:
            if not self._broker.topic_exists(topic):
                _logger.warning(
                    f"Subscribing to non-existent topic {topic!r} — "
                    "it will be visible once created"
                )
                continue
            cfg = self._broker.get_topic_config(topic)
            for p in range(cfg.num_partitions):
                committed = self._broker.get_committed_offset(
                    self._group_id, topic, p
                )
                self._positions[(topic, p)] = committed

        _logger.info(
            f"Consumer subscribed: group={self._group_id!r}, "
            f"topics={topics!r}, initial_positions={self._positions}"
        )

    # ------------------------------------------------------------------
    # Poll
    # ------------------------------------------------------------------

    def poll(self, timeout: float = 1.0) -> Optional[MockMessage]:
        """
        Return the next available message across all subscribed partitions.

        Iterates through all (topic, partition) assignments in round-robin
        order.  If no message is available after a full scan, sleeps for
        *timeout* seconds then returns ``None``.

        Returns
        -------
        MockMessage or None
        """
        if self._closed:
            raise RuntimeError("Cannot poll a closed consumer")

        # Re-initialise positions for topics that may have been created after
        # subscribe() was called.
        for topic in self._subscribed_topics:
            if self._broker.topic_exists(topic):
                cfg = self._broker.get_topic_config(topic)
                for p in range(cfg.num_partitions):
                    if (topic, p) not in self._positions:
                        committed = self._broker.get_committed_offset(
                            self._group_id, topic, p
                        )
                        self._positions[(topic, p)] = committed

        assignments = [
            (t, p)
            for t in self._subscribed_topics
            if self._broker.topic_exists(t)
            for p in range(self._broker.get_topic_config(t).num_partitions)
        ]
        if not assignments:
            time.sleep(timeout)
            return None

        # Full scan for one available message
        for _ in range(len(assignments)):
            idx = self._rr_topic_idx % len(assignments)
            self._rr_topic_idx += 1
            topic, partition = assignments[idx]
            offset = self._positions.get((topic, partition), 0)
            messages = self._broker.fetch(topic, partition, offset, max_messages=1)
            if messages:
                msg = messages[0]
                # Advance local position (not committed yet)
                self._positions[(topic, partition)] = msg.offset + 1
                self._last_message = msg
                self._consumed_count += 1
                _logger.debug(
                    f"poll(): received topic={topic!r}, partition={partition}, "
                    f"offset={msg.offset}, consumed_total={self._consumed_count}"
                )
                return msg

        # Nothing available
        _logger.debug(
            f"poll(): no messages available, sleeping {timeout}s "
            f"(consumed_total={self._consumed_count})"
        )
        time.sleep(timeout)
        return None

    # ------------------------------------------------------------------
    # Commit
    # ------------------------------------------------------------------

    def commit(self) -> None:
        """
        Commit the current local positions to the broker for all assigned
        partitions.

        This implements the at-least-once guarantee: commit only after the
        message has been successfully processed (or intentionally skipped via
        DLQ routing).
        """
        if self._closed:
            raise RuntimeError("Cannot commit on a closed consumer")

        for (topic, partition), offset in self._positions.items():
            self._broker.commit_offset(
                self._group_id, topic, partition, offset
            )
        _logger.info(
            f"Offsets committed: group={self._group_id!r}, "
            f"positions={self._positions}"
        )

    def commit_message(self, message: MockMessage) -> None:
        """
        Commit the offset of a specific *message* (offset + 1).

        Useful when processing individual messages without tracking positions
        manually.
        """
        next_offset = message.offset + 1
        self._broker.commit_offset(
            self._group_id, message.topic, message.partition, next_offset
        )
        self._positions[(message.topic, message.partition)] = next_offset
        _logger.debug(
            f"Message-level commit: topic={message.topic!r}, "
            f"partition={message.partition}, committed_offset={next_offset}"
        )

    # ------------------------------------------------------------------
    # DLQ routing
    # ------------------------------------------------------------------

    def route_to_dlq(
        self,
        message: MockMessage,
        error: Exception,
        retry_count: int = 0,
    ) -> None:
        """
        Forward a failed *message* to the dead-letter queue topic.

        The DLQ message value is a JSON envelope containing:
        * ``original_topic``  : source topic
        * ``original_partition`` : source partition
        * ``original_offset`` : source offset
        * ``original_value``  : base64-decoded string of the original value
        * ``error``           : string representation of the exception
        * ``retry_count``     : how many times this message has been retried
        * ``routed_at``       : UTC timestamp
        """
        if not self._broker.topic_exists(self._dlq_topic):
            _logger.error(
                f"DLQ topic {self._dlq_topic!r} does not exist — "
                "message will be lost"
            )
            return

        try:
            original_value = message.value.decode("utf-8")
        except Exception:
            original_value = repr(message.value)

        envelope = {
            "original_topic": message.topic,
            "original_partition": message.partition,
            "original_offset": message.offset,
            "original_key": message.key.decode("utf-8") if message.key else None,
            "original_value": original_value,
            "error": str(error),
            "retry_count": retry_count,
            "routed_at": time.time(),
        }

        self._broker.produce(
            topic=self._dlq_topic,
            key=message.key,
            value=json.dumps(envelope).encode("utf-8"),
        )
        _logger.warning(
            f"Message routed to DLQ: original_topic={message.topic!r}, "
            f"original_offset={message.offset}, error={error!r}"
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Commit outstanding offsets and mark this consumer as closed."""
        if self._closed:
            return
        self.commit()
        self._closed = True
        _logger.info(
            f"MockKafkaConsumer closed: group={self._group_id!r}, "
            f"total_consumed={self._consumed_count}"
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def lag(self) -> Dict[str, Dict[int, int]]:
        """Return current consumer lag per topic / partition."""
        result: Dict[str, Dict[int, int]] = {}
        for topic in self._subscribed_topics:
            if self._broker.topic_exists(topic):
                result[topic] = self._broker.get_lag(self._group_id, topic)
        return result

    @property
    def consumed_count(self) -> int:
        return self._consumed_count


# ---------------------------------------------------------------------------
# Demo / main
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Demonstrate MockKafkaConsumer: consume messages, handle errors, route to DLQ.
    """
    from kafka_core.mock_kafka import MockKafkaBroker
    from kafka_core.producer import MockKafkaProducer

    _logger.info("=== MockKafkaConsumer demo start ===")

    broker = MockKafkaBroker()
    broker.create_topic("events", num_partitions=2, replication_factor=1)
    broker.create_topic("dlq", num_partitions=1, replication_factor=1)

    producer = MockKafkaProducer(broker=broker)

    # Produce 8 events, one of which will simulate a processing error
    events = [
        {"user_id": f"u{i}", "action": "click", "page": f"/page-{i}"}
        for i in range(8)
    ]
    for i, event in enumerate(events):
        producer.produce("events", key=event["user_id"], value=event)
    producer.flush()
    _logger.info(f"Produced {len(events)} events")

    consumer = MockKafkaConsumer(
        broker=broker,
        group_id="demo-consumer-group",
        dlq_topic="dlq",
    )
    consumer.subscribe(["events"])

    processed = 0
    failed = 0

    for _ in range(len(events)):
        msg = consumer.poll(timeout=0.05)
        if msg is None:
            break

        payload = json.loads(msg.value.decode())
        _logger.info(f"Processing: {payload}")

        # Simulate a processing error for every 4th message
        if processed % 4 == 3:
            exc = ValueError(f"Simulated processing failure for offset={msg.offset}")
            _logger.error(f"Processing failed: {exc}")
            consumer.route_to_dlq(msg, exc, retry_count=0)
            consumer.commit()
            failed += 1
        else:
            consumer.commit()
            processed += 1

    _logger.info(
        f"Consumer demo complete: processed={processed}, "
        f"routed_to_dlq={failed}, lag={consumer.lag()}"
    )
    consumer.close()
    _logger.info("=== MockKafkaConsumer demo complete ===")


if __name__ == "__main__":
    main()
