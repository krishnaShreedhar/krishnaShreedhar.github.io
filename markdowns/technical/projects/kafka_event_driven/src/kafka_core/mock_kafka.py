"""
mock_kafka.py — Thread-safe in-memory Kafka broker simulation.

Simulates the core Kafka broker responsibilities:
  * Topic and partition management
  * Message storage with monotonically increasing offsets per partition
  * Consumer group offset tracking (committed positions)
  * Consumer lag computation

Design decisions
----------------
* Each topic partition is stored as a plain Python list protected by a single
  per-broker ``threading.Lock``.  For a demo workload the coarse-grained lock
  is sufficient; a production-grade mock would use per-partition locks.
* Consumer group state is stored in a nested dict:
    _consumer_offsets[group_id][topic][partition] = committed_offset
* Messages are plain dicts so that no additional dependency on a Kafka client
  library is required when ``use_mock: true``.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# ---------------------------------------------------------------------------
# Bootstrap logging from config.yaml
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

    file_handler = logging.handlers.RotatingFileHandler(
        filename=str(log_file),
        maxBytes=log_cfg["max_bytes"],
        backupCount=log_cfg["backup_count"],
    )
    file_handler.setFormatter(_JSONFormatter())
    file_handler.setLevel(level)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(_JSONFormatter())
    stream_handler.setLevel(level)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


_CONFIG = _load_config()
_logger = _build_logger("kafka_core.mock_kafka", _CONFIG)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MockMessage:
    """Represents a single Kafka message stored in a partition."""
    topic: str
    partition: int
    offset: int
    key: Optional[bytes]
    value: bytes
    timestamp: float = field(default_factory=time.time)
    headers: Dict[str, str] = field(default_factory=dict)

    def __repr__(self) -> str:
        key_repr = self.key.decode() if self.key else None
        return (
            f"MockMessage(topic={self.topic!r}, partition={self.partition}, "
            f"offset={self.offset}, key={key_repr!r})"
        )


@dataclass
class PartitionState:
    """Mutable state for one topic partition."""
    messages: List[MockMessage] = field(default_factory=list)

    @property
    def log_end_offset(self) -> int:
        """Next offset that will be assigned (i.e. current high-water mark)."""
        return len(self.messages)


@dataclass
class TopicConfig:
    """Immutable topic-level configuration."""
    name: str
    num_partitions: int
    replication_factor: int
    retention_ms: int = 604_800_000  # 7 days


# ---------------------------------------------------------------------------
# MockKafkaBroker
# ---------------------------------------------------------------------------

class MockKafkaBroker:
    """
    Thread-safe in-memory Kafka broker.

    Responsibilities
    ----------------
    * Topic lifecycle: create / list / inspect
    * Message produce: atomically append and assign offset
    * Message consume: retrieve messages from a given offset onward
    * Offset commit: persist consumer group positions
    * Lag computation: high-water mark minus committed offset

    All state is protected by a single ``threading.Lock`` acquired on every
    mutating or reading operation.
    """

    def __init__(self) -> None:
        self._lock: threading.Lock = threading.Lock()
        # topic_name -> list[PartitionState]  (index == partition id)
        self._topics: Dict[str, List[PartitionState]] = {}
        # topic_name -> TopicConfig
        self._topic_configs: Dict[str, TopicConfig] = {}
        # group_id -> {topic -> {partition -> committed_offset}}
        self._consumer_offsets: Dict[str, Dict[str, Dict[int, int]]] = {}

        _logger.info("MockKafkaBroker initialised")

    # ------------------------------------------------------------------
    # Topic management
    # ------------------------------------------------------------------

    def create_topic(
        self,
        name: str,
        num_partitions: int = 1,
        replication_factor: int = 1,
        retention_ms: int = 604_800_000,
    ) -> None:
        """Create a topic with the given number of partitions."""
        with self._lock:
            if name in self._topics:
                _logger.warning(
                    f"Topic '{name}' already exists — skipping creation"
                )
                return
            self._topics[name] = [PartitionState() for _ in range(num_partitions)]
            self._topic_configs[name] = TopicConfig(
                name=name,
                num_partitions=num_partitions,
                replication_factor=replication_factor,
                retention_ms=retention_ms,
            )
        _logger.info(
            f"Topic created: name={name!r}, partitions={num_partitions}, "
            f"replication_factor={replication_factor}, retention_ms={retention_ms}"
        )

    def list_topics(self) -> List[str]:
        """Return sorted list of all topic names."""
        with self._lock:
            return sorted(self._topics.keys())

    def get_topic_config(self, name: str) -> TopicConfig:
        """Return the ``TopicConfig`` for *name*.  Raises ``KeyError`` if absent."""
        with self._lock:
            if name not in self._topic_configs:
                raise KeyError(f"Unknown topic: {name!r}")
            return self._topic_configs[name]

    def topic_exists(self, name: str) -> bool:
        with self._lock:
            return name in self._topics

    # ------------------------------------------------------------------
    # Produce
    # ------------------------------------------------------------------

    def produce(
        self,
        topic: str,
        key: Optional[bytes],
        value: bytes,
        partition: Optional[int] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> MockMessage:
        """
        Append *value* to *topic*.

        Partition selection
        -------------------
        * If *partition* is specified, the message is routed there.
        * Otherwise, if *key* is provided the partition is selected via
          ``hash(key) % num_partitions`` (deterministic routing).
        * Fallback: round-robin across partitions (tracked per-topic via a
          simple counter stored in the topic config — implemented as an
          attribute added at runtime to avoid complicating the dataclass).
        """
        with self._lock:
            if topic not in self._topics:
                raise RuntimeError(
                    f"Topic '{topic}' does not exist.  Create it before producing."
                )
            partitions = self._topics[topic]
            num_partitions = len(partitions)

            if partition is not None:
                if partition >= num_partitions:
                    raise ValueError(
                        f"Partition {partition} out of range for topic '{topic}' "
                        f"(num_partitions={num_partitions})"
                    )
                chosen = partition
            elif key is not None:
                chosen = hash(key) % num_partitions
            else:
                # round-robin counter stored as attribute on the config object
                cfg = self._topic_configs[topic]
                rr_attr = "_rr_counter"
                counter = getattr(cfg, rr_attr, 0)
                chosen = counter % num_partitions
                setattr(cfg, rr_attr, counter + 1)

            offset = partitions[chosen].log_end_offset
            msg = MockMessage(
                topic=topic,
                partition=chosen,
                offset=offset,
                key=key,
                value=value,
                headers=headers or {},
            )
            partitions[chosen].messages.append(msg)

        _logger.debug(
            f"Produced message: topic={topic!r}, partition={chosen}, offset={offset}"
        )
        return msg

    # ------------------------------------------------------------------
    # Consume
    # ------------------------------------------------------------------

    def fetch(
        self,
        topic: str,
        partition: int,
        offset: int,
        max_messages: int = 1,
    ) -> List[MockMessage]:
        """
        Return up to *max_messages* starting at *offset* from the given
        *topic* / *partition*.  Returns an empty list when the offset is at
        or beyond the log end.
        """
        with self._lock:
            if topic not in self._topics:
                raise RuntimeError(f"Unknown topic: {topic!r}")
            partition_state = self._topics[topic][partition]
            messages = partition_state.messages[offset : offset + max_messages]
        if messages:
            _logger.debug(
                f"Fetched {len(messages)} message(s) from "
                f"topic={topic!r}, partition={partition}, offset={offset}"
            )
        return messages

    def log_end_offset(self, topic: str, partition: int) -> int:
        """Return the next assignable offset (high-water mark)."""
        with self._lock:
            if topic not in self._topics:
                raise RuntimeError(f"Unknown topic: {topic!r}")
            return self._topics[topic][partition].log_end_offset

    # ------------------------------------------------------------------
    # Offset management
    # ------------------------------------------------------------------

    def commit_offset(
        self,
        group_id: str,
        topic: str,
        partition: int,
        offset: int,
    ) -> None:
        """Persist the committed offset for a consumer group / topic / partition."""
        with self._lock:
            group = self._consumer_offsets.setdefault(group_id, {})
            topic_offsets = group.setdefault(topic, {})
            topic_offsets[partition] = offset
        _logger.debug(
            f"Offset committed: group={group_id!r}, topic={topic!r}, "
            f"partition={partition}, offset={offset}"
        )

    def get_committed_offset(
        self, group_id: str, topic: str, partition: int
    ) -> int:
        """Return committed offset, defaulting to 0 (earliest) if never committed."""
        with self._lock:
            return (
                self._consumer_offsets
                .get(group_id, {})
                .get(topic, {})
                .get(partition, 0)
            )

    # ------------------------------------------------------------------
    # Lag
    # ------------------------------------------------------------------

    def get_lag(self, group_id: str, topic: str) -> Dict[int, int]:
        """
        Return per-partition consumer lag for *group_id* on *topic*.

        lag[p] = log_end_offset[p] - committed_offset[p]
        """
        with self._lock:
            if topic not in self._topics:
                raise RuntimeError(f"Unknown topic: {topic!r}")
            num_partitions = len(self._topics[topic])
            result: Dict[int, int] = {}
            for p in range(num_partitions):
                hw = self._topics[topic][p].log_end_offset
                committed = (
                    self._consumer_offsets
                    .get(group_id, {})
                    .get(topic, {})
                    .get(p, 0)
                )
                result[p] = hw - committed
        _logger.debug(
            f"Lag computed: group={group_id!r}, topic={topic!r}, lag={result}"
        )
        return result


# ---------------------------------------------------------------------------
# Module-level singleton for shared use across demos
# ---------------------------------------------------------------------------

_BROKER: Optional[MockKafkaBroker] = None


def get_broker() -> MockKafkaBroker:
    """Return (or lazily create) the module-level MockKafkaBroker singleton."""
    global _BROKER
    if _BROKER is None:
        _BROKER = MockKafkaBroker()
    return _BROKER


def reset_broker() -> None:
    """Destroy and reset the singleton (useful between demo runs)."""
    global _BROKER
    _BROKER = None
    _logger.info("MockKafkaBroker singleton reset")


# ---------------------------------------------------------------------------
# Demo / main
# ---------------------------------------------------------------------------

def main() -> None:
    """Demonstrate basic broker operations: create topic, produce, fetch, commit."""
    _logger.info("=== MockKafkaBroker demo start ===")

    broker = MockKafkaBroker()

    # Create topics
    broker.create_topic("orders", num_partitions=3, replication_factor=1)
    broker.create_topic("payments", num_partitions=2, replication_factor=1)
    _logger.info(f"Topics: {broker.list_topics()}")

    # Produce messages
    for i in range(6):
        key = f"user-{i % 3}".encode()
        value = json.dumps({"order_id": i, "amount": (i + 1) * 10.0}).encode()
        msg = broker.produce("orders", key=key, value=value)
        _logger.info(f"Produced → {msg}")

    # Fetch and consume
    group = "demo-group"
    for partition in range(3):
        committed = broker.get_committed_offset(group, "orders", partition)
        messages = broker.fetch("orders", partition, committed, max_messages=10)
        for m in messages:
            _logger.info(f"Consumed ← {m}")
            broker.commit_offset(group, "orders", partition, m.offset + 1)

    # Show lag (should be 0 after full consumption)
    lag = broker.get_lag(group, "orders")
    _logger.info(f"Lag after commit: {lag}")

    _logger.info("=== MockKafkaBroker demo complete ===")


if __name__ == "__main__":
    main()
