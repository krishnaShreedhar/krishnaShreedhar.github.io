"""
topic_manager.py — TopicManager for creating and inspecting broker topics.

Responsibilities
----------------
* Provide a clean, high-level API for topic lifecycle management.
* Read topic defaults from ``config.yaml`` (topics section).
* Log every create / list / inspect operation at appropriate levels.
* Support bulk creation from config (``create_from_config()``).

This class acts as the administrative client analogue; in a real Kafka setup
it would wrap the ``confluent_kafka.admin.AdminClient``.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from kafka_core.mock_kafka import MockKafkaBroker, TopicConfig

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
_logger = _build_logger("kafka_core.topic_manager", _CONFIG)


# ---------------------------------------------------------------------------
# TopicManager
# ---------------------------------------------------------------------------

class TopicManager:
    """
    High-level topic administration for the MockKafkaBroker.

    Parameters
    ----------
    broker : The ``MockKafkaBroker`` instance to administer.

    Methods
    -------
    create_topic(name, partitions, replication_factor, retention_ms)
    create_from_config()
        Reads ``topics:`` section of config.yaml and creates all listed topics.
    list_topics() -> List[str]
    get_topic_info(name) -> dict
    topic_exists(name) -> bool
    describe_all() -> List[dict]
    """

    def __init__(self, broker: MockKafkaBroker) -> None:
        self._broker = broker
        _logger.info("TopicManager initialised")

    # ------------------------------------------------------------------
    # Creation
    # ------------------------------------------------------------------

    def create_topic(
        self,
        name: str,
        partitions: int = 1,
        replication_factor: int = 1,
        retention_ms: int = 604_800_000,
    ) -> bool:
        """
        Create a topic on the broker.

        Returns
        -------
        True if the topic was freshly created, False if it already existed.
        """
        _logger.info(
            f"create_topic: name={name!r}, partitions={partitions}, "
            f"replication_factor={replication_factor}, retention_ms={retention_ms}"
        )
        existed = self._broker.topic_exists(name)
        if existed:
            _logger.warning(
                f"Topic {name!r} already exists — skipping (idempotent create)"
            )
            return False

        self._broker.create_topic(
            name=name,
            num_partitions=partitions,
            replication_factor=replication_factor,
            retention_ms=retention_ms,
        )
        _logger.info(f"Topic {name!r} created successfully")
        return True

    def create_from_config(self) -> List[str]:
        """
        Read the ``topics:`` section of config.yaml and create all topics.

        Returns the names of topics that were freshly created (skips existing).
        """
        topics_cfg = _CONFIG.get("topics", {})
        created: List[str] = []

        _logger.info(
            f"create_from_config: found {len(topics_cfg)} topic(s) in config"
        )

        for topic_name, topic_opts in topics_cfg.items():
            partitions = topic_opts.get("partitions", 1)
            replication_factor = topic_opts.get("replication_factor", 1)
            retention_ms = topic_opts.get("retention_ms", 604_800_000)

            newly_created = self.create_topic(
                name=topic_name,
                partitions=partitions,
                replication_factor=replication_factor,
                retention_ms=retention_ms,
            )
            if newly_created:
                created.append(topic_name)

        _logger.info(
            f"create_from_config complete: created={created}"
        )
        return created

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def list_topics(self) -> List[str]:
        """Return sorted list of all topic names on the broker."""
        topics = self._broker.list_topics()
        _logger.info(f"list_topics: {topics}")
        return topics

    def topic_exists(self, name: str) -> bool:
        """Return True if *name* exists on the broker."""
        exists = self._broker.topic_exists(name)
        _logger.debug(f"topic_exists: name={name!r}, result={exists}")
        return exists

    def get_topic_info(self, name: str) -> Dict[str, Any]:
        """
        Return a dict describing the topic configuration.

        Raises ``KeyError`` if the topic does not exist.
        """
        cfg: TopicConfig = self._broker.get_topic_config(name)
        info = {
            "name": cfg.name,
            "num_partitions": cfg.num_partitions,
            "replication_factor": cfg.replication_factor,
            "retention_ms": cfg.retention_ms,
        }
        _logger.info(f"get_topic_info: {info}")
        return info

    def describe_all(self) -> List[Dict[str, Any]]:
        """Return a list of topic info dicts for every topic on the broker."""
        all_info = []
        for name in self._broker.list_topics():
            try:
                info = self.get_topic_info(name)
                all_info.append(info)
            except KeyError:
                _logger.error(
                    f"describe_all: topic {name!r} disappeared between list and describe"
                )
        _logger.info(f"describe_all: returned {len(all_info)} topic(s)")
        return all_info

    def delete_topic(self, name: str) -> None:
        """
        Log a warning that deletion is not supported in the mock broker.

        In a real Kafka AdminClient this would call
        ``AdminClient.delete_topics()``.  The mock broker does not support
        deletion to keep the implementation simple and avoid data-loss surprises
        in demos.
        """
        _logger.warning(
            f"delete_topic({name!r}): MockKafkaBroker does not support topic "
            "deletion — operation is a no-op in mock mode"
        )


# ---------------------------------------------------------------------------
# Demo / main
# ---------------------------------------------------------------------------

def main() -> None:
    """Demonstrate TopicManager: create topics from config, list, and inspect."""
    from kafka_core.mock_kafka import MockKafkaBroker

    _logger.info("=== TopicManager demo start ===")

    broker = MockKafkaBroker()
    manager = TopicManager(broker=broker)

    # Create topics defined in config.yaml
    created = manager.create_from_config()
    _logger.info(f"Topics created from config: {created}")

    # Create an additional ad-hoc topic
    manager.create_topic(
        name="ad_hoc_topic",
        partitions=2,
        replication_factor=1,
        retention_ms=3_600_000,  # 1 hour
    )

    # List all topics
    all_topics = manager.list_topics()
    _logger.info(f"All topics on broker: {all_topics}")

    # Describe each topic
    for topic_info in manager.describe_all():
        _logger.info(f"Topic info: {topic_info}")

    # Idempotent re-creation
    manager.create_topic("user_events", partitions=4)

    # Attempted deletion (no-op)
    manager.delete_topic("ad_hoc_topic")

    _logger.info("=== TopicManager demo complete ===")


if __name__ == "__main__":
    main()
