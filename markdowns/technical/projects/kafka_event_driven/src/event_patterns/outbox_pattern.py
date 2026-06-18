"""
outbox_pattern.py — Transactional Outbox Pattern with polling relay.

Concept
-------
The Transactional Outbox Pattern solves the dual-write problem: reliably
publishing a Kafka event *and* updating a database in a single atomic
operation.

Instead of writing to Kafka directly from the business transaction, the
application writes a record to an ``outbox`` table *in the same DB transaction*
as the domain update.  A separate ``OutboxRelay`` process polls the outbox,
publishes pending entries to Kafka, and marks them as published.

This guarantees:
  * If the DB transaction rolls back, no event is published.
  * If the relay crashes after publishing but before marking, the entry is
    re-published on restart (at-least-once delivery).

Components
----------
OutboxEntry  : Single outbox record (id, topic, key, value, published flag).
OutboxTable  : In-memory database simulation for the outbox table.
OutboxRelay  : Polls OutboxTable, publishes to Kafka, marks published.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import yaml

from kafka_core.mock_kafka import MockKafkaBroker
from kafka_core.producer import MockKafkaProducer

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
_logger = _build_logger("event_patterns.outbox_pattern", _CONFIG)


# ---------------------------------------------------------------------------
# OutboxEntry
# ---------------------------------------------------------------------------

@dataclass
class OutboxEntry:
    """
    A single row in the outbox table.

    Attributes
    ----------
    id         : Unique entry identifier (UUID4).
    topic      : Kafka topic to publish to.
    key        : Message key (string).
    value      : Message payload dict.
    created_at : Unix epoch timestamp when the entry was written.
    published  : True once the relay has published and acknowledged the entry.
    published_at : Unix epoch timestamp of successful publish.
    retry_count  : Number of publish attempts (incremented by relay on retry).
    """
    id: str
    topic: str
    key: str
    value: Dict[str, Any]
    created_at: float = field(default_factory=time.time)
    published: bool = False
    published_at: Optional[float] = None
    retry_count: int = 0

    @classmethod
    def create(
        cls,
        topic: str,
        key: str,
        value: Dict[str, Any],
    ) -> "OutboxEntry":
        """Factory: auto-generate id and created_at."""
        return cls(
            id=str(uuid.uuid4()),
            topic=topic,
            key=key,
            value=value,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "topic": self.topic,
            "key": self.key,
            "value": self.value,
            "created_at": self.created_at,
            "published": self.published,
            "published_at": self.published_at,
            "retry_count": self.retry_count,
        }


# ---------------------------------------------------------------------------
# OutboxTable  (simulated DB table)
# ---------------------------------------------------------------------------

class OutboxTable:
    """
    In-memory simulation of the outbox DB table.

    Provides the same atomicity guarantee as a real SQL outbox table:
    ``add_entry`` simulates writing the outbox row inside the same DB
    transaction as the domain update (represented here as a fake
    ``domain_db`` dict).

    Parameters
    ----------
    domain_db : Mutable dict simulating the application's main DB.
    """

    def __init__(self, domain_db: Optional[Dict[str, Any]] = None) -> None:
        self._lock = threading.Lock()
        self._entries: Dict[str, OutboxEntry] = {}
        self._domain_db: Dict[str, Any] = domain_db if domain_db is not None else {}
        self._total_added: int = 0
        _logger.info("OutboxTable initialised")

    def atomic_write(
        self,
        domain_key: str,
        domain_value: Any,
        outbox_entry: OutboxEntry,
    ) -> None:
        """
        Simulate an atomic DB transaction:
          1. Write *domain_value* to the application domain table.
          2. Insert *outbox_entry* into the outbox table.

        Both writes happen under the same lock, ensuring consistency.
        In a real application this would be a single SQL transaction.
        """
        with self._lock:
            # Step 1: domain write
            self._domain_db[domain_key] = domain_value
            # Step 2: outbox write (same "transaction")
            self._entries[outbox_entry.id] = outbox_entry
            self._total_added += 1

        _logger.info(
            f"OutboxTable.atomic_write: domain_key={domain_key!r}, "
            f"outbox_id={outbox_entry.id!r}, topic={outbox_entry.topic!r}"
        )

    def add_entry(self, entry: OutboxEntry) -> None:
        """Add a single outbox entry (without an explicit domain write)."""
        with self._lock:
            self._entries[entry.id] = entry
            self._total_added += 1
        _logger.debug(
            f"OutboxTable.add_entry: id={entry.id!r}, topic={entry.topic!r}"
        )

    def get_unpublished(self, limit: int = 100) -> List[OutboxEntry]:
        """Return up to *limit* unpublished entries ordered by creation time."""
        with self._lock:
            unpublished = [
                e for e in self._entries.values() if not e.published
            ]
        unpublished.sort(key=lambda e: e.created_at)
        result = unpublished[:limit]
        if result:
            _logger.debug(
                f"get_unpublished: found {len(result)} unpublished entries "
                f"(total_unpublished={len(unpublished)})"
            )
        return result

    def mark_published(self, entry_id: str) -> None:
        """Mark an outbox entry as published."""
        with self._lock:
            if entry_id not in self._entries:
                raise KeyError(f"Outbox entry {entry_id!r} not found")
            self._entries[entry_id].published = True
            self._entries[entry_id].published_at = time.time()
        _logger.debug(f"OutboxTable.mark_published: id={entry_id!r}")

    def increment_retry(self, entry_id: str) -> None:
        """Increment the retry count for a failed publish attempt."""
        with self._lock:
            if entry_id in self._entries:
                self._entries[entry_id].retry_count += 1

    def stats(self) -> Dict[str, int]:
        with self._lock:
            total = len(self._entries)
            published = sum(1 for e in self._entries.values() if e.published)
        return {
            "total": total,
            "published": published,
            "pending": total - published,
        }

    @property
    def domain_db(self) -> Dict[str, Any]:
        return dict(self._domain_db)


# ---------------------------------------------------------------------------
# OutboxRelay
# ---------------------------------------------------------------------------

class OutboxRelay:
    """
    Polls the OutboxTable and publishes pending entries to Kafka.

    The relay runs in a background thread.  On each poll cycle it:
      1. Fetches unpublished entries from the outbox table.
      2. Publishes each to the Kafka broker via MockKafkaProducer.
      3. Marks each as published in the outbox table.

    If publishing fails, the retry count is incremented and the entry
    remains pending for the next poll cycle.

    Parameters
    ----------
    outbox_table     : The ``OutboxTable`` to poll.
    broker           : Shared ``MockKafkaBroker`` instance.
    poll_interval_s  : Seconds between polling cycles.
    batch_size       : Max entries to process per cycle.
    """

    def __init__(
        self,
        outbox_table: OutboxTable,
        broker: MockKafkaBroker,
        poll_interval_s: float = 1.0,
        batch_size: int = 100,
    ) -> None:
        self._outbox = outbox_table
        self._broker = broker
        self._producer = MockKafkaProducer(broker=broker)
        self._poll_interval = poll_interval_s
        self._batch_size = batch_size
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._published_count: int = 0
        self._failed_count: int = 0

        _logger.info(
            f"OutboxRelay initialised: poll_interval={poll_interval_s}s, "
            f"batch_size={batch_size}"
        )

    def start(self) -> None:
        """Start the relay background thread."""
        if self._running:
            _logger.warning("OutboxRelay.start() called but relay is already running")
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._relay_loop,
            name="outbox-relay",
            daemon=True,
        )
        self._thread.start()
        _logger.info("OutboxRelay started (background thread)")

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the relay and wait for the background thread to finish."""
        _logger.info("OutboxRelay stopping...")
        self._running = False
        if self._thread:
            self._thread.join(timeout=timeout)
        _logger.info(
            f"OutboxRelay stopped: published={self._published_count}, "
            f"failed_attempts={self._failed_count}"
        )

    def run_once(self) -> Tuple[int, int]:
        """
        Execute a single relay cycle synchronously.

        Returns
        -------
        (published_count, failed_count) for this cycle.
        """
        return self._process_batch()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _relay_loop(self) -> None:
        _logger.info("OutboxRelay relay loop started")
        while self._running:
            self._process_batch()
            time.sleep(self._poll_interval)
        _logger.info("OutboxRelay relay loop exited")

    def _process_batch(self) -> tuple:
        entries = self._outbox.get_unpublished(limit=self._batch_size)
        if not entries:
            return (0, 0)

        _logger.info(
            f"OutboxRelay processing batch of {len(entries)} entries"
        )
        cycle_published = 0
        cycle_failed = 0

        for entry in entries:
            try:
                # Ensure the topic exists on the broker
                if not self._broker.topic_exists(entry.topic):
                    self._broker.create_topic(
                        entry.topic, num_partitions=2, replication_factor=1
                    )
                self._producer.produce(
                    topic=entry.topic,
                    key=entry.key,
                    value=entry.value,
                )
                self._outbox.mark_published(entry.id)
                self._published_count += 1
                cycle_published += 1
                _logger.info(
                    f"OutboxRelay published: id={entry.id!r}, "
                    f"topic={entry.topic!r}, key={entry.key!r}"
                )
            except Exception as exc:
                self._outbox.increment_retry(entry.id)
                self._failed_count += 1
                cycle_failed += 1
                _logger.error(
                    f"OutboxRelay publish failed: id={entry.id!r}, "
                    f"topic={entry.topic!r}, error={exc!r}"
                )

        self._producer.flush()
        return (cycle_published, cycle_failed)

    @property
    def published_count(self) -> int:
        return self._published_count

    @property
    def failed_count(self) -> int:
        return self._failed_count


# ---------------------------------------------------------------------------
# Demo / main
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Demonstrate the Transactional Outbox Pattern:
      * Write domain records + outbox entries atomically.
      * Run OutboxRelay to publish pending entries to Kafka.
      * Verify all entries are published.
    """
    from kafka_core.mock_kafka import MockKafkaBroker

    _logger.info("=== OutboxPattern demo start ===")

    broker = MockKafkaBroker()
    broker.create_topic("user_events", num_partitions=4, replication_factor=1)
    broker.create_topic("payments", num_partitions=2, replication_factor=1)

    domain_db: Dict[str, Any] = {}
    outbox = OutboxTable(domain_db=domain_db)
    relay = OutboxRelay(outbox_table=outbox, broker=broker, poll_interval_s=0.5)

    # --- Simulate 5 domain operations with atomic outbox writes ---
    orders = [
        {
            "order_id": f"ORD-{i:03d}",
            "customer_id": f"CUST-{i % 3 + 1}",
            "amount": round(10.0 * (i + 1), 2),
            "status": "created",
        }
        for i in range(5)
    ]

    for order in orders:
        entry = OutboxEntry.create(
            topic="user_events",
            key=order["customer_id"],
            value={"event_type": "OrderCreated", "payload": order},
        )
        outbox.atomic_write(
            domain_key=f"order:{order['order_id']}",
            domain_value=order,
            outbox_entry=entry,
        )
        _logger.info(
            f"Domain write + outbox entry: order_id={order['order_id']!r}"
        )

    # Add a payment event (different topic)
    payment_entry = OutboxEntry.create(
        topic="payments",
        key="CUST-1",
        value={"event_type": "PaymentInitiated", "amount": 100.00},
    )
    outbox.add_entry(payment_entry)

    _logger.info(f"OutboxTable stats before relay: {outbox.stats()}")
    _logger.info(f"Domain DB size: {len(outbox.domain_db)} records")

    # --- Run relay synchronously ---
    published, failed = relay.run_once()
    _logger.info(
        f"Relay cycle: published={published}, failed={failed}"
    )

    _logger.info(f"OutboxTable stats after relay: {outbox.stats()}")

    # --- Verify all published ---
    remaining = outbox.get_unpublished()
    assert len(remaining) == 0, f"Expected 0 pending entries, got {len(remaining)}"

    # --- Check Kafka topic has the messages ---
    lag = broker.get_lag("__none__", "user_events")
    _logger.info(f"user_events high-water marks: {lag}")

    # --- Demonstrate background relay ---
    relay.start()
    extra_entry = OutboxEntry.create(
        topic="user_events",
        key="CUST-5",
        value={"event_type": "LateEvent", "note": "Published via background relay"},
    )
    outbox.add_entry(extra_entry)
    _logger.info("Added extra entry for background relay to pick up...")
    time.sleep(1.2)
    relay.stop()

    _logger.info(f"Total published by relay: {relay.published_count}")
    _logger.info(f"Final outbox stats: {outbox.stats()}")
    _logger.info("=== OutboxPattern demo complete ===")


if __name__ == "__main__":
    main()
