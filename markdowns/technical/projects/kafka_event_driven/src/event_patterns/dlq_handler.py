"""
dlq_handler.py — Dead Letter Queue with exponential backoff retry.

Concept
-------
A Dead Letter Queue (DLQ) captures messages that could not be processed
after N attempts.  Instead of blocking the main consumer loop, failed
messages are forwarded to the DLQ topic.  A dedicated DLQ handler retries
them with exponential backoff.  After exceeding the maximum retry count,
messages are quarantined and require manual intervention.

Backoff formula
---------------
  next_retry_at = now + backoff_base ^ retry_count

  E.g. with backoff_base=2.0:
    retry 0 → +1s  (2^0)
    retry 1 → +2s  (2^1)
    retry 2 → +4s  (2^2)
    retry 3 → quarantine

Components
----------
DLQMessage  : Wraps the original MockMessage with retry metadata.
DLQHandler  : Manages the DLQ: add, retry with backoff, quarantine.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

from kafka_core.mock_kafka import MockKafkaBroker, MockMessage
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
_logger = _build_logger("event_patterns.dlq_handler", _CONFIG)


# ---------------------------------------------------------------------------
# DLQMessage
# ---------------------------------------------------------------------------

@dataclass
class DLQMessage:
    """
    Wraps an original ``MockMessage`` with DLQ tracking metadata.

    Attributes
    ----------
    original_message  : The raw MockMessage that failed processing.
    original_error    : String description of why processing failed.
    retry_count       : Number of times this message has been retried.
    next_retry_at     : Unix timestamp when this message should next be retried.
    quarantined       : True when retry_count >= max_retries.
    added_at          : When this entry was first added to the DLQ.
    last_error        : The most recent error (may differ from original_error).
    """
    original_message: MockMessage
    original_error: str
    retry_count: int = 0
    next_retry_at: float = field(default_factory=time.time)
    quarantined: bool = False
    added_at: float = field(default_factory=time.time)
    last_error: str = ""

    def __post_init__(self) -> None:
        if not self.last_error:
            self.last_error = self.original_error

    def is_ready_for_retry(self) -> bool:
        """Return True if the message's next_retry_at is in the past."""
        return not self.quarantined and time.time() >= self.next_retry_at

    def to_dict(self) -> Dict[str, Any]:
        msg = self.original_message
        return {
            "original_topic": msg.topic,
            "original_partition": msg.partition,
            "original_offset": msg.offset,
            "original_key": msg.key.decode() if msg.key else None,
            "original_error": self.original_error,
            "last_error": self.last_error,
            "retry_count": self.retry_count,
            "next_retry_at": self.next_retry_at,
            "quarantined": self.quarantined,
            "added_at": self.added_at,
        }


# ---------------------------------------------------------------------------
# DLQHandler
# ---------------------------------------------------------------------------

class DLQHandler:
    """
    Manages the Dead Letter Queue: ingestion, retry, quarantine.

    Parameters
    ----------
    max_retries      : Maximum retry attempts before quarantine.
    backoff_base_s   : Base for exponential backoff in seconds.
    processor        : Optional callable ``(MockMessage) -> None`` that
                       attempts to re-process a message.  If not provided,
                       a no-op that always succeeds is used (for testing).
    broker           : Optional broker for publishing quarantine notifications.
    quarantine_topic : Topic to publish quarantined messages to.
    """

    def __init__(
        self,
        max_retries: int = 3,
        backoff_base_s: float = 2.0,
        processor: Optional[Callable[[MockMessage], None]] = None,
        broker: Optional[MockKafkaBroker] = None,
        quarantine_topic: str = "dlq",
    ) -> None:
        self._max_retries = max_retries
        self._backoff_base = backoff_base_s
        self._processor = processor
        self._broker = broker
        self._quarantine_topic = quarantine_topic
        self._producer: Optional[MockKafkaProducer] = (
            MockKafkaProducer(broker=broker) if broker else None
        )

        # Internal DLQ storage: list to maintain order
        self._queue: List[DLQMessage] = []
        self._quarantine: List[DLQMessage] = []

        self._stats = {
            "total_added": 0,
            "total_retried": 0,
            "total_succeeded": 0,
            "total_quarantined": 0,
        }

        _logger.info(
            f"DLQHandler initialised: max_retries={max_retries}, "
            f"backoff_base_s={backoff_base_s}"
        )

    # ------------------------------------------------------------------
    # Add to DLQ
    # ------------------------------------------------------------------

    def add_to_dlq(
        self,
        message: MockMessage,
        error: Exception,
        retry_count: int = 0,
    ) -> DLQMessage:
        """
        Add a failed *message* to the DLQ.

        The ``next_retry_at`` is set immediately (no delay for first entry
        unless retry_count > 0).

        Parameters
        ----------
        message      : The original MockMessage that failed.
        error        : The exception that caused the failure.
        retry_count  : If this message is being re-added after a retry, pass
                       the current retry count so backoff is computed correctly.

        Returns
        -------
        The newly created DLQMessage.
        """
        backoff_delay = self._backoff_base ** retry_count
        dlq_msg = DLQMessage(
            original_message=message,
            original_error=str(error),
            retry_count=retry_count,
            next_retry_at=time.time() + backoff_delay,
        )

        if retry_count >= self._max_retries:
            dlq_msg.quarantined = True
            self._quarantine.append(dlq_msg)
            self._stats["total_quarantined"] += 1
            self._publish_quarantine_notification(dlq_msg)
            _logger.error(
                f"Message QUARANTINED: topic={message.topic!r}, "
                f"offset={message.offset}, retry_count={retry_count}, "
                f"error={error!r}"
            )
        else:
            self._queue.append(dlq_msg)
            self._stats["total_added"] += 1
            _logger.warning(
                f"Message added to DLQ: topic={message.topic!r}, "
                f"offset={message.offset}, retry_count={retry_count}, "
                f"next_retry_in={backoff_delay:.1f}s, error={error!r}"
            )

        return dlq_msg

    # ------------------------------------------------------------------
    # Process DLQ
    # ------------------------------------------------------------------

    def process_dlq(self) -> Dict[str, int]:
        """
        Process all DLQ messages that are ready for retry.

        Returns
        -------
        Dict with counts: ``succeeded``, ``re_queued``, ``quarantined``.
        """
        now = time.time()
        ready = [m for m in self._queue if m.is_ready_for_retry()]
        not_ready = [m for m in self._queue if not m.is_ready_for_retry()]

        if not ready:
            _logger.debug(
                f"process_dlq: no messages ready (queue_size={len(self._queue)})"
            )
            return {"succeeded": 0, "re_queued": 0, "quarantined": 0}

        _logger.info(
            f"process_dlq: processing {len(ready)} ready message(s) "
            f"({len(not_ready)} still in backoff)"
        )

        cycle_succeeded = 0
        cycle_re_queued = 0
        cycle_quarantined = 0
        new_queue = list(not_ready)

        for dlq_msg in ready:
            self._stats["total_retried"] += 1
            _logger.info(
                f"Retrying: topic={dlq_msg.original_message.topic!r}, "
                f"offset={dlq_msg.original_message.offset}, "
                f"attempt={dlq_msg.retry_count + 1}/{self._max_retries}"
            )

            try:
                if self._processor:
                    self._processor(dlq_msg.original_message)
                # Success
                self._stats["total_succeeded"] += 1
                cycle_succeeded += 1
                _logger.info(
                    f"Retry SUCCESS: topic={dlq_msg.original_message.topic!r}, "
                    f"offset={dlq_msg.original_message.offset}"
                )
            except Exception as exc:
                dlq_msg.retry_count += 1
                dlq_msg.last_error = str(exc)

                if dlq_msg.retry_count >= self._max_retries:
                    dlq_msg.quarantined = True
                    self._quarantine.append(dlq_msg)
                    self._stats["total_quarantined"] += 1
                    cycle_quarantined += 1
                    self._publish_quarantine_notification(dlq_msg)
                    _logger.error(
                        f"Message QUARANTINED after {dlq_msg.retry_count} retries: "
                        f"topic={dlq_msg.original_message.topic!r}, "
                        f"offset={dlq_msg.original_message.offset}"
                    )
                else:
                    # Re-queue with increased backoff
                    backoff_delay = self._backoff_base ** dlq_msg.retry_count
                    dlq_msg.next_retry_at = time.time() + backoff_delay
                    new_queue.append(dlq_msg)
                    cycle_re_queued += 1
                    _logger.warning(
                        f"Retry FAILED, re-queued with backoff={backoff_delay:.1f}s: "
                        f"topic={dlq_msg.original_message.topic!r}, "
                        f"offset={dlq_msg.original_message.offset}, "
                        f"error={exc!r}"
                    )

        self._queue = new_queue
        result = {
            "succeeded": cycle_succeeded,
            "re_queued": cycle_re_queued,
            "quarantined": cycle_quarantined,
        }
        _logger.info(f"process_dlq cycle result: {result}")
        return result

    # ------------------------------------------------------------------
    # Stats and inspection
    # ------------------------------------------------------------------

    def get_dlq_stats(self) -> Dict[str, Any]:
        """Return DLQ statistics snapshot."""
        stats = dict(self._stats)
        stats["queue_size"] = len(self._queue)
        stats["quarantine_size"] = len(self._quarantine)
        stats["ready_for_retry"] = sum(
            1 for m in self._queue if m.is_ready_for_retry()
        )
        _logger.info(f"DLQ stats: {stats}")
        return stats

    def list_queue(self) -> List[Dict[str, Any]]:
        """Return serialised view of all messages currently in the DLQ queue."""
        return [m.to_dict() for m in self._queue]

    def list_quarantine(self) -> List[Dict[str, Any]]:
        """Return serialised view of all quarantined messages."""
        return [m.to_dict() for m in self._quarantine]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _publish_quarantine_notification(self, dlq_msg: DLQMessage) -> None:
        """Publish a quarantine notification to the DLQ Kafka topic (if broker set)."""
        if self._producer is None or self._broker is None:
            return
        if not self._broker.topic_exists(self._quarantine_topic):
            self._broker.create_topic(
                self._quarantine_topic, num_partitions=1, replication_factor=1
            )
        notification = dlq_msg.to_dict()
        notification["quarantine_notification"] = True
        self._producer.produce(
            topic=self._quarantine_topic,
            key=dlq_msg.original_message.key,
            value=notification,
        )
        self._producer.flush()
        _logger.debug(
            f"Quarantine notification published for offset="
            f"{dlq_msg.original_message.offset}"
        )


# ---------------------------------------------------------------------------
# Demo / main
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Demonstrate DLQ with exponential backoff:
      * Generate 10 messages, 4 of which always fail processing.
      * Show messages moving through DLQ → retry → quarantine lifecycle.
    """
    from kafka_core.mock_kafka import MockKafkaBroker, MockMessage

    _logger.info("=== DLQHandler demo start ===")

    cfg = _CONFIG.get("event_patterns", {})
    max_retries: int = cfg.get("dlq_max_retries", 3)
    backoff_base: float = cfg.get("dlq_backoff_base_s", 2.0)

    broker = MockKafkaBroker()
    broker.create_topic("user_events", num_partitions=2, replication_factor=1)
    broker.create_topic("dlq", num_partitions=1, replication_factor=1)

    # Track which message keys are "always bad" (simulate persistent failure)
    always_fail_keys = {"bad-msg-1", "bad-msg-2"}

    def flaky_processor(msg: MockMessage) -> None:
        """Fail deterministically for certain keys."""
        key = msg.key.decode() if msg.key else "unknown"
        if key in always_fail_keys:
            raise ValueError(f"Permanent processing failure for key={key!r}")
        _logger.info(f"Successfully processed message key={key!r}")

    handler = DLQHandler(
        max_retries=max_retries,
        backoff_base_s=0.01,  # use tiny backoff so demo runs quickly
        processor=flaky_processor,
        broker=broker,
        quarantine_topic="dlq",
    )

    # Build mock messages (simulating what would come from the broker)
    def make_msg(topic: str, key: str, offset: int) -> MockMessage:
        return MockMessage(
            topic=topic,
            partition=0,
            offset=offset,
            key=key.encode(),
            value=json.dumps({"key": key, "offset": offset}).encode(),
        )

    # Add messages to DLQ (some recoverable, some permanent failures)
    messages = [
        make_msg("user_events", "good-msg-1", 0),
        make_msg("user_events", "bad-msg-1", 1),
        make_msg("user_events", "good-msg-2", 2),
        make_msg("user_events", "bad-msg-2", 3),
        make_msg("user_events", "good-msg-3", 4),
    ]

    for msg in messages:
        key = msg.key.decode()
        if key in always_fail_keys:
            handler.add_to_dlq(msg, ValueError(f"Initial failure for {key!r}"))
        # good messages don't go to DLQ initially

    _logger.info(f"Initial DLQ stats: {handler.get_dlq_stats()}")

    # Process DLQ across multiple cycles
    for cycle in range(max_retries + 2):
        _logger.info(f"--- DLQ processing cycle {cycle + 1} ---")
        result = handler.process_dlq()
        _logger.info(f"Cycle {cycle + 1} result: {result}")
        stats = handler.get_dlq_stats()
        _logger.info(f"Stats after cycle {cycle + 1}: {stats}")

        if stats["queue_size"] == 0 and stats["ready_for_retry"] == 0:
            _logger.info("DLQ queue drained — stopping processing cycles")
            break

        # Tiny sleep between cycles (in real usage this would be seconds)
        time.sleep(0.05)

    final_stats = handler.get_dlq_stats()
    _logger.info(f"Final DLQ stats: {final_stats}")
    _logger.info(f"Quarantined messages: {handler.list_quarantine()}")

    assert final_stats["queue_size"] == 0
    assert final_stats["quarantine_size"] == len(always_fail_keys)

    _logger.info("All assertions passed")
    _logger.info("=== DLQHandler demo complete ===")


if __name__ == "__main__":
    main()
