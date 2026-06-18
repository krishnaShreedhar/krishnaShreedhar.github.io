"""
producer.py — MockKafkaProducer wrapping the in-memory broker.

Responsibilities
----------------
* Serialize Python objects to JSON bytes before handing off to the broker.
* Accept an optional delivery callback that mirrors the confluent-kafka API
  (called with ``(error, message)`` — ``error`` is ``None`` on success).
* Maintain an internal buffer of un-flushed produce calls and drain it on
  ``flush()``.
* Log every produce operation at DEBUG level and flush operations at INFO.

The class is intentionally designed so that swapping it for a real
``confluent_kafka.Producer`` requires only a configuration change in
``config.yaml`` (``kafka.use_mock: false``) and a thin adapter layer —
the public interface is identical.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import yaml

from kafka_core.mock_kafka import MockKafkaBroker, MockMessage, get_broker

# ---------------------------------------------------------------------------
# Logging bootstrap (mirrors mock_kafka.py — shared helper would require a
# separate utils module; keeping each file self-contained per project rules)
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
_logger = _build_logger("kafka_core.producer", _CONFIG)

# ---------------------------------------------------------------------------
# Delivery receipt
# ---------------------------------------------------------------------------

class DeliveryReceipt:
    """
    Carries the outcome of a single produce call.

    Attributes
    ----------
    error   : None on success; an Exception on failure.
    message : The MockMessage that was stored in the broker on success.
    """

    def __init__(
        self,
        error: Optional[Exception],
        message: Optional[MockMessage],
    ) -> None:
        self.error = error
        self.message = message

    def __repr__(self) -> str:
        if self.error:
            return f"DeliveryReceipt(error={self.error!r})"
        return f"DeliveryReceipt(ok, offset={self.message.offset})"


# type alias for the optional delivery callback
DeliveryCallback = Callable[[Optional[Exception], Optional[MockMessage]], None]


# ---------------------------------------------------------------------------
# MockKafkaProducer
# ---------------------------------------------------------------------------

class MockKafkaProducer:
    """
    Wraps ``MockKafkaBroker`` with a producer interface.

    Parameters
    ----------
    broker  : Shared ``MockKafkaBroker`` instance.
    config  : Producer config section from ``config.yaml``
              (used for documentation / future real-Kafka parity).

    Public API
    ----------
    produce(topic, key, value, partition=None, callback=None)
        Serialize and deliver one message.  If *callback* is supplied it is
        invoked synchronously with ``(error, message)``.
    flush(timeout=None)
        Drain all pending delivery callbacks.  With the mock broker every
        produce is synchronous so ``flush()`` is a no-op beyond callback
        draining.
    pending_count()
        Number of messages produced but not yet flushed.
    """

    def __init__(
        self,
        broker: MockKafkaBroker,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._broker = broker
        self._config = config or _CONFIG.get("producer", {})
        self._lock = threading.Lock()
        self._pending_callbacks: List[Tuple[DeliveryCallback, DeliveryReceipt]] = []
        _logger.info(
            f"MockKafkaProducer initialised with config: {self._config}"
        )

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def produce(
        self,
        topic: str,
        key: Any = None,
        value: Any = None,
        partition: Optional[int] = None,
        headers: Optional[Dict[str, str]] = None,
        callback: Optional[DeliveryCallback] = None,
    ) -> MockMessage:
        """
        Serialize *value* to JSON bytes and append it to the broker.

        Parameters
        ----------
        topic     : Destination Kafka topic name.
        key       : Optional message key.  If not bytes, will be JSON-serialised.
        value     : Message payload.  If not bytes, will be JSON-serialised.
        partition : Explicit partition override (default: broker-selected).
        headers   : Optional dict of string headers.
        callback  : Invoked as ``callback(error, message)`` after produce.
                    ``error`` is ``None`` on success.

        Returns
        -------
        MockMessage stored in the broker.
        """
        _logger.debug(
            f"Producing to topic={topic!r}, key={key!r}"
        )

        key_bytes = self._serialize(key)
        value_bytes = self._serialize(value)

        try:
            msg = self._broker.produce(
                topic=topic,
                key=key_bytes,
                value=value_bytes,
                partition=partition,
                headers=headers,
            )
            receipt = DeliveryReceipt(error=None, message=msg)
            _logger.debug(
                f"Produce success: topic={topic!r}, partition={msg.partition}, "
                f"offset={msg.offset}"
            )
        except Exception as exc:
            receipt = DeliveryReceipt(error=exc, message=None)
            _logger.error(
                f"Produce failed: topic={topic!r}, key={key!r}, error={exc!r}"
            )
            if callback:
                callback(exc, None)
            raise

        if callback:
            with self._lock:
                self._pending_callbacks.append((callback, receipt))

        return msg

    def flush(self, timeout: Optional[float] = None) -> int:
        """
        Drain all pending delivery callbacks.

        Returns the number of callbacks that were invoked.
        """
        with self._lock:
            pending = list(self._pending_callbacks)
            self._pending_callbacks.clear()

        count = 0
        for cb, receipt in pending:
            try:
                cb(receipt.error, receipt.message)
            except Exception as exc:
                _logger.error(f"Delivery callback raised an exception: {exc!r}")
            count += 1

        _logger.info(f"flush() invoked — drained {count} delivery callback(s)")
        return count

    def pending_count(self) -> int:
        """Number of messages with outstanding delivery callbacks."""
        with self._lock:
            return len(self._pending_callbacks)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize(value: Any) -> Optional[bytes]:
        """Convert *value* to bytes via JSON if it is not already bytes."""
        if value is None:
            return None
        if isinstance(value, bytes):
            return value
        if isinstance(value, str):
            return value.encode("utf-8")
        return json.dumps(value).encode("utf-8")


# ---------------------------------------------------------------------------
# Demo / main
# ---------------------------------------------------------------------------

def main() -> None:
    """Demonstrate MockKafkaProducer: produce messages with delivery callbacks."""
    from kafka_core.mock_kafka import MockKafkaBroker

    _logger.info("=== MockKafkaProducer demo start ===")

    broker = MockKafkaBroker()
    broker.create_topic("orders", num_partitions=3, replication_factor=1)

    producer = MockKafkaProducer(broker=broker)

    delivered: List[str] = []

    def on_delivery(error: Optional[Exception], message: Optional[MockMessage]) -> None:
        if error:
            _logger.error(f"Delivery failed: {error!r}")
        else:
            record = (
                f"topic={message.topic!r}, partition={message.partition}, "
                f"offset={message.offset}"
            )
            delivered.append(record)
            _logger.info(f"Delivery callback: {record}")

    # Produce a variety of payload types
    payloads = [
        ("user-1", {"event": "click", "page": "/home"}),
        ("user-2", {"event": "purchase", "item_id": 42, "price": 9.99}),
        ("user-1", {"event": "logout"}),
        (None, "raw-string-event"),
        ("user-3", b"raw-bytes-event"),
    ]

    for key, value in payloads:
        producer.produce(
            topic="orders",
            key=key,
            value=value,
            callback=on_delivery,
        )

    flushed = producer.flush()
    _logger.info(f"Flushed {flushed} callbacks.  Delivered records: {delivered}")
    _logger.info("=== MockKafkaProducer demo complete ===")


if __name__ == "__main__":
    main()
