"""
saga_orchestrator.py — Choreography-based Saga with compensating transactions.

Concept
-------
A Saga is a sequence of local transactions where each step publishes an event
that triggers the next step.  If any step fails, previously completed steps
are compensated (rolled back) in reverse order.

This implementation uses a simple sequential (orchestrator-style) execution for
clarity, but publishes Kafka events at each step so the pattern is fully
observable in the event log.

OrderSaga steps
---------------
1. PlaceOrder           → compensating: CancelOrder
2. ReserveInventory     → compensating: ReleaseInventory
3. ProcessPayment       → compensating: RefundPayment

On success   : OrderCompleted event published.
On failure   : Compensating events published in reverse order, then
               OrderFailed event published.

All saga state transitions are published to the Kafka broker so downstream
consumers can react to them.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
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
_logger = _build_logger("event_patterns.saga_orchestrator", _CONFIG)


# ---------------------------------------------------------------------------
# Saga primitives
# ---------------------------------------------------------------------------

class SagaStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    COMPENSATING = "COMPENSATING"
    FAILED = "FAILED"


@dataclass
class SagaStep:
    """
    A single step in a Saga.

    Attributes
    ----------
    name                 : Human-readable step name.
    action               : Callable ``(context) -> context`` that executes the
                           step's local transaction.  Should raise on failure.
    compensating_action  : Callable ``(context) -> None`` that undoes the step.
    event_type           : Kafka event type published on success.
    compensating_event   : Kafka event type published on compensation.
    """
    name: str
    action: Callable[[Dict[str, Any]], Dict[str, Any]]
    compensating_action: Callable[[Dict[str, Any]], None]
    event_type: str
    compensating_event_type: str


@dataclass
class SagaExecution:
    """Tracks the runtime state of a single saga instance."""
    saga_id: str
    order_id: str
    status: SagaStatus = SagaStatus.PENDING
    completed_steps: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None


# ---------------------------------------------------------------------------
# OrderSaga
# ---------------------------------------------------------------------------

class OrderSaga:
    """
    Saga orchestrator for the Order → Inventory → Payment flow.

    Parameters
    ----------
    broker   : Shared MockKafkaBroker to publish saga events.
    topic    : Topic on which saga events are published (default: user_events).
    """

    SAGA_EVENTS_TOPIC = "user_events"

    def __init__(
        self,
        broker: MockKafkaBroker,
        topic: str = "user_events",
    ) -> None:
        self._broker = broker
        self._topic = topic
        self._producer = MockKafkaProducer(broker=broker)

        # --- Ensure topic exists ---
        if not broker.topic_exists(topic):
            broker.create_topic(topic, num_partitions=4, replication_factor=1)

        # --- Build saga steps ---
        self._steps: List[SagaStep] = self._build_steps()

        # Execution history: saga_id -> SagaExecution
        self._executions: Dict[str, SagaExecution] = {}

        _logger.info(
            f"OrderSaga initialised: topic={topic!r}, "
            f"steps={[s.name for s in self._steps]}"
        )

    # ------------------------------------------------------------------
    # Step definitions
    # ------------------------------------------------------------------

    def _build_steps(self) -> List[SagaStep]:
        return [
            SagaStep(
                name="PlaceOrder",
                action=self._place_order,
                compensating_action=self._cancel_order,
                event_type="OrderPlaced",
                compensating_event_type="OrderCancelled",
            ),
            SagaStep(
                name="ReserveInventory",
                action=self._reserve_inventory,
                compensating_action=self._release_inventory,
                event_type="InventoryReserved",
                compensating_event_type="InventoryReleased",
            ),
            SagaStep(
                name="ProcessPayment",
                action=self._process_payment,
                compensating_action=self._refund_payment,
                event_type="PaymentProcessed",
                compensating_event_type="PaymentRefunded",
            ),
        ]

    # --- Forward actions ---

    def _place_order(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Create the order record."""
        _logger.info(
            f"[PlaceOrder] order_id={ctx['order_id']!r}, "
            f"items={ctx.get('items')}, total={ctx.get('total')}"
        )
        ctx["order_status"] = "placed"
        ctx["placed_at"] = time.time()
        return ctx

    def _reserve_inventory(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Reserve inventory for each line item."""
        items = ctx.get("items", [])
        _logger.info(
            f"[ReserveInventory] order_id={ctx['order_id']!r}, items={items}"
        )
        # Simulate failure if item_id == 999
        for item in items:
            if item.get("item_id") == 999:
                raise RuntimeError(
                    f"Inventory unavailable for item_id={item['item_id']}"
                )
        ctx["reserved_items"] = items
        ctx["reservation_id"] = str(uuid.uuid4())[:8]
        return ctx

    def _process_payment(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Charge the customer."""
        total = ctx.get("total", 0.0)
        _logger.info(
            f"[ProcessPayment] order_id={ctx['order_id']!r}, total={total}"
        )
        # Simulate failure if total > 9999
        if total > 9_999:
            raise RuntimeError(
                f"Payment declined: amount {total} exceeds limit"
            )
        ctx["payment_id"] = str(uuid.uuid4())[:8]
        ctx["payment_status"] = "charged"
        return ctx

    # --- Compensating actions ---

    def _cancel_order(self, ctx: Dict[str, Any]) -> None:
        _logger.info(
            f"[CancelOrder] Compensating order_id={ctx['order_id']!r}"
        )
        ctx["order_status"] = "cancelled"

    def _release_inventory(self, ctx: Dict[str, Any]) -> None:
        reservation_id = ctx.get("reservation_id", "unknown")
        _logger.info(
            f"[ReleaseInventory] Releasing reservation_id={reservation_id!r} "
            f"for order_id={ctx['order_id']!r}"
        )
        ctx.pop("reserved_items", None)
        ctx.pop("reservation_id", None)

    def _refund_payment(self, ctx: Dict[str, Any]) -> None:
        payment_id = ctx.get("payment_id", "unknown")
        _logger.info(
            f"[RefundPayment] Refunding payment_id={payment_id!r} "
            f"for order_id={ctx['order_id']!r}"
        )
        ctx["payment_status"] = "refunded"

    # ------------------------------------------------------------------
    # Saga execution
    # ------------------------------------------------------------------

    def execute(self, order_context: Dict[str, Any]) -> SagaExecution:
        """
        Execute the OrderSaga for a given *order_context*.

        Parameters
        ----------
        order_context : Must contain at least ``order_id``, ``items``, ``total``.

        Returns
        -------
        SagaExecution with final status (COMPLETED or FAILED).
        """
        saga_id = str(uuid.uuid4())
        execution = SagaExecution(
            saga_id=saga_id,
            order_id=order_context["order_id"],
            status=SagaStatus.RUNNING,
            context=dict(order_context),
        )
        self._executions[saga_id] = execution

        _logger.info(
            f"Saga started: saga_id={saga_id!r}, order_id={order_context['order_id']!r}"
        )

        failed_at: Optional[int] = None  # index of step that failed

        # --- Forward pass ---
        for i, step in enumerate(self._steps):
            _logger.info(
                f"Saga step [{i+1}/{len(self._steps)}] START: {step.name!r} "
                f"saga_id={saga_id!r}"
            )
            try:
                execution.context = step.action(execution.context)
                execution.completed_steps.append(step.name)
                self._publish_event(
                    event_type=step.event_type,
                    saga_id=saga_id,
                    order_id=execution.order_id,
                    context_snapshot=execution.context,
                )
                _logger.info(
                    f"Saga step SUCCESS: {step.name!r} saga_id={saga_id!r}"
                )
            except Exception as exc:
                failed_at = i
                execution.error = str(exc)
                _logger.error(
                    f"Saga step FAILED: {step.name!r} saga_id={saga_id!r}, "
                    f"error={exc!r}"
                )
                break

        if failed_at is None:
            # All steps succeeded
            execution.status = SagaStatus.COMPLETED
            execution.finished_at = time.time()
            self._publish_event(
                event_type="OrderCompleted",
                saga_id=saga_id,
                order_id=execution.order_id,
                context_snapshot=execution.context,
            )
            _logger.info(
                f"Saga COMPLETED: saga_id={saga_id!r}, order_id={execution.order_id!r}"
            )
        else:
            # --- Compensation pass (reverse order, only completed steps) ---
            execution.status = SagaStatus.COMPENSATING
            _logger.warning(
                f"Saga COMPENSATING: saga_id={saga_id!r}, "
                f"failed_step={self._steps[failed_at].name!r}"
            )

            for step in reversed(self._steps[:failed_at]):
                if step.name in execution.completed_steps:
                    _logger.info(
                        f"Compensation step: {step.name!r} → "
                        f"{step.compensating_event_type!r}"
                    )
                    try:
                        step.compensating_action(execution.context)
                        self._publish_event(
                            event_type=step.compensating_event_type,
                            saga_id=saga_id,
                            order_id=execution.order_id,
                            context_snapshot=execution.context,
                        )
                    except Exception as comp_exc:
                        _logger.error(
                            f"Compensation failed for {step.name!r}: {comp_exc!r} "
                            "— manual intervention required"
                        )

            execution.status = SagaStatus.FAILED
            execution.finished_at = time.time()
            self._publish_event(
                event_type="OrderFailed",
                saga_id=saga_id,
                order_id=execution.order_id,
                context_snapshot={"error": execution.error},
            )
            _logger.error(
                f"Saga FAILED: saga_id={saga_id!r}, error={execution.error!r}"
            )

        self._producer.flush()
        return execution

    # ------------------------------------------------------------------
    # Event publishing
    # ------------------------------------------------------------------

    def _publish_event(
        self,
        event_type: str,
        saga_id: str,
        order_id: str,
        context_snapshot: Dict[str, Any],
    ) -> None:
        payload = {
            "event_type": event_type,
            "saga_id": saga_id,
            "order_id": order_id,
            "timestamp": time.time(),
            "context": context_snapshot,
        }
        self._producer.produce(
            topic=self._topic,
            key=order_id,
            value=payload,
        )
        _logger.debug(
            f"Saga event published: event_type={event_type!r}, "
            f"saga_id={saga_id!r}"
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_execution(self, saga_id: str) -> Optional[SagaExecution]:
        return self._executions.get(saga_id)

    def list_executions(self) -> List[SagaExecution]:
        return list(self._executions.values())


# ---------------------------------------------------------------------------
# Demo / main
# ---------------------------------------------------------------------------

def main() -> None:
    """Demonstrate OrderSaga with a success path and a failure/compensation path."""
    from kafka_core.mock_kafka import MockKafkaBroker

    _logger.info("=== OrderSaga demo start ===")
    broker = MockKafkaBroker()
    saga = OrderSaga(broker=broker)

    # --- Success path ---
    _logger.info("--- Path 1: Successful order ---")
    result = saga.execute({
        "order_id": "ORD-001",
        "customer_id": "CUST-42",
        "items": [
            {"item_id": 101, "qty": 2, "price": 19.99},
            {"item_id": 202, "qty": 1, "price": 49.99},
        ],
        "total": 89.97,
    })
    _logger.info(
        f"Saga result: status={result.status.value}, "
        f"completed_steps={result.completed_steps}"
    )
    assert result.status == SagaStatus.COMPLETED

    # --- Failure path: inventory unavailable ---
    _logger.info("--- Path 2: Inventory failure → compensation ---")
    result2 = saga.execute({
        "order_id": "ORD-002",
        "customer_id": "CUST-17",
        "items": [
            {"item_id": 999, "qty": 1, "price": 5.00},  # triggers failure
        ],
        "total": 5.00,
    })
    _logger.info(
        f"Saga result: status={result2.status.value}, "
        f"completed_steps={result2.completed_steps}, "
        f"error={result2.error!r}"
    )
    assert result2.status == SagaStatus.FAILED
    assert "PlaceOrder" in result2.completed_steps

    # --- Failure path: payment declined ---
    _logger.info("--- Path 3: Payment failure → compensation ---")
    result3 = saga.execute({
        "order_id": "ORD-003",
        "customer_id": "CUST-99",
        "items": [{"item_id": 777, "qty": 1, "price": 15000.0}],
        "total": 15_000.0,  # triggers payment decline
    })
    _logger.info(
        f"Saga result: status={result3.status.value}, "
        f"completed_steps={result3.completed_steps}"
    )
    assert result3.status == SagaStatus.FAILED
    assert "ReserveInventory" in result3.completed_steps

    _logger.info(
        f"Total saga executions: {len(saga.list_executions())}"
    )
    _logger.info("=== OrderSaga demo complete ===")


if __name__ == "__main__":
    main()
