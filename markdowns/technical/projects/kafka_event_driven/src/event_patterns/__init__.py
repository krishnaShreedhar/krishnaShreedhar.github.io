"""
event_patterns — implementations of common Kafka event-driven patterns.

Exports:
    Event, EventStore          : append-only event log (event sourcing)
    SagaStep, OrderSaga        : choreography-based saga with compensation
    OutboxEntry, OutboxTable, OutboxRelay : transactional outbox pattern
    DLQMessage, DLQHandler     : dead-letter queue with exponential backoff
"""

from event_patterns.event_sourcing import Event, EventStore
from event_patterns.saga_orchestrator import SagaStep, OrderSaga
from event_patterns.outbox_pattern import OutboxEntry, OutboxTable, OutboxRelay
from event_patterns.dlq_handler import DLQMessage, DLQHandler

__all__ = [
    "Event",
    "EventStore",
    "SagaStep",
    "OrderSaga",
    "OutboxEntry",
    "OutboxTable",
    "OutboxRelay",
    "DLQMessage",
    "DLQHandler",
]
