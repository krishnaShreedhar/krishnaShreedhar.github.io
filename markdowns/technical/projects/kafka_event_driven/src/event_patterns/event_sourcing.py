"""
event_sourcing.py — Append-only event log with state reconstruction.

Concept
-------
Event Sourcing stores every state change as an immutable event rather than
overwriting a "current state" record.  The authoritative state of any
aggregate can be rebuilt at any point in time by replaying its event history.

Key components implemented
--------------------------
Event
    Immutable data record: event_id, event_type, aggregate_id, data, timestamp.

EventStore
    Append-only log.  Supports:
      * append(event)              — record a new event
      * get_events(aggregate_id)   — retrieve full history for an aggregate
      * replay_state(aggregate_id) — fold events into current state dict

Demo flow
---------
  UserRegistered  → user exists with email
  EmailVerified   → email_verified = True
  ProfileUpdated  → display_name and bio set
  Final state reconstructed by replaying the three events.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

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
_logger = _build_logger("event_patterns.event_sourcing", _CONFIG)


# ---------------------------------------------------------------------------
# Domain: Event
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Event:
    """
    Immutable domain event.

    Attributes
    ----------
    event_id      : Globally unique identifier (UUID4 string).
    event_type    : Human-readable type, e.g. ``"UserRegistered"``.
    aggregate_id  : ID of the entity this event belongs to.
    data          : Arbitrary payload dict (serialisable to JSON).
    timestamp     : Unix epoch float when the event was created.
    version       : Monotonically increasing sequence number within the
                    aggregate (set by EventStore on append).
    """
    event_id: str
    event_type: str
    aggregate_id: str
    data: Dict[str, Any]
    timestamp: float
    version: int = 0

    @classmethod
    def create(
        cls,
        event_type: str,
        aggregate_id: str,
        data: Dict[str, Any],
    ) -> "Event":
        """Factory: generate event_id and timestamp automatically."""
        return cls(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            aggregate_id=aggregate_id,
            data=data,
            timestamp=time.time(),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "aggregate_id": self.aggregate_id,
            "data": self.data,
            "timestamp": self.timestamp,
            "version": self.version,
        }


# ---------------------------------------------------------------------------
# EventStore
# ---------------------------------------------------------------------------

class EventStore:
    """
    Append-only in-memory event log.

    The store is partitioned by ``aggregate_id``.  Each aggregate has its own
    ordered list of events; the version field is set on append to the current
    list length (1-based).

    State reconstruction
    --------------------
    ``replay_state`` folds over an aggregate's events using a registry of
    event handler functions.  Handlers are registered with
    ``register_handler(event_type, fn)`` where ``fn(state, event) -> state``.

    Parameters
    ----------
    name : Human-readable name for this store instance (used in logs).
    """

    def __init__(self, name: str = "default") -> None:
        self._name = name
        # aggregate_id -> List[Event] (ordered by version)
        self._store: Dict[str, List[Event]] = {}
        # event_type -> handler function
        self._handlers: Dict[str, Callable[[Dict, Event], Dict]] = {}
        self._total_events: int = 0
        _logger.info(f"EventStore '{name}' initialised")

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def append(self, event: Event) -> Event:
        """
        Append *event* to the store.

        The event's ``version`` is set to the next sequence number for its
        aggregate.  Returns the versioned ``Event`` stored.
        """
        agg_id = event.aggregate_id
        if agg_id not in self._store:
            self._store[agg_id] = []

        version = len(self._store[agg_id]) + 1
        # Create a new frozen instance with the assigned version
        versioned = Event(
            event_id=event.event_id,
            event_type=event.event_type,
            aggregate_id=event.aggregate_id,
            data=event.data,
            timestamp=event.timestamp,
            version=version,
        )
        self._store[agg_id].append(versioned)
        self._total_events += 1

        _logger.info(
            f"EventStore[{self._name}] appended: "
            f"aggregate_id={agg_id!r}, event_type={event.event_type!r}, "
            f"version={version}, event_id={event.event_id!r}"
        )
        return versioned

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def get_events(
        self,
        aggregate_id: str,
        from_version: int = 1,
    ) -> List[Event]:
        """
        Return events for *aggregate_id* starting from *from_version*.

        Returns an empty list if the aggregate does not exist or has no events
        at or after *from_version*.
        """
        events = self._store.get(aggregate_id, [])
        result = [e for e in events if e.version >= from_version]
        _logger.debug(
            f"get_events: aggregate_id={aggregate_id!r}, "
            f"from_version={from_version}, returned={len(result)} events"
        )
        return result

    def get_all_aggregate_ids(self) -> List[str]:
        """Return all aggregate IDs that have at least one event."""
        return list(self._store.keys())

    # ------------------------------------------------------------------
    # State reconstruction
    # ------------------------------------------------------------------

    def register_handler(
        self,
        event_type: str,
        handler: Callable[[Dict[str, Any], Event], Dict[str, Any]],
    ) -> None:
        """
        Register a handler function for *event_type*.

        The handler signature is ``handler(current_state, event) -> new_state``.
        """
        self._handlers[event_type] = handler
        _logger.debug(f"Registered handler for event_type={event_type!r}")

    def replay_state(
        self,
        aggregate_id: str,
        initial_state: Optional[Dict[str, Any]] = None,
        up_to_version: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Reconstruct aggregate state by replaying its event history.

        Parameters
        ----------
        aggregate_id  : The aggregate whose events to replay.
        initial_state : Starting state dict (defaults to empty dict).
        up_to_version : Replay only events up to and including this version.
                        Defaults to all events (point-in-time restore).

        Returns
        -------
        A state dict built by successively applying each registered handler.
        Events with no registered handler are logged as warnings and skipped.
        """
        state = dict(initial_state or {})
        events = self._store.get(aggregate_id, [])

        _logger.info(
            f"replay_state: aggregate_id={aggregate_id!r}, "
            f"total_events={len(events)}, "
            f"up_to_version={up_to_version or 'all'}"
        )

        for event in events:
            if up_to_version is not None and event.version > up_to_version:
                break

            handler = self._handlers.get(event.event_type)
            if handler is None:
                _logger.warning(
                    f"No handler for event_type={event.event_type!r} — skipping"
                )
                continue

            previous_state = dict(state)
            state = handler(state, event)
            _logger.debug(
                f"Applied {event.event_type!r} v{event.version}: "
                f"state delta={_dict_diff(previous_state, state)}"
            )

        _logger.info(
            f"replay_state complete: aggregate_id={aggregate_id!r}, "
            f"final_state={state}"
        )
        return state

    @property
    def total_events(self) -> int:
        return self._total_events


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _dict_diff(before: dict, after: dict) -> dict:
    """Return keys that changed between *before* and *after*."""
    diff = {}
    for key in set(before) | set(after):
        bv = before.get(key, "<absent>")
        av = after.get(key, "<absent>")
        if bv != av:
            diff[key] = {"before": bv, "after": av}
    return diff


# ---------------------------------------------------------------------------
# User aggregate domain logic
# ---------------------------------------------------------------------------

def _apply_user_registered(state: dict, event: Event) -> dict:
    state = dict(state)
    state.update({
        "user_id": event.aggregate_id,
        "email": event.data["email"],
        "email_verified": False,
        "display_name": None,
        "bio": None,
        "status": "pending_verification",
        "registered_at": event.timestamp,
    })
    return state


def _apply_email_verified(state: dict, event: Event) -> dict:
    state = dict(state)
    state["email_verified"] = True
    state["status"] = "active"
    state["verified_at"] = event.timestamp
    return state


def _apply_profile_updated(state: dict, event: Event) -> dict:
    state = dict(state)
    state.update(event.data)
    state["profile_updated_at"] = event.timestamp
    return state


def _apply_user_deactivated(state: dict, event: Event) -> dict:
    state = dict(state)
    state["status"] = "deactivated"
    state["deactivated_at"] = event.timestamp
    state["deactivation_reason"] = event.data.get("reason", "unspecified")
    return state


def build_user_event_store() -> EventStore:
    """Build and return an EventStore pre-configured with user aggregate handlers."""
    store = EventStore(name="user_store")
    store.register_handler("UserRegistered", _apply_user_registered)
    store.register_handler("EmailVerified", _apply_email_verified)
    store.register_handler("ProfileUpdated", _apply_profile_updated)
    store.register_handler("UserDeactivated", _apply_user_deactivated)
    return store


# ---------------------------------------------------------------------------
# Demo / main
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Demonstrate event sourcing: append events, reconstruct state at various
    points in time.
    """
    _logger.info("=== EventSourcing demo start ===")

    store = build_user_event_store()
    user_id = "user-" + str(uuid.uuid4())[:8]

    # --- Stream of domain events ---
    e1 = store.append(Event.create(
        event_type="UserRegistered",
        aggregate_id=user_id,
        data={"email": "alice@example.com"},
    ))
    _logger.info(f"Appended: {e1.event_type} v{e1.version}")

    e2 = store.append(Event.create(
        event_type="EmailVerified",
        aggregate_id=user_id,
        data={"verified_by": "email_link"},
    ))
    _logger.info(f"Appended: {e2.event_type} v{e2.version}")

    e3 = store.append(Event.create(
        event_type="ProfileUpdated",
        aggregate_id=user_id,
        data={"display_name": "Alice Smith", "bio": "ML engineer"},
    ))
    _logger.info(f"Appended: {e3.event_type} v{e3.version}")

    e4 = store.append(Event.create(
        event_type="ProfileUpdated",
        aggregate_id=user_id,
        data={"bio": "Senior ML engineer at Acme Corp"},
    ))
    _logger.info(f"Appended: {e4.event_type} v{e4.version}")

    # --- State at registration only (point-in-time v1) ---
    state_v1 = store.replay_state(user_id, up_to_version=1)
    _logger.info(f"State at v1 (registered): status={state_v1['status']!r}")
    assert state_v1["email_verified"] is False
    assert state_v1["status"] == "pending_verification"

    # --- State after email verification (v2) ---
    state_v2 = store.replay_state(user_id, up_to_version=2)
    _logger.info(f"State at v2 (verified): status={state_v2['status']!r}")
    assert state_v2["email_verified"] is True
    assert state_v2["status"] == "active"

    # --- Current state (all events) ---
    current = store.replay_state(user_id)
    _logger.info(
        f"Current state: display_name={current['display_name']!r}, "
        f"bio={current['bio']!r}, status={current['status']!r}"
    )
    assert current["bio"] == "Senior ML engineer at Acme Corp"

    # --- Total events in store ---
    _logger.info(f"Total events in store: {store.total_events}")

    # --- Second aggregate (another user) ---
    user2_id = "user-" + str(uuid.uuid4())[:8]
    store.append(Event.create("UserRegistered", user2_id, {"email": "bob@example.com"}))
    store.append(Event.create("EmailVerified", user2_id, {}))
    store.append(Event.create("UserDeactivated", user2_id, {"reason": "spam"}))
    state_u2 = store.replay_state(user2_id)
    _logger.info(f"User2 final state: {state_u2}")
    assert state_u2["status"] == "deactivated"

    _logger.info("All assertions passed")
    _logger.info("=== EventSourcing demo complete ===")


if __name__ == "__main__":
    main()
