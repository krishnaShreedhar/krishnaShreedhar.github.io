"""
windowed_aggregation.py — Tumbling and Session windows over event streams.

Concepts
--------
Tumbling Window
    Fixed-size, non-overlapping time buckets.  Every event falls into exactly
    one bucket.  At the end of each window the bucket is sealed and aggregated.

      |--- window 0 ---|--- window 1 ---|--- window 2 ---|
      t=0             t=300           t=600             t=900

Session Window
    Groups events that are close in time (inactivity gap < session_gap_s).
    A session ends when no event arrives for session_gap_s seconds.
    Session length is variable and depends on user activity patterns.

Both window types compute per-window aggregates: count, sum(value), min, max,
and mean of the ``value`` field in each event.

Configuration
-------------
All timing parameters come from config.yaml:
  streaming.window_size_s   : Tumbling window size in seconds.
  streaming.session_gap_s   : Inactivity gap to close a session window.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

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
_logger = _build_logger("streaming_analytics.windowed_aggregation", _CONFIG)


# ---------------------------------------------------------------------------
# Window aggregate dataclass
# ---------------------------------------------------------------------------

@dataclass
class WindowAggregate:
    """Aggregated statistics for a single time window."""
    window_id: str
    window_start: float
    window_end: float
    count: int = 0
    total: float = 0.0
    min_value: float = float("inf")
    max_value: float = float("-inf")
    events: List[Dict[str, Any]] = field(default_factory=list, repr=False)

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count > 0 else 0.0

    @property
    def duration_s(self) -> float:
        return self.window_end - self.window_start

    def add(self, value: float, event: Dict[str, Any]) -> None:
        self.count += 1
        self.total += value
        self.min_value = min(self.min_value, value)
        self.max_value = max(self.max_value, value)
        self.events.append(event)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_id": self.window_id,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "duration_s": round(self.duration_s, 2),
            "count": self.count,
            "total": round(self.total, 4),
            "mean": round(self.mean, 4),
            "min": round(self.min_value, 4) if self.count > 0 else None,
            "max": round(self.max_value, 4) if self.count > 0 else None,
        }


# ---------------------------------------------------------------------------
# TumblingWindow
# ---------------------------------------------------------------------------

class TumblingWindow:
    """
    Fixed-size, non-overlapping tumbling window processor.

    Each event is assigned to a bucket based on:
      bucket_index = floor(event_timestamp / window_size_s)

    When ``flush()`` is called (or automatically on ``add()`` when the
    current bucket's time has passed), all sealed buckets are aggregated
    and returned.

    Parameters
    ----------
    window_size_s : Duration of each window in seconds.
    value_key     : Key in the event dict used as the numeric value to aggregate.
    """

    def __init__(
        self,
        window_size_s: float = 300.0,
        value_key: str = "value",
    ) -> None:
        self._window_size = window_size_s
        self._value_key = value_key
        # bucket_index -> WindowAggregate
        self._buckets: Dict[int, WindowAggregate] = {}
        self._sealed: List[WindowAggregate] = []
        self._total_events = 0

        _logger.info(
            f"TumblingWindow initialised: window_size_s={window_size_s}, "
            f"value_key={value_key!r}"
        )

    def add(self, event: Dict[str, Any]) -> None:
        """
        Add *event* to the appropriate time bucket.

        Parameters
        ----------
        event : Must contain a ``timestamp`` key (Unix epoch float) and
                the key specified by ``value_key``.
        """
        ts = float(event.get("timestamp", time.time()))
        value = float(event.get(self._value_key, 0.0))
        bucket_idx = int(ts // self._window_size)

        if bucket_idx not in self._buckets:
            window_start = bucket_idx * self._window_size
            window_end = window_start + self._window_size
            self._buckets[bucket_idx] = WindowAggregate(
                window_id=f"tumbling-{bucket_idx}",
                window_start=window_start,
                window_end=window_end,
            )
            _logger.debug(
                f"TumblingWindow: opened bucket {bucket_idx} "
                f"[{window_start:.0f}, {window_end:.0f})"
            )

        self._buckets[bucket_idx].add(value, event)
        self._total_events += 1
        _logger.debug(
            f"TumblingWindow.add: bucket={bucket_idx}, value={value:.4f}, "
            f"total_in_bucket={self._buckets[bucket_idx].count}"
        )

    def flush(self, up_to_ts: Optional[float] = None) -> List[WindowAggregate]:
        """
        Seal and return all completed windows up to *up_to_ts*.

        A window is considered complete if its ``window_end`` is <= ``up_to_ts``.
        Defaults to ``time.time()`` if not provided.

        Returns
        -------
        List of sealed ``WindowAggregate`` objects (sorted by window_start).
        """
        cutoff = up_to_ts if up_to_ts is not None else time.time()
        newly_sealed: List[WindowAggregate] = []
        remaining: Dict[int, WindowAggregate] = {}

        for idx, agg in sorted(self._buckets.items()):
            if agg.window_end <= cutoff:
                newly_sealed.append(agg)
                self._sealed.append(agg)
                _logger.info(
                    f"TumblingWindow sealed: {agg.window_id}, "
                    f"count={agg.count}, mean={agg.mean:.4f}, "
                    f"total={agg.total:.4f}"
                )
            else:
                remaining[idx] = agg

        self._buckets = remaining
        return newly_sealed

    def all_sealed(self) -> List[WindowAggregate]:
        """Return all previously sealed windows."""
        return list(self._sealed)

    @property
    def total_events(self) -> int:
        return self._total_events

    @property
    def open_bucket_count(self) -> int:
        return len(self._buckets)


# ---------------------------------------------------------------------------
# SessionWindow
# ---------------------------------------------------------------------------

@dataclass
class Session:
    """A single user session grouped by inactivity gap."""
    session_id: str
    user_id: str
    start_ts: float
    end_ts: float
    events: List[Dict[str, Any]] = field(default_factory=list)
    closed: bool = False

    def add(self, event: Dict[str, Any], ts: float) -> None:
        self.events.append(event)
        self.end_ts = ts

    def aggregate(self) -> Dict[str, Any]:
        values = [float(e.get("value", 0.0)) for e in self.events]
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "duration_s": round(self.end_ts - self.start_ts, 2),
            "event_count": len(self.events),
            "total_value": round(sum(values), 4),
            "mean_value": round(sum(values) / len(values), 4) if values else 0.0,
        }


class SessionWindow:
    """
    Session window processor: groups events by per-user inactivity gap.

    A session is open as long as events keep arriving within *session_gap_s*
    seconds of each other.  A call to ``close_inactive_sessions()`` seals
    sessions that have been quiet for longer than the gap.

    Parameters
    ----------
    session_gap_s : Inactivity gap (seconds) that triggers session close.
    user_key      : Key in event dict containing the user identifier.
    value_key     : Key for the numeric value to aggregate.
    """

    def __init__(
        self,
        session_gap_s: float = 30.0,
        user_key: str = "user_id",
        value_key: str = "value",
    ) -> None:
        self._gap = session_gap_s
        self._user_key = user_key
        self._value_key = value_key
        # user_id -> open Session
        self._open_sessions: Dict[str, Session] = {}
        self._closed_sessions: List[Session] = []
        self._session_counter = 0
        self._total_events = 0

        _logger.info(
            f"SessionWindow initialised: session_gap_s={session_gap_s}, "
            f"user_key={user_key!r}"
        )

    def add(self, event: Dict[str, Any]) -> None:
        """
        Add *event* to the appropriate user session.

        If the user has no open session, a new one is started.
        If the user has an open session and this event arrives within
        *session_gap_s* of the last event, it extends that session.
        Otherwise the old session is closed and a new one is started.
        """
        ts = float(event.get("timestamp", time.time()))
        user_id = str(event.get(self._user_key, "anonymous"))

        existing = self._open_sessions.get(user_id)

        if existing is not None:
            if ts - existing.end_ts <= self._gap:
                # Extend existing session
                existing.add(event, ts)
                _logger.debug(
                    f"SessionWindow: extended session {existing.session_id!r} "
                    f"for user={user_id!r}, gap={ts - existing.end_ts:.1f}s"
                )
            else:
                # Gap exceeded — close existing, start new
                self._close_session(existing)
                self._start_session(user_id, event, ts)
        else:
            self._start_session(user_id, event, ts)

        self._total_events += 1

    def close_inactive_sessions(
        self, current_ts: Optional[float] = None
    ) -> List[Session]:
        """
        Close all sessions where ``current_ts - last_event_ts > session_gap_s``.

        Returns the list of newly closed sessions.
        """
        cutoff = (current_ts or time.time()) - self._gap
        to_close = [
            s for s in self._open_sessions.values() if s.end_ts < cutoff
        ]
        for session in to_close:
            self._close_session(session)

        if to_close:
            _logger.info(
                f"SessionWindow.close_inactive_sessions: closed {len(to_close)} session(s)"
            )
        return to_close

    def flush_all(self) -> List[Session]:
        """Close all open sessions immediately."""
        sessions = list(self._open_sessions.values())
        for s in sessions:
            self._close_session(s)
        _logger.info(f"SessionWindow.flush_all: closed {len(sessions)} session(s)")
        return sessions

    def closed_sessions(self) -> List[Session]:
        return list(self._closed_sessions)

    @property
    def total_events(self) -> int:
        return self._total_events

    @property
    def open_session_count(self) -> int:
        return len(self._open_sessions)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _start_session(
        self, user_id: str, event: Dict[str, Any], ts: float
    ) -> Session:
        self._session_counter += 1
        session_id = f"sess-{user_id}-{self._session_counter}"
        session = Session(
            session_id=session_id,
            user_id=user_id,
            start_ts=ts,
            end_ts=ts,
        )
        session.add(event, ts)
        self._open_sessions[user_id] = session
        _logger.debug(
            f"SessionWindow: started session {session_id!r} for user={user_id!r}"
        )
        return session

    def _close_session(self, session: Session) -> None:
        session.closed = True
        self._open_sessions.pop(session.user_id, None)
        self._closed_sessions.append(session)
        agg = session.aggregate()
        _logger.info(
            f"SessionWindow closed: session_id={session.session_id!r}, "
            f"user={session.user_id!r}, "
            f"duration_s={agg['duration_s']:.1f}, "
            f"event_count={agg['event_count']}, "
            f"total_value={agg['total_value']:.4f}"
        )


# ---------------------------------------------------------------------------
# Demo / main
# ---------------------------------------------------------------------------

def generate_stream_events(
    num_events: int,
    num_users: int,
    base_ts: float,
    window_size_s: float,
) -> List[Dict[str, Any]]:
    """Generate synthetic stream events for windowing demos."""
    rng = random.Random(42)
    events = []
    for i in range(num_events):
        ts = base_ts + i * (window_size_s * 3 / num_events)
        events.append({
            "event_id": i,
            "user_id": f"user-{rng.randint(1, num_users):03d}",
            "timestamp": ts,
            "value": round(rng.uniform(1.0, 100.0), 2),
            "event_type": rng.choice(["click", "purchase", "view"]),
        })
    return events


def main() -> None:
    """Demonstrate TumblingWindow and SessionWindow aggregations."""
    _logger.info("=== WindowedAggregation demo start ===")

    cfg = _CONFIG.get("streaming", {})
    window_size_s: float = float(cfg.get("window_size_s", 300))
    session_gap_s: float = float(cfg.get("session_gap_s", 30))
    num_events: int = int(cfg.get("num_events_to_generate", 200))

    base_ts = time.time() - window_size_s * 5  # events start 5 windows ago

    events = generate_stream_events(
        num_events=num_events,
        num_users=10,
        base_ts=base_ts,
        window_size_s=window_size_s,
    )
    _logger.info(f"Generated {num_events} stream events")

    # --- Tumbling Window ---
    _logger.info("--- TumblingWindow aggregation ---")
    tumbling = TumblingWindow(window_size_s=window_size_s, value_key="value")
    for event in events:
        tumbling.add(event)

    # Flush all windows up to now
    sealed = tumbling.flush(up_to_ts=time.time())
    _logger.info(f"TumblingWindow: sealed {len(sealed)} windows")
    for agg in sealed:
        _logger.info(f"  Window aggregate: {agg.to_dict()}")

    # --- Session Window ---
    _logger.info("--- SessionWindow aggregation ---")
    # Use a smaller gap relative to the synthetic event spacing
    session_win = SessionWindow(session_gap_s=window_size_s * 0.3, user_key="user_id")
    for event in events:
        session_win.add(event)

    # Close all remaining open sessions
    session_win.flush_all()
    closed_sessions = session_win.closed_sessions()
    _logger.info(f"SessionWindow: {len(closed_sessions)} closed sessions")
    for sess in closed_sessions[:10]:  # log first 10
        _logger.info(f"  Session: {sess.aggregate()}")

    # Summary stats
    durations = [s.aggregate()["duration_s"] for s in closed_sessions]
    counts = [s.aggregate()["event_count"] for s in closed_sessions]
    if durations:
        _logger.info(
            f"Session summary: "
            f"avg_duration_s={sum(durations)/len(durations):.1f}, "
            f"avg_events_per_session={sum(counts)/len(counts):.1f}, "
            f"total_sessions={len(closed_sessions)}"
        )

    _logger.info("=== WindowedAggregation demo complete ===")


if __name__ == "__main__":
    main()
