"""
Structured logger for RL experiments.

Provides file + stream logging with a consistent timestamp format and
convenience helpers for metrics and episode summaries.
"""
from __future__ import annotations

import logging
import os
from typing import Dict, Optional


_LEVEL_MAP: Dict[str, int] = {
    "DEBUG": logging.DEBUG,       # 10
    "INFO": logging.INFO,          # 20
    "WARNING": logging.WARNING,    # 30
    "ERROR": logging.ERROR,        # 40
}


class _MillisecondFormatter(logging.Formatter):
    """Custom formatter that includes milliseconds in the timestamp."""

    def formatTime(self, record: logging.LogRecord, datefmt: Optional[str] = None) -> str:  # noqa: N802
        import datetime
        ct = datetime.datetime.fromtimestamp(record.created)
        return ct.strftime("%Y-%m-%d %H:%M:%S.") + f"{ct.microsecond // 1000:03d}"


_FORMAT = "[%(asctime)s] [%(levelname)-7s] [%(name)s] %(message)s"


class RLLogger:
    """
    Dual-output (file + stream) logger for reinforcement learning runs.

    Log format:
        [YYYY-MM-DD HH:MM:SS.mmm] [LEVEL  ] [name] message

    Responsibilities (SRP):
    - File handler: writes to {log_dir}/{name}.log
    - Stream handler: writes to stdout
    - Structured helpers: log_metrics, log_episode
    """

    def __init__(self, name: str, log_dir: str, level: str = "INFO") -> None:
        if level not in _LEVEL_MAP:
            raise ValueError(f"Unsupported log level '{level}'. Choose from {list(_LEVEL_MAP)}.")

        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"{name}.log")

        numeric_level = _LEVEL_MAP[level]

        self._logger = logging.getLogger(name)
        # Avoid adding duplicate handlers if this logger is re-used.
        if not self._logger.handlers:
            self._logger.setLevel(numeric_level)

            formatter = _MillisecondFormatter(_FORMAT)

            # File handler
            fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
            fh.setLevel(numeric_level)
            fh.setFormatter(formatter)

            # Stream handler
            sh = logging.StreamHandler()
            sh.setLevel(numeric_level)
            sh.setFormatter(formatter)

            self._logger.addHandler(fh)
            self._logger.addHandler(sh)

        self._logger.info("Logger initialised. log_path=%s level=%s", log_path, level)

    # ------------------------------------------------------------------
    # Basic logging
    # ------------------------------------------------------------------

    def debug(self, msg: str) -> None:
        self._logger.debug(msg)

    def info(self, msg: str) -> None:
        self._logger.info(msg)

    def warning(self, msg: str) -> None:
        self._logger.warning(msg)

    def error(self, msg: str) -> None:
        self._logger.error(msg)

    # ------------------------------------------------------------------
    # Structured helpers
    # ------------------------------------------------------------------

    def log_metrics(self, step: int, metrics: Dict[str, float]) -> None:
        """
        Log scalar metrics at a given step.

        Output example:
            [METRICS] step=500 loss=0.0423 epsilon=0.72 mean_q=1.34
        """
        parts = " ".join(f"{k}={v:.6g}" for k, v in metrics.items())
        self._logger.info("[METRICS] step=%d %s", step, parts)

    def log_episode(
        self,
        episode: int,
        total_reward: float,
        steps: int,
        info: Optional[Dict] = None,
    ) -> None:
        """
        Log per-episode summary.

        Output example:
            [EPISODE] episode=42 reward=7.35 steps=120 goal_reached=True
        """
        info_str = ""
        if info:
            info_str = " " + " ".join(f"{k}={v}" for k, v in info.items())
        self._logger.info(
            "[EPISODE] episode=%d reward=%.2f steps=%d%s",
            episode,
            total_reward,
            steps,
            info_str,
        )
