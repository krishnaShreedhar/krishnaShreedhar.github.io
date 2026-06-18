"""
monitoring.py — Prediction monitoring with Population Stability Index (PSI).

Concept
-------
PSI measures how much the distribution of a model's scores has shifted
between a reference period (training or validation) and the current
production period.

PSI formula
-----------
  PSI = Σ (actual_pct_i - expected_pct_i) × ln(actual_pct_i / expected_pct_i)

Interpretation
--------------
  PSI < 0.10  : No significant change
  0.10 ≤ PSI < 0.25 : Moderate shift — investigate
  PSI ≥ 0.25  : Major shift — retrain model

Components
----------
DriftDetector      : Computes PSI between reference and current score distributions.
MonitoringPipeline : Consumes predictions topic, accumulates scores, computes PSI
                     on a configurable schedule, logs alerts when PSI > threshold.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from kafka_core.mock_kafka import MockKafkaBroker
from kafka_core.consumer import MockKafkaConsumer
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
_logger = _build_logger("ml_pipeline.monitoring", _CONFIG)


# ---------------------------------------------------------------------------
# DriftDetector
# ---------------------------------------------------------------------------

class DriftDetector:
    """
    Computes the Population Stability Index (PSI) between two score distributions.

    Parameters
    ----------
    n_bins          : Number of equal-width bins in [0, 1] for bucketing scores.
    alert_threshold : PSI value above which a drift alert is logged.
    """

    # Severity thresholds
    THRESHOLD_MODERATE = 0.10
    THRESHOLD_MAJOR = 0.25

    def __init__(
        self,
        n_bins: int = 10,
        alert_threshold: float = 0.10,
    ) -> None:
        self._n_bins = n_bins
        self._alert_threshold = alert_threshold
        self._psi_history: List[Tuple[float, float]] = []  # (timestamp, psi)
        _logger.info(
            f"DriftDetector initialised: n_bins={n_bins}, "
            f"alert_threshold={alert_threshold}"
        )

    def compute_psi(
        self,
        reference_scores: List[float],
        current_scores: List[float],
    ) -> float:
        """
        Compute PSI between *reference_scores* and *current_scores*.

        Parameters
        ----------
        reference_scores : Scores from the reference period (baseline).
        current_scores   : Scores from the current monitoring window.

        Returns
        -------
        PSI as a non-negative float.  Returns 0.0 if either list is empty.
        """
        if not reference_scores or not current_scores:
            _logger.warning(
                "PSI computation skipped: empty reference or current scores"
            )
            return 0.0

        bins = [i / self._n_bins for i in range(self._n_bins + 1)]
        ref_counts = self._bin_scores(reference_scores, bins)
        cur_counts = self._bin_scores(current_scores, bins)

        ref_total = sum(ref_counts)
        cur_total = sum(cur_counts)

        psi = 0.0
        epsilon = 1e-6  # avoid log(0)
        bin_contributions: List[Dict[str, Any]] = []

        for i in range(self._n_bins):
            ref_pct = (ref_counts[i] + epsilon) / (ref_total + epsilon * self._n_bins)
            cur_pct = (cur_counts[i] + epsilon) / (cur_total + epsilon * self._n_bins)
            contribution = (cur_pct - ref_pct) * math.log(cur_pct / ref_pct)
            psi += contribution
            bin_contributions.append({
                "bin": f"[{bins[i]:.1f}, {bins[i+1]:.1f})",
                "ref_pct": round(ref_pct, 4),
                "cur_pct": round(cur_pct, 4),
                "contribution": round(contribution, 6),
            })

        self._psi_history.append((time.time(), psi))
        severity = self._classify_psi(psi)

        _logger.info(
            f"PSI computed: psi={psi:.4f}, severity={severity!r}, "
            f"ref_n={len(reference_scores)}, cur_n={len(current_scores)}, "
            f"n_bins={self._n_bins}"
        )

        for bc in bin_contributions:
            _logger.debug(f"PSI bin: {bc}")

        if psi >= self._alert_threshold:
            _logger.warning(
                f"PSI ALERT: psi={psi:.4f} >= threshold={self._alert_threshold}, "
                f"severity={severity!r} — model drift detected, consider retraining"
            )

        return psi

    def compute_feature_psi(
        self,
        feature_name: str,
        reference_values: List[float],
        current_values: List[float],
        bin_edges: Optional[List[float]] = None,
    ) -> float:
        """
        Compute PSI for a single feature (not score).

        If *bin_edges* are not provided, equal-width bins are derived from the
        combined range of reference and current values.
        """
        if not reference_values or not current_values:
            return 0.0

        if bin_edges is None:
            min_val = min(min(reference_values), min(current_values))
            max_val = max(max(reference_values), max(current_values))
            step = (max_val - min_val) / self._n_bins or 1.0
            bin_edges = [min_val + i * step for i in range(self._n_bins + 1)]

        ref_counts = self._bin_scores(reference_values, bin_edges)
        cur_counts = self._bin_scores(current_values, bin_edges)
        ref_total = sum(ref_counts)
        cur_total = sum(cur_counts)
        epsilon = 1e-6
        psi = 0.0
        for i in range(len(bin_edges) - 1):
            rp = (ref_counts[i] + epsilon) / (ref_total + epsilon * len(ref_counts))
            cp = (cur_counts[i] + epsilon) / (cur_total + epsilon * len(cur_counts))
            psi += (cp - rp) * math.log(cp / rp)

        _logger.info(
            f"Feature PSI: feature={feature_name!r}, psi={psi:.4f}"
        )
        return psi

    def _bin_scores(self, scores: List[float], bins: List[float]) -> List[int]:
        """Count scores falling into each bin (right-exclusive except last)."""
        counts = [0] * (len(bins) - 1)
        for s in scores:
            # Clamp to [0, 1] range
            s = max(bins[0], min(bins[-1], s))
            for i in range(len(bins) - 1):
                if i == len(bins) - 2:
                    if bins[i] <= s <= bins[i + 1]:
                        counts[i] += 1
                        break
                else:
                    if bins[i] <= s < bins[i + 1]:
                        counts[i] += 1
                        break
        return counts

    def _classify_psi(self, psi: float) -> str:
        if psi < self.THRESHOLD_MODERATE:
            return "stable"
        elif psi < self.THRESHOLD_MAJOR:
            return "moderate_drift"
        else:
            return "major_drift"

    @property
    def psi_history(self) -> List[Tuple[float, float]]:
        return list(self._psi_history)

    def latest_psi(self) -> Optional[float]:
        if self._psi_history:
            return self._psi_history[-1][1]
        return None


# ---------------------------------------------------------------------------
# MonitoringPipeline
# ---------------------------------------------------------------------------

class MonitoringPipeline:
    """
    Consumes the predictions topic and computes PSI on accumulated scores.

    Parameters
    ----------
    broker              : Shared ``MockKafkaBroker`` instance.
    drift_detector      : ``DriftDetector`` instance.
    reference_scores    : Baseline score distribution from training/validation.
    group_id            : Kafka consumer group ID.
    source_topic        : Topic to consume predictions from.
    monitoring_topic    : Topic to publish monitoring events to.
    psi_compute_every_n : Compute PSI every N predictions consumed.
    sample_rate         : Fraction of predictions sampled for monitoring (0–1).
    """

    def __init__(
        self,
        broker: MockKafkaBroker,
        drift_detector: DriftDetector,
        reference_scores: List[float],
        group_id: str = "monitoring-pipeline",
        source_topic: str = "predictions",
        monitoring_topic: str = "monitoring",
        psi_compute_every_n: int = 10,
        sample_rate: float = 1.0,
    ) -> None:
        self._broker = broker
        self._detector = drift_detector
        self._reference_scores = reference_scores
        self._source_topic = source_topic
        self._monitoring_topic = monitoring_topic
        self._psi_every_n = psi_compute_every_n
        self._sample_rate = sample_rate

        self._consumer = MockKafkaConsumer(
            broker=broker,
            group_id=group_id,
            dlq_topic="dlq",
        )
        self._consumer.subscribe([source_topic])
        self._producer = MockKafkaProducer(broker=broker)

        if not broker.topic_exists(monitoring_topic):
            broker.create_topic(monitoring_topic, num_partitions=2, replication_factor=1)

        self._current_scores: List[float] = []
        self._consumed_count = 0
        self._psi_events: List[Dict[str, Any]] = []

        import random
        self._rng = random.Random(1234)

        _logger.info(
            f"MonitoringPipeline initialised: "
            f"source={source_topic!r}, "
            f"psi_every_n={psi_compute_every_n}, "
            f"sample_rate={sample_rate}, "
            f"reference_n={len(reference_scores)}"
        )

    def run(
        self,
        num_events: int,
        poll_timeout: float = 0.05,
    ) -> List[Dict[str, Any]]:
        """
        Consume up to *num_events* predictions and compute PSI periodically.

        Returns a list of PSI event records.
        """
        _logger.info(f"MonitoringPipeline.run: target={num_events}")

        for _ in range(num_events):
            msg = self._consumer.poll(timeout=poll_timeout)
            if msg is None:
                break

            try:
                payload = json.loads(msg.value.decode("utf-8"))
                score = float(payload.get("score", 0.0))
                user_id = payload.get("user_id", "unknown")
                model_id = payload.get("model_id", "unknown")

                # Apply sample rate
                if self._rng.random() <= self._sample_rate:
                    self._current_scores.append(score)

                self._consumed_count += 1
                self._consumer.commit()

                _logger.debug(
                    f"MonitoringPipeline consumed: user_id={user_id!r}, "
                    f"model={model_id!r}, score={score:.4f}"
                )

                # Compute PSI every N predictions
                if (
                    len(self._current_scores) > 0
                    and self._consumed_count % self._psi_every_n == 0
                ):
                    psi_value = self._detector.compute_psi(
                        self._reference_scores,
                        self._current_scores,
                    )
                    psi_event = {
                        "event_type": "PSIComputed",
                        "psi": round(psi_value, 6),
                        "current_n": len(self._current_scores),
                        "reference_n": len(self._reference_scores),
                        "timestamp": time.time(),
                        "severity": self._detector._classify_psi(psi_value),
                    }
                    self._psi_events.append(psi_event)
                    self._producer.produce(
                        topic=self._monitoring_topic,
                        key="psi_event",
                        value=psi_event,
                    )
                    _logger.info(f"PSI event published: {psi_event}")

            except Exception as exc:
                _logger.error(
                    f"MonitoringPipeline error: {exc!r}"
                )
                self._consumer.route_to_dlq(msg, exc)
                self._consumer.commit()

        self._producer.flush()
        _logger.info(
            f"MonitoringPipeline.run complete: "
            f"consumed={self._consumed_count}, "
            f"psi_events={len(self._psi_events)}"
        )
        return self._psi_events

    def close(self) -> None:
        self._consumer.close()

    @property
    def current_scores(self) -> List[float]:
        return list(self._current_scores)

    @property
    def consumed_count(self) -> int:
        return self._consumed_count


# ---------------------------------------------------------------------------
# Demo / main
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Demonstrate DriftDetector and MonitoringPipeline:
      * Stable baseline vs stable current → low PSI.
      * Stable baseline vs shifted current → high PSI.
    """
    import random
    from kafka_core.mock_kafka import MockKafkaBroker
    from kafka_core.producer import MockKafkaProducer
    from ml_pipeline.model_inference import MockModel
    from ml_pipeline.feature_pipeline import FeatureStore

    _logger.info("=== Monitoring demo start ===")

    rng = random.Random(42)

    # --- Scenario 1: No drift ---
    _logger.info("--- Scenario 1: Stable distribution (expect PSI < 0.10) ---")
    detector1 = DriftDetector(n_bins=10, alert_threshold=0.10)

    reference_scores = [rng.gauss(0.4, 0.1) for _ in range(200)]
    reference_scores = [max(0.0, min(1.0, s)) for s in reference_scores]

    current_scores_stable = [rng.gauss(0.41, 0.10) for _ in range(100)]
    current_scores_stable = [max(0.0, min(1.0, s)) for s in current_scores_stable]

    psi_stable = detector1.compute_psi(reference_scores, current_scores_stable)
    _logger.info(
        f"Scenario 1 PSI (stable): {psi_stable:.4f} "
        f"({'ALERT' if psi_stable >= 0.10 else 'OK'})"
    )

    # --- Scenario 2: Major drift ---
    _logger.info("--- Scenario 2: Shifted distribution (expect PSI >= 0.25) ---")
    detector2 = DriftDetector(n_bins=10, alert_threshold=0.10)

    current_scores_shifted = [rng.gauss(0.75, 0.08) for _ in range(100)]
    current_scores_shifted = [max(0.0, min(1.0, s)) for s in current_scores_shifted]

    psi_shifted = detector2.compute_psi(reference_scores, current_scores_shifted)
    _logger.info(
        f"Scenario 2 PSI (shifted): {psi_shifted:.4f} "
        f"({'ALERT' if psi_shifted >= 0.10 else 'OK'})"
    )

    assert psi_stable < psi_shifted, (
        f"Expected stable PSI ({psi_stable:.4f}) < shifted PSI ({psi_shifted:.4f})"
    )

    # --- Scenario 3: MonitoringPipeline consuming predictions topic ---
    _logger.info("--- Scenario 3: MonitoringPipeline end-to-end ---")

    broker = MockKafkaBroker()
    broker.create_topic("predictions", num_partitions=4, replication_factor=1)
    broker.create_topic("monitoring", num_partitions=2, replication_factor=1)
    broker.create_topic("dlq", num_partitions=1, replication_factor=1)

    # Produce synthetic prediction messages to the predictions topic
    producer = MockKafkaProducer(broker=broker)
    model = MockModel(model_id="v1", seed=42)
    feature_store = FeatureStore()

    for i in range(50):
        user_id = f"user-{i % 10:03d}"
        features = {
            "click_count": rng.randint(0, 20),
            "purchase_count": rng.randint(0, 3),
            "session_count": rng.randint(0, 10),
            "last_event_ts": time.time() - rng.uniform(0, 7200),
        }
        score = model.predict(features)
        producer.produce(
            topic="predictions",
            key=user_id,
            value={
                "user_id": user_id,
                "model_id": "v1",
                "score": score,
                "features": features,
                "timestamp": time.time(),
            },
        )
    producer.flush()
    _logger.info("Produced 50 prediction messages to predictions topic")

    # Reference: stable baseline
    reference = [rng.gauss(0.4, 0.1) for _ in range(200)]
    reference = [max(0.0, min(1.0, s)) for s in reference]

    detector3 = DriftDetector(n_bins=10, alert_threshold=0.10)
    monitor = MonitoringPipeline(
        broker=broker,
        drift_detector=detector3,
        reference_scores=reference,
        group_id="monitoring-demo",
        psi_compute_every_n=10,
        sample_rate=1.0,
    )

    psi_events = monitor.run(num_events=50)
    _logger.info(f"Monitoring pipeline PSI events: {psi_events}")
    _logger.info(
        f"Total consumed: {monitor.consumed_count}, "
        f"current_scores_n: {len(monitor.current_scores)}"
    )
    monitor.close()

    _logger.info("All scenarios complete")
    _logger.info("=== Monitoring demo complete ===")


if __name__ == "__main__":
    main()
