"""
statistics_calculator.py

Descriptive statistics module implementing measures of central tendency,
spread, and shape. Reads all configuration from config.yaml.

Concepts illustrated:
  - Mean, median, mode (central tendency)
  - Trimmed mean, geometric mean, harmonic mean (robust / specialised means)
  - Variance, standard deviation, IQR, MAD, coefficient of variation (spread)
  - Percentiles, skewness, kurtosis (shape)
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import pathlib
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy import stats


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _build_logger(cfg: dict[str, Any]) -> logging.Logger:
    """Construct a logger with rotating-file + console handlers."""
    log_cfg = cfg["logging"]
    log_file = pathlib.Path(log_cfg["log_file"])
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("descriptive_stats.statistics_calculator")
    logger.setLevel(getattr(logging, log_cfg["level"].upper()))

    formatter = logging.Formatter(
        fmt=(
            '{"time": "%(asctime)s", "level": "%(levelname)s", '
            '"logger": "%(name)s", "message": "%(message)s"}'
        ),
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # Rotating file handler
    fh = logging.handlers.RotatingFileHandler(
        filename=log_file,
        maxBytes=log_cfg["max_bytes"],
        backupCount=log_cfg["backup_count"],
    )
    fh.setFormatter(formatter)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(config_path: str | pathlib.Path) -> dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, "r") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# StatisticsCalculator
# ---------------------------------------------------------------------------

class StatisticsCalculator:
    """
    Compute a comprehensive suite of descriptive statistics for a numeric
    array or Pandas Series.

    Responsibilities (SRP):
      - Central tendency measures
      - Spread / variability measures
      - Shape measures (skewness, kurtosis)
      - Percentile computation

    All hyper-parameters are sourced from the YAML config.
    """

    def __init__(self, cfg: dict[str, Any], logger: logging.Logger) -> None:
        self._cfg = cfg
        self._logger = logger
        self._ds_cfg = cfg["descriptive_stats"]
        self._trimmed_pct = self._ds_cfg["trimmed_mean_pct"]
        self._percentiles = self._ds_cfg["percentiles"]
        self._logger.info(
            "StatisticsCalculator initialised | "
            f"trimmed_pct={self._trimmed_pct} | "
            f"percentiles={self._percentiles}"
        )

    # ------------------------------------------------------------------
    # Central Tendency
    # ------------------------------------------------------------------

    def arithmetic_mean(self, data: np.ndarray) -> float:
        result = float(np.mean(data))
        self._logger.debug(f"arithmetic_mean={result:.6f}")
        return result

    def median(self, data: np.ndarray) -> float:
        result = float(np.median(data))
        self._logger.debug(f"median={result:.6f}")
        return result

    def mode(self, data: np.ndarray) -> float:
        """Return the most frequent value; uses scipy for continuous data."""
        result = float(stats.mode(data, keepdims=True).mode[0])
        self._logger.debug(f"mode={result:.6f}")
        return result

    def trimmed_mean(self, data: np.ndarray) -> float:
        """
        Trimmed (truncated) mean - excludes the lowest and highest
        `trimmed_pct` fraction of observations, making it robust to outliers.
        """
        result = float(stats.trim_mean(data, self._trimmed_pct))
        self._logger.debug(
            f"trimmed_mean(pct={self._trimmed_pct})={result:.6f}"
        )
        return result

    def geometric_mean(self, data: np.ndarray) -> float:
        """
        Geometric mean - appropriate for ratios and growth rates.
        Requires all positive values.
        """
        if np.any(data <= 0):
            raise ValueError(
                "Geometric mean requires strictly positive values."
            )
        result = float(stats.gmean(data))
        self._logger.debug(f"geometric_mean={result:.6f}")
        return result

    def harmonic_mean(self, data: np.ndarray) -> float:
        """
        Harmonic mean - appropriate for rates (e.g., speeds, P/E ratios).
        Requires all positive values.
        """
        if np.any(data <= 0):
            raise ValueError(
                "Harmonic mean requires strictly positive values."
            )
        result = float(stats.hmean(data))
        self._logger.debug(f"harmonic_mean={result:.6f}")
        return result

    # ------------------------------------------------------------------
    # Spread / Variability
    # ------------------------------------------------------------------

    def variance(self, data: np.ndarray, ddof: int = 1) -> float:
        """Sample variance (ddof=1 by default)."""
        result = float(np.var(data, ddof=ddof))
        self._logger.debug(f"variance(ddof={ddof})={result:.6f}")
        return result

    def std_dev(self, data: np.ndarray, ddof: int = 1) -> float:
        """Sample standard deviation (ddof=1 by default)."""
        result = float(np.std(data, ddof=ddof))
        self._logger.debug(f"std_dev(ddof={ddof})={result:.6f}")
        return result

    def iqr(self, data: np.ndarray) -> float:
        """
        Inter-Quartile Range = Q3 - Q1.
        Robust measure of spread - unaffected by outliers.
        """
        result = float(stats.iqr(data))
        self._logger.debug(f"IQR={result:.6f}")
        return result

    def mad(self, data: np.ndarray) -> float:
        """
        Median Absolute Deviation - the most robust scale estimator.
        MAD = median(|x_i - median(x)|)
        """
        med = np.median(data)
        result = float(np.median(np.abs(data - med)))
        self._logger.debug(f"MAD={result:.6f}")
        return result

    def coefficient_of_variation(self, data: np.ndarray) -> float:
        """
        CV = std / mean - dimensionless measure of relative variability.
        Useful for comparing datasets with different units or magnitudes.
        """
        mu = np.mean(data)
        if mu == 0:
            raise ValueError("CV undefined when mean is zero.")
        result = float(np.std(data, ddof=1) / mu)
        self._logger.debug(f"CV={result:.6f}")
        return result

    # ------------------------------------------------------------------
    # Shape
    # ------------------------------------------------------------------

    def skewness(self, data: np.ndarray) -> float:
        """
        Pearson's moment coefficient of skewness.
        > 0 : right-skewed (long tail on the right)
        < 0 : left-skewed  (long tail on the left)
        = 0 : symmetric
        """
        result = float(stats.skew(data))
        self._logger.debug(f"skewness={result:.6f}")
        return result

    def kurtosis(self, data: np.ndarray) -> float:
        """
        Excess kurtosis (Fisher definition; normal distribution = 0).
        > 0 : leptokurtic (heavy tails)
        < 0 : platykurtic (light tails)
        """
        result = float(stats.kurtosis(data))
        self._logger.debug(f"excess_kurtosis={result:.6f}")
        return result

    # ------------------------------------------------------------------
    # Percentiles
    # ------------------------------------------------------------------

    def percentiles(self, data: np.ndarray) -> dict[str, float]:
        """Compute all configured percentiles."""
        results: dict[str, float] = {}
        for p in self._percentiles:
            label = f"P{int(p * 100)}"
            results[label] = float(np.quantile(data, p))
        self._logger.info(f"Percentiles computed: {results}")
        return results

    # ------------------------------------------------------------------
    # Summary report
    # ------------------------------------------------------------------

    def full_summary(self, data: np.ndarray, label: str = "data") -> dict[str, Any]:
        """
        Compute and return all statistics in a single dictionary.
        Also logs each statistic at INFO level.
        """
        self._logger.info(
            f"Computing full summary statistics for '{label}' "
            f"(n={len(data)})"
        )

        # Positive-only stats guarded separately
        try:
            geo_mean = self.geometric_mean(data)
        except ValueError as exc:
            geo_mean = None
            self._logger.warning(f"geometric_mean skipped: {exc}")

        try:
            har_mean = self.harmonic_mean(data)
        except ValueError as exc:
            har_mean = None
            self._logger.warning(f"harmonic_mean skipped: {exc}")

        try:
            cv = self.coefficient_of_variation(data)
        except ValueError as exc:
            cv = None
            self._logger.warning(f"coefficient_of_variation skipped: {exc}")

        summary: dict[str, Any] = {
            "label": label,
            "n": len(data),
            # Central tendency
            "mean": self.arithmetic_mean(data),
            "median": self.median(data),
            "mode": self.mode(data),
            "trimmed_mean": self.trimmed_mean(data),
            "geometric_mean": geo_mean,
            "harmonic_mean": har_mean,
            # Spread
            "variance": self.variance(data),
            "std_dev": self.std_dev(data),
            "iqr": self.iqr(data),
            "mad": self.mad(data),
            "coefficient_of_variation": cv,
            # Shape
            "skewness": self.skewness(data),
            "kurtosis": self.kurtosis(data),
            # Percentiles
            **self.percentiles(data),
        }

        self._logger.info(
            f"Summary for '{label}': mean={summary['mean']:.4f}, "
            f"std={summary['std_dev']:.4f}, skew={summary['skewness']:.4f}, "
            f"kurt={summary['kurtosis']:.4f}"
        )
        return summary

    def summary_to_dataframe(self, summary: dict[str, Any]) -> pd.DataFrame:
        """Convert summary dict to a single-row DataFrame for display."""
        return pd.DataFrame([summary])


# ---------------------------------------------------------------------------
# Synthetic data generator (for standalone execution)
# ---------------------------------------------------------------------------

class SyntheticDataGenerator:
    """
    Generate realistic synthetic datasets for demonstration.
    SRP: only responsible for data generation.
    """

    def __init__(self, cfg: dict[str, Any], logger: logging.Logger) -> None:
        self._cfg = cfg
        self._logger = logger
        self._n = cfg["data"]["sample_size"]
        self._seed = cfg["data"]["random_seed"]
        self._rng = np.random.default_rng(self._seed)
        self._logger.info(
            f"SyntheticDataGenerator | n={self._n}, seed={self._seed}"
        )

    def normal_sample(self, mu: float = 50.0, sigma: float = 10.0) -> np.ndarray:
        """Standard normal-distributed sample."""
        data = self._rng.normal(mu, sigma, self._n)
        self._logger.info(
            f"Generated normal sample | mu={mu}, sigma={sigma}, n={self._n}"
        )
        return data

    def skewed_sample(self, a: float = 5.0) -> np.ndarray:
        """Chi-squared sample (positively skewed)."""
        data = self._rng.chisquare(df=a, size=self._n)
        self._logger.info(
            f"Generated skewed (chi2) sample | df={a}, n={self._n}"
        )
        return data

    def with_outliers(self, base: np.ndarray, fraction: float = 0.02) -> np.ndarray:
        """Inject heavy outliers into a copy of *base*."""
        data = base.copy()
        n_out = max(1, int(fraction * len(data)))
        indices = self._rng.choice(len(data), n_out, replace=False)
        data[indices] = self._rng.uniform(
            base.mean() + 5 * base.std(),
            base.mean() + 10 * base.std(),
            n_out,
        )
        self._logger.info(
            f"Injected {n_out} outliers ({fraction*100:.1f}% of {len(data)})"
        )
        return data


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    config_path = pathlib.Path(__file__).parents[2] / "config.yaml"
    cfg = load_config(config_path)
    logger = _build_logger(cfg)

    logger.info("=== StatisticsCalculator demo start ===")

    generator = SyntheticDataGenerator(cfg, logger)
    calculator = StatisticsCalculator(cfg, logger)

    # Dataset 1: Normal
    normal_data = generator.normal_sample(mu=50.0, sigma=10.0)
    summary_normal = calculator.full_summary(normal_data, label="Normal(50,10)")

    # Dataset 2: Skewed
    skewed_data = generator.skewed_sample(a=3.0)
    summary_skewed = calculator.full_summary(skewed_data, label="Chi2(df=3)")

    # Dataset 3: With outliers
    outlier_data = generator.with_outliers(normal_data, fraction=0.03)
    summary_outlier = calculator.full_summary(outlier_data, label="Normal+Outliers")

    # Print comparison
    df = pd.concat(
        [
            calculator.summary_to_dataframe(summary_normal),
            calculator.summary_to_dataframe(summary_skewed),
            calculator.summary_to_dataframe(summary_outlier),
        ],
        ignore_index=True,
    )
    print("\n=== Descriptive Statistics Summary ===")
    cols = [
        "label", "n", "mean", "median", "trimmed_mean",
        "std_dev", "iqr", "mad", "skewness", "kurtosis",
        "P25", "P50", "P75", "P99",
    ]
    print(df[cols].to_string(index=False))

    logger.info("=== StatisticsCalculator demo complete ===")


if __name__ == "__main__":
    main()
