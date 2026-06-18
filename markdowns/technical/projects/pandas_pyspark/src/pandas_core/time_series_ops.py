"""
Pandas Time-Series Operations
==============================
Covers:
  - Rolling windows (mean, std, min/max)
  - Exponentially-weighted mean (ewm)
  - Expanding windows
  - Shift / lag / lead
  - Resampling (resample vs asfreq)
  - Period arithmetic
  - Rolling with custom window functions

All constants loaded from config.yaml.

Run:
    python src/pandas_core/time_series_ops.py
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config_loader import load_config, setup_logging  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data generator
# ---------------------------------------------------------------------------

class TimeSeriesGenerator:
    """Generates an hourly time-series DataFrame for time-series demos."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self._n = cfg["data"]["num_rows"]
        self._seed = cfg["data"]["random_seed"]
        self._rng = np.random.default_rng(self._seed)

    def build(self) -> pd.DataFrame:
        """Daily OHLCV-style synthetic market data."""
        dates = pd.date_range("2020-01-01", periods=self._n, freq="h")
        noise = self._rng.normal(0, 1, size=self._n).cumsum()
        close = 100 + noise
        high = close + self._rng.uniform(0, 2, size=self._n)
        low = close - self._rng.uniform(0, 2, size=self._n)
        open_ = close + self._rng.normal(0, 0.5, size=self._n)
        volume = self._rng.integers(1_000, 10_000, size=self._n)

        df = pd.DataFrame(
            {
                "timestamp": dates,
                "open": open_.round(4),
                "high": high.round(4),
                "low": low.round(4),
                "close": close.round(4),
                "volume": volume,
            }
        ).set_index("timestamp")
        logger.info("TimeSeriesGenerator.build() -> shape=%s", df.shape)
        return df


# ---------------------------------------------------------------------------
# Rolling windows
# ---------------------------------------------------------------------------

class RollingWindowDemo:
    """Demonstrates rolling, min_periods, and custom functions."""

    def __init__(self, df: pd.DataFrame, window: int) -> None:
        self._df = df
        self._window = window
        logger.info("RollingWindowDemo: window=%d", window)

    def simple_rolling(self) -> pd.DataFrame:
        logger.info("--- Simple rolling mean / std ---")
        df = self._df[["close"]].copy()
        df[f"sma_{self._window}"] = df["close"].rolling(self._window).mean().round(4)
        df[f"rolling_std_{self._window}"] = df["close"].rolling(self._window).std().round(4)
        df[f"rolling_min_{self._window}"] = df["close"].rolling(self._window).min().round(4)
        df[f"rolling_max_{self._window}"] = df["close"].rolling(self._window).max().round(4)

        # Bollinger bands
        df["bb_upper"] = (df[f"sma_{self._window}"] + 2 * df[f"rolling_std_{self._window}"]).round(4)
        df["bb_lower"] = (df[f"sma_{self._window}"] - 2 * df[f"rolling_std_{self._window}"]).round(4)

        logger.info("Rolling indicators (last 5 non-NaN rows):\n%s", df.dropna().tail(5).to_string())
        return df

    def rolling_with_min_periods(self) -> pd.Series:
        logger.info("--- Rolling with min_periods=1 (no leading NaN) ---")
        result = self._df["close"].rolling(self._window, min_periods=1).mean()
        logger.info("First 5 values: %s", result.head(5).round(4).tolist())
        return result

    def rolling_custom_func(self) -> pd.Series:
        """Rolling range = max - min."""
        logger.info("--- Rolling custom function (high - low range) ---")
        price_range = self._df["close"].rolling(self._window).apply(
            lambda x: x.max() - x.min(), raw=True
        )
        logger.info(
            "Rolling range: mean=%.4f, max=%.4f",
            price_range.mean(),
            price_range.max(),
        )
        return price_range

    def run(self) -> None:
        self.simple_rolling()
        self.rolling_with_min_periods()
        self.rolling_custom_func()


# ---------------------------------------------------------------------------
# EWM
# ---------------------------------------------------------------------------

class EWMDemo:
    """Exponentially-weighted functions."""

    def __init__(self, df: pd.DataFrame, span: int) -> None:
        self._df = df
        self._span = span

    def run(self) -> pd.DataFrame:
        logger.info("--- EWM demo: span=%d ---", self._span)
        df = self._df[["close"]].copy()
        df[f"ema_{self._span}"] = df["close"].ewm(span=self._span, adjust=False).mean().round(4)
        df[f"ewm_std_{self._span}"] = df["close"].ewm(span=self._span, adjust=False).std().round(4)

        # MACD = EMA12 - EMA26
        ema12 = df["close"].ewm(span=12, adjust=False).mean()
        ema26 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = (ema12 - ema26).round(4)
        df["signal"] = df["macd"].ewm(span=9, adjust=False).mean().round(4)

        logger.info("EWM last 3 rows:\n%s", df.tail(3).to_string())
        return df


# ---------------------------------------------------------------------------
# Expanding windows
# ---------------------------------------------------------------------------

class ExpandingWindowDemo:
    """Expanding (cumulative) statistics."""

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def run(self) -> pd.DataFrame:
        logger.info("--- Expanding window demo ---")
        df = self._df[["close", "volume"]].copy()
        df["expanding_mean"] = df["close"].expanding().mean().round(4)
        df["expanding_max"] = df["close"].expanding().max().round(4)
        df["expanding_cumvol"] = df["volume"].expanding().sum()

        # Running Sharpe-like ratio (return / std)
        df["hourly_return"] = df["close"].pct_change()
        df["expanding_sharpe"] = (
            df["hourly_return"].expanding().mean()
            / df["hourly_return"].expanding().std()
        ).round(6)

        logger.info("Expanding stats tail:\n%s", df.dropna().tail(3).to_string())
        return df


# ---------------------------------------------------------------------------
# Shift / lag / lead
# ---------------------------------------------------------------------------

class ShiftDemo:
    """Lag and lead engineering features."""

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def run(self) -> pd.DataFrame:
        logger.info("--- Shift / lag / lead demo ---")
        df = self._df[["close"]].copy()
        df["lag_1"] = df["close"].shift(1)
        df["lag_7"] = df["close"].shift(7)
        df["lead_1"] = df["close"].shift(-1)

        df["return_1h"] = (df["close"] - df["lag_1"]) / df["lag_1"]
        df["return_7h"] = (df["close"] - df["lag_7"]) / df["lag_7"]

        logger.info(
            "return_1h: mean=%.6f, std=%.6f",
            df["return_1h"].mean(),
            df["return_1h"].std(),
        )
        logger.info("Shift sample:\n%s", df.dropna().head(3).to_string())
        return df


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

class ResampleDemo:
    """Downsample hourly data to daily OHLCV."""

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def run(self) -> pd.DataFrame:
        logger.info("--- Resample: hourly -> daily OHLCV ---")
        daily = self._df.resample("D").agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        daily.columns = ["d_open", "d_high", "d_low", "d_close", "d_volume"]

        logger.info("Daily OHLCV shape=%s, head:\n%s", daily.shape, daily.head(5).to_string())

        # asfreq vs resample: asfreq only reindexes, no aggregation
        weekly = self._df["close"].resample("W").last()
        logger.info("Weekly close shape=%d, head: %s", len(weekly), weekly.head(4).round(4).tolist())
        return daily


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class TimeSeriesRunner:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self._cfg = cfg

    def run(self) -> None:
        logger.info("=== Time Series Operations START ===")
        df = TimeSeriesGenerator(self._cfg).build()
        window = self._cfg["window"]["rolling_window"]
        span = self._cfg["window"]["ewm_span"]

        RollingWindowDemo(df, window).run()
        EWMDemo(df, span).run()
        ExpandingWindowDemo(df).run()
        ShiftDemo(df).run()
        ResampleDemo(df).run()
        logger.info("=== Time Series Operations END ===")


if __name__ == "__main__":
    _cfg = load_config()
    setup_logging(_cfg)
    TimeSeriesRunner(_cfg).run()
