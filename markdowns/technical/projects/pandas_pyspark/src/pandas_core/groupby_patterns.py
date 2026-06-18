"""
Pandas GroupBy Patterns
=======================
Covers:
  - agg() with multiple functions
  - Named aggregations (pandas >= 0.25)
  - transform() for group-level broadcast
  - filter() to drop groups by predicate
  - apply() for arbitrary per-group logic
  - Groupby on time-based (Grouper) and multi-column keys

All constants loaded from config.yaml.

Run:
    python src/pandas_core/groupby_patterns.py
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
# Data generation (single responsibility)
# ---------------------------------------------------------------------------

class GroupByDataGenerator:
    """Generates reproducible data for groupby demonstrations."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self._n = cfg["data"]["num_rows"]
        self._rng = np.random.default_rng(cfg["data"]["random_seed"])

    def build(self) -> pd.DataFrame:
        dates = pd.date_range("2022-01-01", periods=self._n, freq="h")
        df = pd.DataFrame(
            {
                "timestamp": dates,
                "region": self._rng.choice(["North", "South", "East", "West"], size=self._n),
                "product": self._rng.choice(["Widget", "Gadget", "Doohickey"], size=self._n),
                "salesperson_id": self._rng.integers(1, 21, size=self._n),
                "revenue": self._rng.exponential(scale=500, size=self._n).round(2),
                "units": self._rng.integers(1, 100, size=self._n),
                "discount_pct": self._rng.uniform(0, 0.30, size=self._n).round(4),
            }
        )
        logger.info("GroupByDataGenerator.build() -> shape=%s", df.shape)
        return df


# ---------------------------------------------------------------------------
# Agg patterns
# ---------------------------------------------------------------------------

class AggPatterns:
    """Multiple aggregation functions applied in a single pass."""

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def multi_agg(self) -> pd.DataFrame:
        """Apply list of functions per column via dict-style agg."""
        logger.info("--- multi_agg: agg with list of functions ---")
        result = self._df.groupby("region")["revenue"].agg(
            ["sum", "mean", "std", "min", "max"]
        )
        result.columns = [f"revenue_{f}" for f in result.columns]
        logger.info("multi_agg result:\n%s", result.to_string())
        return result

    def named_agg(self) -> pd.DataFrame:
        """Named aggregations — explicit output column names."""
        logger.info("--- named_agg: pandas NamedAgg syntax ---")
        result = (
            self._df.groupby(["region", "product"])
            .agg(
                total_revenue=pd.NamedAgg(column="revenue", aggfunc="sum"),
                mean_revenue=pd.NamedAgg(column="revenue", aggfunc="mean"),
                max_units=pd.NamedAgg(column="units", aggfunc="max"),
                num_transactions=pd.NamedAgg(column="revenue", aggfunc="count"),
                avg_discount=pd.NamedAgg(column="discount_pct", aggfunc="mean"),
            )
            .round(2)
            .reset_index()
        )
        logger.info("named_agg result shape=%s, columns=%s", result.shape, result.columns.tolist())
        logger.info("named_agg head:\n%s", result.head(8).to_string(index=False))
        return result

    def custom_agg_func(self) -> pd.DataFrame:
        """Custom lambda aggregation — inter-quartile range per group."""
        logger.info("--- custom_agg_func: IQR per group ---")
        iqr = lambda x: x.quantile(0.75) - x.quantile(0.25)  # noqa: E731
        result = (
            self._df.groupby("region")["revenue"]
            .agg(iqr)
            .rename("revenue_iqr")
            .round(2)
            .reset_index()
        )
        logger.info("IQR result:\n%s", result.to_string(index=False))
        return result

    def run(self) -> None:
        self.multi_agg()
        self.named_agg()
        self.custom_agg_func()


# ---------------------------------------------------------------------------
# Transform patterns
# ---------------------------------------------------------------------------

class TransformPatterns:
    """transform() broadcasts group-level results back to original index."""

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def group_mean_normalise(self) -> pd.DataFrame:
        """Subtract group mean (z-score per region-product)."""
        logger.info("--- transform: group-mean normalisation ---")
        df = self._df.copy()
        df["revenue_group_mean"] = df.groupby(["region", "product"])["revenue"].transform("mean")
        df["revenue_zscore"] = (
            (df["revenue"] - df["revenue_group_mean"])
            / df.groupby(["region", "product"])["revenue"].transform("std")
        ).round(4)
        logger.info(
            "zscore stats: mean=%.4f, std=%.4f",
            df["revenue_zscore"].mean(),
            df["revenue_zscore"].std(),
        )
        return df

    def cumsum_within_group(self) -> pd.DataFrame:
        """Cumulative revenue within each salesperson."""
        logger.info("--- transform: cumulative sum per salesperson ---")
        df = self._df.sort_values("timestamp").copy()
        df["cumrev"] = df.groupby("salesperson_id")["revenue"].transform("cumsum")
        logger.info("cumrev tail per salesperson:\n%s",
            df.groupby("salesperson_id")["cumrev"].last().head(5).to_string())
        return df

    def rank_within_group(self) -> pd.DataFrame:
        """Rank transactions within each region by revenue (descending)."""
        logger.info("--- transform: dense_rank within region ---")
        df = self._df.copy()
        df["rank"] = df.groupby("region")["revenue"].rank(method="dense", ascending=False)
        top = df[df["rank"] == 1].groupby("region")["revenue"].max()
        logger.info("Top revenue per region:\n%s", top.to_string())
        return df

    def run(self) -> None:
        self.group_mean_normalise()
        self.cumsum_within_group()
        self.rank_within_group()


# ---------------------------------------------------------------------------
# Filter patterns
# ---------------------------------------------------------------------------

class FilterPatterns:
    """filter() drops entire groups that fail a predicate."""

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def keep_large_groups(self) -> pd.DataFrame:
        """Keep only salesperson groups with >= 5000 transactions."""
        logger.info("--- filter: keep salesperson groups with >= 5000 rows ---")
        filtered = self._df.groupby("salesperson_id").filter(lambda g: len(g) >= 5000)
        logger.info(
            "Rows before=%d, after=%d (groups kept: %s)",
            len(self._df),
            len(filtered),
            sorted(filtered["salesperson_id"].unique().tolist()),
        )
        return filtered

    def keep_high_revenue_groups(self) -> pd.DataFrame:
        """Keep only region-product groups where mean revenue > 450."""
        logger.info("--- filter: keep groups where mean_revenue > 450 ---")
        filtered = self._df.groupby(["region", "product"]).filter(
            lambda g: g["revenue"].mean() > 450
        )
        logger.info(
            "Rows before=%d, after=%d",
            len(self._df),
            len(filtered),
        )
        return filtered

    def run(self) -> None:
        self.keep_large_groups()
        self.keep_high_revenue_groups()


# ---------------------------------------------------------------------------
# Apply patterns
# ---------------------------------------------------------------------------

class ApplyPatterns:
    """apply() for arbitrary per-group logic returning DataFrames."""

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def top_n_per_group(self, n: int = 3) -> pd.DataFrame:
        """Return top-n highest-revenue rows per region."""
        logger.info("--- apply: top-%d rows per region by revenue ---", n)
        top = (
            self._df.groupby("region", group_keys=False)
            .apply(lambda g: g.nlargest(n, "revenue"))
            .reset_index(drop=True)
        )
        logger.info("top_%d_per_region shape=%s", n, top.shape)
        logger.info("\n%s", top[["region", "product", "revenue"]].to_string(index=False))
        return top

    def run(self) -> None:
        self.top_n_per_group(3)


# ---------------------------------------------------------------------------
# Time-based grouping
# ---------------------------------------------------------------------------

class TimeGrouper:
    """Groupby with pd.Grouper for temporal resampling."""

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df.set_index("timestamp").sort_index()

    def monthly_totals(self) -> pd.DataFrame:
        logger.info("--- pd.Grouper: monthly revenue totals ---")
        monthly = (
            self._df.groupby([pd.Grouper(freq="ME"), "region"])["revenue"]
            .sum()
            .round(2)
            .reset_index()
        )
        logger.info("Monthly totals shape=%s, head:\n%s", monthly.shape, monthly.head(8).to_string(index=False))
        return monthly

    def run(self) -> None:
        self.monthly_totals()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class GroupByRunner:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self._cfg = cfg

    def run(self) -> None:
        logger.info("=== GroupBy Patterns START ===")
        df = GroupByDataGenerator(self._cfg).build()
        AggPatterns(df).run()
        TransformPatterns(df).run()
        FilterPatterns(df).run()
        ApplyPatterns(df).run()
        TimeGrouper(df).run()
        logger.info("=== GroupBy Patterns END ===")


if __name__ == "__main__":
    _cfg = load_config()
    setup_logging(_cfg)
    GroupByRunner(_cfg).run()
