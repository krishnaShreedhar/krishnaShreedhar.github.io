"""
Pandas DataFrame Operations
===========================
Demonstrates indexing (.loc, .iloc, .at, .iat), merging (inner/left/right/outer),
reshaping (melt, pivot, stack, unstack, crosstab), MultiIndex creation and operations,
and the copy-vs-view distinction — all driven from config.yaml.

Run this module directly:
    python src/pandas_core/dataframe_operations.py
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Bootstrap path so that sibling imports work when run directly
# ---------------------------------------------------------------------------
_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config_loader import load_config, setup_logging  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------

class SyntheticDataGenerator:
    """Generates reproducible synthetic DataFrames for all examples.

    Responsibilities (SRP): only generates data, no analysis logic.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self._seed: int = cfg["data"]["random_seed"]
        self._n: int = cfg["data"]["num_rows"]
        self._rng = np.random.default_rng(self._seed)
        logger.info(
            "SyntheticDataGenerator initialised: seed=%d, n=%d",
            self._seed,
            self._n,
        )

    def sales_df(self) -> pd.DataFrame:
        """Return a sales-style DataFrame with dates, categories, and numerics."""
        dates = pd.date_range("2022-01-01", periods=self._n, freq="h")
        regions = self._rng.choice(["North", "South", "East", "West"], size=self._n)
        products = self._rng.choice(["Widget", "Gadget", "Doohickey"], size=self._n)
        salesperson_ids = self._rng.integers(1, 21, size=self._n)
        revenue = self._rng.exponential(scale=500, size=self._n).round(2)
        units = self._rng.integers(1, 100, size=self._n)

        df = pd.DataFrame(
            {
                "timestamp": dates,
                "region": regions,
                "product": products,
                "salesperson_id": salesperson_ids,
                "revenue": revenue,
                "units": units,
            }
        )
        logger.debug("sales_df shape: %s", df.shape)
        return df

    def employee_df(self) -> pd.DataFrame:
        """Return a small employee lookup DataFrame for join demonstrations."""
        n = 20
        df = pd.DataFrame(
            {
                "salesperson_id": range(1, n + 1),
                "name": [f"Employee_{i}" for i in range(1, n + 1)],
                "department": self._rng.choice(
                    ["Sales", "Marketing", "Operations"], size=n
                ),
                "hire_year": self._rng.integers(2010, 2023, size=n),
            }
        )
        return df


# ---------------------------------------------------------------------------
# Indexing demonstrations
# ---------------------------------------------------------------------------

class IndexingDemo:
    """Shows loc, iloc, at, iat usage on a real DataFrame.

    Open/Closed: extend by adding new demo methods without changing existing ones.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df.copy()
        logger.info("IndexingDemo initialised with DataFrame shape %s", df.shape)

    def demo_loc(self) -> None:
        """Label-based selection with .loc."""
        logger.info("--- .loc demo ---")

        # Row slice by label (default RangeIndex, so same as positional here)
        first_ten = self._df.loc[0:9, ["region", "revenue"]]
        logger.info(".loc first 10 rows: shape=%s", first_ten.shape)

        # Boolean mask
        high_revenue = self._df.loc[self._df["revenue"] > 1000, :]
        logger.info(
            ".loc boolean filter (revenue > 1000): %d rows", len(high_revenue)
        )

        # Set value via .loc — avoids SettingWithCopyWarning
        sample = self._df.copy()
        sample.loc[sample["region"] == "North", "revenue"] *= 1.1
        logger.info(
            ".loc scalar set — North revenue increased by 10%%: mean=%.2f",
            sample.loc[sample["region"] == "North", "revenue"].mean(),
        )

    def demo_iloc(self) -> None:
        """Integer-position-based selection with .iloc."""
        logger.info("--- .iloc demo ---")

        first_col = self._df.iloc[:, 0]
        logger.info(".iloc first column ('%s') head: %s", first_col.name, first_col.head(3).tolist())

        block = self._df.iloc[100:110, 2:5]
        logger.info(".iloc block [100:110, 2:5]: shape=%s, columns=%s", block.shape, block.columns.tolist())

    def demo_at_iat(self) -> None:
        """Scalar accessors .at and .iat (fastest for single-cell access)."""
        logger.info("--- .at / .iat demo ---")

        val_at = self._df.at[0, "revenue"]
        logger.info(".at[0, 'revenue'] = %.4f", val_at)

        val_iat = self._df.iat[0, 4]
        logger.info(".iat[0, 4] = %.4f (same cell)", val_iat)

        assert abs(val_at - val_iat) < 1e-9, "at and iat must return same value"
        logger.info("Assertion passed: .at == .iat for same cell")

    def run(self) -> None:
        self.demo_loc()
        self.demo_iloc()
        self.demo_at_iat()


# ---------------------------------------------------------------------------
# Merge / Join demonstrations
# ---------------------------------------------------------------------------

class MergeDemo:
    """Demonstrates inner, left, right, outer joins and indicator usage."""

    def __init__(self, sales: pd.DataFrame, employees: pd.DataFrame) -> None:
        self._sales = sales
        self._emp = employees
        logger.info(
            "MergeDemo: sales=%s, employees=%s", sales.shape, employees.shape
        )

    def _log_merge(self, kind: str, result: pd.DataFrame) -> None:
        logger.info(
            "  %s join -> shape=%s, nulls_in_name=%d",
            kind,
            result.shape,
            result["name"].isna().sum() if "name" in result.columns else -1,
        )

    def run(self) -> None:
        logger.info("--- Merge/Join demo ---")
        sales_sample = self._sales.head(500)

        inner = pd.merge(sales_sample, self._emp, on="salesperson_id", how="inner")
        self._log_merge("inner", inner)

        left = pd.merge(sales_sample, self._emp, on="salesperson_id", how="left")
        self._log_merge("left", left)

        right = pd.merge(sales_sample, self._emp, on="salesperson_id", how="right")
        self._log_merge("right", right)

        outer = pd.merge(sales_sample, self._emp, on="salesperson_id", how="outer")
        self._log_merge("outer", outer)

        # indicator column
        indicator = pd.merge(
            sales_sample, self._emp, on="salesperson_id", how="outer", indicator=True
        )
        logger.info(
            "Merge indicator value_counts:\n%s", indicator["_merge"].value_counts().to_string()
        )


# ---------------------------------------------------------------------------
# Reshaping demonstrations
# ---------------------------------------------------------------------------

class ReshapingDemo:
    """melt, pivot, stack, unstack, crosstab."""

    def __init__(self, df: pd.DataFrame) -> None:
        # Use a small summary for reshaping to keep it tractable
        self._summary = (
            df.groupby(["region", "product"])["revenue"]
            .mean()
            .round(2)
            .reset_index()
            .rename(columns={"revenue": "avg_revenue"})
        )
        logger.info("ReshapingDemo summary shape: %s", self._summary.shape)

    def demo_pivot(self) -> pd.DataFrame:
        """Wide-format: products as columns."""
        logger.info("--- pivot demo ---")
        pivoted = self._summary.pivot(
            index="region", columns="product", values="avg_revenue"
        )
        logger.info("Pivoted shape: %s\n%s", pivoted.shape, pivoted.to_string())
        return pivoted

    def demo_melt(self, wide: pd.DataFrame) -> pd.DataFrame:
        """Back to long format from the pivoted DataFrame."""
        logger.info("--- melt demo ---")
        long = wide.reset_index().melt(
            id_vars="region", var_name="product", value_name="avg_revenue"
        )
        logger.info("Melted shape: %s", long.shape)
        return long

    def demo_stack_unstack(self, wide: pd.DataFrame) -> None:
        """MultiIndex stack/unstack roundtrip."""
        logger.info("--- stack / unstack demo ---")
        stacked = wide.stack()
        logger.info("Stacked type=%s, len=%d", type(stacked).__name__, len(stacked))
        unstacked = stacked.unstack()
        logger.info("Unstacked back to shape=%s", unstacked.shape)

    def demo_crosstab(self, df: pd.DataFrame) -> None:
        logger.info("--- crosstab demo ---")
        ct = pd.crosstab(
            df["region"],
            df["product"],
            values=df["revenue"],
            aggfunc="mean",
        ).round(2)
        logger.info("Crosstab:\n%s", ct.to_string())

    def run(self, df: pd.DataFrame) -> None:
        wide = self.demo_pivot()
        self.demo_melt(wide)
        self.demo_stack_unstack(wide)
        self.demo_crosstab(df)


# ---------------------------------------------------------------------------
# MultiIndex demonstrations
# ---------------------------------------------------------------------------

class MultiIndexDemo:
    """MultiIndex creation and .xs / .swaplevel / .sortlevel operations."""

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df
        logger.info("MultiIndexDemo initialised")

    def run(self) -> None:
        logger.info("--- MultiIndex demo ---")

        # Create MultiIndex from groupby
        mi_df = (
            self._df.groupby(["region", "product", "salesperson_id"])["revenue"]
            .sum()
            .round(2)
        )
        logger.info("MultiIndex Series shape: %s, levels: %s", mi_df.shape, mi_df.index.names)

        # .xs — cross-section
        north_slice = mi_df.xs("North", level="region")
        logger.info(".xs('North', level='region') shape: %s", north_slice.shape)

        # swaplevel
        swapped = mi_df.swaplevel("region", "product")
        logger.info("After swaplevel: first index names=%s", swapped.index.names)

        # Unstack last level
        unstacked = mi_df.unstack(level="salesperson_id")
        logger.info("Unstacked salesperson_id -> shape: %s", unstacked.shape)

        # Create MultiIndex from scratch
        arrays = [
            np.array(["A", "A", "B", "B"]),
            np.array(["x", "y", "x", "y"]),
        ]
        mi = pd.MultiIndex.from_arrays(arrays, names=["first", "second"])
        s = pd.Series([10, 20, 30, 40], index=mi)
        logger.info("Manually created MultiIndex Series:\n%s", s.to_string())


# ---------------------------------------------------------------------------
# Copy vs View
# ---------------------------------------------------------------------------

class CopyVsViewDemo:
    """Illustrates pandas copy-on-write semantics and the SettingWithCopyWarning trap."""

    def run(self) -> None:
        logger.info("--- Copy vs View demo ---")
        base = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        # Slice — may be a view or copy depending on pandas version / operation
        # Always use .copy() when you intend an independent copy
        safe_copy = base[["a"]].copy()
        safe_copy["a"] = 99
        logger.info(
            "base['a'] unchanged after modifying safe_copy: %s",
            base["a"].tolist(),
        )

        # Using .loc assignment on the original (safe in-place mutation)
        direct = base.copy()
        direct.loc[0, "b"] = 999
        logger.info("direct.loc mutation: b[0]=%d (base b[0]=%d)", direct.at[0, "b"], base.at[0, "b"])

        # Demonstrate that chained indexing can be unpredictable
        logger.info(
            "Copy-vs-view rule of thumb: always call .copy() when slicing "
            "and you intend a new DataFrame; use .loc for in-place mutations."
        )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class DataFrameOperationsRunner:
    """Top-level orchestrator — composes and runs all sub-demos."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self._cfg = cfg
        self._gen = SyntheticDataGenerator(cfg)

    def run(self) -> None:
        logger.info("=== DataFrame Operations START ===")

        sales = self._gen.sales_df()
        employees = self._gen.employee_df()

        IndexingDemo(sales).run()
        MergeDemo(sales, employees).run()
        ReshapingDemo(sales).run(sales)
        MultiIndexDemo(sales).run()
        CopyVsViewDemo().run()

        logger.info("=== DataFrame Operations END ===")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _cfg = load_config()
    setup_logging(_cfg)
    DataFrameOperationsRunner(_cfg).run()
