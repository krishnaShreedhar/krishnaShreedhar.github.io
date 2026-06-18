"""
Pandas Performance Patterns
============================
Benchmarks and demonstrates:
  - Vectorised operations vs iterrows (with timing)
  - query() and eval() with numexpr backend
  - pipe() for clean transformation chains
  - Polars comparison (syntax-level, no execution dependency required)

All constants loaded from config.yaml.

Run:
    python src/pandas_optimization/performance_patterns.py
"""

import logging
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config_loader import load_config, setup_logging  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _timeit(label: str, fn: Callable, *args, **kwargs):
    """Time a function call and log the result."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    logger.info("  [TIMER] %-40s %.4f s", label, elapsed)
    return result, elapsed


# ---------------------------------------------------------------------------
# Data generator
# ---------------------------------------------------------------------------

class PerfDataGenerator:
    """Synthetic DataFrame for performance benchmarks."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self._n = cfg["data"]["num_rows"]
        self._rng = np.random.default_rng(cfg["data"]["random_seed"])

    def build(self) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "a": self._rng.uniform(0, 100, self._n),
                "b": self._rng.uniform(0, 100, self._n),
                "c": self._rng.integers(1, 10, self._n),
                "category": self._rng.choice(["X", "Y", "Z", "W"], self._n),
                "revenue": self._rng.exponential(500, self._n),
                "cost": self._rng.exponential(300, self._n),
            }
        )
        logger.info("PerfDataGenerator.build() -> %s rows", self._n)
        return df


# ---------------------------------------------------------------------------
# Vectorised vs iterrows
# ---------------------------------------------------------------------------

class VectorisedVsIterrows:
    """Direct benchmark: compute (a*b + c) using three approaches."""

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def _compute_iterrows(self) -> pd.Series:
        results = []
        for _, row in self._df.iterrows():
            results.append(row["a"] * row["b"] + row["c"])
        return pd.Series(results, dtype=float)

    def _compute_itertuples(self) -> List[float]:
        return [row.a * row.b + row.c for row in self._df.itertuples(index=False)]

    def _compute_vectorised(self) -> pd.Series:
        return self._df["a"] * self._df["b"] + self._df["c"]

    def run(self) -> Dict[str, float]:
        logger.info("--- Vectorised vs iterrows benchmark ---")
        n = min(5_000, len(self._df))  # iterrows is O(n) and slow — cap for demo
        sample = self._df.head(n).reset_index(drop=True)

        _, t_iter = _timeit("iterrows (n=%d)" % n, self.__class__._compute_iterrows, self)
        _, t_ituples = _timeit(
            "itertuples (n=%d)" % n,
            lambda s: [r.a * r.b + r.c for r in s.itertuples(index=False)],
            sample,
        )
        _, t_vec = _timeit("vectorised (full n=%d)" % len(self._df), self._compute_vectorised)

        speedup = t_iter / (t_vec + 1e-12)
        logger.info("  Speedup of vectorised vs iterrows: %.0fx", speedup)
        return {"iterrows": t_iter, "itertuples": t_ituples, "vectorised": t_vec}


# ---------------------------------------------------------------------------
# query() and eval()
# ---------------------------------------------------------------------------

class QueryEvalDemo:
    """Demonstrates query() and eval() — can use numexpr if installed."""

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def demo_query(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        logger.info("--- query() vs boolean mask ---")
        expr = "a > 50 and b < 30 and revenue > 400"

        _, t_bool = _timeit(
            "boolean mask",
            lambda: self._df[(self._df["a"] > 50) & (self._df["b"] < 30) & (self._df["revenue"] > 400)],
        )
        result_q, t_query = _timeit(
            "query()",
            lambda: self._df.query(expr),
        )
        logger.info(
            "query result shape=%s | bool_t=%.4fs  query_t=%.4fs",
            result_q.shape,
            t_bool,
            t_query,
        )
        return result_q

    def demo_eval(self) -> None:
        logger.info("--- eval() for in-place column creation ---")
        df = self._df.copy()

        _, t_eval = _timeit(
            "eval() new column",
            lambda: df.eval("profit = revenue - cost", inplace=True),
        )
        _, t_native = _timeit(
            "native assignment",
            lambda: df.__setitem__("profit2", df["revenue"] - df["cost"]),
        )
        logger.info(
            "eval_t=%.4fs  native_t=%.4fs | profit head=%s",
            t_eval,
            t_native,
            df["profit"].head(3).round(2).tolist(),
        )

    def run(self) -> None:
        self.demo_query()
        self.demo_eval()


# ---------------------------------------------------------------------------
# pipe()
# ---------------------------------------------------------------------------

class PipeDemo:
    """Demonstrates pipe() for readable, composable transformation chains."""

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    @staticmethod
    def _add_profit(df: pd.DataFrame) -> pd.DataFrame:
        return df.assign(profit=df["revenue"] - df["cost"])

    @staticmethod
    def _add_margin(df: pd.DataFrame) -> pd.DataFrame:
        return df.assign(margin=(df["profit"] / df["revenue"]).round(4))

    @staticmethod
    def _flag_high_margin(df: pd.DataFrame, threshold: float = 0.4) -> pd.DataFrame:
        return df.assign(high_margin=df["margin"] > threshold)

    @staticmethod
    def _filter_positive_profit(df: pd.DataFrame) -> pd.DataFrame:
        return df[df["profit"] > 0]

    def run(self) -> pd.DataFrame:
        logger.info("--- pipe() transformation chain ---")
        result = (
            self._df
            .pipe(self._add_profit)
            .pipe(self._filter_positive_profit)
            .pipe(self._add_margin)
            .pipe(self._flag_high_margin, threshold=0.4)
        )
        logger.info(
            "pipe() result: shape=%s, high_margin_count=%d",
            result.shape,
            result["high_margin"].sum(),
        )
        logger.info("Sample:\n%s", result[["revenue", "cost", "profit", "margin", "high_margin"]].head(5).to_string(index=False))
        return result


# ---------------------------------------------------------------------------
# Polars syntax comparison
# ---------------------------------------------------------------------------

class PolarsComparison:
    """
    Shows the Polars equivalent syntax for common Pandas operations.

    Polars is NOT executed here (avoids a hard dependency) — the code is
    presented as string documentation so learners can see the API difference.
    """

    def run(self) -> None:
        logger.info("--- Polars vs Pandas syntax comparison ---")

        comparisons = {
            "Filter rows": (
                "pandas: df[df['a'] > 50]",
                "polars: df.filter(pl.col('a') > 50)",
            ),
            "Select columns": (
                "pandas: df[['a', 'b']]",
                "polars: df.select(['a', 'b'])",
            ),
            "Add column": (
                "pandas: df.assign(profit=df['revenue'] - df['cost'])",
                "polars: df.with_columns((pl.col('revenue') - pl.col('cost')).alias('profit'))",
            ),
            "GroupBy agg": (
                "pandas: df.groupby('category').agg(total=('revenue', 'sum'))",
                "polars: df.group_by('category').agg(pl.col('revenue').sum().alias('total'))",
            ),
            "Sort": (
                "pandas: df.sort_values('revenue', ascending=False)",
                "polars: df.sort('revenue', descending=True)",
            ),
            "Lazy evaluation": (
                "pandas: eager by default (no lazy graph)",
                "polars: df.lazy().filter(...).collect()  # deferred execution",
            ),
        }
        for concept, (pd_api, pl_api) in comparisons.items():
            logger.info("  %-20s | %-55s | %s", concept, pd_api, pl_api)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class PerformanceRunner:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self._cfg = cfg

    def run(self) -> None:
        logger.info("=== Performance Patterns START ===")
        df = PerfDataGenerator(self._cfg).build()
        VectorisedVsIterrows(df).run()
        QueryEvalDemo(df).run()
        PipeDemo(df).run()
        PolarsComparison().run()
        logger.info("=== Performance Patterns END ===")


if __name__ == "__main__":
    _cfg = load_config()
    setup_logging(_cfg)
    PerformanceRunner(_cfg).run()
