"""
Pandas Dtype Optimizer
======================
Demonstrates:
  - Category dtype for low-cardinality string columns (memory saving)
  - Downcast numerics with pd.to_numeric(downcast=)
  - int8/int16/float32 conversions
  - Memory usage reporting before and after

All constants loaded from config.yaml.

Run:
    python src/pandas_optimization/dtype_optimizer.py
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

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

class DtypeDataGenerator:
    """Creates a wide DataFrame with many string and numeric columns."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self._n = cfg["data"]["num_rows"]
        self._seed = cfg["data"]["random_seed"]
        self._cat_threshold = cfg["pandas"]["category_threshold"]
        self._rng = np.random.default_rng(self._seed)

    def build(self) -> pd.DataFrame:
        # Low-cardinality strings -> good candidates for category dtype
        region = self._rng.choice(["North", "South", "East", "West"], self._n)
        product = self._rng.choice(["Widget", "Gadget", "Doohickey"], self._n)
        status = self._rng.choice(["Active", "Inactive", "Pending"], self._n)
        # High-cardinality string -> NOT a candidate
        order_id = [f"ORD-{i:08d}" for i in self._rng.integers(0, self._n * 10, self._n)]

        df = pd.DataFrame(
            {
                "region": region,          # object -> category
                "product": product,        # object -> category
                "status": status,          # object -> category
                "order_id": order_id,      # high-card object — keep as object
                "revenue": self._rng.exponential(500, self._n).astype("float64"),
                "units": self._rng.integers(1, 1000, self._n).astype("int64"),
                "discount_pct": self._rng.uniform(0, 0.5, self._n).astype("float64"),
                "age": self._rng.integers(0, 120, self._n).astype("int64"),
                "score": self._rng.integers(0, 100, self._n).astype("int64"),
            }
        )
        logger.info("DtypeDataGenerator.build() -> shape=%s", df.shape)
        return df


# ---------------------------------------------------------------------------
# Memory reporter
# ---------------------------------------------------------------------------

class MemoryReporter:
    """Single responsibility: report DataFrame memory usage."""

    @staticmethod
    def report(df: pd.DataFrame, label: str) -> float:
        """Log and return total memory in MB."""
        total_bytes = df.memory_usage(deep=True).sum()
        mb = total_bytes / 1024 ** 2
        per_col = df.memory_usage(deep=True) / 1024 ** 2
        logger.info(
            "[MEM] %-25s: %.2f MB total | dtypes: %s",
            label,
            mb,
            df.dtypes.value_counts().to_dict(),
        )
        logger.debug("Per-column MB:\n%s", per_col.round(3).to_string())
        return mb


# ---------------------------------------------------------------------------
# Category dtype optimizer
# ---------------------------------------------------------------------------

class CategoryOptimizer:
    """Convert low-cardinality object columns to category dtype."""

    def __init__(self, threshold: int) -> None:
        self._threshold = threshold
        logger.info("CategoryOptimizer: threshold=%d unique values", threshold)

    def optimize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy with eligible object columns converted to category."""
        df = df.copy()
        converted: List[str] = []
        skipped: List[str] = []

        for col in df.select_dtypes(include="object").columns:
            n_unique = df[col].nunique()
            if n_unique < self._threshold:
                df[col] = df[col].astype("category")
                converted.append(col)
                logger.debug(
                    "  '%s': %d unique -> category", col, n_unique
                )
            else:
                skipped.append(col)
                logger.debug(
                    "  '%s': %d unique -> kept as object (>= threshold)", col, n_unique
                )

        logger.info(
            "CategoryOptimizer: converted=%s, skipped=%s", converted, skipped
        )
        return df


# ---------------------------------------------------------------------------
# Numeric downcast optimizer
# ---------------------------------------------------------------------------

class NumericDowncaster:
    """Downcast int64 -> smallest int type; float64 -> float32."""

    def optimize(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        int_cols = df.select_dtypes(include=["int64", "int32"]).columns
        float_cols = df.select_dtypes(include=["float64"]).columns

        for col in int_cols:
            original_dtype = df[col].dtype
            df[col] = pd.to_numeric(df[col], downcast="unsigned")
            logger.debug(
                "  int col '%s': %s -> %s", col, original_dtype, df[col].dtype
            )

        for col in float_cols:
            original_dtype = df[col].dtype
            df[col] = pd.to_numeric(df[col], downcast="float")
            logger.debug(
                "  float col '%s': %s -> %s", col, original_dtype, df[col].dtype
            )

        logger.info(
            "NumericDowncaster: processed %d int cols, %d float cols",
            len(int_cols),
            len(float_cols),
        )
        return df


# ---------------------------------------------------------------------------
# Full pipeline runner
# ---------------------------------------------------------------------------

class DtypeOptimizationRunner:
    """Orchestrates the full dtype optimization pipeline and reports savings."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self._cfg = cfg

    def run(self) -> None:
        logger.info("=== Dtype Optimizer START ===")
        cat_threshold = self._cfg["pandas"]["category_threshold"]

        df_raw = DtypeDataGenerator(self._cfg).build()
        reporter = MemoryReporter()
        cat_opt = CategoryOptimizer(threshold=cat_threshold)
        num_opt = NumericDowncaster()

        mb_before = reporter.report(df_raw, "original")

        df_cat = cat_opt.optimize(df_raw)
        mb_after_cat = reporter.report(df_cat, "after category")

        df_final = num_opt.optimize(df_cat)
        mb_after_num = reporter.report(df_final, "after numeric downcast")

        saving_pct = (1 - mb_after_num / mb_before) * 100
        logger.info(
            "Total memory reduction: %.2f MB -> %.2f MB (%.1f%% saved)",
            mb_before,
            mb_after_num,
            saving_pct,
        )
        logger.info("Final dtypes:\n%s", df_final.dtypes.to_string())
        logger.info("=== Dtype Optimizer END ===")


if __name__ == "__main__":
    _cfg = load_config()
    setup_logging(_cfg)
    DtypeOptimizationRunner(_cfg).run()
