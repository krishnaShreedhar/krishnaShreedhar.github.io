"""
PySpark Window Functions
=========================
Demonstrates:
  - rank(), dense_rank(), row_number()
  - lag(), lead()
  - Rolling mean (rangeBetween / rowsBetween)
  - Cumulative sum within partition
  - ntile()

All constants loaded from config.yaml.

Run:
    python src/pyspark_core/window_functions.py
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

class WindowDataGenerator:
    def __init__(self, spark, cfg: Dict[str, Any]) -> None:
        self._spark = spark
        self._n = cfg["data"]["num_rows"]
        self._rng = np.random.default_rng(cfg["data"]["random_seed"])

    def build(self):
        from pyspark.sql.types import (
            DoubleType,
            IntegerType,
            StringType,
            StructField,
            StructType,
            TimestampType,
        )

        dates = pd.date_range("2022-01-01", periods=self._n, freq="h")
        pdf = pd.DataFrame(
            {
                "timestamp": dates,
                "region": self._rng.choice(["North", "South", "East", "West"], self._n),
                "salesperson_id": self._rng.integers(1, 21, self._n).astype(int),
                "revenue": self._rng.exponential(500, self._n).round(2),
            }
        )
        schema = StructType(
            [
                StructField("timestamp", TimestampType(), False),
                StructField("region", StringType(), False),
                StructField("salesperson_id", IntegerType(), False),
                StructField("revenue", DoubleType(), False),
            ]
        )
        sdf = self._spark.createDataFrame(pdf, schema=schema)
        logger.info("WindowDataGenerator.build(): %d rows", sdf.count())
        return sdf


# ---------------------------------------------------------------------------
# Ranking functions
# ---------------------------------------------------------------------------

class RankingWindowDemo:
    """rank, dense_rank, row_number, ntile."""

    def __init__(self, sdf) -> None:
        self._df = sdf

    def run(self):
        from pyspark.sql import Window
        from pyspark.sql import functions as F

        logger.info("--- Ranking window functions ---")

        window_region_rev = (
            Window.partitionBy("region")
            .orderBy(F.desc("revenue"))
        )

        result = (
            self._df
            .withColumn("rank", F.rank().over(window_region_rev))
            .withColumn("dense_rank", F.dense_rank().over(window_region_rev))
            .withColumn("row_number", F.row_number().over(window_region_rev))
            .withColumn("revenue_quartile", F.ntile(4).over(window_region_rev))
        )

        logger.info("Ranking window applied. Showing top-3 per region:")
        result.filter(F.col("row_number") <= 3).orderBy("region", "row_number").show(12, truncate=False)
        return result


# ---------------------------------------------------------------------------
# Lag / Lead
# ---------------------------------------------------------------------------

class LagLeadDemo:
    """Time-ordered lag/lead per salesperson."""

    def __init__(self, sdf) -> None:
        self._df = sdf

    def run(self):
        from pyspark.sql import Window
        from pyspark.sql import functions as F

        logger.info("--- lag / lead window functions ---")

        window_person_time = (
            Window.partitionBy("salesperson_id")
            .orderBy("timestamp")
        )

        result = (
            self._df
            .withColumn("prev_revenue", F.lag("revenue", 1).over(window_person_time))
            .withColumn("next_revenue", F.lead("revenue", 1).over(window_person_time))
            .withColumn(
                "revenue_change",
                F.col("revenue") - F.col("prev_revenue"),
            )
        )

        logger.info("Lag/lead sample (salesperson_id=1):")
        result.filter(F.col("salesperson_id") == 1).orderBy("timestamp").show(10, truncate=False)
        return result


# ---------------------------------------------------------------------------
# Rolling aggregation
# ---------------------------------------------------------------------------

class RollingWindowDemo:
    """rowsBetween rolling mean, cumulative sum."""

    def __init__(self, sdf, rolling_rows: int) -> None:
        self._df = sdf
        self._rolling_rows = rolling_rows

    def run(self):
        from pyspark.sql import Window
        from pyspark.sql import functions as F

        logger.info("--- Rolling window (rowsBetween) ---")

        window_rolling = (
            Window.partitionBy("salesperson_id")
            .orderBy("timestamp")
            .rowsBetween(-(self._rolling_rows - 1), 0)
        )

        window_cumsum = (
            Window.partitionBy("salesperson_id")
            .orderBy("timestamp")
            .rowsBetween(Window.unboundedPreceding, 0)
        )

        result = (
            self._df
            .withColumn(
                f"rolling_mean_{self._rolling_rows}",
                F.round(F.avg("revenue").over(window_rolling), 2),
            )
            .withColumn("cumulative_revenue", F.round(F.sum("revenue").over(window_cumsum), 2))
        )

        logger.info("Rolling window sample (salesperson_id=1):")
        result.filter(F.col("salesperson_id") == 1).orderBy("timestamp").show(10, truncate=False)
        return result


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class WindowFunctionsRunner:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self._cfg = cfg

    def run(self) -> None:
        logger.info("=== PySpark Window Functions START ===")
        from pyspark_core.spark_session import build_spark_session, stop_spark_session

        spark = build_spark_session(self._cfg)
        try:
            sdf = WindowDataGenerator(spark, self._cfg).build()
            rolling_rows = self._cfg["window"]["rolling_window"]

            RankingWindowDemo(sdf).run()
            LagLeadDemo(sdf).run()
            RollingWindowDemo(sdf, rolling_rows).run()
        finally:
            stop_spark_session(spark)

        logger.info("=== PySpark Window Functions END ===")


if __name__ == "__main__":
    _cfg = load_config()
    setup_logging(_cfg)
    WindowFunctionsRunner(_cfg).run()
