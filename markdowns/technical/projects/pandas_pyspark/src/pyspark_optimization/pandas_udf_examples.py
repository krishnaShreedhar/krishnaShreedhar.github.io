"""
Pandas UDF (Vectorized UDF) vs Python UDF
==========================================
Demonstrates:
  - Regular Python UDF (row-by-row, slow)
  - Pandas UDF (Series -> Series, vectorized via Apache Arrow)
  - Grouped Map Pandas UDF (mapInPandas) for distributed inference
  - Performance comparison between the two UDF types

All constants loaded from config.yaml.

Run:
    python src/pyspark_optimization/pandas_udf_examples.py
"""

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator

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

class UDFDataGenerator:
    def __init__(self, spark, cfg: Dict[str, Any]) -> None:
        self._spark = spark
        self._n = cfg["data"]["num_rows"]
        self._rng = np.random.default_rng(cfg["data"]["random_seed"])

    def build(self):
        pdf = pd.DataFrame(
            {
                "revenue": self._rng.exponential(500, self._n).round(2),
                "cost": self._rng.exponential(300, self._n).round(2),
                "region": self._rng.choice(["North", "South", "East", "West"], self._n),
                "feature_a": self._rng.normal(0, 1, self._n).round(6),
                "feature_b": self._rng.uniform(0, 1, self._n).round(6),
            }
        )
        sdf = self._spark.createDataFrame(pdf)
        logger.info("UDFDataGenerator.build(): %d rows", sdf.count())
        return sdf


# ---------------------------------------------------------------------------
# Regular Python UDF
# ---------------------------------------------------------------------------

class PythonUDFDemo:
    """Row-by-row Python UDF — serialises each row to Python and back."""

    def __init__(self, sdf) -> None:
        self._sdf = sdf

    def run(self):
        from pyspark.sql import functions as F
        from pyspark.sql.types import DoubleType

        logger.info("--- Regular Python UDF (row-by-row) ---")

        @F.udf(returnType=DoubleType())
        def compute_profit_margin(revenue: float, cost: float) -> float:
            if revenue == 0:
                return 0.0
            return round((revenue - cost) / revenue, 6)

        t0 = time.perf_counter()
        result = self._sdf.withColumn(
            "profit_margin",
            compute_profit_margin(F.col("revenue"), F.col("cost")),
        )
        count = result.count()
        elapsed = time.perf_counter() - t0

        logger.info(
            "Python UDF: rows=%d, elapsed=%.3f s", count, elapsed
        )
        result.select("revenue", "cost", "profit_margin").show(5, truncate=False)
        return result, elapsed


# ---------------------------------------------------------------------------
# Pandas UDF (Series -> Series)
# ---------------------------------------------------------------------------

class PandasUDFDemo:
    """Pandas UDF uses Apache Arrow for batch serialisation — much faster."""

    def __init__(self, sdf) -> None:
        self._sdf = sdf

    def run(self):
        from pyspark.sql import functions as F
        from pyspark.sql.functions import pandas_udf
        from pyspark.sql.types import DoubleType

        logger.info("--- Pandas UDF (vectorized, Arrow-backed) ---")

        @pandas_udf(DoubleType())
        def compute_profit_margin_vec(revenue: pd.Series, cost: pd.Series) -> pd.Series:
            """Vectorised version: operates on Pandas Series."""
            margin = (revenue - cost) / revenue.replace(0, float("nan"))
            return margin.round(6).fillna(0.0)

        t0 = time.perf_counter()
        result = self._sdf.withColumn(
            "profit_margin_vec",
            compute_profit_margin_vec(F.col("revenue"), F.col("cost")),
        )
        count = result.count()
        elapsed = time.perf_counter() - t0

        logger.info(
            "Pandas UDF: rows=%d, elapsed=%.3f s", count, elapsed
        )
        result.select("revenue", "cost", "profit_margin_vec").show(5, truncate=False)
        return result, elapsed


# ---------------------------------------------------------------------------
# mapInPandas — distributed inference pattern
# ---------------------------------------------------------------------------

class MapInPandasDemo:
    """
    mapInPandas: process each partition as a Pandas DataFrame.
    Ideal for distributed ML inference where a model is loaded once per partition.
    """

    def __init__(self, sdf) -> None:
        self._sdf = sdf

    def run(self):
        from pyspark.sql.types import (
            DoubleType,
            StringType,
            StructField,
            StructType,
        )

        logger.info("--- mapInPandas (distributed inference pattern) ---")

        # Output schema — must be declared explicitly
        output_schema = StructType(
            [
                StructField("region", StringType(), True),
                StructField("revenue", DoubleType(), True),
                StructField("score", DoubleType(), True),
                StructField("prediction", StringType(), True),
            ]
        )

        def score_partition(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
            """
            Simulated inference: load a 'model' once per partition,
            then score all rows in that partition efficiently.
            """
            # Model loading would happen here (e.g., joblib.load or torch.load)
            logger.debug("score_partition: starting partition")
            for pdf in iterator:
                # Simulate a scoring function: score = weighted feature combination
                pdf["score"] = (
                    0.6 * pdf["feature_a"] + 0.4 * pdf["feature_b"]
                ).round(6)
                pdf["prediction"] = pdf["score"].apply(
                    lambda s: "positive" if s > 0 else "negative"
                )
                yield pdf[["region", "revenue", "score", "prediction"]]

        result = self._sdf.mapInPandas(score_partition, schema=output_schema)
        result_count = result.count()
        logger.info("mapInPandas result: %d rows", result_count)
        result.show(8, truncate=False)

        pred_dist = result.groupBy("prediction").count()
        logger.info("Prediction distribution:")
        pred_dist.show(truncate=False)
        return result


# ---------------------------------------------------------------------------
# Performance comparison report
# ---------------------------------------------------------------------------

class UDFPerformanceComparison:
    """Logs a side-by-side timing summary."""

    def __init__(self, python_elapsed: float, pandas_elapsed: float) -> None:
        self._python = python_elapsed
        self._pandas = pandas_elapsed

    def report(self) -> None:
        logger.info("--- UDF Performance Comparison ---")
        logger.info("  %-20s: %.3f s", "Python UDF", self._python)
        logger.info("  %-20s: %.3f s", "Pandas UDF", self._pandas)
        speedup = self._python / (self._pandas + 1e-12)
        logger.info("  Pandas UDF speedup: ~%.1fx", speedup)
        logger.info(
            "  Note: Pandas UDF uses Apache Arrow batch transfer; Python UDF "
            "serialises every row via pickle -> slower for large datasets."
        )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class PandasUDFRunner:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self._cfg = cfg

    def run(self) -> None:
        logger.info("=== Pandas UDF Examples START ===")
        from pyspark_core.spark_session import build_spark_session, stop_spark_session

        spark = build_spark_session(self._cfg)
        try:
            sdf = UDFDataGenerator(spark, self._cfg).build()

            _, t_python = PythonUDFDemo(sdf).run()
            _, t_pandas = PandasUDFDemo(sdf).run()
            MapInPandasDemo(sdf).run()
            UDFPerformanceComparison(t_python, t_pandas).report()
        finally:
            stop_spark_session(spark)

        logger.info("=== Pandas UDF Examples END ===")


if __name__ == "__main__":
    _cfg = load_config()
    setup_logging(_cfg)
    PandasUDFRunner(_cfg).run()
