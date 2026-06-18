"""
PySpark DataFrame Operations
=============================
Demonstrates:
  - Schema definition with StructType / StructField
  - Column operations: withColumn, filter, select
  - GroupBy aggregations
  - Broadcast join vs regular join
  - SparkSQL with createOrReplaceTempView

All constants loaded from config.yaml.

Run:
    python src/pyspark_core/dataframe_ops.py
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
# Schema definition
# ---------------------------------------------------------------------------

def build_sales_schema():
    """Return a StructType schema for the sales DataFrame."""
    from pyspark.sql.types import (
        DoubleType,
        IntegerType,
        StringType,
        StructField,
        StructType,
        TimestampType,
    )

    schema = StructType(
        [
            StructField("timestamp", TimestampType(), nullable=False),
            StructField("region", StringType(), nullable=False),
            StructField("product", StringType(), nullable=False),
            StructField("salesperson_id", IntegerType(), nullable=False),
            StructField("revenue", DoubleType(), nullable=False),
            StructField("units", IntegerType(), nullable=False),
        ]
    )
    logger.info("Sales schema: %s", [f.name for f in schema.fields])
    return schema


def build_employee_schema():
    """Return a StructType schema for employee lookup."""
    from pyspark.sql.types import IntegerType, StringType, StructField, StructType

    return StructType(
        [
            StructField("salesperson_id", IntegerType(), nullable=False),
            StructField("name", StringType(), nullable=True),
            StructField("department", StringType(), nullable=True),
            StructField("hire_year", IntegerType(), nullable=True),
        ]
    )


# ---------------------------------------------------------------------------
# Synthetic data -> Spark DataFrame
# ---------------------------------------------------------------------------

class SparkDataGenerator:
    """Creates Spark DataFrames from synthetic Pandas DataFrames."""

    def __init__(self, spark, cfg: Dict[str, Any]) -> None:
        self._spark = spark
        self._n = cfg["data"]["num_rows"]
        self._rng = np.random.default_rng(cfg["data"]["random_seed"])

    def sales_df(self):
        dates = pd.date_range("2022-01-01", periods=self._n, freq="h")
        pdf = pd.DataFrame(
            {
                "timestamp": dates,
                "region": self._rng.choice(["North", "South", "East", "West"], self._n),
                "product": self._rng.choice(["Widget", "Gadget", "Doohickey"], self._n),
                "salesperson_id": self._rng.integers(1, 21, self._n).astype(int),
                "revenue": self._rng.exponential(500, self._n).round(2),
                "units": self._rng.integers(1, 100, self._n).astype(int),
            }
        )
        sdf = self._spark.createDataFrame(pdf, schema=build_sales_schema())
        logger.info("SparkDataGenerator.sales_df(): %d rows", sdf.count())
        return sdf

    def employee_df(self):
        n = 20
        pdf = pd.DataFrame(
            {
                "salesperson_id": list(range(1, n + 1)),
                "name": [f"Employee_{i}" for i in range(1, n + 1)],
                "department": self._rng.choice(["Sales", "Marketing", "Operations"], n).tolist(),
                "hire_year": self._rng.integers(2010, 2023, n).astype(int).tolist(),
            }
        )
        sdf = self._spark.createDataFrame(pdf, schema=build_employee_schema())
        logger.info("SparkDataGenerator.employee_df(): %d rows", sdf.count())
        return sdf


# ---------------------------------------------------------------------------
# Column operations
# ---------------------------------------------------------------------------

class ColumnOperationsDemo:
    """withColumn, filter, select, when/otherwise."""

    def __init__(self, sales_sdf) -> None:
        self._df = sales_sdf

    def run(self):
        from pyspark.sql import functions as F

        logger.info("--- PySpark column operations ---")

        # withColumn: derive new column
        df = self._df.withColumn("revenue_tax", F.col("revenue") * F.lit(1.15))
        df = df.withColumn(
            "revenue_tier",
            F.when(F.col("revenue") > 1000, "High")
             .when(F.col("revenue") > 400, "Medium")
             .otherwise("Low"),
        )
        logger.info("Added revenue_tax and revenue_tier columns")

        # filter / where
        high = df.filter(F.col("revenue_tier") == "High")
        logger.info("High-tier rows: %d", high.count())

        # select
        selected = df.select("region", "product", "revenue", "revenue_tier")
        logger.info("Select 4 columns: schema=%s", selected.schema.simpleString())

        # distinct count
        n_distinct = df.select("salesperson_id").distinct().count()
        logger.info("Distinct salespersons: %d", n_distinct)

        # GroupBy aggregation
        agg_df = (
            df.groupBy("region", "product")
            .agg(
                F.sum("revenue").alias("total_revenue"),
                F.mean("revenue").alias("mean_revenue"),
                F.count("*").alias("num_transactions"),
                F.max("units").alias("max_units"),
            )
            .orderBy("region", "product")
        )
        logger.info("GroupBy aggregation schema: %s", agg_df.schema.simpleString())
        agg_df.show(8, truncate=False)
        return agg_df


# ---------------------------------------------------------------------------
# Broadcast join
# ---------------------------------------------------------------------------

class JoinDemo:
    """Regular join vs explicit broadcast join."""

    def __init__(self, sales_sdf, employee_sdf) -> None:
        self._sales = sales_sdf
        self._emp = employee_sdf

    def run(self):
        from pyspark.sql import functions as F

        logger.info("--- PySpark join demo ---")

        # Regular inner join
        regular = self._sales.join(self._emp, on="salesperson_id", how="inner")
        logger.info("Regular inner join rows: %d", regular.count())

        # Explicit broadcast hint (force broadcast of the small employee table)
        broadcast = self._sales.join(
            F.broadcast(self._emp), on="salesperson_id", how="left"
        )
        logger.info("Broadcast left join rows: %d", broadcast.count())
        broadcast.select("region", "product", "name", "department").show(5, truncate=False)

        # Explain to show physical plan difference (logged at DEBUG)
        logger.debug("Broadcast join explain plan:\n%s", broadcast._jdf.queryExecution().toString())

        return broadcast


# ---------------------------------------------------------------------------
# SparkSQL
# ---------------------------------------------------------------------------

class SparkSQLDemo:
    """createOrReplaceTempView and SQL queries."""

    def __init__(self, spark, sales_sdf) -> None:
        self._df = sales_sdf
        self._spark = spark

    def run(self):
        logger.info("--- SparkSQL demo ---")
        self._df.createOrReplaceTempView("sales")

        top_regions = self._spark.sql(
            """
            SELECT
                region,
                ROUND(SUM(revenue), 2)   AS total_revenue,
                ROUND(AVG(revenue), 2)   AS avg_revenue,
                COUNT(*)                 AS transactions
            FROM sales
            GROUP BY region
            ORDER BY total_revenue DESC
            """
        )
        logger.info("Top regions by revenue:")
        top_regions.show(truncate=False)

        product_rank = self._spark.sql(
            """
            SELECT
                region,
                product,
                ROUND(SUM(revenue), 2) AS total_rev,
                RANK() OVER (PARTITION BY region ORDER BY SUM(revenue) DESC) AS rev_rank
            FROM sales
            GROUP BY region, product
            """
        )
        logger.info("Product rank within region:")
        product_rank.orderBy("region", "rev_rank").show(12, truncate=False)
        return top_regions


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class DataFrameOpsRunner:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self._cfg = cfg

    def run(self) -> None:
        logger.info("=== PySpark DataFrame Ops START ===")
        from pyspark_core.spark_session import build_spark_session, stop_spark_session

        spark = build_spark_session(self._cfg)
        try:
            gen = SparkDataGenerator(spark, self._cfg)
            sales = gen.sales_df()
            employees = gen.employee_df()

            ColumnOperationsDemo(sales).run()
            JoinDemo(sales, employees).run()
            SparkSQLDemo(spark, sales).run()
        finally:
            stop_spark_session(spark)

        logger.info("=== PySpark DataFrame Ops END ===")


if __name__ == "__main__":
    _cfg = load_config()
    setup_logging(_cfg)
    DataFrameOpsRunner(_cfg).run()
