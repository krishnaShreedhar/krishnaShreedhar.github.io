"""
PySpark Optimization Patterns
==============================
Demonstrates:
  - AQE (Adaptive Query Execution) configuration
  - Repartition vs coalesce trade-offs
  - Caching / persistence strategies (MEMORY_AND_DISK, DISK_ONLY, etc.)
  - Broadcast variables for lookup tables
  - explain() to show the Catalyst physical plan

All constants loaded from config.yaml.

Run:
    python src/pyspark_optimization/optimization_patterns.py
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

class OptDataGenerator:
    def __init__(self, spark, cfg: Dict[str, Any]) -> None:
        self._spark = spark
        self._n = cfg["data"]["num_rows"]
        self._rng = np.random.default_rng(cfg["data"]["random_seed"])

    def sales_sdf(self):
        pdf = pd.DataFrame(
            {
                "region": self._rng.choice(["North", "South", "East", "West"], self._n),
                "product": self._rng.choice(["Widget", "Gadget", "Doohickey"], self._n),
                "salesperson_id": self._rng.integers(1, 21, self._n).astype(int),
                "revenue": self._rng.exponential(500, self._n).round(2),
                "units": self._rng.integers(1, 100, self._n).astype(int),
            }
        )
        sdf = self._spark.createDataFrame(pdf)
        logger.info("OptDataGenerator.sales_sdf(): %d rows, %d partitions",
                    sdf.count(), sdf.rdd.getNumPartitions())
        return sdf

    def employee_dict(self) -> dict:
        """Small lookup table as a Python dict (for broadcast variable)."""
        return {i: f"Employee_{i}" for i in range(1, 21)}


# ---------------------------------------------------------------------------
# AQE demo
# ---------------------------------------------------------------------------

class AQEDemo:
    """
    Adaptive Query Execution (AQE) — configuration and observability.

    AQE re-optimises queries at runtime using actual partition statistics.
    Key features:
      - Coalesces small shuffle partitions automatically
      - Converts sort-merge joins to broadcast joins on-the-fly
      - Handles skewed data by splitting large partitions
    """

    def __init__(self, spark, cfg: Dict[str, Any]) -> None:
        self._spark = spark
        self._cfg = cfg

    def show_aqe_settings(self) -> None:
        logger.info("--- AQE configuration ---")
        settings = {
            "spark.sql.adaptive.enabled":
                self._spark.conf.get("spark.sql.adaptive.enabled", "N/A"),
            "spark.sql.adaptive.coalescePartitions.enabled":
                self._spark.conf.get("spark.sql.adaptive.coalescePartitions.enabled", "N/A"),
            "spark.sql.adaptive.skewJoin.enabled":
                self._spark.conf.get("spark.sql.adaptive.skewJoin.enabled", "N/A"),
            "spark.sql.shuffle.partitions":
                self._spark.conf.get("spark.sql.shuffle.partitions", "N/A"),
        }
        for k, v in settings.items():
            logger.info("  %-60s = %s", k, v)

    def run(self) -> None:
        self.show_aqe_settings()
        logger.info(
            "AQE benefit: with %d shuffle partitions configured, AQE will "
            "automatically coalesce empty/tiny partitions after a shuffle, "
            "reducing task overhead.",
            self._cfg["spark"]["shuffle_partitions"],
        )


# ---------------------------------------------------------------------------
# Repartition vs Coalesce
# ---------------------------------------------------------------------------

class PartitioningDemo:
    """Demonstrates repartition (shuffle) vs coalesce (no shuffle)."""

    def __init__(self, sdf) -> None:
        self._sdf = sdf

    def run(self) -> None:
        from pyspark.sql import functions as F

        logger.info("--- Repartition vs Coalesce ---")

        n_orig = self._sdf.rdd.getNumPartitions()
        logger.info("Original partitions: %d", n_orig)

        # Repartition: full shuffle, distributes data evenly or by column hash
        sdf_rep = self._sdf.repartition(8, "region")
        n_rep = sdf_rep.rdd.getNumPartitions()
        logger.info("After repartition(8, 'region'): %d partitions (full shuffle)", n_rep)

        # Coalesce: narrows partitions without shuffle — only makes sense to go smaller
        sdf_coal = self._sdf.coalesce(2)
        n_coal = sdf_coal.rdd.getNumPartitions()
        logger.info("After coalesce(2): %d partitions (no shuffle)", n_coal)

        # Partition statistics
        partition_sizes = (
            self._sdf
            .withColumn("partition_id", F.spark_partition_id())
            .groupBy("partition_id")
            .count()
            .orderBy("partition_id")
        )
        logger.info("Rows per partition (original):")
        partition_sizes.show(truncate=False)


# ---------------------------------------------------------------------------
# Caching / Persistence
# ---------------------------------------------------------------------------

class CachingDemo:
    """Shows different StorageLevel options and when to use each."""

    def __init__(self, sdf) -> None:
        self._sdf = sdf

    def run(self) -> None:
        from pyspark import StorageLevel
        from pyspark.sql import functions as F

        logger.info("--- Caching / Persistence ---")

        # MEMORY_AND_DISK — default cache(): spills to disk if RAM is full
        cached = self._sdf.cache()
        count1 = cached.count()  # triggers caching
        logger.info("MEMORY_AND_DISK (cache()): %d rows cached", count1)

        # Re-use cached DF for two downstream operations without re-reading
        agg1 = cached.groupBy("region").agg(F.sum("revenue").alias("total_rev"))
        agg2 = cached.groupBy("product").agg(F.count("*").alias("txn_count"))
        logger.info("Derived two aggregations from single cached DF")
        agg1.show(4, truncate=False)
        agg2.show(3, truncate=False)

        # Unpersist to free memory
        cached.unpersist()
        logger.info("Cache unpersisted")

        # DISK_ONLY — useful when data is large and memory is limited
        disk_sdf = self._sdf.persist(StorageLevel.DISK_ONLY)
        _ = disk_sdf.count()
        logger.info("DISK_ONLY persist: rows=%d", _)
        disk_sdf.unpersist()

        logger.info(
            "Caching rule of thumb: cache() when a DF is used 2+ times in "
            "a DAG. Always unpersist after use to avoid executor OOM."
        )


# ---------------------------------------------------------------------------
# Broadcast variables
# ---------------------------------------------------------------------------

class BroadcastVariableDemo:
    """Broadcast a Python dict lookup to all executors."""

    def __init__(self, spark, sdf, lookup: dict) -> None:
        self._spark = spark
        self._sdf = sdf
        self._lookup = lookup

    def run(self):
        from pyspark.sql import functions as F
        from pyspark.sql.types import StringType

        logger.info("--- Broadcast variable demo ---")

        # Broadcast the lookup dictionary
        bc_lookup = self._spark.sparkContext.broadcast(self._lookup)
        logger.info("Broadcast variable created: %d entries", len(bc_lookup.value))

        # Use in a UDF
        @F.udf(returnType=StringType())
        def map_id_to_name(sid: int) -> str:
            return bc_lookup.value.get(sid, "Unknown")

        result = self._sdf.withColumn("salesperson_name", map_id_to_name(F.col("salesperson_id")))
        logger.info("After broadcast lookup UDF:")
        result.select("salesperson_id", "salesperson_name", "revenue").show(5, truncate=False)

        bc_lookup.unpersist()
        logger.info("Broadcast variable unpersisted")
        return result


# ---------------------------------------------------------------------------
# Explain plan
# ---------------------------------------------------------------------------

class ExplainDemo:
    """Show the Catalyst query plan using explain()."""

    def __init__(self, sdf) -> None:
        self._sdf = sdf

    def run(self) -> None:
        from pyspark.sql import functions as F

        logger.info("--- explain() query plan demo ---")

        query = (
            self._sdf
            .filter(F.col("revenue") > 400)
            .groupBy("region")
            .agg(F.sum("revenue").alias("total_revenue"))
            .orderBy(F.desc("total_revenue"))
        )

        logger.info("Physical plan for filter -> groupBy -> orderBy:")
        # Capture explain output (printed to stdout by Spark)
        query.explain(mode="formatted")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class OptimizationRunner:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self._cfg = cfg

    def run(self) -> None:
        logger.info("=== PySpark Optimization Patterns START ===")
        from pyspark_core.spark_session import build_spark_session, stop_spark_session

        spark = build_spark_session(self._cfg)
        try:
            gen = OptDataGenerator(spark, self._cfg)
            sdf = gen.sales_sdf()
            lookup = gen.employee_dict()

            AQEDemo(spark, self._cfg).run()
            PartitioningDemo(sdf).run()
            CachingDemo(sdf).run()
            BroadcastVariableDemo(spark, sdf, lookup).run()
            ExplainDemo(sdf).run()
        finally:
            stop_spark_session(spark)

        logger.info("=== PySpark Optimization Patterns END ===")


if __name__ == "__main__":
    _cfg = load_config()
    setup_logging(_cfg)
    OptimizationRunner(_cfg).run()
