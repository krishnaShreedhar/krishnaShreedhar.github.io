"""
PySpark SparkSession Factory
=============================
Provides a single factory function that builds a SparkSession from the
config.yaml spark section.  All other pyspark_core modules import from here
to ensure one consistent session per process.

All constants loaded from config.yaml.

Usage (import):
    from pyspark_core.spark_session import build_spark_session
    spark = build_spark_session(cfg)
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config_loader import load_config, setup_logging  # noqa: E402

logger = logging.getLogger(__name__)


def build_spark_session(cfg: Dict[str, Any]):
    """
    Build and return a SparkSession configured from cfg['spark'].

    Parameters
    ----------
    cfg:
        Full config dict as returned by load_config().

    Returns
    -------
    pyspark.sql.SparkSession
    """
    from pyspark.sql import SparkSession  # deferred import — PySpark may not be installed

    spark_cfg = cfg["spark"]
    app_name: str = spark_cfg["app_name"]
    master: str = spark_cfg["master"]
    executor_memory: str = str(spark_cfg["executor_memory"])
    executor_cores: int = int(spark_cfg["executor_cores"])
    shuffle_partitions: int = int(spark_cfg["shuffle_partitions"])
    aqe_enabled: bool = bool(spark_cfg["adaptive_enabled"])
    broadcast_mb: int = int(spark_cfg["broadcast_threshold_mb"])
    arrow_enabled: bool = bool(spark_cfg["arrow_enabled"])

    logger.info(
        "Building SparkSession: app=%s master=%s executor_memory=%s",
        app_name,
        master,
        executor_memory,
    )

    builder = (
        SparkSession.builder
        .appName(app_name)
        .master(master)
        .config("spark.executor.memory", executor_memory)
        .config("spark.executor.cores", str(executor_cores))
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.sql.adaptive.enabled", str(aqe_enabled).lower())
        .config("spark.sql.autoBroadcastJoinThreshold", str(broadcast_mb * 1024 * 1024))
        .config("spark.sql.execution.arrow.pyspark.enabled", str(arrow_enabled).lower())
        # Reduce log noise in local mode
        .config("spark.ui.showConsoleProgress", "false")
    )

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    logger.info(
        "SparkSession ready: version=%s, master=%s",
        spark.version,
        spark.sparkContext.master,
    )
    return spark


def stop_spark_session(spark) -> None:
    """Gracefully stop the SparkSession."""
    if spark is not None:
        logger.info("Stopping SparkSession")
        spark.stop()


if __name__ == "__main__":
    _cfg = load_config()
    setup_logging(_cfg)
    _spark = build_spark_session(_cfg)
    logger.info("Spark version: %s", _spark.version)
    stop_spark_session(_spark)
