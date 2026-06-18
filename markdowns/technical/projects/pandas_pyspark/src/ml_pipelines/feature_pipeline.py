"""
PySpark MLlib Feature Pipeline
================================
Demonstrates the standard feature engineering chain:
  StringIndexer -> OneHotEncoder -> VectorAssembler -> StandardScaler

All assembled as a PySpark Pipeline so transformations are fit-once,
applied consistently to train and test sets.

All constants loaded from config.yaml.

Run:
    python src/ml_pipelines/feature_pipeline.py
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

class MLDataGenerator:
    """Generates synthetic classification data."""

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
        )

        region = self._rng.choice(["North", "South", "East", "West"], self._n)
        product = self._rng.choice(["Widget", "Gadget", "Doohickey"], self._n)
        age = self._rng.integers(18, 65, self._n).astype(float)
        revenue = self._rng.exponential(500, self._n)
        units = self._rng.integers(1, 100, self._n).astype(float)
        discount = self._rng.uniform(0, 0.5, self._n)

        # Binary target: high-value customer (revenue > 600 and units > 50)
        label = ((revenue > 600) & (units > 50)).astype(float)

        pdf = pd.DataFrame(
            {
                "region": region,
                "product": product,
                "age": age,
                "revenue": revenue.round(2),
                "units": units,
                "discount_pct": discount.round(4),
                "label": label,
            }
        )
        schema = StructType(
            [
                StructField("region", StringType(), False),
                StructField("product", StringType(), False),
                StructField("age", DoubleType(), False),
                StructField("revenue", DoubleType(), False),
                StructField("units", DoubleType(), False),
                StructField("discount_pct", DoubleType(), False),
                StructField("label", DoubleType(), False),
            ]
        )
        sdf = self._spark.createDataFrame(pdf, schema=schema)
        pos_rate = pdf["label"].mean()
        logger.info(
            "MLDataGenerator.build(): %d rows, positive_rate=%.3f",
            sdf.count(),
            pos_rate,
        )
        return sdf


# ---------------------------------------------------------------------------
# Feature pipeline builder
# ---------------------------------------------------------------------------

class FeaturePipelineBuilder:
    """
    Builds a PySpark ML Pipeline: StringIndexer -> OHE -> VectorAssembler -> StandardScaler.

    Single Responsibility: pipeline construction only.
    """

    def __init__(self, categorical_cols: List[str], numeric_cols: List[str]) -> None:
        self._cat_cols = categorical_cols
        self._num_cols = numeric_cols
        logger.info(
            "FeaturePipelineBuilder: cat_cols=%s, num_cols=%s",
            categorical_cols,
            numeric_cols,
        )

    def build(self):
        from pyspark.ml import Pipeline
        from pyspark.ml.feature import (
            OneHotEncoder,
            StandardScaler,
            StringIndexer,
            VectorAssembler,
        )

        stages = []

        # Step 1: StringIndexer for each categorical column
        indexed_cols = []
        for col in self._cat_cols:
            output_col = f"{col}_index"
            indexer = StringIndexer(
                inputCol=col,
                outputCol=output_col,
                handleInvalid="keep",
            )
            stages.append(indexer)
            indexed_cols.append(output_col)
            logger.debug("Added StringIndexer: %s -> %s", col, output_col)

        # Step 2: OneHotEncoder for indexed categoricals
        ohe_cols = []
        for col in indexed_cols:
            output_col = col.replace("_index", "_ohe")
            ohe = OneHotEncoder(inputCol=col, outputCol=output_col, dropLast=True)
            stages.append(ohe)
            ohe_cols.append(output_col)
            logger.debug("Added OneHotEncoder: %s -> %s", col, output_col)

        # Step 3: VectorAssembler combines OHE + numeric columns into a single features vector
        all_feature_cols = ohe_cols + self._num_cols
        assembler = VectorAssembler(
            inputCols=all_feature_cols,
            outputCol="raw_features",
            handleInvalid="keep",
        )
        stages.append(assembler)
        logger.debug("Added VectorAssembler: %s -> raw_features", all_feature_cols)

        # Step 4: StandardScaler normalises the feature vector
        scaler = StandardScaler(
            inputCol="raw_features",
            outputCol="features",
            withMean=True,
            withStd=True,
        )
        stages.append(scaler)
        logger.debug("Added StandardScaler: raw_features -> features")

        pipeline = Pipeline(stages=stages)
        logger.info("Feature pipeline built with %d stages", len(stages))
        return pipeline


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

class FeaturePipelineRunner:
    """Fits and transforms data using the feature pipeline."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self._cfg = cfg

    def run(self) -> None:
        logger.info("=== Feature Pipeline START ===")
        from pyspark_core.spark_session import build_spark_session, stop_spark_session

        spark = build_spark_session(self._cfg)
        try:
            sdf = MLDataGenerator(spark, self._cfg).build()

            train_ratio = self._cfg["ml_pipeline"]["train_ratio"]
            train_sdf, test_sdf = sdf.randomSplit(
                [train_ratio, 1 - train_ratio],
                seed=self._cfg["data"]["random_seed"],
            )
            logger.info(
                "Train/Test split: %d / %d rows", train_sdf.count(), test_sdf.count()
            )

            cat_cols = ["region", "product"]
            num_cols = ["age", "revenue", "units", "discount_pct"]
            pipeline = FeaturePipelineBuilder(cat_cols, num_cols).build()

            # Fit on train only
            model = pipeline.fit(train_sdf)
            logger.info("Feature pipeline fitted on training data")

            # Transform both sets
            train_transformed = model.transform(train_sdf)
            test_transformed = model.transform(test_sdf)

            logger.info("Train transformed schema (feature columns):")
            feature_cols = ["features", "label"]
            train_transformed.select(feature_cols).show(3, truncate=True)
            logger.info("Feature vector size: %d", train_transformed.select("features").first()[0].size)

            # Save pipeline model
            output_dir = Path(self._cfg["data"]["output_dir"]) / "feature_pipeline_model"
            output_dir.mkdir(parents=True, exist_ok=True)
            model.write().overwrite().save(str(output_dir))
            logger.info("Feature pipeline model saved to %s", output_dir)

            # Load back to verify
            from pyspark.ml import PipelineModel
            loaded = PipelineModel.load(str(output_dir))
            verify = loaded.transform(test_sdf.limit(5))
            logger.info("Loaded model verification: %d rows transformed", verify.count())

        finally:
            stop_spark_session(spark)

        logger.info("=== Feature Pipeline END ===")


if __name__ == "__main__":
    _cfg = load_config()
    setup_logging(_cfg)
    FeaturePipelineRunner(_cfg).run()
