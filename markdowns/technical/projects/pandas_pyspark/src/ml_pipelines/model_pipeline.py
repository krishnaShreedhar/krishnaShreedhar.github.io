"""
PySpark MLlib Model Pipeline
==============================
Demonstrates:
  - GradientBoostedTrees + RandomForest training with Pipeline API
  - CrossValidator + ParamGridBuilder for hyperparameter search
  - Model evaluation (AUC, accuracy)
  - Model save / load

All constants loaded from config.yaml.

Run:
    python src/ml_pipelines/model_pipeline.py
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
# Data generator (same as feature_pipeline but inline for independence)
# ---------------------------------------------------------------------------

class ModelMLDataGenerator:
    def __init__(self, spark, cfg: Dict[str, Any]) -> None:
        self._spark = spark
        self._n = cfg["data"]["num_rows"]
        self._rng = np.random.default_rng(cfg["data"]["random_seed"])

    def build(self):
        from pyspark.sql.types import (
            DoubleType,
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
        logger.info("ModelMLDataGenerator.build(): %d rows", sdf.count())
        return sdf


# ---------------------------------------------------------------------------
# Feature preprocessing (minimal, inline)
# ---------------------------------------------------------------------------

def build_feature_stages(cat_cols, num_cols):
    """Return pipeline stages for feature preprocessing."""
    from pyspark.ml.feature import (
        OneHotEncoder,
        StandardScaler,
        StringIndexer,
        VectorAssembler,
    )

    stages = []
    indexed_cols = []
    for col in cat_cols:
        idx_col = f"{col}_index"
        stages.append(StringIndexer(inputCol=col, outputCol=idx_col, handleInvalid="keep"))
        indexed_cols.append(idx_col)

    ohe_cols = []
    for col in indexed_cols:
        ohe_col = col.replace("_index", "_ohe")
        stages.append(OneHotEncoder(inputCol=col, outputCol=ohe_col, dropLast=True))
        ohe_cols.append(ohe_col)

    assembler = VectorAssembler(
        inputCols=ohe_cols + num_cols,
        outputCol="raw_features",
        handleInvalid="keep",
    )
    stages.append(assembler)

    scaler = StandardScaler(
        inputCol="raw_features",
        outputCol="features",
        withMean=True,
        withStd=True,
    )
    stages.append(scaler)
    return stages


# ---------------------------------------------------------------------------
# GBT Classifier pipeline
# ---------------------------------------------------------------------------

class GBTPipelineTrainer:
    """Trains a GradientBoostedTrees classifier with Pipeline API."""

    def __init__(self, spark, train_sdf, test_sdf, cfg: Dict[str, Any]) -> None:
        self._spark = spark
        self._train = train_sdf
        self._test = test_sdf
        self._cfg = cfg

    def train_and_evaluate(self):
        from pyspark.ml import Pipeline
        from pyspark.ml.classification import GBTClassifier
        from pyspark.ml.evaluation import BinaryClassificationEvaluator

        logger.info("--- GBT Classifier pipeline ---")
        cat_cols = ["region", "product"]
        num_cols = ["age", "revenue", "units", "discount_pct"]

        feature_stages = build_feature_stages(cat_cols, num_cols)
        gbt = GBTClassifier(
            featuresCol="features",
            labelCol="label",
            maxDepth=5,
            maxIter=20,
            seed=self._cfg["data"]["random_seed"],
        )
        pipeline = Pipeline(stages=feature_stages + [gbt])

        logger.info("Fitting GBT pipeline...")
        model = pipeline.fit(self._train)

        predictions = model.transform(self._test)
        evaluator = BinaryClassificationEvaluator(labelCol="label")
        auc = evaluator.evaluate(predictions)
        logger.info("GBT AUC on test set: %.4f", auc)

        # Feature importances from GBT stage
        gbt_model = model.stages[-1]
        importances = gbt_model.featureImportances
        logger.info("GBT feature importances (sparse): %s", importances)

        # Save model
        output_dir = Path(self._cfg["data"]["output_dir"]) / "gbt_pipeline_model"
        output_dir.mkdir(parents=True, exist_ok=True)
        model.write().overwrite().save(str(output_dir))
        logger.info("GBT model saved to %s", output_dir)

        return model, auc


# ---------------------------------------------------------------------------
# Random Forest + CrossValidator
# ---------------------------------------------------------------------------

class RFCrossValidationTrainer:
    """Random Forest with CrossValidator + ParamGridBuilder."""

    def __init__(self, spark, train_sdf, test_sdf, cfg: Dict[str, Any]) -> None:
        self._spark = spark
        self._train = train_sdf
        self._test = test_sdf
        self._cfg = cfg

    def train_and_evaluate(self):
        from pyspark.ml import Pipeline
        from pyspark.ml.classification import RandomForestClassifier
        from pyspark.ml.evaluation import BinaryClassificationEvaluator
        from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

        logger.info("--- RandomForest + CrossValidator ---")
        cat_cols = ["region", "product"]
        num_cols = ["age", "revenue", "units", "discount_pct"]
        feature_stages = build_feature_stages(cat_cols, num_cols)

        rf = RandomForestClassifier(
            featuresCol="features",
            labelCol="label",
            seed=self._cfg["data"]["random_seed"],
        )
        pipeline = Pipeline(stages=feature_stages + [rf])

        max_depths = self._cfg["ml_pipeline"]["max_depth_options"]
        param_grid = (
            ParamGridBuilder()
            .addGrid(rf.maxDepth, max_depths)
            .addGrid(rf.numTrees, [10, 20])
            .build()
        )

        evaluator = BinaryClassificationEvaluator(labelCol="label")
        cv = CrossValidator(
            estimator=pipeline,
            estimatorParamMaps=param_grid,
            evaluator=evaluator,
            numFolds=self._cfg["ml_pipeline"]["cv_folds"],
            parallelism=self._cfg["ml_pipeline"]["parallelism"],
            seed=self._cfg["data"]["random_seed"],
        )

        logger.info(
            "Running CrossValidator: %d folds, %d parameter combinations",
            self._cfg["ml_pipeline"]["cv_folds"],
            len(param_grid),
        )
        cv_model = cv.fit(self._train)

        best_auc = max(cv_model.avgMetrics)
        logger.info("Best CV AUC: %.4f", best_auc)

        # Evaluate best model on held-out test
        test_predictions = cv_model.transform(self._test)
        test_auc = evaluator.evaluate(test_predictions)
        logger.info("RF best model test AUC: %.4f", test_auc)

        # Save best model
        output_dir = Path(self._cfg["data"]["output_dir"]) / "rf_cv_model"
        output_dir.mkdir(parents=True, exist_ok=True)
        cv_model.bestModel.write().overwrite().save(str(output_dir))
        logger.info("RF best model saved to %s", output_dir)

        return cv_model, test_auc


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class ModelPipelineRunner:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self._cfg = cfg

    def run(self) -> None:
        logger.info("=== Model Pipeline START ===")
        from pyspark_core.spark_session import build_spark_session, stop_spark_session

        spark = build_spark_session(self._cfg)
        try:
            sdf = ModelMLDataGenerator(spark, self._cfg).build()
            train_ratio = self._cfg["ml_pipeline"]["train_ratio"]
            train, test = sdf.randomSplit(
                [train_ratio, 1 - train_ratio],
                seed=self._cfg["data"]["random_seed"],
            )
            logger.info("Train: %d rows, Test: %d rows", train.count(), test.count())

            _, gbt_auc = GBTPipelineTrainer(spark, train, test, self._cfg).train_and_evaluate()
            _, rf_auc = RFCrossValidationTrainer(spark, train, test, self._cfg).train_and_evaluate()

            logger.info("=== Summary ===")
            logger.info("  GBT AUC:           %.4f", gbt_auc)
            logger.info("  RandomForest AUC:  %.4f", rf_auc)
            logger.info("  Winner: %s", "GBT" if gbt_auc > rf_auc else "RandomForest")

        finally:
            stop_spark_session(spark)

        logger.info("=== Model Pipeline END ===")


if __name__ == "__main__":
    _cfg = load_config()
    setup_logging(_cfg)
    ModelPipelineRunner(_cfg).run()
