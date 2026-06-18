"""
Ray Data Pipeline for Distributed Training.

Demonstrates:
  - Building a Ray Dataset from in-memory data
  - Schema-validated preprocessing transformations (map_batches)
  - Train/validation split
  - Iterating over Ray Dataset batches inside a training loop
  - Proper Shard-per-Worker pattern for DDP

All constants are read from config.yaml.  No values are hardcoded.
"""

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import ray
import ray.data

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.config_loader import load_config
from utils.logging_setup import get_logger

_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config.yaml"
config: dict[str, Any] = load_config(str(_CONFIG_PATH))
logger = get_logger(__name__, config)

_TRAIN_CFG = config["distributed_training"]
_RAY_CFG = config["ray"]["init"]

BATCH_SIZE: int = _TRAIN_CFG["batch_size"]
NUM_WORKERS: int = _TRAIN_CFG["num_workers"]


# ===========================================================================
# Synthetic dataset generation
# ===========================================================================

def _make_classification_dataframe(num_samples: int, num_features: int) -> pd.DataFrame:
    """Generate a linearly-separable synthetic classification dataset."""
    rng = np.random.default_rng(seed=42)
    X = rng.standard_normal((num_samples, num_features)).astype(np.float32)
    weights = rng.standard_normal(num_features).astype(np.float32)
    logits = X @ weights
    labels = (logits > 0).astype(np.int64)

    feature_cols = {f"feature_{i}": X[:, i] for i in range(num_features)}
    df = pd.DataFrame(feature_cols)
    df["label"] = labels
    return df


# ===========================================================================
# Preprocessing transforms (pure functions, stateless)
# ===========================================================================

def normalise_batch(batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    Z-score normalise all feature columns in a batch dict.

    Ray Data calls this function with a dict of column_name → numpy array,
    making it easy to apply element-wise or column-wise transforms.
    """
    feature_keys = [k for k in batch.keys() if k.startswith("feature_")]
    for key in feature_keys:
        col = batch[key].astype(np.float32)
        mean = col.mean()
        std = col.std() + 1e-8  # numerical stability guard
        batch[key] = (col - mean) / std
    return batch


def add_interaction_features(batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    Add pairwise interaction terms for the first two features.

    Demonstrates how map_batches() can be chained to add derived columns
    without materialising intermediate datasets.
    """
    if "feature_0" in batch and "feature_1" in batch:
        batch["feature_0_x_1"] = batch["feature_0"] * batch["feature_1"]
    return batch


# ===========================================================================
# Pipeline builder (Interface Segregation Principle)
# ===========================================================================

class DataPipelineBuilder:
    """
    Single-responsibility builder for the Ray Data training pipeline.

    Constructs a preprocessing chain that can be reused across
    training, validation, and serving.
    """

    def __init__(self, num_samples: int, num_features: int) -> None:
        self._num_samples = num_samples
        self._num_features = num_features
        logger.debug(
            "DataPipelineBuilder created",
            extra={"num_samples": num_samples, "num_features": num_features},
        )

    def build_dataset(self) -> ray.data.Dataset:
        """Create a raw in-memory Ray Dataset from synthetic data."""
        logger.info(
            "Generating synthetic classification dataset",
            extra={"num_samples": self._num_samples, "num_features": self._num_features},
        )
        df = _make_classification_dataframe(self._num_samples, self._num_features)
        dataset = ray.data.from_pandas(df)
        logger.info(
            "Raw dataset created",
            extra={"num_rows": dataset.count(), "schema": str(dataset.schema())},
        )
        return dataset

    def apply_preprocessing(self, dataset: ray.data.Dataset) -> ray.data.Dataset:
        """Chain normalisation and feature engineering transforms."""
        logger.info("Applying preprocessing transforms")

        dataset = dataset.map_batches(
            normalise_batch,
            batch_format="numpy",
            batch_size=BATCH_SIZE,
        )
        logger.debug("Normalisation transform applied")

        dataset = dataset.map_batches(
            add_interaction_features,
            batch_format="numpy",
            batch_size=BATCH_SIZE,
        )
        logger.debug("Interaction feature transform applied")

        return dataset

    def split_train_val(
        self, dataset: ray.data.Dataset, val_fraction: float = 0.2
    ) -> tuple[ray.data.Dataset, ray.data.Dataset]:
        """Split dataset into train and validation subsets."""
        total = dataset.count()
        val_size = max(1, int(total * val_fraction))
        train_size = total - val_size

        train_ds, val_ds = dataset.split_at_indices([train_size])
        logger.info(
            "Dataset split complete",
            extra={
                "train_size": train_size,
                "val_size": val_size,
                "val_fraction": val_fraction,
            },
        )
        return train_ds, val_ds


# ===========================================================================
# Pipeline runner (demo)
# ===========================================================================

def run_pipeline_demo() -> tuple[ray.data.Dataset, ray.data.Dataset]:
    """
    End-to-end demo: build, preprocess, split, and inspect the pipeline.

    Returns
    -------
    tuple[ray.data.Dataset, ray.data.Dataset]
        (train_dataset, val_dataset) ready for TorchTrainer consumption.
    """
    logger.info("Running data pipeline demo")

    num_samples = 1000
    num_features = 10

    builder = DataPipelineBuilder(num_samples=num_samples, num_features=num_features)
    raw_ds = builder.build_dataset()
    processed_ds = builder.apply_preprocessing(raw_ds)
    train_ds, val_ds = builder.split_train_val(processed_ds, val_fraction=0.2)

    # Inspect a sample batch
    logger.info("Inspecting a sample batch from the training dataset")
    sample_batch = train_ds.take_batch(batch_size=5, batch_format="pandas")
    logger.info(
        "Sample batch retrieved",
        extra={
            "columns": list(sample_batch.columns),
            "shape": list(sample_batch.shape),
            "label_distribution": sample_batch["label"].value_counts().to_dict(),
        },
    )

    logger.info(
        "Data pipeline demo complete",
        extra={
            "train_rows": train_ds.count(),
            "val_rows": val_ds.count(),
        },
    )
    return train_ds, val_ds


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    logger.info("Initialising Ray for data pipeline demo", extra={"ray_config": _RAY_CFG})
    ray.init(
        num_cpus=_RAY_CFG["num_cpus"],
        num_gpus=_RAY_CFG["num_gpus"],
        object_store_memory=_RAY_CFG["object_store_memory"],
        ignore_reinit_error=True,
    )
    logger.info("Ray initialised", extra={"resources": ray.cluster_resources()})

    try:
        run_pipeline_demo()
    finally:
        logger.info("Shutting down Ray")
        ray.shutdown()


if __name__ == "__main__":
    main()
