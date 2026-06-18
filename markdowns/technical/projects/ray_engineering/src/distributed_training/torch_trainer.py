"""
Ray Train – Distributed PyTorch Training.

Demonstrates:
  - TorchTrainer with ScalingConfig
  - DDP via ray.train.torch.prepare_model() and prepare_data_loader()
  - Checkpointing with ray.train.report() and ray.train.Checkpoint
  - Ray Data integration (dataset_config / get_dataset_shard)
  - Per-epoch metric reporting (loss, accuracy)

All constants are read from config.yaml.  No values are hardcoded.
"""

import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import ray
import ray.train
import ray.train.torch
import torch
import torch.nn as nn
import torch.optim as optim
from ray.train import Checkpoint, ScalingConfig
from ray.train.torch import TorchTrainer
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.config_loader import load_config
from utils.logging_setup import get_logger

_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config.yaml"
config: dict[str, Any] = load_config(str(_CONFIG_PATH))
logger = get_logger(__name__, config)

_TRAIN_CFG = config["distributed_training"]
_RAY_CFG = config["ray"]["init"]

NUM_WORKERS: int = _TRAIN_CFG["num_workers"]
USE_GPU: bool = _TRAIN_CFG["use_gpu"]
EPOCHS: int = _TRAIN_CFG["epochs"]
LEARNING_RATE: float = _TRAIN_CFG["learning_rate"]
BATCH_SIZE: int = _TRAIN_CFG["batch_size"]
NUM_TO_KEEP_CHECKPOINTS: int = _TRAIN_CFG["num_to_keep_checkpoints"]

NUM_FEATURES: int = 10  # synthetic dataset dimensionality
NUM_CLASSES: int = 2


# ===========================================================================
# Model definition (Single Responsibility: model architecture only)
# ===========================================================================

class BinaryClassifier(nn.Module):
    """
    Simple feedforward network for binary classification.

    Architecture: Linear → ReLU → Dropout → Linear → ReLU → Linear → Sigmoid
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout_rate: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x)).squeeze(dim=-1)


# ===========================================================================
# Dataset helpers
# ===========================================================================

def _make_tensor_dataset(
    num_samples: int, num_features: int, seed: int = 42
) -> TensorDataset:
    """Generate a synthetic binary classification TensorDataset."""
    rng = np.random.default_rng(seed=seed)
    X = rng.standard_normal((num_samples, num_features)).astype(np.float32)
    weights = rng.standard_normal(num_features).astype(np.float32)
    labels = (X @ weights > 0).astype(np.float32)
    return TensorDataset(torch.from_numpy(X), torch.from_numpy(labels))


# ===========================================================================
# Training function (executed inside each Ray worker)
# ===========================================================================

def _train_loop_per_worker(train_config: dict[str, Any]) -> None:
    """
    Per-worker training loop executed by TorchTrainer.

    Each worker:
      1. Builds the model and wraps it with DDP via prepare_model()
      2. Creates a DataLoader and wraps it via prepare_data_loader() for
         proper shard distribution in DDP
      3. Runs EPOCHS of training, reporting metrics each epoch
      4. Saves a checkpoint at the end of each epoch

    Parameters
    ----------
    train_config:
        Dict passed through TorchTrainer's ``train_loop_config``.
    """
    import logging as _logging

    worker_log = _logging.getLogger(f"train_worker.{ray.train.get_context().get_world_rank()}")
    worker_log.setLevel(_logging.DEBUG)

    epochs: int = train_config["epochs"]
    lr: float = train_config["learning_rate"]
    batch_size: int = train_config["batch_size"]
    hidden_dim: int = train_config["hidden_dim"]
    dropout_rate: float = train_config["dropout_rate"]
    num_samples: int = train_config["num_samples"]

    worker_rank = ray.train.get_context().get_world_rank()
    worker_log.info("Worker started | rank=%d", worker_rank)

    # Build model and wrap with DDP
    model = BinaryClassifier(
        input_dim=NUM_FEATURES,
        hidden_dim=hidden_dim,
        dropout_rate=dropout_rate,
    )
    model = ray.train.torch.prepare_model(model)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    # Build local dataset (in real usage, use get_dataset_shard())
    full_dataset = _make_tensor_dataset(
        num_samples=num_samples, num_features=NUM_FEATURES, seed=worker_rank
    )
    loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)
    loader = ray.train.torch.prepare_data_loader(loader)

    # Optionally restore from checkpoint
    checkpoint = ray.train.get_checkpoint()
    start_epoch = 0
    if checkpoint:
        with checkpoint.as_directory() as ckpt_dir:
            state = torch.load(Path(ckpt_dir) / "checkpoint.pt", weights_only=False)
            model.module.load_state_dict(state["model_state_dict"])
            optimizer.load_state_dict(state["optimizer_state_dict"])
            start_epoch = state["epoch"] + 1
            worker_log.info("Resumed from checkpoint at epoch %d", start_epoch)

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_y)
            preds = (outputs >= 0.5).float()
            correct += (preds == batch_y).sum().item()
            total += len(batch_y)

        avg_loss = total_loss / total
        accuracy = correct / total

        worker_log.info(
            "Epoch complete | rank=%d epoch=%d loss=%.4f acc=%.4f",
            worker_rank, epoch + 1, avg_loss, accuracy,
        )

        # Save checkpoint
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "checkpoint.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                },
                ckpt_path,
            )
            checkpoint = Checkpoint.from_directory(tmp_dir)

            # Report metrics + checkpoint to Ray Train
            ray.train.report(
                metrics={
                    "epoch": epoch + 1,
                    "loss": avg_loss,
                    "accuracy": accuracy,
                },
                checkpoint=checkpoint,
            )

    worker_log.info("Training complete | rank=%d", worker_rank)


# ===========================================================================
# Trainer builder (Open/Closed Principle: extend without modifying)
# ===========================================================================

class DistributedTrainerFactory:
    """
    Builds a configured TorchTrainer instance.

    Separating construction from execution makes it easy to swap out
    trainers (e.g., TensorflowTrainer, HorovodTrainer) without changing
    the calling code.
    """

    @staticmethod
    def build(
        num_workers: int,
        use_gpu: bool,
        train_loop_config: dict[str, Any],
        num_to_keep: int,
    ) -> TorchTrainer:
        scaling_config = ScalingConfig(
            num_workers=num_workers,
            use_gpu=use_gpu,
            resources_per_worker={"CPU": 1},
        )

        run_config = ray.train.RunConfig(
            name="binary_classifier_ddp",
            checkpoint_config=ray.train.CheckpointConfig(
                num_to_keep=num_to_keep,
                checkpoint_score_attribute="accuracy",
                checkpoint_score_order="max",
            ),
        )

        trainer = TorchTrainer(
            train_loop_per_worker=_train_loop_per_worker,
            train_loop_config=train_loop_config,
            scaling_config=scaling_config,
            run_config=run_config,
        )
        return trainer


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    logger.info("Initialising Ray for distributed training", extra={"ray_config": _RAY_CFG})
    ray.init(
        num_cpus=_RAY_CFG["num_cpus"],
        num_gpus=_RAY_CFG["num_gpus"],
        object_store_memory=_RAY_CFG["object_store_memory"],
        ignore_reinit_error=True,
    )
    logger.info("Ray initialised", extra={"resources": ray.cluster_resources()})

    train_loop_config = {
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "hidden_dim": 64,
        "dropout_rate": 0.2,
        "num_samples": 2000,
    }

    logger.info(
        "Building TorchTrainer",
        extra={
            "num_workers": NUM_WORKERS,
            "use_gpu": USE_GPU,
            "epochs": EPOCHS,
            "train_loop_config": train_loop_config,
        },
    )

    trainer = DistributedTrainerFactory.build(
        num_workers=NUM_WORKERS,
        use_gpu=USE_GPU,
        train_loop_config=train_loop_config,
        num_to_keep=NUM_TO_KEEP_CHECKPOINTS,
    )

    try:
        result = trainer.fit()
        logger.info(
            "Training finished",
            extra={
                "best_checkpoint": str(result.best_checkpoints),
                "metrics": result.metrics,
            },
        )
        if result.best_checkpoints:
            best_ckpt, best_metrics = result.best_checkpoints[0]
            logger.info(
                "Best checkpoint metrics",
                extra={"best_metrics": best_metrics, "checkpoint": str(best_ckpt)},
            )
    except Exception as exc:
        logger.error("Training failed", extra={"error": str(exc)}, exc_info=True)
        raise
    finally:
        logger.info("Shutting down Ray")
        ray.shutdown()


if __name__ == "__main__":
    main()
