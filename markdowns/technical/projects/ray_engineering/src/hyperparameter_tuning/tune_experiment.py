"""
Ray Tune – Hyperparameter Optimisation Experiment.

Demonstrates:
  - Defining a param_space (search space) using ray.tune primitives
  - ASHAScheduler for early stopping of poor trials
  - OptunaSearch as the Bayesian search algorithm
  - TuneConfig for experiment-level settings
  - Integration of a TorchTrainer inside Tuner
  - Analysing results from ResultGrid

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
import ray.tune
import torch
import torch.nn as nn
import torch.optim as optim
from ray import tune
from ray.train import Checkpoint, ScalingConfig
from ray.train.torch import TorchTrainer
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.config_loader import load_config
from utils.logging_setup import get_logger

_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config.yaml"
config: dict[str, Any] = load_config(str(_CONFIG_PATH))
logger = get_logger(__name__, config)

_TUNE_CFG = config["hyperparameter_tuning"]
_RAY_CFG = config["ray"]["init"]

NUM_SAMPLES: int = _TUNE_CFG["num_samples"]
MAX_EPOCHS: int = _TUNE_CFG["max_epochs"]
GRACE_PERIOD: int = _TUNE_CFG["grace_period"]
REDUCTION_FACTOR: int = _TUNE_CFG["reduction_factor"]
METRIC: str = _TUNE_CFG["metric"]
MODE: str = _TUNE_CFG["mode"]

NUM_FEATURES: int = 10


# ===========================================================================
# Model (same architecture as distributed_training for consistency)
# ===========================================================================

class TunableClassifier(nn.Module):
    """Feedforward classifier with tunable hidden dim and dropout."""

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
# Dataset helper
# ===========================================================================

def _make_tensor_dataset(num_samples: int, num_features: int, seed: int = 0) -> TensorDataset:
    rng = np.random.default_rng(seed=seed)
    X = rng.standard_normal((num_samples, num_features)).astype(np.float32)
    w = rng.standard_normal(num_features).astype(np.float32)
    y = (X @ w > 0).astype(np.float32)
    return TensorDataset(torch.from_numpy(X), torch.from_numpy(y))


# ===========================================================================
# Trainable (per-worker loop injected into TorchTrainer)
# ===========================================================================

def _trainable_loop(train_config: dict[str, Any]) -> None:
    """
    Per-worker training loop used inside TorchTrainer during Tune.

    Reads hyperparameters from train_config (which Tune populates from the
    search space for each trial).  Reports accuracy to Tune so that
    schedulers and search algorithms can act on it.
    """
    import logging as _logging

    w_log = _logging.getLogger("tune_worker")
    w_log.setLevel(_logging.DEBUG)

    lr: float = train_config["lr"]
    hidden_dim: int = train_config["hidden_dim"]
    dropout_rate: float = train_config["dropout_rate"]
    batch_size: int = train_config["batch_size"]
    epochs: int = train_config["epochs"]
    num_samples: int = train_config["num_samples"]

    model = TunableClassifier(NUM_FEATURES, hidden_dim, dropout_rate)
    model = ray.train.torch.prepare_model(model)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    dataset = _make_tensor_dataset(num_samples, NUM_FEATURES)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loader = ray.train.torch.prepare_data_loader(loader)

    # Optionally resume from checkpoint
    checkpoint = ray.train.get_checkpoint()
    start_epoch = 0
    if checkpoint:
        with checkpoint.as_directory() as ckpt_dir:
            state = torch.load(Path(ckpt_dir) / "checkpoint.pt", weights_only=False)
            model.module.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            start_epoch = state["epoch"] + 1

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            out = model(X_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y_batch)
            correct += ((out >= 0.5).float() == y_batch).sum().item()
            total += len(y_batch)

        avg_loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1)

        w_log.debug(
            "Tune trial epoch=%d loss=%.4f acc=%.4f", epoch + 1, avg_loss, accuracy
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                Path(tmp_dir) / "checkpoint.pt",
            )
            ckpt = Checkpoint.from_directory(tmp_dir)
            ray.train.report(
                metrics={"loss": avg_loss, "accuracy": accuracy, "epoch": epoch + 1},
                checkpoint=ckpt,
            )


# ===========================================================================
# Search space definition (Single Responsibility)
# ===========================================================================

def build_param_space() -> dict[str, Any]:
    """
    Define the hyperparameter search space.

    Uses Ray Tune's built-in distributions so that OptunaSearch can
    efficiently explore the space.
    """
    return {
        "train_loop_config": {
            "lr": tune.loguniform(1e-4, 1e-1),
            "hidden_dim": tune.choice([32, 64, 128, 256]),
            "dropout_rate": tune.uniform(0.0, 0.5),
            "batch_size": tune.choice([32, 64, 128]),
            "epochs": MAX_EPOCHS,
            "num_samples": 1500,
        }
    }


# ===========================================================================
# Experiment runner (Open/Closed: swap scheduler or searcher easily)
# ===========================================================================

class TuneExperimentRunner:
    """
    Composes TorchTrainer + Tuner with ASHA + Optuna.

    Responsibilities:
      - Build scheduler
      - Build search algorithm
      - Build TorchTrainer (inner trainable)
      - Build Tuner and run it
      - Log and return ResultGrid
    """

    def __init__(
        self,
        num_samples: int,
        metric: str,
        mode: str,
        grace_period: int,
        reduction_factor: int,
    ) -> None:
        self._num_samples = num_samples
        self._metric = metric
        self._mode = mode
        self._grace_period = grace_period
        self._reduction_factor = reduction_factor

        logger.info(
            "TuneExperimentRunner created",
            extra={
                "num_samples": num_samples,
                "metric": metric,
                "mode": mode,
                "grace_period": grace_period,
                "reduction_factor": reduction_factor,
            },
        )

    def _build_scheduler(self) -> ASHAScheduler:
        scheduler = ASHAScheduler(
            metric=self._metric,
            mode=self._mode,
            max_t=MAX_EPOCHS,
            grace_period=self._grace_period,
            reduction_factor=self._reduction_factor,
        )
        logger.debug("ASHAScheduler built", extra={"max_t": MAX_EPOCHS})
        return scheduler

    def _build_search_algorithm(self) -> OptunaSearch:
        searcher = OptunaSearch(
            metric=self._metric,
            mode=self._mode,
        )
        logger.debug("OptunaSearch algorithm built")
        return searcher

    def _build_trainer(self) -> TorchTrainer:
        trainer = TorchTrainer(
            train_loop_per_worker=_trainable_loop,
            scaling_config=ScalingConfig(
                num_workers=1,  # 1 worker per Tune trial to fit on a laptop
                use_gpu=False,
            ),
        )
        logger.debug("TorchTrainer (inner) built for Tune")
        return trainer

    def run(self) -> ray.tune.ResultGrid:
        scheduler = self._build_scheduler()
        search_alg = self._build_search_algorithm()
        trainer = self._build_trainer()
        param_space = build_param_space()

        logger.info(
            "Building Tuner",
            extra={
                "param_space_keys": list(
                    param_space.get("train_loop_config", {}).keys()
                ),
                "num_samples": self._num_samples,
            },
        )

        tuner = tune.Tuner(
            trainer,
            param_space=param_space,
            tune_config=tune.TuneConfig(
                metric=self._metric,
                mode=self._mode,
                num_samples=self._num_samples,
                scheduler=scheduler,
                search_alg=search_alg,
            ),
            run_config=ray.train.RunConfig(
                name="ray_tune_hpo_experiment",
                verbose=0,
            ),
        )

        logger.info("Starting Tune experiment – this may take a while")
        results: ray.tune.ResultGrid = tuner.fit()

        best_result = results.get_best_result(metric=self._metric, mode=self._mode)
        logger.info(
            "Tune experiment complete",
            extra={
                "best_config": best_result.config,
                "best_metrics": best_result.metrics,
                "num_trials": len(results),
            },
        )

        # Log top-5 trials
        df = results.get_dataframe()
        top5 = df.nlargest(5, f"accuracy")[
            ["accuracy", "loss", "config/train_loop_config/lr",
             "config/train_loop_config/hidden_dim",
             "config/train_loop_config/dropout_rate"]
        ]
        logger.info("Top-5 trials", extra={"top5": top5.to_dict(orient="records")})

        return results


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    logger.info("Initialising Ray for HPO", extra={"ray_config": _RAY_CFG})
    ray.init(
        num_cpus=_RAY_CFG["num_cpus"],
        num_gpus=_RAY_CFG["num_gpus"],
        object_store_memory=_RAY_CFG["object_store_memory"],
        ignore_reinit_error=True,
    )
    logger.info("Ray initialised", extra={"resources": ray.cluster_resources()})

    runner = TuneExperimentRunner(
        num_samples=NUM_SAMPLES,
        metric=METRIC,
        mode=MODE,
        grace_period=GRACE_PERIOD,
        reduction_factor=REDUCTION_FACTOR,
    )

    try:
        runner.run()
    except Exception as exc:
        logger.error("HPO experiment failed", extra={"error": str(exc)}, exc_info=True)
        raise
    finally:
        logger.info("Shutting down Ray")
        ray.shutdown()


if __name__ == "__main__":
    main()
