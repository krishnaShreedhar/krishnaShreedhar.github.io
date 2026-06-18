"""Hyperparameter tuning with Optuna.

Each trial:
  1. Samples hyperparameters from the search space defined in configs/tuning.yaml.
  2. Builds a SmallVLM with those parameters (overriding model.yaml defaults).
  3. Trains for a small number of epochs on a subset of training data.
  4. Returns validation CIDEr (or loss) as the optimisation objective.
  5. Optionally prunes underperforming trials early.

Usage:
  python -m src.tuning.hyperparameter_tuning
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import optuna
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from src.data.collator import ImageCaptionCollator
from src.data.dataset import ImageCaptionDataset, build_tokenizer
from src.data.transforms import train_transform, val_transform
from src.model.config import ModelConfig
from src.model.vlm import SmallVLM
from src.utils.config_utils import get_section, load_yaml
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_CONFIG_PATH = "configs/tuning.yaml"
_MODEL_CONFIG_PATH = "configs/model.yaml"


def _suggest_param(trial: optuna.Trial, name: str, spec: dict[str, Any]) -> Any:
    """Sample a single parameter from an Optuna trial given its search-space spec."""
    kind = spec["type"]
    if kind == "loguniform":
        return trial.suggest_float(name, spec["low"], spec["high"], log=True)
    if kind == "uniform":
        return trial.suggest_float(name, spec["low"], spec["high"])
    if kind == "int":
        return trial.suggest_int(name, spec["low"], spec["high"], step=spec.get("step", 1))
    if kind == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    raise ValueError(f"Unknown search-space type: {kind}")


def _build_model(model_cfg: ModelConfig, overrides: dict[str, Any]) -> SmallVLM:
    """Return a SmallVLM with trial-sampled architecture overrides applied."""
    model_overrides = {
        k: v for k, v in overrides.items()
        if k in model_cfg.__dataclass_fields__  # type: ignore[attr-defined]
    }
    cfg = model_cfg.override(model_overrides)
    return SmallVLM(cfg)


def _train_and_evaluate(
    model: SmallVLM,
    cfg: dict[str, Any],
    params: dict[str, Any],
    trial: optuna.Trial,
    output_dir: Path,
) -> float:
    """Train for a few epochs and return negative validation loss (proxy for CIDEr)."""
    from torch.optim import AdamW
    from src.training.trainer import _cosine_schedule_with_warmup

    trial_cfg = cfg.get("trial", {})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    tokenizer = build_tokenizer()
    image_size = model.cfg.image_size

    train_ds = ImageCaptionDataset(
        dataset_name=cfg["dataset_name"],
        split="train",
        tokenizer=tokenizer,
        transform=train_transform(image_size),
        max_seq_len=model.cfg.max_seq_len,
        max_samples=trial_cfg.get("max_train_samples", 20000),
        cache_dir=cfg.get("data_root"),
    )
    val_ds = ImageCaptionDataset(
        dataset_name=cfg["dataset_name"],
        split="validation",
        tokenizer=tokenizer,
        transform=val_transform(image_size),
        max_seq_len=model.cfg.max_seq_len,
        max_samples=trial_cfg.get("max_val_samples", 2000),
        cache_dir=cfg.get("data_root"),
    )
    collator = ImageCaptionCollator()
    bsz = params.get("batch_size_per_gpu", 32)
    train_loader = DataLoader(
        train_ds, batch_size=bsz, sampler=RandomSampler(train_ds),
        num_workers=4, pin_memory=True, collate_fn=collator, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bsz, sampler=SequentialSampler(val_ds),
        num_workers=4, collate_fn=collator,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=params["learning_rate"],
        weight_decay=params.get("weight_decay", 0.05),
    )
    grad_accum = trial_cfg.get("gradient_accumulation_steps", 4)
    total_steps = len(train_loader) * trial_cfg.get("num_epochs", 3) // grad_accum
    scheduler = _cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=params.get("warmup_steps", 200),
        num_training_steps=total_steps,
    )
    use_bf16 = trial_cfg.get("bf16", True)
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16

    model.train()
    global_step = 0
    for epoch in range(trial_cfg.get("num_epochs", 3)):
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=use_bf16):
                out = model(
                    images=batch["images"],
                    caption_ids=batch["caption_ids"],
                    attention_mask=batch["attention_mask"],
                )
                loss = out["loss"] / grad_accum
            loss.backward()

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

        # Mid-epoch pruning report
        val_loss = _validate(model, val_loader, device, autocast_dtype, use_bf16)
        logger.info("Trial %d epoch %d: val_loss=%.4f", trial.number, epoch, val_loss)
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return val_loss   # Lower is better → Optuna direction="minimize"


def _validate(model, loader, device, dtype, use_autocast) -> float:
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=use_autocast):
                out = model(
                    images=batch["images"],
                    caption_ids=batch["caption_ids"],
                    attention_mask=batch["attention_mask"],
                )
            total += out["loss"].item()
            n += 1
    model.train()
    return total / max(n, 1)


def objective(trial: optuna.Trial, cfg: dict[str, Any], model_cfg: ModelConfig,
              output_dir: Path) -> float:
    """Optuna objective function: sample params → train → return val loss."""
    search_space = cfg["search_space"]
    params = {name: _suggest_param(trial, name, spec)
              for name, spec in search_space.items()}
    logger.info("Trial %d params: %s", trial.number, params)

    # Override architecture params in model config
    model = _build_model(model_cfg, params)
    val_loss = _train_and_evaluate(model, cfg, params, trial, output_dir)

    trial_dir = output_dir / f"trial_{trial.number}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"trial_number": trial.number, "params": params, "val_loss": val_loss},
               trial_dir / "result.pt")
    return val_loss


def main() -> None:
    raw_cfg = load_yaml(_CONFIG_PATH)
    cfg = get_section(raw_cfg, "tuning")
    logger.info("Starting hyperparameter tuning study: %s", cfg["study_name"])

    model_cfg = ModelConfig.from_yaml(_MODEL_CONFIG_PATH)
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    pruner: optuna.pruners.BasePruner
    pruner_name = cfg.get("pruner", "median")
    if pruner_name == "median":
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=cfg.get("pruner_n_startup_trials", 5),
            n_warmup_steps=cfg.get("pruner_n_warmup_steps", 500),
        )
    elif pruner_name == "hyperband":
        pruner = optuna.pruners.HyperbandPruner()
    else:
        pruner = optuna.pruners.NopPruner()

    study = optuna.create_study(
        study_name=cfg["study_name"],
        storage=cfg.get("storage"),
        direction="minimize",   # minimise val_loss
        pruner=pruner,
        load_if_exists=True,
    )

    study.optimize(
        lambda trial: objective(trial, cfg, model_cfg, output_dir),
        n_trials=cfg.get("n_trials", 50),
        n_jobs=cfg.get("n_jobs", 1),
        timeout=cfg.get("timeout"),
    )

    best = study.best_trial
    logger.info("Best trial: number=%d, val_loss=%.4f", best.number, best.value)
    logger.info("Best params: %s", best.params)

    import json
    summary = {"best_trial": best.number, "best_val_loss": best.value, "best_params": best.params}
    with (output_dir / "best_trial.json").open("w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary saved to %s/best_trial.json", output_dir)


if __name__ == "__main__":
    main()
