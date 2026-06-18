"""BaseTrainer: shared DDP training loop used by pre-training and fine-tuning.

Responsibilities:
  - Distributed setup (torch.distributed / DDP)
  - Mixed-precision (BF16 via torch.amp)
  - Gradient accumulation and clipping
  - LR scheduling (cosine with warmup)
  - Periodic checkpointing
  - WandB logging (rank-0 only)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """Return a scheduler with linear warmup followed by cosine decay."""
    import math

    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return step / max(1, num_warmup_steps)
        progress = (step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)


def setup_distributed() -> tuple[int, int, int]:
    """Initialise torch.distributed for DDP. Returns (rank, local_rank, world_size)."""
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    logger.info("DDP initialised: rank=%d/%d (local_rank=%d)", rank, world_size, local_rank)
    return rank, local_rank, world_size


class BaseTrainer:
    """Template trainer; subclasses override `build_dataloaders` and may extend `train_epoch`."""

    def __init__(self, model: nn.Module, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.rank, self.local_rank, self.world_size = setup_distributed()
        self.is_main = self.rank == 0

        self.device = torch.device(f"cuda:{self.local_rank}")
        model = model.to(self.device)
        self.model = DDP(model, device_ids=[self.local_rank],
                         find_unused_parameters=cfg.get("ddp_find_unused_parameters", False))

        self.optimizer = self._build_optimizer()
        self.scaler = GradScaler(enabled=cfg.get("fp16", False))
        self.use_bf16 = cfg.get("bf16", True)
        self.autocast_dtype = torch.bfloat16 if self.use_bf16 else torch.float16

        output_dir = Path(cfg["output_dir"])
        if self.is_main:
            output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir

        self.global_step = 0
        self.best_val_loss = float("inf")

        if self.is_main and cfg.get("wandb_project"):
            self._init_wandb()

        logger.info("BaseTrainer ready on device %s", self.device)

    def _build_optimizer(self) -> AdamW:
        cfg = self.cfg
        # Separate weight-decayed params from biases / LayerNorm
        decay, no_decay = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim <= 1 or "bias" in name:
                no_decay.append(param)
            else:
                decay.append(param)
        param_groups = [
            {"params": decay, "weight_decay": cfg.get("weight_decay", 0.05)},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return AdamW(
            param_groups,
            lr=cfg["learning_rate"],
            betas=(cfg.get("beta1", 0.9), cfg.get("beta2", 0.95)),
            eps=cfg.get("eps", 1e-8),
        )

    def _init_wandb(self) -> None:
        import wandb  # type: ignore[import]
        wandb.init(
            project=self.cfg["wandb_project"],
            name=self.cfg.get("wandb_run_name"),
            config=self.cfg,
        )
        logger.info("WandB initialised: project=%s", self.cfg["wandb_project"])

    def _log(self, metrics: dict[str, float], step: int) -> None:
        if not self.is_main:
            return
        try:
            import wandb
            if wandb.run is not None:
                wandb.log(metrics, step=step)
        except ImportError:
            pass
        logger.info("step=%d | %s", step, " | ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    def _save_checkpoint(self, tag: str) -> None:
        if not self.is_main:
            return
        path = self.output_dir / f"checkpoint_{tag}.pt"
        state = {
            "global_step": self.global_step,
            "model_state_dict": self.model.module.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
        }
        torch.save(state, path)
        logger.info("Checkpoint saved: %s", path)

    def _load_checkpoint(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        self.model.module.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.global_step = state.get("global_step", 0)
        self.best_val_loss = state.get("best_val_loss", float("inf"))
        logger.info("Resumed from %s at step %d", path, self.global_step)

    def build_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        raise NotImplementedError

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def validate(self, val_loader: DataLoader) -> float:
        self.model.eval()
        total_loss, n_batches = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                with torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
                    loss = self.compute_loss(batch)
                total_loss += loss.item()
                n_batches += 1
        self.model.train()
        return total_loss / max(n_batches, 1)

    def train(self) -> None:
        cfg = self.cfg
        train_loader, val_loader = self.build_dataloaders()

        resume = cfg.get("resume_from_checkpoint")
        if resume:
            self._load_checkpoint(resume)

        steps_per_epoch = len(train_loader) // cfg.get("gradient_accumulation_steps", 1)
        total_steps = cfg["num_epochs"] * steps_per_epoch
        scheduler = _cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=cfg.get("warmup_steps", 0),
            num_training_steps=total_steps,
            min_lr_ratio=cfg.get("min_lr_ratio", 0.1),
        )

        grad_accum = cfg.get("gradient_accumulation_steps", 1)
        log_interval = cfg.get("log_interval", 50)
        save_interval = cfg.get("save_interval", 1000)
        grad_clip = cfg.get("grad_clip", 1.0)

        logger.info(
            "Training: epochs=%d, steps/epoch=%d, total_steps=%d",
            cfg["num_epochs"], steps_per_epoch, total_steps,
        )

        self.model.train()
        for epoch in range(cfg["num_epochs"]):
            if isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(epoch)

            accum_loss = 0.0
            self.optimizer.zero_grad()

            for step_in_epoch, batch in enumerate(train_loader):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                with torch.autocast(device_type="cuda", dtype=self.autocast_dtype,
                                    enabled=self.use_bf16 or cfg.get("fp16", False)):
                    loss = self.compute_loss(batch) / grad_accum

                if cfg.get("fp16", False):
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                accum_loss += loss.item()

                if (step_in_epoch + 1) % grad_accum == 0:
                    if cfg.get("fp16", False):
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

                    if cfg.get("fp16", False):
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                    if self.global_step % log_interval == 0:
                        lr = scheduler.get_last_lr()[0]
                        self._log({"train/loss": accum_loss, "train/lr": lr}, self.global_step)
                        accum_loss = 0.0

                    if self.global_step % save_interval == 0:
                        val_loss = self.validate(val_loader)
                        self._log({"val/loss": val_loss}, self.global_step)
                        self._save_checkpoint(f"step_{self.global_step}")
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self._save_checkpoint("best")

            logger.info("Epoch %d/%d complete", epoch + 1, cfg["num_epochs"])

        self._save_checkpoint("final")
        dist.destroy_process_group()
        logger.info("Training complete. Best val loss: %.4f", self.best_val_loss)
