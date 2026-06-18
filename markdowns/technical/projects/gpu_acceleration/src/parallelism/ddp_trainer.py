"""
DistributedDataParallel (DDP) Trainer — Tutorial Module 3a

Demonstrates:
  - Launching with torchrun: `torchrun --nproc_per_node=2 -m src.parallelism.ddp_trainer`
  - Process group initialisation (NCCL backend for GPU-to-GPU)
  - Wrapping a model in DDP
  - DistributedSampler to shard the dataset across ranks
  - All-reduce gradient synchronisation (happens automatically inside DDP)
  - Proper rank-aware logging (only rank 0 logs to avoid duplicate output)

Key concept: Every GPU holds a *full* copy of the model. After each
backward pass, DDP all-reduces gradients across GPUs so every replica
stays in sync. The throughput advantage comes from processing N×more
samples per step.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.logging_utils import setup_logger
from src.transformer_training.model import TransformerLM
from src.transformer_training.dataset import SyntheticTokenDataset

CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "parallelism.yaml"


class DDPTrainer:
    def __init__(self, config: dict[str, Any], rank: int, world_size: int) -> None:
        self._cfg = config
        self._rank = rank
        self._world_size = world_size
        self._device = torch.device(f"cuda:{rank}")

        # Only rank 0 writes logs to avoid interleaved output from multiple processes
        log_cfg = {**config, "logging": {**config["logging"], "level": "INFO" if rank == 0 else "WARNING"}}
        self._log = setup_logger(f"parallelism.ddp.rank{rank}", log_cfg)

        self._build_components()

    def _build_components(self) -> None:
        t_cfg = self._cfg["training"]

        dataset = SyntheticTokenDataset(self._cfg)
        sampler = DistributedSampler(
            dataset, num_replicas=self._world_size, rank=self._rank, shuffle=True
        )
        self._loader = DataLoader(
            dataset,
            batch_size=t_cfg["batch_size"] // self._world_size,
            sampler=sampler,
            pin_memory=True,
            num_workers=2,
        )
        self._sampler = sampler

        model = TransformerLM(self._cfg).to(self._device)
        # DDP wraps the model; gradients are all-reduced after each backward
        self._model = DDP(
            model,
            device_ids=[self._rank],
            find_unused_parameters=self._cfg["distributed"]["find_unused_parameters"],
        )

        self._optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=t_cfg["learning_rate"],
            weight_decay=t_cfg["weight_decay"],
        )
        self._criterion = nn.CrossEntropyLoss()
        self._amp_dtype = torch.bfloat16 if t_cfg["amp_dtype"] == "bfloat16" else torch.float16

        if self._rank == 0:
            raw_model = self._model.module
            self._log.info(
                "DDP: world_size=%d  model_params=%.2fM per GPU",
                self._world_size, sum(p.numel() for p in raw_model.parameters()) / 1e6,
            )

    def _train_step(self, tokens: torch.Tensor, targets: torch.Tensor) -> float:
        tokens = tokens.to(self._device, non_blocking=True)
        targets = targets.to(self._device, non_blocking=True)

        self._optimizer.zero_grad(set_to_none=True)

        amp_dtype = self._amp_dtype
        use_amp = self._cfg["training"]["use_amp"]
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            logits = self._model(tokens)
            loss = self._criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self._model.parameters(), self._cfg["training"]["gradient_clip_norm"]
        )
        self._optimizer.step()
        return loss.item()

    def train(self) -> None:
        num_epochs = self._cfg["training"]["num_epochs"]
        log_every = self._cfg["training"]["log_every_n_steps"]
        self._log.info("DDP training: rank=%d  epochs=%d", self._rank, num_epochs)

        for epoch in range(1, num_epochs + 1):
            self._sampler.set_epoch(epoch)  # ensures different shuffling per epoch
            self._model.train()
            epoch_loss = 0.0
            t0 = time.perf_counter()

            for step, (tokens, targets) in enumerate(self._loader, 1):
                loss = self._train_step(tokens, targets)
                epoch_loss += loss
                if self._rank == 0 and step % log_every == 0:
                    self._log.info("epoch=%d step=%d loss=%.4f", epoch, step, loss)

            elapsed = time.perf_counter() - t0
            if self._rank == 0:
                self._log.info(
                    "=== Epoch %d  avg_loss=%.4f  time=%.1f s ===",
                    epoch, epoch_loss / len(self._loader), elapsed,
                )


def _init_process_group(backend: str) -> tuple[int, int]:
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size


def main() -> None:
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    rank, world_size = _init_process_group(config["distributed"]["backend"])
    trainer = DDPTrainer(config, rank, world_size)
    trainer.train()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
