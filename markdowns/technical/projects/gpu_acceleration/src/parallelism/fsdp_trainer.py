"""
FullyShardedDataParallel (FSDP) Trainer — Tutorial Module 3b

Demonstrates:
  - Launching with torchrun: `torchrun --nproc_per_node=2 -m src.parallelism.fsdp_trainer`
  - FSDP sharding strategies: FULL_SHARD, SHARD_GRAD_OP, NO_SHARD
  - auto_wrap_policy to shard large sub-modules (TransformerBlock)
  - CPU offloading option for activations/parameters
  - Memory savings: each GPU holds only 1/N of the parameters

Key concept: Unlike DDP (full model per GPU), FSDP shards model
parameters, gradients, and optionally optimizer states across GPUs.
During forward/backward, each GPU all-gathers parameters just-in-time
and releases them immediately (FULL_SHARD). This cuts per-GPU memory
by ~N× at the cost of extra all-gather communication.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from functools import partial
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, CPUOffload
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.logging_utils import setup_logger
from src.transformer_training.model import TransformerLM, TransformerBlock
from src.transformer_training.dataset import SyntheticTokenDataset

CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "parallelism.yaml"

_SHARDING_MAP = {
    "FULL_SHARD": ShardingStrategy.FULL_SHARD,
    "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
    "NO_SHARD": ShardingStrategy.NO_SHARD,
}


class FSDPTrainer:
    def __init__(self, config: dict[str, Any], rank: int, world_size: int) -> None:
        self._cfg = config
        self._rank = rank
        self._world_size = world_size
        self._device = torch.device(f"cuda:{rank}")

        log_cfg = {**config, "logging": {**config["logging"], "level": "INFO" if rank == 0 else "WARNING"}}
        self._log = setup_logger(f"parallelism.fsdp.rank{rank}", log_cfg)

        self._build_components()

    def _build_components(self) -> None:
        t_cfg = self._cfg["training"]
        fsdp_cfg = self._cfg["fsdp"]

        dataset = SyntheticTokenDataset(self._cfg)
        sampler = DistributedSampler(dataset, num_replicas=self._world_size, rank=self._rank)
        self._loader = DataLoader(
            dataset,
            batch_size=t_cfg["batch_size"] // self._world_size,
            sampler=sampler,
            pin_memory=True,
            num_workers=2,
        )
        self._sampler = sampler

        # Wrap policy: shard any module with ≥ min_num_params parameters
        wrap_policy = partial(
            size_based_auto_wrap_policy,
            min_num_params=fsdp_cfg["min_num_params"],
        )

        strategy = _SHARDING_MAP[fsdp_cfg["sharding_strategy"]]
        cpu_offload = CPUOffload(offload_params=True) if fsdp_cfg["cpu_offload"] else None

        model = TransformerLM(self._cfg).to(self._device)
        self._model = FSDP(
            model,
            auto_wrap_policy=wrap_policy,
            sharding_strategy=strategy,
            cpu_offload=cpu_offload,
            device_id=self._rank,
            use_orig_params=True,  # required for AdamW weight decay masking
        )

        self._optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=t_cfg["learning_rate"],
            weight_decay=t_cfg["weight_decay"],
        )
        self._criterion = nn.CrossEntropyLoss()
        self._amp_dtype = torch.bfloat16 if t_cfg["amp_dtype"] == "bfloat16" else torch.float16

        if self._rank == 0:
            self._log.info(
                "FSDP: world_size=%d  strategy=%s  cpu_offload=%s",
                self._world_size, fsdp_cfg["sharding_strategy"], fsdp_cfg["cpu_offload"],
            )
            mem = torch.cuda.memory_reserved(self._device) / 1e9
            self._log.info("Reserved GPU memory after FSDP init: %.2f GB", mem)

    def _train_step(self, tokens: torch.Tensor, targets: torch.Tensor) -> float:
        tokens = tokens.to(self._device, non_blocking=True)
        targets = targets.to(self._device, non_blocking=True)

        self._optimizer.zero_grad(set_to_none=True)

        use_amp = self._cfg["training"]["use_amp"]
        with torch.autocast(device_type="cuda", dtype=self._amp_dtype, enabled=use_amp):
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
        self._log.info("FSDP training: rank=%d  epochs=%d", self._rank, num_epochs)

        for epoch in range(1, num_epochs + 1):
            self._sampler.set_epoch(epoch)
            self._model.train()
            epoch_loss = 0.0
            t0 = time.perf_counter()

            for step, (tokens, targets) in enumerate(self._loader, 1):
                loss = self._train_step(tokens, targets)
                epoch_loss += loss
                if self._rank == 0 and step % log_every == 0:
                    self._log.info("epoch=%d step=%d loss=%.4f", epoch, step, loss)

            if self._rank == 0:
                elapsed = time.perf_counter() - t0
                mem_gb = torch.cuda.max_memory_allocated(self._device) / 1e9
                self._log.info(
                    "=== Epoch %d  avg_loss=%.4f  time=%.1f s  peak_mem=%.2f GB ===",
                    epoch, epoch_loss / len(self._loader), elapsed, mem_gb,
                )


def main() -> None:
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    dist.init_process_group(backend=config["distributed"]["backend"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    trainer = FSDPTrainer(config, rank, world_size)
    trainer.train()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
