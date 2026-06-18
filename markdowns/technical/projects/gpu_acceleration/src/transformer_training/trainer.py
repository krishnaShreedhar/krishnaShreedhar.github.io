"""
GPU Trainer — Tutorial Module 2c

Demonstrates:
  - Moving model and data to GPU
  - Forward pass → loss → backward pass → optimizer step
  - Automatic Mixed Precision (AMP) with BF16 on H200
  - Gradient clipping
  - Logging training metrics at configurable intervals
  - Periodic checkpoint saving
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from src.logging_utils import setup_logger
from src.transformer_training.model import TransformerLM
from src.transformer_training.dataset import SyntheticTokenDataset


class Trainer:
    def __init__(self, config: dict[str, Any]) -> None:
        self._cfg = config
        self._log = setup_logger("transformer_training.trainer", config)
        self._device = torch.device(config["gpu"]["device"])
        t_cfg = config["training"]

        self._num_epochs = t_cfg["num_epochs"]
        self._grad_clip = t_cfg["gradient_clip_norm"]
        self._log_every = t_cfg["log_every_n_steps"]
        self._ckpt_dir = Path(t_cfg["checkpoint_dir"])
        self._use_amp = t_cfg["use_amp"]
        self._amp_dtype = torch.bfloat16 if t_cfg["amp_dtype"] == "bfloat16" else torch.float16

        self._build_components()

    def _build_components(self) -> None:
        t_cfg = self._cfg["training"]

        dataset = SyntheticTokenDataset(self._cfg)
        self._loader = DataLoader(
            dataset,
            batch_size=t_cfg["batch_size"],
            shuffle=True,
            pin_memory=True,
            num_workers=2,
        )

        self._model = TransformerLM(self._cfg).to(self._device)
        self._log.info(
            "Model parameters: %.2fM on %s",
            self._model.num_parameters() / 1e6, self._device,
        )

        self._optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=self._cfg["training"]["learning_rate"],
            weight_decay=self._cfg["training"]["weight_decay"],
        )
        self._criterion = nn.CrossEntropyLoss()

        # GradScaler only needed for float16; bfloat16 doesn't underflow
        self._scaler: GradScaler | None = (
            GradScaler() if (self._use_amp and self._amp_dtype == torch.float16) else None
        )

    def _train_step(
        self, tokens: torch.Tensor, targets: torch.Tensor
    ) -> tuple[float, float]:
        """Single forward + backward pass. Returns (loss, tokens_per_sec)."""
        tokens = tokens.to(self._device, non_blocking=True)
        targets = targets.to(self._device, non_blocking=True)

        t0 = time.perf_counter()
        self._optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=self._amp_dtype, enabled=self._use_amp):
            logits = self._model(tokens)               # (B, T, vocab_size)
            loss = self._criterion(
                logits.view(-1, logits.size(-1)),      # (B*T, vocab_size)
                targets.view(-1),                      # (B*T,)
            )

        if self._scaler is not None:
            self._scaler.scale(loss).backward()
            self._scaler.unscale_(self._optimizer)
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._grad_clip)
            self._scaler.step(self._optimizer)
            self._scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._grad_clip)
            self._optimizer.step()

        torch.cuda.synchronize(self._device)
        elapsed = time.perf_counter() - t0
        tok_per_s = tokens.numel() / elapsed

        return loss.item(), tok_per_s

    def train(self) -> None:
        self._log.info("Starting training for %d epochs", self._num_epochs)
        global_step = 0

        for epoch in range(1, self._num_epochs + 1):
            self._model.train()
            epoch_loss = 0.0
            epoch_tokens = 0

            t_epoch = time.perf_counter()
            for step, (tokens, targets) in enumerate(self._loader, 1):
                loss, tok_per_s = self._train_step(tokens, targets)
                epoch_loss += loss
                epoch_tokens += tokens.numel()
                global_step += 1

                if step % self._log_every == 0:
                    self._log.info(
                        "epoch=%d  step=%d  loss=%.4f  tok/s=%.0f  amp=%s",
                        epoch, step, loss, tok_per_s, self._amp_dtype,
                    )

            epoch_elapsed = time.perf_counter() - t_epoch
            self._log.info(
                "=== Epoch %d done  avg_loss=%.4f  time=%.1f s  total_tok=%d ===",
                epoch, epoch_loss / len(self._loader), epoch_elapsed, epoch_tokens,
            )
            self._save_checkpoint(epoch, epoch_loss / len(self._loader))

    def _save_checkpoint(self, epoch: int, loss: float) -> None:
        self._ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = self._ckpt_dir / f"epoch_{epoch:03d}_loss_{loss:.4f}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self._model.state_dict(),
                "optimizer_state_dict": self._optimizer.state_dict(),
                "loss": loss,
            },
            path,
        )
        self._log.info("Checkpoint saved → %s", path)
