"""
Synthetic Token Dataset — Tutorial Module 2b

Generates random token sequences so the training loop can run
without a real corpus, keeping the tutorial self-contained.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset


class SyntheticTokenDataset(Dataset):
    """Random integer token sequences for next-token prediction."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        ds = cfg["dataset"]
        self._n = ds["num_samples"]
        self._seq_len = ds["seq_len"]
        self._vocab_size = ds["vocab_size"]
        # Pre-generate the whole dataset in one shot (fits in CPU RAM for tutorial sizes)
        self._data = torch.randint(0, self._vocab_size, (self._n, self._seq_len + 1))

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self._data[idx]
        # Input: [0..T-1], Target: [1..T]  — standard LM shift
        return tokens[:-1], tokens[1:]
