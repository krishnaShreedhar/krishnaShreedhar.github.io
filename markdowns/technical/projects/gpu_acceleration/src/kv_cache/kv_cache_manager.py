"""
KV-Cache Manager — Tutorial Module 4b

Manages a pre-allocated GPU buffer that stores key and value tensors
for all layers across all sequence positions.

Pre-allocation strategy:
  - Allocate max_batch_size × max_seq_len slots upfront.
  - Avoid repeated alloc/free during generation (critical for latency).
  - Use bfloat16 to halve the memory footprint vs float32.

Memory cost:
  2 (K+V) × num_layers × num_heads × head_dim × max_seq_len × max_batch × dtype_bytes
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class CacheConfig:
    num_layers: int
    num_heads: int
    head_dim: int
    max_batch_size: int
    max_seq_len: int
    dtype: torch.dtype
    device: torch.device


class KVCacheManager:
    """
    Pre-allocated KV cache for a complete Transformer decoder.

    Shape of each buffer: (max_batch_size, num_heads, max_seq_len, head_dim)
    Indexed by layer_idx.
    """

    def __init__(self, cfg: CacheConfig) -> None:
        self._cfg = cfg
        shape = (cfg.max_batch_size, cfg.num_heads, cfg.max_seq_len, cfg.head_dim)

        self._k_cache = torch.zeros(cfg.num_layers, *shape, dtype=cfg.dtype, device=cfg.device)
        self._v_cache = torch.zeros(cfg.num_layers, *shape, dtype=cfg.dtype, device=cfg.device)

        mem_bytes = self._k_cache.numel() * 2 * self._k_cache.element_size()  # K + V

    @classmethod
    def from_config(cls, config: dict[str, Any], device: torch.device) -> "KVCacheManager":
        m = config["model"]
        kv = config["kv_cache"]
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        cfg = CacheConfig(
            num_layers=m["num_layers"],
            num_heads=m["num_heads"],
            head_dim=m["head_dim"],
            max_batch_size=kv["max_batch_size"],
            max_seq_len=kv["max_seq_len"],
            dtype=dtype_map[kv["dtype"]],
            device=device,
        )
        return cls(cfg)

    def update(
        self,
        layer_idx: int,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        start_pos: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Write new K/V slices into the cache at [start_pos : start_pos + seq_len]
        and return the full cache tensors for this layer.
        """
        seq_len = k_new.size(2)  # k_new: (B, H, T, head_dim)
        self._k_cache[layer_idx, :k_new.size(0), :, start_pos: start_pos + seq_len] = k_new.to(self._cfg.dtype)
        self._v_cache[layer_idx, :v_new.size(0), :, start_pos: start_pos + seq_len] = v_new.to(self._cfg.dtype)
        return self._k_cache[layer_idx], self._v_cache[layer_idx]

    def reset(self) -> None:
        """Zero the cache — call between independent generation requests."""
        self._k_cache.zero_()
        self._v_cache.zero_()

    def memory_bytes(self) -> int:
        return (self._k_cache.numel() + self._v_cache.numel()) * self._k_cache.element_size()
