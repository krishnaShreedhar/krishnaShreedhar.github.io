"""
Attention with and without KV-Cache — Tutorial Module 4a

Side-by-side implementations show exactly what the KV-cache saves:
  - Naive: recomputes K and V for the full sequence at every decode step → O(T²)
  - Cached: K and V are computed once per token and reused → O(T) per step

Run standalone for a complexity comparison:
  python -m src.kv_cache.attention
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.kv_cache.kv_cache_manager import KVCacheManager


class MultiHeadAttentionNaive(nn.Module):
    """
    Standard multi-head attention without caching.
    At each decode step, recomputes K,V from the full history.
    Cost: O(T²) FLOPs per token.
    """

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C = x.shape

        def split(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        Q, K, V = split(self.q(x)), split(self.k(x)), split(self.v(x))
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)


class MultiHeadAttentionCached(nn.Module):
    """
    Multi-head attention with an explicit KV-cache.

    During prefill: processes the full prompt and populates the cache.
    During decode: processes a single new token, reads K/V from cache,
                   appends new K/V, and runs attention over the full history.

    Cost: O(T) FLOPs per decode step (K/V not recomputed).
    """

    def __init__(
        self, d_model: int, num_heads: int, layer_idx: int, cache: KVCacheManager
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.layer_idx = layer_idx
        self.cache = cache

        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
    ) -> torch.Tensor:
        B, T, C = x.shape  # T=seq_len during prefill, T=1 during decode

        def split(t: torch.Tensor, seq_len: int) -> torch.Tensor:
            return t.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        Q = split(self.q(x), T)
        K_new = split(self.k(x), T)
        V_new = split(self.v(x), T)

        # Update and fetch full K/V history from cache
        K, V = self.cache.update(self.layer_idx, K_new, V_new, start_pos)

        # Attention over cached + new tokens
        seq_len_kv = start_pos + T
        scores = torch.matmul(Q, K[:, :, :seq_len_kv].transpose(-2, -1)) / self.scale
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V[:, :, :seq_len_kv])
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)
