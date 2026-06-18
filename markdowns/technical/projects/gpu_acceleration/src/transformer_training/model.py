"""
Minimal Transformer Model — Tutorial Module 2a

A standard encoder-only Transformer built from first principles using only
torch.nn primitives. No HuggingFace or external model libraries.

Architecture:
  Embedding → N × (MultiHeadAttention + FFN) → LayerNorm → LinearHead
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    """
    Scaled dot-product multi-head self-attention.
    Explicit Q/K/V projections illustrate the mechanics clearly.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, C = x.shape  # batch, seq_len, d_model

        # Project and reshape to (B, num_heads, T, head_dim)
        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        Q = split_heads(self.q_proj(x))
        K = split_heads(self.k_proj(x))
        V = split_heads(self.v_proj(x))

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, H, T, T)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        # Merge heads
        out = torch.matmul(attn, V)  # (B, H, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # Pre-norm (more stable than post-norm)
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ff(self.norm2(x))
        return x


class TransformerLM(nn.Module):
    """Encoder-only Transformer for masked language modelling / next-token prediction."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        m = cfg["model"]
        self.token_emb = nn.Embedding(m["vocab_size"], m["d_model"])
        self.pos_emb = nn.Embedding(m["max_seq_len"], m["d_model"])
        self.drop = nn.Dropout(m["dropout"])
        self.blocks = nn.ModuleList([
            TransformerBlock(m["d_model"], m["num_heads"], m["d_ff"], m["dropout"])
            for _ in range(m["num_layers"])
        ])
        self.norm = nn.LayerNorm(m["d_model"])
        self.head = nn.Linear(m["d_model"], m["vocab_size"], bias=False)
        # Weight tying: output projection shares weights with token embedding
        self.head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, T = tokens.shape
        positions = torch.arange(T, device=tokens.device).unsqueeze(0)  # (1, T)
        x = self.drop(self.token_emb(tokens) + self.pos_emb(positions))
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.head(x)  # (B, T, vocab_size)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
