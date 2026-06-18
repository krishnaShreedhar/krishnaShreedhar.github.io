"""Language Decoder: a GPT-2-style autoregressive Transformer decoder.

Each DecoderBlock contains:
  1. Causal self-attention (masked so token i only attends to ≤ i)
  2. Cross-attention to vision tokens (optional, enabled via cfg.use_cross_attention)
  3. Position-wise FFN

Pipeline:
  token_ids (B, T)
    → TokenEmbedding + PosEmbedding  → (B, T, D)
    → Dropout
    → L × DecoderBlock               → (B, T, D)
    → LayerNorm
    → LM head (Linear, D → vocab)    → logits (B, T, vocab)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.config import ModelConfig
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a causal (lower-triangular) mask."""

    def __init__(self, hidden_dim: int, num_heads: int, attn_dropout: float,
                 max_seq_len: int) -> None:
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim, bias=True)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_drop = nn.Dropout(attn_dropout)

        # Register a causal mask buffer so it moves with the model (device-aware)
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len)).unsqueeze(0).unsqueeze(0)
        self.register_buffer("causal_mask", mask)  # (1, 1, T, T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)           # each (B, H, T, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        # Apply causal mask: positions beyond the diagonal get -inf
        attn = attn.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.out_proj(x)


class CrossAttention(nn.Module):
    """Multi-head cross-attention: queries from text, keys/values from vision tokens."""

    def __init__(self, hidden_dim: int, num_heads: int, attn_dropout: float) -> None:
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.kv_proj = nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_drop = nn.Dropout(attn_dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:       (B, T, D) — text token queries
            context: (B, N, D) — vision token keys/values
        Returns:
            (B, T, D) — attended text tokens
        """
        B, T, D = x.shape
        N = context.size(1)

        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(context).reshape(B, N, 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)               # each (B, H, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale   # (B, H, T, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.out_proj(x)


class DecoderMLP(nn.Module):
    def __init__(self, hidden_dim: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        inner = int(hidden_dim * mlp_ratio)
        self.fc1 = nn.Linear(hidden_dim, inner)
        self.fc2 = nn.Linear(inner, hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(F.gelu(self.fc1(x)))))


class DecoderBlock(nn.Module):
    """One Transformer decoder block (causal self-attn + optional cross-attn + FFN)."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        D = cfg.lang_hidden_dim
        self.norm1 = nn.LayerNorm(D)
        self.self_attn = CausalSelfAttention(
            D, cfg.lang_num_heads, cfg.lang_attn_dropout, cfg.max_seq_len
        )
        self.use_cross_attention = cfg.use_cross_attention
        if self.use_cross_attention:
            self.norm_cross = nn.LayerNorm(D)
            self.cross_attn = CrossAttention(D, cfg.lang_num_heads, cfg.lang_attn_dropout)
        self.norm2 = nn.LayerNorm(D)
        self.mlp = DecoderMLP(D, cfg.lang_mlp_ratio, cfg.lang_dropout)

    def forward(self, x: torch.Tensor, vision_tokens: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.self_attn(self.norm1(x))
        if self.use_cross_attention and vision_tokens is not None:
            x = x + self.cross_attn(self.norm_cross(x), vision_tokens)
        x = x + self.mlp(self.norm2(x))
        return x


class LanguageDecoder(nn.Module):
    """Autoregressive decoder that conditions on projected vision tokens.

    Returns:
        logits: (B, T, vocab_size) — raw unnormalised next-token scores.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        D = cfg.lang_hidden_dim
        self.tok_embed = nn.Embedding(cfg.vocab_size, D)
        self.pos_embed = nn.Embedding(cfg.max_seq_len, D)
        self.drop = nn.Dropout(cfg.lang_dropout)
        self.blocks = nn.ModuleList([DecoderBlock(cfg) for _ in range(cfg.lang_num_layers)])
        self.norm = nn.LayerNorm(D)
        self.lm_head = nn.Linear(D, cfg.vocab_size, bias=False)
        # Weight tying: embedding matrix = LM head weight (saves params, improves quality)
        self.lm_head.weight = self.tok_embed.weight

        self._init_weights()
        logger.info(
            "LanguageDecoder: %d layers, dim=%d, heads=%d, vocab=%d, max_len=%d",
            cfg.lang_num_layers, D, cfg.lang_num_heads, cfg.vocab_size, cfg.max_seq_len,
        )

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        vision_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids:    (B, T) long tensor of token ids.
            vision_tokens:(B, N, lang_hidden_dim) projected vision context.
        Returns:
            logits: (B, T, vocab_size)
        """
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)   # (1, T)
        x = self.drop(self.tok_embed(input_ids) + self.pos_embed(pos))

        for block in self.blocks:
            x = block(x, vision_tokens)

        x = self.norm(x)
        return self.lm_head(x)
