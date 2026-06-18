"""Vision Encoder: a minimal ViT that converts an image into a sequence of patch embeddings.

Pipeline:
  Image (B, C, H, W)
    → PatchEmbedding  → (B, N, D)   N = (H/P)*(W/P) patch tokens
    → + cls_token     → (B, N+1, D)
    → + pos_embed     → (B, N+1, D)
    → Dropout
    → N × ViTBlock    → (B, N+1, D)
    → LayerNorm
  output:
    patch_tokens  (B, N, D)  — vision tokens fed to the projection layer
    cls_token     (B, D)     — optional global image representation
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.config import ModelConfig
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class PatchEmbedding(nn.Module):
    """Split an image into non-overlapping patches and linearly project each patch."""

    def __init__(self, image_size: int, patch_size: int, in_channels: int, hidden_dim: int) -> None:
        super().__init__()
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        self.num_patches = (image_size // patch_size) ** 2
        # A single Conv2d with stride=patch_size extracts and flattens patches in one step
        self.proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)
        logger.debug("PatchEmbedding: %d patches of dim %d", self.num_patches, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) → (B, D, H/P, W/P) → (B, N, D)
        x = self.proj(x)               # (B, D, grid, grid)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x


class MultiHeadSelfAttention(nn.Module):
    """Standard scaled dot-product multi-head self-attention."""

    def __init__(self, hidden_dim: int, num_heads: int, attn_dropout: float = 0.0) -> None:
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim, bias=True)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_drop = nn.Dropout(attn_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)   # (3, B, H, N, head_dim)
        q, k, v = qkv.unbind(0)             # each (B, H, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale   # (B, H, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, D)  # (B, N, D)
        return self.proj(x)


class MLP(nn.Module):
    """Position-wise feed-forward network used inside Transformer blocks."""

    def __init__(self, hidden_dim: int, mlp_ratio: float, dropout: float = 0.0) -> None:
        super().__init__()
        inner_dim = int(hidden_dim * mlp_ratio)
        self.fc1 = nn.Linear(hidden_dim, inner_dim)
        self.fc2 = nn.Linear(inner_dim, hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ViTBlock(nn.Module):
    """Single ViT Transformer block: LayerNorm → MHSA → residual → LayerNorm → MLP → residual."""

    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float,
                 dropout: float, attn_dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = MultiHeadSelfAttention(hidden_dim, num_heads, attn_dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionEncoder(nn.Module):
    """ViT-based image encoder.

    Returns:
        patch_tokens: (B, N, vision_hidden_dim) — one token per image patch.
        cls_token:    (B, vision_hidden_dim)     — aggregated [CLS] representation.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        D = cfg.vision_hidden_dim
        N = cfg.num_vision_tokens

        self.patch_embed = PatchEmbedding(
            cfg.image_size, cfg.patch_size, cfg.in_channels, D
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, D))
        # +1 for the cls_token position
        self.pos_embed = nn.Parameter(torch.zeros(1, N + 1, D))
        self.pos_drop = nn.Dropout(cfg.vision_dropout)

        self.blocks = nn.ModuleList([
            ViTBlock(D, cfg.vision_num_heads, cfg.vision_mlp_ratio,
                     cfg.vision_dropout, cfg.vision_attn_dropout)
            for _ in range(cfg.vision_num_layers)
        ])
        self.norm = nn.LayerNorm(D)

        self._init_weights()
        logger.info(
            "VisionEncoder: %d layers, dim=%d, heads=%d, patches=%d",
            cfg.vision_num_layers, D, cfg.vision_num_heads, N,
        )

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            images: (B, C, H, W) float tensor in [0, 1] (or normalised).
        Returns:
            patch_tokens: (B, N, D)
            cls_token:    (B, D)
        """
        B = images.size(0)
        x = self.patch_embed(images)                    # (B, N, D)
        cls = self.cls_token.expand(B, -1, -1)          # (B, 1, D)
        x = torch.cat([cls, x], dim=1)                  # (B, N+1, D)
        x = self.pos_drop(x + self.pos_embed)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        cls_out = x[:, 0]       # (B, D)
        patch_out = x[:, 1:]    # (B, N, D)
        return patch_out, cls_out
