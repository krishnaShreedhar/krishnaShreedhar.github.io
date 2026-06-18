"""Vision-to-Language Projection.

Maps vision encoder output (vision_hidden_dim) into the language decoder
embedding space (lang_hidden_dim) via a two-layer MLP with GELU activation.

  patch_tokens (B, N, vision_dim)
      → Linear(vision_dim, projection_hidden_dim) → GELU
      → Linear(projection_hidden_dim, lang_dim)
  → projected_tokens (B, N, lang_dim)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.config import ModelConfig
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class VisionProjection(nn.Module):
    """Two-layer MLP connector between vision encoder and language decoder."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(cfg.vision_hidden_dim, cfg.projection_hidden_dim)
        self.fc2 = nn.Linear(cfg.projection_hidden_dim, cfg.lang_hidden_dim)
        self._init_weights()
        logger.info(
            "VisionProjection: %d → %d → %d",
            cfg.vision_hidden_dim, cfg.projection_hidden_dim, cfg.lang_hidden_dim,
        )

    def _init_weights(self) -> None:
        for m in [self.fc1, self.fc2]:
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_tokens: (B, N, vision_hidden_dim)
        Returns:
            projected: (B, N, lang_hidden_dim)
        """
        x = F.gelu(self.fc1(patch_tokens))
        return self.fc2(x)
