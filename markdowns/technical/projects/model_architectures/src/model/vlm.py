"""SmallVLM: top-level Vision-Language Model for image captioning.

Forward pass:
  image       (B, C, H, W)
  caption_ids (B, T)                   ← teacher-forced input during training
    ↓
  VisionEncoder   → patch_tokens (B, N, vision_dim)
  VisionProjection → proj_tokens (B, N, lang_dim)
  LanguageDecoder(caption_ids, proj_tokens) → logits (B, T, vocab)

Loss:
  Cross-entropy between logits[:, :-1] and caption_ids[:, 1:]
  (standard next-token prediction, ignoring pad tokens).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.config import ModelConfig
from src.model.language_decoder import LanguageDecoder
from src.model.projection import VisionProjection
from src.model.vision_encoder import VisionEncoder
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class SmallVLM(nn.Module):
    """Minimal Vision-Language Model for image captioning."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.vision_encoder = VisionEncoder(cfg)
        self.projection = VisionProjection(cfg)
        self.language_decoder = LanguageDecoder(cfg)
        logger.info("SmallVLM initialised. Total params: %s", self._param_count())

    def _param_count(self) -> str:
        total = sum(p.numel() for p in self.parameters())
        return f"{total / 1e6:.1f}M"

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to projected language-space tokens.

        Args:
            images: (B, C, H, W)
        Returns:
            proj_tokens: (B, N, lang_hidden_dim)
        """
        patch_tokens, _ = self.vision_encoder(images)
        return self.projection(patch_tokens)

    def forward(
        self,
        images: torch.Tensor,
        caption_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Teacher-forced forward pass for training.

        Args:
            images:         (B, C, H, W)
            caption_ids:    (B, T) including BOS at position 0
            attention_mask: (B, T) 1 for real tokens, 0 for padding (optional)
        Returns:
            dict with keys:
              "loss"   — scalar cross-entropy loss
              "logits" — (B, T-1, vocab_size) raw next-token predictions
        """
        proj_tokens = self.encode_image(images)           # (B, N, lang_dim)

        # Decoder input: all tokens except the last one
        decoder_input = caption_ids[:, :-1]               # (B, T-1)
        # Target: all tokens except the first (BOS)
        targets = caption_ids[:, 1:]                      # (B, T-1)

        logits = self.language_decoder(decoder_input, proj_tokens)  # (B, T-1, vocab)

        # Build ignore mask: pad positions should not contribute to loss
        ignore_index = self.cfg.pad_token_id
        if attention_mask is not None:
            # Shift mask to align with targets
            target_mask = attention_mask[:, 1:]            # (B, T-1)
            targets = targets.masked_fill(target_mask == 0, -100)

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-100,
        )

        logger.debug("Forward pass: loss=%.4f", loss.item())
        return {"loss": loss, "logits": logits}

    @classmethod
    def from_config_yaml(cls, path: str) -> "SmallVLM":
        cfg = ModelConfig.from_yaml(path)
        return cls(cfg)
