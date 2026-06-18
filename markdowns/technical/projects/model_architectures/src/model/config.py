"""VLM model configuration dataclass loaded from configs/model.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.utils.config_utils import get_section, load_yaml
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class ModelConfig:
    # Vision Encoder
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    vision_hidden_dim: int = 384
    vision_num_heads: int = 6
    vision_num_layers: int = 6
    vision_mlp_ratio: float = 4.0
    vision_dropout: float = 0.0
    vision_attn_dropout: float = 0.0

    # Projection MLP
    projection_hidden_dim: int = 768

    # Language Decoder
    vocab_size: int = 50257
    max_seq_len: int = 128
    lang_hidden_dim: int = 384
    lang_num_heads: int = 6
    lang_num_layers: int = 6
    lang_mlp_ratio: float = 4.0
    lang_dropout: float = 0.1
    lang_attn_dropout: float = 0.0
    use_cross_attention: bool = True

    # Special tokens
    bos_token_id: int = 50256
    eos_token_id: int = 50256
    pad_token_id: int = 50256

    # Derived
    num_vision_tokens: int = field(init=False)

    def __post_init__(self) -> None:
        self.num_vision_tokens = (self.image_size // self.patch_size) ** 2
        logger.debug(
            "ModelConfig: vision_tokens=%d, vision_dim=%d, lang_dim=%d",
            self.num_vision_tokens, self.vision_hidden_dim, self.lang_hidden_dim,
        )

    @classmethod
    def from_yaml(cls, path: str) -> "ModelConfig":
        raw = load_yaml(path)
        section = get_section(raw, "model")
        section.pop("num_vision_tokens", None)  # Always derived
        logger.info("Building ModelConfig from %s", path)
        return cls(**section)

    def override(self, overrides: dict[str, Any]) -> "ModelConfig":
        """Return a new ModelConfig with specified fields overridden."""
        import dataclasses
        d = dataclasses.asdict(self)
        d.pop("num_vision_tokens")
        d.update(overrides)
        return ModelConfig(**d)
