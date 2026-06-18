"""Pre-training entry point.

Runs next-token-prediction on image-caption pairs (Conceptual Captions by default).

Usage (from project root, 2 GPUs):
  torchrun --nproc_per_node=2 -m src.training.pretrain
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, DistributedSampler

from src.data.collator import ImageCaptionCollator
from src.data.dataset import ImageCaptionDataset, build_tokenizer
from src.data.transforms import train_transform, val_transform
from src.model.config import ModelConfig
from src.model.vlm import SmallVLM
from src.training.trainer import BaseTrainer
from src.utils.config_utils import get_section, load_yaml
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_CONFIG_PATH = "configs/pretrain.yaml"
_MODEL_CONFIG_PATH = "configs/model.yaml"


class PreTrainer(BaseTrainer):
    """Pre-trains SmallVLM on image-caption pairs with next-token prediction."""

    def __init__(self, model: SmallVLM, cfg: dict) -> None:
        super().__init__(model, cfg)
        self.tokenizer = build_tokenizer()
        self.model_cfg = model.cfg

    def build_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        cfg = self.cfg
        image_size = self.model_cfg.image_size

        train_ds = ImageCaptionDataset(
            dataset_name=cfg["dataset_name"],
            split=cfg["train_split"],
            tokenizer=self.tokenizer,
            transform=train_transform(image_size),
            max_seq_len=self.model_cfg.max_seq_len,
            max_samples=cfg.get("max_train_samples"),
            cache_dir=cfg.get("data_root"),
        )
        val_ds = ImageCaptionDataset(
            dataset_name=cfg["dataset_name"],
            split=cfg["val_split"],
            tokenizer=self.tokenizer,
            transform=val_transform(image_size),
            max_seq_len=self.model_cfg.max_seq_len,
            max_samples=cfg.get("max_val_samples"),
            cache_dir=cfg.get("data_root"),
        )

        collator = ImageCaptionCollator()
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg["batch_size_per_gpu"],
            sampler=DistributedSampler(train_ds, shuffle=True),
            num_workers=cfg.get("num_workers", 8),
            pin_memory=cfg.get("pin_memory", True),
            collate_fn=collator,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg["batch_size_per_gpu"],
            sampler=DistributedSampler(val_ds, shuffle=False),
            num_workers=cfg.get("num_workers", 8),
            pin_memory=cfg.get("pin_memory", True),
            collate_fn=collator,
            drop_last=False,
        )
        logger.info("Train batches: %d | Val batches: %d", len(train_loader), len(val_loader))
        return train_loader, val_loader

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        out = self.model(
            images=batch["images"],
            caption_ids=batch["caption_ids"],
            attention_mask=batch["attention_mask"],
        )
        return out["loss"]


def main() -> None:
    raw_cfg = load_yaml(_CONFIG_PATH)
    cfg = get_section(raw_cfg, "pretrain")

    logger.setLevel(cfg.get("log_level", "INFO"))
    logger.info("Starting pre-training")

    model_cfg = ModelConfig.from_yaml(_MODEL_CONFIG_PATH)
    model = SmallVLM(model_cfg)

    trainer = PreTrainer(model, cfg)
    trainer.train()


if __name__ == "__main__":
    main()
