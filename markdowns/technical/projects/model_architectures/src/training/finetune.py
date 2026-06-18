"""Fine-tuning entry point.

Loads a pre-trained SmallVLM checkpoint and fine-tunes on a downstream
captioning dataset (COCO Captions by default). Supports per-module LR scaling
so the vision encoder can train at a lower rate than the decoder.

Usage (2 GPUs):
  torchrun --nproc_per_node=2 -m src.training.finetune
"""

from __future__ import annotations

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler

from src.data.collator import ImageCaptionCollator
from src.data.dataset import ImageCaptionDataset, build_tokenizer
from src.data.transforms import train_transform, val_transform
from src.model.config import ModelConfig
from src.model.vlm import SmallVLM
from src.training.trainer import BaseTrainer, _cosine_schedule_with_warmup
from src.utils.config_utils import get_section, load_yaml
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_CONFIG_PATH = "configs/finetune.yaml"
_MODEL_CONFIG_PATH = "configs/model.yaml"


class FineTuner(BaseTrainer):
    """Fine-tunes a pre-trained SmallVLM checkpoint."""

    def __init__(self, model: SmallVLM, cfg: dict) -> None:
        self.model_cfg = model.cfg
        super().__init__(model, cfg)
        self.tokenizer = build_tokenizer()

    def _build_optimizer(self) -> AdamW:
        """Build optimizer with a smaller LR for the vision encoder."""
        cfg = self.cfg
        base_lr = cfg["learning_rate"]
        vision_lr = base_lr * cfg.get("vision_encoder_lr_scale", 0.1)

        vision_params = list(self.model.module.vision_encoder.parameters())
        other_params = [
            p for n, p in self.model.named_parameters()
            if p not in set(vision_params) and p.requires_grad
        ]
        param_groups = [
            {"params": vision_params, "lr": vision_lr,
             "weight_decay": cfg.get("weight_decay", 0.01)},
            {"params": other_params, "lr": base_lr,
             "weight_decay": cfg.get("weight_decay", 0.01)},
        ]
        logger.info("FineTuner optimizer: base_lr=%.2e, vision_lr=%.2e", base_lr, vision_lr)
        return AdamW(param_groups, betas=(cfg.get("beta1", 0.9), cfg.get("beta2", 0.999)),
                     eps=cfg.get("eps", 1e-8))

    def _load_pretrained(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        missing, unexpected = self.model.module.load_state_dict(
            state["model_state_dict"], strict=False
        )
        logger.info("Loaded pretrained checkpoint: %s", path)
        if missing:
            logger.warning("Missing keys: %s", missing)
        if unexpected:
            logger.warning("Unexpected keys: %s", unexpected)

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
        )
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
    cfg = get_section(raw_cfg, "finetune")

    logger.info("Starting fine-tuning")

    model_cfg = ModelConfig.from_yaml(_MODEL_CONFIG_PATH)
    model = SmallVLM(model_cfg)

    trainer = FineTuner(model, cfg)

    pretrained = cfg.get("pretrained_checkpoint")
    if pretrained:
        trainer._load_pretrained(pretrained)
    else:
        logger.warning("No pretrained_checkpoint specified; fine-tuning from scratch")

    trainer.train()


if __name__ == "__main__":
    main()
