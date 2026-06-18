"""RL/SFT Trainer for SmallVLM.

Two modes (set via configs/rl_sft.yaml → rl_sft.mode):

SFT mode:
  Standard supervised fine-tuning using cross-entropy loss on high-quality
  caption annotations. Identical in structure to finetune.py but with its
  own config section and output directory.

RL mode (REINFORCE):
  For each image in the batch:
    1. Sample K captions from the model at temperature T (rollouts).
    2. Score each caption with a reward metric (BLEU-4 or CIDEr).
    3. Compute advantage = reward - baseline (running mean).
    4. Policy gradient loss: -log_prob * advantage
    5. KL penalty against a frozen reference model copy.
    6. Entropy bonus to prevent collapse.

Usage (2 GPUs):
  torchrun --nproc_per_node=2 -m src.training.rl_sft_trainer
"""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler

from src.data.collator import ImageCaptionCollator
from src.data.dataset import ImageCaptionDataset, build_tokenizer
from src.data.transforms import train_transform, val_transform
from src.inference.generator import CaptionGenerator
from src.model.config import ModelConfig
from src.model.vlm import SmallVLM
from src.training.trainer import BaseTrainer
from src.utils.config_utils import get_section, load_yaml
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_CONFIG_PATH = "configs/rl_sft.yaml"
_MODEL_CONFIG_PATH = "configs/model.yaml"


# ---------------------------------------------------------------------------
# Reward metrics
# ---------------------------------------------------------------------------

def _bleu4_reward(hypotheses: list[str], references: list[list[str]]) -> list[float]:
    """Compute sentence-level BLEU-4 for each hypothesis."""
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu  # type: ignore

    sf = SmoothingFunction().method1
    rewards = []
    for hyp, refs in zip(hypotheses, references):
        hyp_tokens = hyp.split()
        ref_tokens = [r.split() for r in refs]
        score = sentence_bleu(ref_tokens, hyp_tokens, smoothing_function=sf)
        rewards.append(score)
    return rewards


def _cider_reward(hypotheses: list[str], references: list[list[str]]) -> list[float]:
    """Approximate CIDEr using pycocoevalcap (falls back to BLEU-4 if unavailable)."""
    try:
        from pycocoevalcap.cider.cider import Cider  # type: ignore

        gts = {i: refs for i, refs in enumerate(references)}
        res = {i: [hyp] for i, hyp in enumerate(hypotheses)}
        scorer = Cider()
        _, scores = scorer.compute_score(gts, res)
        return list(scores)
    except ImportError:
        logger.warning("pycocoevalcap not installed; falling back to BLEU-4 reward")
        return _bleu4_reward(hypotheses, references)


def compute_reward(
    hypotheses: list[str],
    references: list[list[str]],
    metric: str,
) -> list[float]:
    if metric == "cider":
        return _cider_reward(hypotheses, references)
    return _bleu4_reward(hypotheses, references)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class RLSFTTrainer(BaseTrainer):
    """Unified SFT + REINFORCE trainer for SmallVLM."""

    def __init__(self, model: SmallVLM, cfg: dict[str, Any]) -> None:
        self.mode = cfg["mode"]
        self.model_cfg = model.cfg
        # Active sub-config (sft or rl section merged with top-level keys)
        sub_cfg = {**cfg, **cfg.get(self.mode, {})}
        super().__init__(model, sub_cfg)
        self.tokenizer = build_tokenizer()

        if self.mode == "rl":
            # Frozen reference model for KL penalty
            ref_model = copy.deepcopy(model).to(self.device)
            for p in ref_model.parameters():
                p.requires_grad_(False)
            self.ref_model = ref_model
            self._reward_baseline = 0.0
            self._baseline_momentum = 0.99
            logger.info("RL mode: reference model frozen")

    def build_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        cfg = self.cfg
        image_size = self.model_cfg.image_size
        sub_cfg = cfg.get(self.mode, cfg)
        bsz = sub_cfg.get("batch_size_per_gpu", 8)

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
            batch_size=bsz,
            sampler=DistributedSampler(train_ds, shuffle=True),
            num_workers=4,
            pin_memory=True,
            collate_fn=collator,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=bsz,
            sampler=DistributedSampler(val_ds, shuffle=False),
            num_workers=4,
            collate_fn=collator,
        )
        return train_loader, val_loader

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.mode == "sft":
            return self._sft_loss(batch)
        return self._rl_loss(batch)

    # ---- SFT ----

    def _sft_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        out = self.model(
            images=batch["images"],
            caption_ids=batch["caption_ids"],
            attention_mask=batch["attention_mask"],
        )
        return out["loss"]

    # ---- RL (REINFORCE) ----

    def _rl_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        rl_cfg = self.cfg.get("rl", self.cfg)
        images = batch["images"]                  # (B, C, H, W)
        ref_captions_ids = batch["caption_ids"]   # (B, T) — ground-truth for reward

        K = rl_cfg.get("num_rollouts_per_sample", 4)
        temperature = rl_cfg.get("temperature", 1.0)
        top_p = rl_cfg.get("top_p", 0.9)
        kl_coeff = rl_cfg.get("kl_coeff", 0.01)
        entropy_coeff = rl_cfg.get("entropy_coeff", 0.001)
        reward_metric = rl_cfg.get("reward_metric", "cider")

        B = images.size(0)
        generator = CaptionGenerator(self.model.module, self.model_cfg, self.device)

        # Decode reference captions for reward scoring
        ref_texts = [
            self.tokenizer.decode(ref_captions_ids[i], skip_special_tokens=True)
            for i in range(B)
        ]

        total_pg_loss = torch.tensor(0.0, device=self.device)
        total_kl_loss = torch.tensor(0.0, device=self.device)
        total_entropy = torch.tensor(0.0, device=self.device)
        n_rollouts = 0

        for _ in range(K):
            # Sample K captions per image (nucleus sampling)
            sampled_ids = generator.generate_nucleus(
                images, temperature=temperature, top_p=top_p,
                max_new_tokens=self.model_cfg.max_seq_len - 1,
            )   # (B, T_gen)

            hyp_texts = [
                self.tokenizer.decode(sampled_ids[i], skip_special_tokens=True)
                for i in range(B)
            ]
            rewards = compute_reward(hyp_texts, [[r] for r in ref_texts], reward_metric)
            reward_tensor = torch.tensor(rewards, device=self.device, dtype=torch.float32)

            # Running-mean baseline
            batch_mean = reward_tensor.mean().item()
            self._reward_baseline = (
                self._baseline_momentum * self._reward_baseline
                + (1 - self._baseline_momentum) * batch_mean
            )
            advantage = (reward_tensor - self._reward_baseline).detach()

            # Compute policy log-probs for the sampled sequence
            proj_tokens = self.model.module.encode_image(images)
            logits = self.model.module.language_decoder(sampled_ids[:, :-1], proj_tokens)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs.gather(2, sampled_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            seq_log_prob = selected.mean(dim=-1)   # (B,)

            pg_loss = -(advantage * seq_log_prob).mean()
            total_pg_loss = total_pg_loss + pg_loss

            # KL penalty vs. frozen reference model
            with torch.no_grad():
                ref_proj = self.ref_model.encode_image(images)
                ref_logits = self.ref_model.language_decoder(sampled_ids[:, :-1], ref_proj)
                ref_log_probs = F.log_softmax(ref_logits, dim=-1)
            kl = (log_probs.exp() * (log_probs - ref_log_probs)).sum(-1).mean()
            total_kl_loss = total_kl_loss + kl

            # Entropy bonus
            entropy = -(log_probs.exp() * log_probs).sum(-1).mean()
            total_entropy = total_entropy + entropy

            n_rollouts += 1

        loss = (
            total_pg_loss / n_rollouts
            + kl_coeff * total_kl_loss / n_rollouts
            - entropy_coeff * total_entropy / n_rollouts
        )
        logger.debug("RL loss=%.4f  baseline=%.4f", loss.item(), self._reward_baseline)
        return loss


def main() -> None:
    raw_cfg = load_yaml(_CONFIG_PATH)
    cfg = get_section(raw_cfg, "rl_sft")

    logger.info("Starting RL/SFT training in mode='%s'", cfg["mode"])

    model_cfg = ModelConfig.from_yaml(_MODEL_CONFIG_PATH)
    model = SmallVLM(model_cfg)

    pretrained = cfg.get("pretrained_checkpoint")
    if pretrained:
        state = torch.load(pretrained, map_location="cpu")
        model.load_state_dict(state["model_state_dict"], strict=False)
        logger.info("Loaded pretrained checkpoint: %s", pretrained)

    trainer = RLSFTTrainer(model, cfg)
    trainer.train()


if __name__ == "__main__":
    main()
