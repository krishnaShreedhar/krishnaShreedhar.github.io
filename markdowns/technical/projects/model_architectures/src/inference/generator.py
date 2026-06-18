"""CaptionGenerator: inference-time caption generation for SmallVLM.

Three decoding strategies:
  greedy       — pick argmax at each step
  beam_search  — keep the top-K partial sequences at each step
  nucleus      — top-p / top-k sampling with temperature

Usage (from project root):
  python -m src.inference.generator
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from src.model.config import ModelConfig
from src.model.vlm import SmallVLM
from src.utils.config_utils import get_section, load_yaml
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_CONFIG_PATH = "configs/inference.yaml"
_MODEL_CONFIG_PATH = "configs/model.yaml"


class CaptionGenerator:
    """Wraps a trained SmallVLM and exposes captioning methods."""

    def __init__(self, model: SmallVLM, cfg: ModelConfig, device: torch.device) -> None:
        self.model = model
        self.cfg = cfg
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def generate_greedy(self, images: torch.Tensor, max_new_tokens: int = 64) -> torch.Tensor:
        """Greedy decoding: always pick the highest-probability next token.

        Args:
            images: (B, C, H, W)
            max_new_tokens: Maximum number of tokens to generate.
        Returns:
            generated_ids: (B, T_gen) long tensor
        """
        B = images.size(0)
        proj_tokens = self.model.encode_image(images)   # (B, N, lang_dim)
        bos = self.cfg.bos_token_id
        eos = self.cfg.eos_token_id

        input_ids = torch.full((B, 1), bos, dtype=torch.long, device=self.device)
        finished = torch.zeros(B, dtype=torch.bool, device=self.device)

        for _ in range(max_new_tokens):
            logits = self.model.language_decoder(input_ids, proj_tokens)  # (B, T, vocab)
            next_token_logits = logits[:, -1, :]                           # (B, vocab)
            next_tokens = next_token_logits.argmax(dim=-1, keepdim=True)  # (B, 1)

            finished = finished | (next_tokens.squeeze(-1) == eos)
            input_ids = torch.cat([input_ids, next_tokens], dim=1)

            if finished.all():
                break

        return input_ids[:, 1:]   # Strip leading BOS

    @torch.no_grad()
    def generate_beam_search(
        self,
        images: torch.Tensor,
        num_beams: int = 5,
        max_new_tokens: int = 64,
        length_penalty: float = 1.0,
        no_repeat_ngram_size: int = 3,
    ) -> torch.Tensor:
        """Beam-search decoding.  Processes one image at a time for clarity.

        Args:
            images: (B, C, H, W) — processed sequentially (batched beams per image)
        Returns:
            generated_ids: (B, T_best)  — best beam for each image, zero-padded
        """
        B = images.size(0)
        results: list[torch.Tensor] = []

        for b in range(B):
            image_b = images[b: b + 1]                       # (1, C, H, W)
            proj = self.model.encode_image(image_b)           # (1, N, D)
            proj_expanded = proj.expand(num_beams, -1, -1)   # (K, N, D)

            bos = self.cfg.bos_token_id
            eos = self.cfg.eos_token_id

            # beams: list of (score, token_ids_list)
            beams = [(0.0, [bos])]
            completed = []

            for _ in range(max_new_tokens):
                candidates = []
                for score, ids in beams:
                    if ids[-1] == eos:
                        completed.append((score, ids))
                        continue
                    id_tensor = torch.tensor([ids], dtype=torch.long, device=self.device)
                    logits = self.model.language_decoder(id_tensor, proj[:1])  # (1, T, vocab)
                    log_probs = F.log_softmax(logits[0, -1], dim=-1)  # (vocab,)

                    # Apply no-repeat n-gram constraint
                    if no_repeat_ngram_size > 1:
                        log_probs = self._block_ngrams(
                            log_probs, ids, no_repeat_ngram_size
                        )

                    topk_lp, topk_ids = log_probs.topk(num_beams)
                    for lp, tok in zip(topk_lp.tolist(), topk_ids.tolist()):
                        candidates.append((score + lp, ids + [tok]))

                if not candidates:
                    break

                # Keep top num_beams by length-penalised score
                def penalised(item: tuple) -> float:
                    sc, ids = item
                    return sc / (len(ids) ** length_penalty)

                candidates.sort(key=penalised, reverse=True)
                beams = candidates[:num_beams]

                if all(ids[-1] == eos for _, ids in beams):
                    completed.extend(beams)
                    break

            completed.extend(beams)
            best_ids = max(completed, key=lambda x: x[0] / (len(x[1]) ** length_penalty))[1]
            # Strip BOS/EOS
            best_ids = [t for t in best_ids if t not in (bos, eos)]
            results.append(torch.tensor(best_ids, dtype=torch.long, device=self.device))

        # Pad results to same length
        max_len = max(r.size(0) for r in results) if results else 1
        padded = torch.zeros(B, max_len, dtype=torch.long, device=self.device)
        for i, r in enumerate(results):
            padded[i, : r.size(0)] = r
        return padded

    def _block_ngrams(
        self, log_probs: torch.Tensor, ids: list[int], n: int
    ) -> torch.Tensor:
        """Set log-prob to -inf for tokens that would create a repeated n-gram."""
        if len(ids) < n - 1:
            return log_probs
        context = tuple(ids[-(n - 1):])
        banned = set()
        for i in range(len(ids) - n + 1):
            if tuple(ids[i: i + n - 1]) == context:
                banned.add(ids[i + n - 1])
        for tok in banned:
            log_probs[tok] = float("-inf")
        return log_probs

    @torch.no_grad()
    def generate_nucleus(
        self,
        images: torch.Tensor,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 50,
        max_new_tokens: int = 64,
    ) -> torch.Tensor:
        """Top-p (nucleus) sampling with optional top-k filtering.

        Args:
            images: (B, C, H, W)
        Returns:
            generated_ids: (B, T_gen)
        """
        B = images.size(0)
        proj_tokens = self.model.encode_image(images)
        bos = self.cfg.bos_token_id
        eos = self.cfg.eos_token_id

        input_ids = torch.full((B, 1), bos, dtype=torch.long, device=self.device)
        finished = torch.zeros(B, dtype=torch.bool, device=self.device)

        for _ in range(max_new_tokens):
            logits = self.model.language_decoder(input_ids, proj_tokens)[:, -1, :]  # (B, vocab)
            logits = logits / temperature

            # top-k filtering
            if top_k > 0:
                topk_vals = logits.topk(top_k, dim=-1).values
                logits = logits.masked_fill(logits < topk_vals[:, -1:], float("-inf"))

            # top-p filtering
            sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
            cum_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            remove_mask = cum_probs - sorted_logits.softmax(dim=-1) > top_p
            sorted_logits[remove_mask] = float("-inf")
            logits = logits.scatter(1, sorted_idx, sorted_logits)

            probs = logits.softmax(dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)   # (B, 1)
            finished = finished | (next_tokens.squeeze(-1) == eos)
            input_ids = torch.cat([input_ids, next_tokens], dim=1)

            if finished.all():
                break

        return input_ids[:, 1:]


def main() -> None:
    """Run batch inference on a directory of images and save captions to JSON."""
    import json
    from pathlib import Path

    from PIL import Image
    from transformers import GPT2Tokenizer

    from src.data.dataset import build_tokenizer
    from src.data.transforms import val_transform

    raw_cfg = load_yaml(_CONFIG_PATH)
    cfg = get_section(raw_cfg, "inference")
    logger.info("Starting inference")

    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    model_cfg = ModelConfig.from_yaml(_MODEL_CONFIG_PATH)
    model = SmallVLM(model_cfg)

    ckpt = torch.load(cfg["checkpoint_path"], map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    logger.info("Model loaded from %s", cfg["checkpoint_path"])

    tokenizer = build_tokenizer()
    transform = val_transform(model_cfg.image_size)
    generator = CaptionGenerator(model, model_cfg, device)
    strategy = cfg.get("decoding_strategy", "beam_search")

    image_dir = cfg.get("image_dir")
    if image_dir is None and cfg.get("image_path") is not None:
        image_paths = [Path(cfg["image_path"])]
    else:
        image_paths = sorted(Path(image_dir).glob("*.jpg")) + sorted(Path(image_dir).glob("*.png"))

    batch_size = cfg.get("batch_size", 16)
    output_file = Path(cfg["output_file"])
    output_file.parent.mkdir(parents=True, exist_ok=True)

    results = {}
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i: i + batch_size]
        images = torch.stack([
            transform(Image.open(p).convert("RGB")) for p in batch_paths
        ]).to(device)

        if strategy == "greedy":
            s_cfg = cfg.get("greedy", {})
            ids = generator.generate_greedy(images, **s_cfg)
        elif strategy == "nucleus":
            s_cfg = cfg.get("nucleus", {})
            ids = generator.generate_nucleus(images, **s_cfg)
        else:
            s_cfg = cfg.get("beam_search", {})
            ids = generator.generate_beam_search(images, **s_cfg)

        for path, caption_ids in zip(batch_paths, ids):
            caption = tokenizer.decode(caption_ids[caption_ids != 0], skip_special_tokens=True)
            results[path.name] = caption
            logger.info("%s → %s", path.name, caption)

    with output_file.open("w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved %d captions to %s", len(results), output_file)


if __name__ == "__main__":
    main()
