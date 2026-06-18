"""
KV-Cache Inference Engine — Tutorial Module 4c

Demonstrates:
  - Two-phase generation: prefill (process prompt) + decode (one token at a time)
  - How the KV-cache eliminates redundant K/V computation during decode
  - Throughput comparison: cached vs. naive recomputation
  - Memory cost of the cache vs. saved compute

Run: python -m src.kv_cache.inference_engine
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.logging_utils import setup_logger
from src.kv_cache.kv_cache_manager import KVCacheManager
from src.kv_cache.attention import MultiHeadAttentionCached, MultiHeadAttentionNaive

CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "kv_cache.yaml"


# ---------------------------------------------------------------------------
# Minimal decoder-only Transformer with KV-cache support
# ---------------------------------------------------------------------------

class CachedDecoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, layer_idx: int, cache: KVCacheManager) -> None:
        super().__init__()
        self.attn = MultiHeadAttentionCached(d_model, num_heads, layer_idx, cache)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), start_pos)
        x = x + self.ff(self.norm2(x))
        return x


class CachedDecoderLM(nn.Module):
    def __init__(self, config: dict[str, Any], cache: KVCacheManager, device: torch.device) -> None:
        super().__init__()
        m = config["model"]
        self.token_emb = nn.Embedding(m["vocab_size"], m["d_model"])
        self.pos_emb = nn.Embedding(m["max_seq_len"], m["d_model"])
        self.blocks = nn.ModuleList([
            CachedDecoderBlock(m["d_model"], m["num_heads"], m["d_ff"], i, cache)
            for i in range(m["num_layers"])
        ])
        self.norm = nn.LayerNorm(m["d_model"])
        self.head = nn.Linear(m["d_model"], m["vocab_size"], bias=False)
        self.head.weight = self.token_emb.weight

    def forward(self, tokens: torch.Tensor, start_pos: int) -> torch.Tensor:
        B, T = tokens.shape
        positions = torch.arange(start_pos, start_pos + T, device=tokens.device).unsqueeze(0)
        x = self.token_emb(tokens) + self.pos_emb(positions)
        for block in self.blocks:
            x = block(x, start_pos)
        return self.head(self.norm(x))  # (B, T, vocab_size)


class NaiveDecoderLM(nn.Module):
    """Naive model that recomputes K/V from full history every step."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        m = config["model"]
        self.token_emb = nn.Embedding(m["vocab_size"], m["d_model"])
        self.pos_emb = nn.Embedding(m["max_seq_len"], m["d_model"])
        self.blocks = nn.ModuleList([
            _NaiveBlock(m["d_model"], m["num_heads"], m["d_ff"])
            for _ in range(m["num_layers"])
        ])
        self.norm = nn.LayerNorm(m["d_model"])
        self.head = nn.Linear(m["d_model"], m["vocab_size"], bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, T = tokens.shape
        positions = torch.arange(T, device=tokens.device).unsqueeze(0)
        x = self.token_emb(tokens) + self.pos_emb(positions)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


class _NaiveBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int) -> None:
        super().__init__()
        self.attn = MultiHeadAttentionNaive(d_model, num_heads)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Inference engine
# ---------------------------------------------------------------------------

class InferenceEngine:
    def __init__(self, config: dict[str, Any]) -> None:
        self._cfg = config
        self._log = setup_logger("kv_cache.inference_engine", config)
        self._device = torch.device(config["gpu"]["device"])
        self._temperature = config["inference"]["temperature"]

    def _sample(self, logits: torch.Tensor) -> torch.Tensor:
        """Greedy sampling (temperature=1 → multinomial, else argmax)."""
        if self._temperature == 1.0:
            return torch.argmax(logits[:, -1], dim=-1, keepdim=True)
        return torch.argmax((logits[:, -1] / self._temperature), dim=-1, keepdim=True)

    def run_cached(self) -> float:
        """Prefill + decode with KV-cache. Returns tokens/sec."""
        inf_cfg = self._cfg["inference"]
        batch_size = self._cfg["kv_cache"]["max_batch_size"]
        prompt_len = inf_cfg["prompt_tokens"]
        max_new = inf_cfg["max_new_tokens"]
        iters = inf_cfg["benchmark_iters"]

        cache = KVCacheManager.from_config(self._cfg, self._device)
        self._log.info(
            "KV-cache allocated: %.1f MB", cache.memory_bytes() / 1e6
        )

        model = CachedDecoderLM(self._cfg, cache, self._device).to(self._device)
        model.eval()

        prompt = torch.randint(
            0, self._cfg["model"]["vocab_size"], (batch_size, prompt_len), device=self._device
        )

        elapsed_total = 0.0
        total_tokens = 0

        with torch.no_grad():
            for _ in range(iters):
                cache.reset()
                t0 = time.perf_counter()

                # Prefill: process entire prompt at once
                _ = model(prompt, start_pos=0)
                tokens = prompt[:, -1:]   # last token → first decode input
                start_pos = prompt_len

                # Decode: one token at a time
                generated: list[torch.Tensor] = []
                for _ in range(max_new):
                    logits = model(tokens, start_pos=start_pos)
                    tokens = self._sample(logits)
                    generated.append(tokens)
                    start_pos += 1

                torch.cuda.synchronize(self._device)
                elapsed_total += time.perf_counter() - t0
                total_tokens += batch_size * max_new

        tok_per_s = total_tokens / elapsed_total
        self._log.info(
            "CACHED:  %d tokens generated  time=%.2f s  throughput=%.0f tok/s",
            total_tokens, elapsed_total, tok_per_s,
        )
        return tok_per_s

    def run_naive(self) -> float:
        """Naive decode: recomputes full K/V every step. Returns tokens/sec."""
        inf_cfg = self._cfg["inference"]
        batch_size = self._cfg["kv_cache"]["max_batch_size"]
        prompt_len = inf_cfg["prompt_tokens"]
        max_new = inf_cfg["max_new_tokens"]
        iters = inf_cfg["benchmark_iters"]

        model = NaiveDecoderLM(self._cfg).to(self._device)
        model.eval()

        elapsed_total = 0.0
        total_tokens = 0

        with torch.no_grad():
            for _ in range(iters):
                history = torch.randint(
                    0, self._cfg["model"]["vocab_size"], (batch_size, prompt_len), device=self._device
                )
                t0 = time.perf_counter()

                for _ in range(max_new):
                    logits = model(history)           # full sequence every time
                    new_tok = self._sample(logits)
                    history = torch.cat([history, new_tok], dim=1)

                torch.cuda.synchronize(self._device)
                elapsed_total += time.perf_counter() - t0
                total_tokens += batch_size * max_new

        tok_per_s = total_tokens / elapsed_total
        self._log.info(
            "NAIVE:   %d tokens generated  time=%.2f s  throughput=%.0f tok/s",
            total_tokens, elapsed_total, tok_per_s,
        )
        return tok_per_s

    def compare(self) -> None:
        self._log.info("=== KV-cache throughput comparison ===")
        cached_tps = self.run_cached()
        naive_tps = self.run_naive()
        speedup = cached_tps / naive_tps
        self._log.info(
            "Speedup from KV-cache: %.2fx  (cached=%.0f tok/s vs naive=%.0f tok/s)",
            speedup, cached_tps, naive_tps,
        )


def main() -> None:
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device available")

    engine = InferenceEngine(config)
    engine.compare()


if __name__ == "__main__":
    main()
