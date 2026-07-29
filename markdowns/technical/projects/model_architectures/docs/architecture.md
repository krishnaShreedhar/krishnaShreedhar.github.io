---
title: "SmallVLM Architecture"
subtitle: "SmallVLM is a minimal Vision-Language Model built from scratch in PyTorch for the image captioning task. Every component is implemented explicitly so the data flow, parameter counts, and design decisions are easy to..."
category: technical
project: model_architectures
project_title: "SmallVLM — Minimal Vision-Language Model Tutorial"
date: 2025-10-19
reading_time: 3
tags:
  - model-architectures
  - docs
author: "Shreedhar Kodate"
output: "blogs/technical/posts/model_architectures/docs/architecture.html"
---
## Overview

SmallVLM is a minimal Vision-Language Model built from scratch in PyTorch for the image captioning task. Every component is implemented explicitly so the data flow, parameter counts, and design decisions are easy to follow.

**Total parameters:** ~25M (default config)

---

## Component breakdown

### 1. Vision Encoder (`src/model/vision_encoder.py`)

A Vision Transformer (ViT) that converts a raw image into a sequence of contextualised patch embeddings.

| Sub-module | Role | Output shape |
|---|---|---|
| `PatchEmbedding` | `Conv2d(3, 384, kernel=16, stride=16)` splits the 224×224 image into 196 non-overlapping 16×16 patches and linearly projects each patch | `(B, 196, 384)` |
| Prepend `[CLS]` token | Learnable global image summary token | `(B, 197, 384)` |
| Learnable positional embedding | Distinguishes spatial positions | `(B, 197, 384)` |
| 6 × `ViTBlock` | Multi-head self-attention (6 heads) + FFN (ratio 4×) with pre-LayerNorm | `(B, 197, 384)` |
| Final `LayerNorm` | Stabilises activations | `(B, 197, 384)` |

**Output exported to the projection layer:**
- `patch_tokens` → `(B, 196, 384)` — one embedding per image patch
- `cls_token` → `(B, 384)` — unused in the captioning head but available for retrieval tasks

**Why ViT?** Patch-based processing gives fixed-length output regardless of image content, which simplifies the decoder interface.

---

### 2. Vision Projection (`src/model/projection.py`)

A two-layer MLP connector that maps visual features into the language decoder's embedding space.

```
384 → Linear → GELU → 768 → Linear → 384
```

This bottleneck design (384 → 768 → 384) allows the projection to learn a non-linear alignment between the vision and language representation spaces. A single linear layer would restrict the mapping to be affine.

---

### 3. Language Decoder (`src/model/language_decoder.py`)

A GPT-2-style autoregressive Transformer decoder with one key addition: **cross-attention to vision tokens**.

| Sub-module | Role |
|---|---|
| `nn.Embedding(50257, 384)` | Maps token IDs to dense vectors |
| Learnable positional embedding | Position-aware token representations (up to 128 tokens) |
| 6 × `DecoderBlock` | Three-stage block (see below) |
| Final `LayerNorm` + `Linear(384, 50257)` | LM head; **weight-tied** to the token embedding matrix |

**DecoderBlock internals:**
1. **Causal self-attention** — Token `i` can only attend to tokens `≤ i` (enforced by a pre-computed causal mask buffer). Ensures autoregressive generation.
2. **Cross-attention** — Query from the text stream, Key/Value from the projected vision tokens. This is how the model "looks at" the image while generating each word.
3. **Position-wise FFN** — Two linear layers with GELU, ratio 4×.

**Weight tying:** The LM head weight matrix is shared with the token embedding matrix. This halves the parameter count of the output projection (~20M → ~15M) and empirically improves generation quality.

---

### 4. Forward pass during training

```
Image (B, C, H, W)
  → VisionEncoder   → patch_tokens (B, 196, 384)
  → VisionProjection → proj_tokens (B, 196, 384)

Caption IDs (B, T)  ← teacher-forced input
  → decoder_input = caption_ids[:, :-1]  # everything except last token
  → target        = caption_ids[:, 1:]   # everything except BOS

LanguageDecoder(decoder_input, proj_tokens)
  → logits (B, T-1, 50257)

Loss = CrossEntropy(logits.reshape(-1, V), target.reshape(-1), ignore_index=pad_id)
```

Teacher-forcing feeds the ground-truth prefix at every step during training, which is faster and more stable than auto-regressive training.

---

## Parameter count (default config)

| Module | ~Params |
|---|---|
| Vision Encoder (ViT-6L-384) | 13.5M |
| Vision Projection (MLP) | 0.6M |
| Language Decoder (GPT-6L-384) | 13.5M (−5M with weight tying) |
| **Total** | **~22M** |

---

## Design principles

- **Minimal by default.** Every component is purpose-built for this task; no inherited `transformers` model classes.
- **Config-driven.** All architectural hyperparameters live in `configs/model.yaml`.
- **Explainable shapes.** Every module's docstring states its exact input/output tensor shapes.
- **Extensible.** Cross-attention can be disabled (`use_cross_attention: false`) for a simpler prefix-based VLM variant.