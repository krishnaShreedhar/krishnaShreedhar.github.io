---
title: "Module 2 — Transformer Training on GPU"
subtitle: "This module implements a complete Transformer training loop from scratch, showing exactly how data and model tensors flow through the GPU."
category: technical
project: gpu_acceleration
project_title: "GPU Acceleration — Transformer Tutorials"
date: 2025-01-12
reading_time: 2
tags:
  - gpu-acceleration
  - docs
author: "Shreedhar Kodate"
output: "blogs/technical/posts/gpu_acceleration/docs/02_transformer_training.html"
---
## Overview

This module implements a complete Transformer training loop from scratch,
showing exactly how data and model tensors flow through the GPU.

| File | Teaches |
|------|---------|
| `model.py` | Transformer architecture: embedding, MHA, FFN, residuals |
| `dataset.py` | Synthetic token dataset (no external corpus needed) |
| `trainer.py` | Training loop with AMP, gradient clipping, checkpointing |
| `main.py` | Entry point — reads config, starts training |

---

## Architecture

```mermaid
flowchart TD
    IN["Input tokens (B, T)"]
    E["Token Embedding (B, T, d_model)"]
    PE["+ Positional Embedding"]
    DROP["Dropout"]

    IN --> E --> PE --> DROP

    subgraph "N × TransformerBlock"
        LN1["LayerNorm"]
        MHA["MultiHeadSelfAttention\nQ·Kᵀ / √d_head → softmax → · V"]
        ADD1["Residual +"]
        LN2["LayerNorm"]
        FFN["FeedForward\nLinear → GELU → Linear"]
        ADD2["Residual +"]
        LN1 --> MHA --> ADD1 --> LN2 --> FFN --> ADD2
    end

    DROP --> LN1
    ADD2 --> LNF["Final LayerNorm"]
    LNF --> HEAD["Linear head (tied weights)"]
    HEAD --> LOGITS["Logits (B, T, vocab_size)"]
    LOGITS --> LOSS["CrossEntropyLoss"]
```

### Multi-Head Attention (MHA) — Step by Step

```mermaid
flowchart LR
    X["x (B,T,C)"]
    Q["Q = xWᵀ_q\n(B,H,T,d_h)"]
    K["K = xWᵀ_k\n(B,H,T,d_h)"]
    V["V = xWᵀ_v\n(B,H,T,d_h)"]
    S["Scores = Q·Kᵀ / √d_h\n(B,H,T,T)"]
    A["Attn = softmax(Scores)\n(B,H,T,T)"]
    O["Out = Attn·V → concat heads → W_o"]

    X --> Q & K & V
    Q & K --> S --> A --> O
    V --> O
```

**Weight tying**: the output projection head shares weights with the token
embedding matrix — reduces parameter count and improves perplexity.

---

## Training Loop

```mermaid
sequenceDiagram
    participant DataLoader
    participant CPU
    participant GPU

    loop Each step
        DataLoader->>CPU: next batch (tokens, targets)
        CPU->>GPU: .to(device, non_blocking=True)
        GPU->>GPU: forward pass (with AMP autocast)
        GPU->>GPU: CrossEntropyLoss
        GPU->>GPU: loss.backward()
        GPU->>GPU: clip_grad_norm_
        GPU->>GPU: optimizer.step()
        GPU-->>CPU: loss.item() (sync)
    end
```

### Automatic Mixed Precision (AMP)

H200 natively supports **BF16** (better dynamic range than FP16, no loss scaling needed):

```mermaid
flowchart LR
    FP32_PARAMS["FP32 params\n(master copy)"] -->|autocast| BF16_FWD["BF16 forward\n(lower memory, faster tensor cores)"]
    BF16_FWD --> LOSS["FP32 loss"]
    LOSS -->|backward| BF16_GRADS["BF16 gradients"]
    BF16_GRADS -->|optimizer step in FP32| FP32_PARAMS
```

- With `bfloat16`: no `GradScaler` needed (no underflow risk).
- With `float16`: `GradScaler` wraps backward to prevent underflow.

### Gradient Clipping

Prevents exploding gradients common in deep Transformers:
```
‖g‖₂ = sqrt(Σ gᵢ²)
if ‖g‖₂ > max_norm:
    g ← g × (max_norm / ‖g‖₂)
```
Configured via `training.gradient_clip_norm` in the YAML.

---

## Memory Breakdown (per GPU, 6-layer model, d_model=512)

| Component | Approx size |
|-----------|------------|
| Parameters (FP32) | ~100 MB |
| Activations (BF16) | ~200 MB (batch=32, seq=128) |
| Gradients (BF16) | ~100 MB |
| Optimizer states (FP32 Adam) | ~200 MB |
| **Total** | **~600 MB** |

H200 has 141 GB — this model fits easily; the config can scale up.

---

## Running

```bash
python -m src.transformer_training.main
```

### Key config knobs (`configs/transformer_training.yaml`)

| Key | Effect |
|-----|--------|
| `training.use_amp` | Enable/disable mixed precision |
| `training.amp_dtype` | `bfloat16` (recommended on H200) or `float16` |
| `training.batch_size` | Increase until GPU memory saturates |
| `model.num_layers` | Depth vs speed tradeoff |
| `logging.level` | `DEBUG` logs every step; `INFO` logs every N steps |