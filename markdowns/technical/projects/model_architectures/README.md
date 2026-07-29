---
title: "SmallVLM — Minimal Vision-Language Model Tutorial"
subtitle: "A from-scratch, explainable implementation of a small Vision-Language Model (VLM) for **image captioning**, built entirely with PyTorch. The goal is to understand every design decision rather than to achieve..."
category: technical
project: model_architectures
project_title: "SmallVLM — Minimal Vision-Language Model Tutorial"
date: 2025-03-16
reading_time: 3
tags:
  - model-architectures
author: "Shreedhar Kodate"
output: "blogs/technical/posts/model_architectures/index.html"
---
A from-scratch, explainable implementation of a small Vision-Language Model (VLM) for **image captioning**, built entirely with PyTorch. The goal is to understand every design decision rather than to achieve state-of-the-art results.

---

## What you'll learn

- How patch embeddings turn a 224×224 image into 196 discrete tokens
- How a ViT encoder contextualises those tokens with self-attention
- How a two-layer MLP bridges vision and language representation spaces
- How a GPT-style decoder generates captions word-by-word using cross-attention to image features
- How to pre-train, fine-tune, and align a VLM with RL (REINFORCE)
- Three decoding strategies: greedy, beam search, nucleus sampling
- Hyperparameter search with Optuna

---

## Architecture at a glance

```
Image (B,3,224,224)
  → ViT Vision Encoder (6 layers, 384 dim)  →  patch_tokens (B,196,384)
  → MLP Projection (384→768→384)            →  proj_tokens  (B,196,384)
                                                     ↓ cross-attention
Text tokens (B,T)
  → GPT Decoder (6 layers, 384 dim)
  → LM Head (384→50257)                     →  logits (B,T,50257)
```

Total parameters: ~22M  
Full architecture explainer: [`docs/architecture.md`](docs/architecture.md)

---

## Project structure

```
model_architectures/
├── src/
│   ├── model/
│   │   ├── config.py           # ModelConfig dataclass
│   │   ├── vision_encoder.py   # ViT encoder
│   │   ├── projection.py       # MLP connector
│   │   ├── language_decoder.py # GPT decoder with cross-attention
│   │   └── vlm.py              # SmallVLM top-level module
│   ├── data/
│   │   ├── dataset.py          # HuggingFace-backed image-caption dataset
│   │   ├── transforms.py       # Train / val image augmentations
│   │   └── collator.py         # Batch collation
│   ├── training/
│   │   ├── trainer.py          # BaseTrainer (DDP, BF16, checkpointing)
│   │   ├── pretrain.py         # Stage 1: pre-training entry point
│   │   ├── finetune.py         # Stage 2: fine-tuning entry point
│   │   └── rl_sft_trainer.py   # Stage 3: SFT / REINFORCE entry point
│   ├── inference/
│   │   └── generator.py        # Greedy / beam search / nucleus sampling
│   ├── tuning/
│   │   └── hyperparameter_tuning.py  # Optuna HPT
│   └── utils/
│       ├── logging_utils.py    # Structured logger factory
│       └── config_utils.py     # YAML loading and merging
├── configs/
│   ├── model.yaml              # Architecture hyperparameters
│   ├── pretrain.yaml           # Pre-training settings
│   ├── finetune.yaml           # Fine-tuning settings
│   ├── rl_sft.yaml             # RL / SFT settings
│   ├── inference.yaml          # Inference settings
│   └── tuning.yaml             # Optuna search space
├── docs/
│   ├── architecture.md         # Detailed architecture walkthrough
│   ├── training_pipeline.md    # Training stages explained
│   └── diagrams/
│       ├── vlm_architecture.mmd   # Mermaid architecture diagram
│       └── training_flow.mmd      # Mermaid training pipeline diagram
├── docker/
│   ├── Dockerfile              # nvidia/cuda:13.0.1-devel-ubuntu22.04 base
│   ├── .env                    # Environment variable template
│   └── docker-compose.yml      # Services: pretrain, finetune, rl_sft, inference, tuning
└── pyproject.toml
```

---

## Quick start

### 1. Build and start the container

```bash
cd docker
cp .env .env.local          # fill in DATA_DIR, OUTPUT_DIR, WANDB_API_KEY
docker compose build
```

### 2. Pre-train

```bash
docker compose run pretrain
```

Equivalent bare-metal command (from project root):

```bash
torchrun --nproc_per_node=2 -m src.training.pretrain
```

### 3. Fine-tune

```bash
docker compose run finetune
```

### 4. RL or SFT alignment

Edit `configs/rl_sft.yaml` → set `mode: rl` or `mode: sft`, then:

```bash
docker compose run rl_sft
```

### 5. Generate captions

```bash
docker compose run inference
```

### 6. Tune hyperparameters

```bash
docker compose run tuning
```

---

## Configuration

All constants and hyperparameters are in `configs/`. No command-line arguments — edit the YAML files instead.

---

## GPU requirements

| Stage | GPUs | Approx. VRAM |
|---|---|---|
| Pre-training | 2× H200 | ~40 GB/GPU |
| Fine-tuning | 2× H200 | ~30 GB/GPU |
| RL / SFT | 2× H200 | ~35 GB/GPU |
| Inference | 1× H200 | ~5 GB |
| HPT (per trial) | 1× H200 | ~10 GB |

---

## Documentation

| Document | Description |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | Detailed component walkthrough with tensor shapes |
| [`docs/training_pipeline.md`](docs/training_pipeline.md) | Training stage descriptions and algorithms |
| `docs/diagrams/vlm_architecture.mmd` | Mermaid data-flow diagram |
| `docs/diagrams/training_flow.mmd` | Mermaid training pipeline diagram |