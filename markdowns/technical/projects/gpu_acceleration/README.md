# GPU Acceleration — Transformer Tutorials

Minimal, hands-on tutorials for understanding Transformer model training and inference on GPU.

## Goals

| Module | What you learn |
|--------|----------------|
| `gpu_basics` | GPU memory management, pinned memory, CPU↔GPU transfers, CUDA streams |
| `transformer_training` | Transformer training loop on GPU with mixed-precision (AMP) |
| `parallelism` | Data parallelism (DDP) and model parallelism (FSDP) on 2x H200 |
| `kv_cache` | KV-cache from scratch for efficient autoregressive inference |

## Hardware

- **GPUs**: 2x NVIDIA H200 (141 GB HBM3e each)
- **CUDA**: 12.8+ recommended

## Docker

```bash
cd docker/
docker compose up --build
```

> **Note**: The prompt specified `nvidia/cuda:13.0.1-devel-ubuntu24.04`. As of this writing,
> the latest stable CUDA image is `12.8.x`. Please verify the tag exists on Docker Hub before
> building; the Dockerfile falls back to `12.8.1-devel-ubuntu24.04`.

## Running Tutorials

All entry points read from a YAML config file — **no CLI arguments** are used.

```bash
# GPU basics — data transfer benchmark
python -m src.gpu_basics.data_transfer

# Transformer training
python -m src.transformer_training.main

# DDP parallelism (2 GPUs)
torchrun --nproc_per_node=2 -m src.parallelism.ddp_trainer

# FSDP parallelism (2 GPUs)
torchrun --nproc_per_node=2 -m src.parallelism.fsdp_trainer

# KV-cache inference
python -m src.kv_cache.inference_engine
```

## Project Structure

```
gpu_acceleration/
├── configs/                  # YAML configs for each module
├── docs/                     # Explainers + Mermaid flow diagrams
├── docker/                   # Dockerfile + compose
├── logs/                     # Runtime log output (gitignored)
└── src/
    ├── logging_utils.py      # Shared logging setup
    ├── gpu_basics/           # Module 1
    ├── transformer_training/ # Module 2
    ├── parallelism/          # Module 3
    └── kv_cache/             # Module 4
```
