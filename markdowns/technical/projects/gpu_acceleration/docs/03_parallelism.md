---
title: "Module 3 — Parallelism: DDP & FSDP on 2× H200"
subtitle: "This module demonstrates two complementary parallelism strategies for scaling Transformer training across both H200 GPUs connected via NVLink."
category: technical
project: gpu_acceleration
project_title: "GPU Acceleration — Transformer Tutorials"
date: 2025-04-05
reading_time: 3
tags:
  - gpu-acceleration
  - docs
author: "Shreedhar Kodate"
output: "blogs/technical/posts/gpu_acceleration/docs/03_parallelism.html"
---
## Overview

This module demonstrates two complementary parallelism strategies for
scaling Transformer training across both H200 GPUs connected via NVLink.

| File | Strategy |
|------|----------|
| `ddp_trainer.py` | **Data parallelism** — each GPU has a full model copy |
| `fsdp_trainer.py` | **Model + data parallelism** — parameters sharded across GPUs |

---

## DistributedDataParallel (DDP)

### Concept

Each GPU holds a **complete copy** of the model. The dataset is split across
GPUs via `DistributedSampler`. After each backward pass, DDP **all-reduces**
gradients so every replica stays identical.

```mermaid
flowchart TD
    subgraph "GPU 0 (rank 0)"
        M0["Full Model Copy"]
        D0["Batch shard 0"]
        D0 --> M0
        M0 --> G0["Gradients"]
    end
    subgraph "GPU 1 (rank 1)"
        M1["Full Model Copy"]
        D1["Batch shard 1"]
        D1 --> M1
        M1 --> G1["Gradients"]
    end

    G0 & G1 -->|"NCCL All-Reduce\n(NVLink: ~900 GB/s)"| AVG["Averaged Gradients"]
    AVG --> M0 & M1
```

### Communication Pattern

```mermaid
sequenceDiagram
    participant R0 as GPU 0 (rank 0)
    participant R1 as GPU 1 (rank 1)

    R0->>R0: forward + backward (local gradients)
    R1->>R1: forward + backward (local gradients)

    note over R0,R1: DDP triggers all-reduce automatically
    R0->>R1: send grad shard (ring all-reduce)
    R1->>R0: send grad shard (ring all-reduce)

    R0->>R0: optimizer.step()
    R1->>R1: optimizer.step()
    note over R0,R1: models stay in sync
```

### When to use DDP

- Model fits in a single GPU's memory.
- Want maximum simplicity — DDP is a one-line wrapper.
- On H200 with NVLink, all-reduce bandwidth is ~900 GB/s — communication
  overhead is negligible for most models.

---

## FullyShardedDataParallel (FSDP)

### Concept

FSDP shards **parameters, gradients, and optimizer states** across GPUs.
Each GPU owns only `1/N` of each tensor. Just-in-time all-gather before
each forward pass reconstructs the full parameter, then immediately frees it.

```mermaid
flowchart TD
    subgraph "Before training"
        P["Full model parameters\n(e.g. 10 GB)"]
        P -->|"FSDP sharding"| S0["Shard 0 on GPU 0\n(5 GB)"]
        P -->|"FSDP sharding"| S1["Shard 1 on GPU 1\n(5 GB)"]
    end

    subgraph "Forward pass (layer L)"
        S0 & S1 -->|"All-gather"| FL["Full layer params\n(temporary, freed after use)"]
        FL --> FWD["Forward computation"]
    end

    subgraph "Backward pass"
        FWD --> GS0["Reduce-scatter gradients\nback to shards"]
    end
```

### Sharding Strategies

| Strategy | What's sharded | Memory saving | Communication cost |
|----------|---------------|---------------|--------------------|
| `FULL_SHARD` | Params + grads + optimizer | 1/N per GPU | All-gather + reduce-scatter each layer |
| `SHARD_GRAD_OP` | Grads + optimizer (params kept full) | ~2/3 | Reduce-scatter only |
| `NO_SHARD` | Nothing (equivalent to DDP) | None | All-reduce |

### auto_wrap_policy

FSDP needs to know which sub-modules to shard independently:

```mermaid
flowchart TD
    ROOT["TransformerLM"]
    ROOT --> EB["token_emb\n(small → not sharded)"]
    ROOT --> B0["TransformerBlock 0\n(≥1M params → SHARD)"]
    ROOT --> B1["TransformerBlock 1\n(≥1M params → SHARD)"]
    ROOT --> BN["..."]
    ROOT --> HEAD["head\n(small → not sharded)"]

    style B0 fill:#f96,color:#000
    style B1 fill:#f96,color:#000
```

Configured via `fsdp.min_num_params` in YAML.

---

## NVLink vs PCIe for Multi-GPU

The 2× H200 server uses **NVLink 4.0** between GPUs:

| Link | Bandwidth | Latency |
|------|-----------|---------|
| NVLink 4.0 (H200) | ~900 GB/s bidirectional | ~1 µs |
| PCIe 5.0 ×16 | ~128 GB/s | ~5 µs |

NVLink makes FSDP's all-gather/reduce-scatter overhead very small,
making FULL_SHARD viable even for moderate-size models.

---

## Launching

```bash
# DDP — 2 GPUs
torchrun --nproc_per_node=2 -m src.parallelism.ddp_trainer

# FSDP — 2 GPUs
torchrun --nproc_per_node=2 -m src.parallelism.fsdp_trainer
```

`torchrun` sets `RANK`, `LOCAL_RANK`, `WORLD_SIZE` env vars automatically.
Each process calls `dist.init_process_group()` to form the group, then
`torch.cuda.set_device(local_rank)` to bind to its GPU.

### Key config knobs (`configs/parallelism.yaml`)

| Key | Effect |
|-----|--------|
| `distributed.backend` | `nccl` (GPU-to-GPU via NVLink/PCIe) |
| `fsdp.sharding_strategy` | `FULL_SHARD` → most memory efficient |
| `fsdp.cpu_offload` | Move params to CPU RAM when not in use (slower) |
| `fsdp.min_num_params` | Threshold for wrapping sub-modules |
| `training.batch_size` | **Global** batch size; each GPU sees `batch_size/N` |