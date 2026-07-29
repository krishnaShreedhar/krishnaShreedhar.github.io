---
title: "Training Pipeline"
subtitle: "Pre-training → Fine-tuning → SFT or RL → Inference ↑ Hyperparameter Tuning (feeds best config back to Stage 1)"
category: technical
project: model_architectures
project_title: "SmallVLM — Minimal Vision-Language Model Tutorial"
date: 2025-12-09
reading_time: 3
tags:
  - model-architectures
  - docs
author: "Shreedhar Kodate"
output: "blogs/technical/posts/model_architectures/docs/training_pipeline.html"
---
## Stage overview

```
Pre-training → Fine-tuning → SFT or RL → Inference
                 ↑
        Hyperparameter Tuning (feeds best config back to Stage 1)
```

All stages run on 2× NVIDIA H200 GPUs via PyTorch DDP (`torchrun --nproc_per_node=2`).

---

## Stage 1 — Pre-training (`src/training/pretrain.py`)

**Goal:** Learn vision-language alignment on a large, noisy dataset.

**Dataset:** Conceptual Captions 3M (`conceptual_captions` on HuggingFace Hub).

**Loss:** Standard cross-entropy next-token prediction (teacher-forcing).

**Key settings** (see `configs/pretrain.yaml`):

| Param | Value | Rationale |
|---|---|---|
| LR | 1e-4 | AdamW with cosine decay |
| Effective batch | 256 | 64/GPU × 2 steps accum × 2 GPUs |
| BF16 | true | H200 native BF16 tensor cores |
| Warmup | 2000 steps | Stabilises early training |
| Epochs | 30 | Large noisy dataset needs more passes |

**Optimizer:** AdamW with parameter-group weight decay: biases and LayerNorm parameters are exempt from weight decay.

**Scheduler:** Cosine annealing with linear warmup, decaying to 10% of peak LR.

---

## Stage 2 — Fine-tuning (`src/training/finetune.py`)

**Goal:** Adapt the pre-trained model to a high-quality captioning dataset.

**Dataset:** COCO Captions 2017 (`coco_captions` on HuggingFace Hub, ~120k training images, 5 captions each).

**Key difference from pre-training:** The vision encoder trains at 10× lower LR (`vision_encoder_lr_scale: 0.1`) to prevent catastrophic forgetting of visual representations.

**Key settings** (see `configs/finetune.yaml`):

| Param | Value |
|---|---|
| LR | 2e-5 |
| Vision encoder LR | 2e-6 |
| Effective batch | 256 |
| Epochs | 10 |

---

## Stage 3a — SFT (`src/training/rl_sft_trainer.py`, `mode: sft`)

Supervised fine-tuning on a curated subset. Structurally identical to Stage 2 but with a lower LR (5e-6) and dedicated config section. Use this stage to specialise the model on a specific caption style or domain.

---

## Stage 3b — RL with REINFORCE (`src/training/rl_sft_trainer.py`, `mode: rl`)

**Goal:** Optimise non-differentiable caption quality metrics (CIDEr, BLEU-4) directly.

**Algorithm:** REINFORCE with a running-mean variance-reduction baseline.

**Per-step procedure:**

1. **Rollout:** For each image in the batch, sample `K=4` candidate captions using nucleus sampling (temperature=1.0, top-p=0.9).
2. **Reward:** Score each candidate with CIDEr (or BLEU-4) against the ground-truth captions.
3. **Advantage:** `A = reward − baseline` where baseline is an exponential moving average of past rewards.
4. **Policy gradient loss:** `L_PG = −mean(A × log_prob_of_sampled_tokens)`
5. **KL penalty:** `L_KL = KL(π_θ ‖ π_ref)` — penalises divergence from the frozen reference model copy loaded at init. Prevents reward hacking.
6. **Entropy bonus:** `L_ent = −H(π_θ)` — encourages exploration to avoid caption collapse.

**Total loss:**
```
L = L_PG + kl_coeff × L_KL − entropy_coeff × L_ent
```

---

## Stage 4 — Inference (`src/inference/generator.py`)

Three decoding strategies:

| Strategy | Description | When to use |
|---|---|---|
| **Greedy** | argmax at every step | Fastest; lowest quality |
| **Beam search** | Keep top-K partial sequences | Best for deterministic evaluation (e.g., CIDEr benchmark) |
| **Nucleus sampling** | Top-p / top-k with temperature | Diversity; also used for RL rollouts |

All three strategies are implemented with `@torch.no_grad()` and are GPU-accelerated. Beam search processes one image at a time for readability; batch beam search can be added as an optimisation.

---

## Stage 5 — Hyperparameter Tuning (`src/tuning/hyperparameter_tuning.py`)

Uses **Optuna** with a **MedianPruner** to search over:

- Learning rate (log-uniform 1e-5 to 5e-4)
- Batch size (16 / 32 / 64)
- Number of encoder/decoder layers (4–8)
- Hidden dimension (256 / 384 / 512)
- Dropout (0.0–0.2)
- Weight decay (log-uniform 1e-4 to 1e-1)
- Warmup steps (100–2000)

Each trial runs for 3 epochs on 20k training samples. The pruner stops unpromising trials after the first epoch. Results are stored in an SQLite database for resumption across sessions.

---

## Distributed training

All multi-GPU stages use **PyTorch DDP** launched with `torchrun`. Key points:

- `DistributedSampler` ensures each GPU sees a non-overlapping subset of the data.
- Gradient synchronisation happens automatically via DDP's all-reduce.
- Logging, checkpointing, and WandB calls are gated on `rank == 0`.
- `ipc: host` in docker-compose enables shared-memory for DataLoader workers.

---

## Checkpointing

A checkpoint is saved every `save_interval` steps and at the end of each epoch. The checkpoint with the lowest validation loss is also saved as `checkpoint_best.pt`. The last `keep_last_n_checkpoints` checkpoints are retained; older ones are removed to save disk space.