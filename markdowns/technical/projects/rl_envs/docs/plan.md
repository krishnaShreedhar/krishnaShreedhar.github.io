---
title: "RL Environments — Project Plan"
subtitle: "This project provides 10 minimal, educational reinforcement learning algorithm tutorials implemented in Python with PyTorch. Each algorithm:"
category: technical
project: rl_envs
project_title: "RL Environments — Educational Reinforcement Learning Tutorials"
date: 2025-07-03
reading_time: 5
tags:
  - rl-envs
  - docs
author: "Shreedhar Kodate"
output: "blogs/technical/posts/rl_envs/docs/plan.html"
---
## Overview

This project provides 10 minimal, educational reinforcement learning algorithm
tutorials implemented in Python with PyTorch. Each algorithm:

- Is implemented in a dedicated module under `src/algorithms/`.
- Reads all hyper-parameters from a YAML config file (no CLI args).
- Logs extensively to `logs/` using a structured format.
- Runs headless by default; optional `rgb_array` rendering via matplotlib.
- Adheres to SOLID programming principles with ABCs for all interfaces.

**Goals:**
1. Provide clear, readable reference implementations of canonical RL algorithms.
2. Demonstrate YAML-config-driven experiment management.
3. Show GPU-accelerated deep RL on 2x H200 hardware.
4. Cover three distinct use-case domains: path planning, autonomous vehicles, and VLM finetuning.

---

## 10 RL Algorithms

| # | Algorithm | Type | Key Concept | Use Case |
|---|-----------|------|-------------|----------|
| 01 | Q-Learning | Tabular | Bellman optimality, TD(0) | Path planning (GridWorld) |
| 02 | DQN | Value-based Deep RL | Experience replay + target network | Path planning (continuous state) |
| 03 | Double DQN | Value-based Deep RL | Decoupled action selection/evaluation | Improved path planning |
| 04 | A2C | Policy Gradient | Advantage estimation, shared actor-critic | Autonomous vehicles |
| 05 | PPO | Policy Gradient | Clipped surrogate objective, trust region | Autonomous vehicles |
| 06 | DDPG | Continuous Control | Deterministic policy + replay buffer | Continuous vehicle control |
| 07 | TD3 | Continuous Control | Twin critics + delayed policy update | Stable continuous control |
| 08 | SAC | Continuous Control | Maximum entropy, automatic temperature | Sample-efficient vehicle control |
| 09 | PPO / GRPO | LLM/VLM Finetuning | RLHF reward signal, token-level policy | VLM reward alignment |
| 10 | Dreamer | Model-based RL | World model, latent imagination | Sample-efficient AV control |

### Algorithm Groupings

**Tabular RL**
- Q-Learning: exact table representation; guaranteed convergence on small finite MDPs.

**Value-based Deep RL**
- DQN: neural network approximates Q(s,a); stabilised by experience replay and target network.
- Double DQN: fixes Q-value overestimation bias in DQN.

**Policy Gradient**
- A2C: synchronous advantage actor-critic; lower variance than REINFORCE via baseline.
- PPO: proximal policy optimisation with clipped objective; widely used in RLHF.

**Continuous Control**
- DDPG: off-policy, deterministic actor-critic for continuous action spaces.
- TD3: twin delayed deep deterministic policy gradient; addresses function approximation error.
- SAC: soft actor-critic with entropy regularisation; state-of-the-art sample efficiency.

**LLM/VLM Finetuning**
- PPO/GRPO: applied to language model token distributions using reward model scores.

**Model-based RL**
- Dreamer: learns a compact world model in latent space; plans via imagination rollouts.

---

## Environments

### GridWorldEnv (`src/envs/path_planning/`)
- NxN grid (default 10x10)
- Discrete action space: up/down/left/right
- State: normalised [row/N, col/N, goal_row/N, goal_col/N]
- Rewards: +10 goal, -1 wall/obstacle, -0.01 step
- Used by: Q-Learning, DQN, Double DQN

### VehicleEnv (`src/envs/autonomous_vehicle/`) — planned
- Continuous state (position, velocity, heading, sensor readings)
- Continuous or discrete action space (steering, throttle)
- Integrates with ROS2 Humble via docker-compose service
- Used by: A2C, PPO, DDPG, TD3, SAC, Dreamer

### VLMEnv (`src/envs/vlm_finetuning/`) — planned
- Wraps a Hugging Face VLM (e.g. LLaVA, InstructBLIP)
- State: image + text prompt token IDs
- Action: next token selection
- Reward: from a separate reward model or human preference signal
- Used by: PPO, GRPO

---

## Config System

Every experiment merges two YAML files:

```
global.yaml          — shared defaults (logging, device, buffer, training schedule)
algorithms/NN.yaml   — algorithm + env overrides
```

Deep merge rules:
- Both files are loaded as Python dicts.
- Algorithm values recursively override global values at matching keys.
- New keys in the algorithm file are added to the merged config.

Access pattern (dot notation):
```python
cfg = ConfigLoader.merge("configs/global.yaml", "configs/algorithms/02_dqn.yaml")
batch_size = cfg["training.batch_size"]          # raises KeyError if missing
lr = cfg.get("training.learning_rate", 1e-3)     # returns default if missing
raw_dict = cfg.raw                               # plain Python dict
```

No CLI argument parsing anywhere in the codebase.

---

## Logging Format

All loggers write to `logs/<name>.log` and stdout simultaneously.

Timestamp format: `[YYYY-MM-DD HH:MM:SS.mmm]`

Log level codes and intended use:
| Level | Code | When to use |
|-------|------|-------------|
| DEBUG | 10 | Per-step: action, reward, Q-values, grid state |
| INFO  | 20 | Per-episode: reward, steps, checkpoints, target updates |
| WARNING | 30 | NaN losses, unexpected state, degraded performance |
| ERROR | 40 | Unrecoverable errors |

Structured log examples:
```
[2026-06-02 10:00:01.123] [INFO   ] [dqn] [EPISODE] episode=100 reward=7.35 steps=88
[2026-06-02 10:00:01.456] [INFO   ] [dqn] [METRICS] step=8800 loss=0.0042 epsilon=0.45 mean_q=2.31
[2026-06-02 10:00:01.789] [WARNING] [dqn] NaN/Inf detected in loss. key=loss step=8801
```

---

## Docker Setup

### Services

**rl_envs** — main GPU training container
- Base image: `nvidia/cuda:13.0.1-devel-ubuntu22.04`
- Python 3.11 + uv package manager
- Mounts: `src/`, `configs/`, `logs/`, `docs/`
- GPU: 2x H200 via NVIDIA Container Runtime

**ros2** — ROS2 Humble for autonomous vehicle experiments
- Base image: `osrf/ros:humble-desktop`
- Network mode: host (for ROS2 DDS discovery)
- ROS_DOMAIN_ID: 42

### Commands

```bash
# Build images
docker compose -f docker/docker-compose.yml build

# Start services
docker compose -f docker/docker-compose.yml up -d

# Attach to training container
docker compose -f docker/docker-compose.yml exec rl_envs bash

# Attach to ROS2 container
docker compose -f docker/docker-compose.yml exec ros2 bash

# Stop all
docker compose -f docker/docker-compose.yml down
```

Inside the container, `PYTHONPATH=/workspace/rl_envs/src` is set, so imports
like `from core.config_loader import ConfigLoader` work directly.

---

## Hardware Requirements

| Component | Specification |
|-----------|---------------|
| GPUs | 2x NVIDIA H200 (80GB HBM3 each) |
| CUDA | 13.0.1 |
| CPU | Any modern x86-64, 16+ cores recommended |
| RAM | 64 GB+ recommended |
| Storage | 100 GB+ for model checkpoints and logs |

Multi-GPU usage:
- `device.gpu_ids: [0, 1]` in `configs/global.yaml`
- `CUDA_VISIBLE_DEVICES=0,1` set in Docker environment
- Algorithms can use `torch.nn.DataParallel` or `DistributedDataParallel`
  (implementation left to individual algorithm modules)

---

## Learning Path

For beginners to deep RL, follow the algorithms in order:

1. **Q-Learning** — Understand the Bellman equation and TD learning on a toy problem.
2. **DQN** — See how neural networks replace the Q-table; learn about replay buffers and target networks.
3. **Double DQN** — One-line change to DQN that fixes a fundamental bias; good lesson in ablation.
4. **A2C** — First policy gradient algorithm; introduces actor-critic and advantage estimation.
5. **PPO** — Production-grade policy gradient; used in ChatGPT/RLHF pipelines.
6. **DDPG** — Extend DQN ideas to continuous actions.
7. **TD3** — Understand why DDPG is unstable and how three simple fixes address it.
8. **SAC** — Maximum-entropy RL; currently state-of-the-art for continuous control.
9. **PPO/GRPO for VLMs** — Apply RL to token generation; connects to modern LLM training.
10. **Dreamer** — World models and latent imagination; the frontier of sample efficiency.