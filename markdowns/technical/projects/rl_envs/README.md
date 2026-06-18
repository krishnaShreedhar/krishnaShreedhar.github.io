# RL Environments — Educational Reinforcement Learning Tutorials

A collection of 10 reinforcement learning algorithm implementations built for
educational purposes. Each algorithm is minimal, well-documented, and tied to a
concrete use case. All hyper-parameters live in YAML config files; there are no
command-line arguments.

---

## Algorithms

| # | Algorithm | Type | Use Case |
|---|-----------|------|----------|
| 01 | Q-Learning | Tabular | Path planning (GridWorld) |
| 02 | DQN | Value-based Deep RL | Path planning (continuous state) |
| 03 | Double DQN | Value-based Deep RL | Improved path planning |
| 04 | A2C | Policy Gradient | Autonomous vehicles |
| 05 | PPO | Policy Gradient | Autonomous vehicles |
| 06 | DDPG | Continuous Control | Autonomous vehicles |
| 07 | TD3 | Continuous Control | Autonomous vehicles |
| 08 | SAC | Continuous Control | Autonomous vehicles |
| 09 | PPO / GRPO | LLM/VLM Finetuning | VLM reward alignment |
| 10 | Dreamer | Model-based RL | Autonomous vehicles |

---

## Use Cases

### 1. Path Planning
Agent navigates a 10x10 GridWorld with obstacles from a random start to a
random goal. Discrete action space (up/down/left/right). Algorithms 01–03.

### 2. Autonomous Vehicles
Simulated vehicle control with continuous state and action spaces. Integrates
with a ROS2 Humble container for realistic sensor/actuator interfaces.
Algorithms 04–08, 10.

### 3. VLM Finetuning
Reinforcement learning from human feedback (RLHF) style fine-tuning of
vision-language models using PPO and GRPO reward signals. Algorithm 09.

---

## Hardware

- 2x NVIDIA H200 GPUs
- CUDA 13.0.1
- Ubuntu 22.04

GPU IDs are configured in `configs/global.yaml` (`device.gpu_ids: [0, 1]`).

---

## Quick Start

### 1. Build and enter the container

```bash
cd docker
docker compose up -d
docker compose exec rl_envs bash
```

### 2. Run Q-Learning on GridWorld

```python
from core.config_loader import ConfigLoader
from core.logger import RLLogger
from algorithms.q_learning import QLearningAgent
from envs.path_planning import GridWorldEnv
from training import Trainer
import torch

cfg = ConfigLoader.merge("configs/global.yaml", "configs/algorithms/01_q_learning.yaml")
logger = RLLogger("q_learning", cfg["logging.log_dir"], cfg["logging.level"])
device = torch.device("cpu")

env = GridWorldEnv(cfg.raw, logger)
agent = QLearningAgent(cfg.raw, device, logger)
trainer = Trainer(agent, env, cfg, logger)
history = trainer.train()
```

### 3. Run DQN on GridWorld

```python
cfg = ConfigLoader.merge("configs/global.yaml", "configs/algorithms/02_dqn.yaml")
# ... same pattern, use DQNAgent
```

---

## Config System

Every experiment is driven by two YAML files that are deep-merged at startup:

```
configs/global.yaml               # shared defaults (logging, device, buffer, training schedule)
configs/algorithms/NN_algo.yaml   # algorithm + env overrides
```

Algorithm values override global values at every nesting level. Access values
via dot notation:

```python
cfg.get("training.batch_size", 64)   # returns default if missing
cfg["training.batch_size"]           # raises KeyError if missing
```

No CLI arguments. No fallbacks inside algorithm code.

---

## Logging

All logs are written to `logs/<name>.log` and mirrored to stdout.

Log format:
```
[YYYY-MM-DD HH:MM:SS.mmm] [LEVEL  ] [name] message
```

Structured log lines:
```
[METRICS] step=500 loss=0.0423 epsilon=0.72 mean_q=1.34
[EPISODE] episode=42 reward=7.35 steps=120 goal_reached=True
```

Log levels:
- `DEBUG`: step-level detail (action, reward, Q-values)
- `INFO`: episode summaries, checkpoints, target network updates
- `WARNING`: NaN/Inf losses, unexpected events

---

## Project Structure

```
rl_envs/
├── configs/
│   ├── global.yaml
│   └── algorithms/
│       ├── 01_q_learning.yaml
│       ├── 02_dqn.yaml
│       └── 03_double_dqn.yaml
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── .env
├── docs/
│   ├── plan.md
│   └── algorithms/
│       ├── 01_q_learning.md
│       ├── 02_dqn.md
│       └── 03_double_dqn.md
├── logs/              # runtime log files (git-ignored except .gitkeep)
├── src/
│   ├── core/          # BaseAgent, BaseEnv, ReplayBuffer, RLLogger, ConfigLoader
│   ├── training/      # Trainer
│   ├── algorithms/    # q_learning, dqn, double_dqn, a2c, ppo, ddpg, td3, sac, grpo, dreamer
│   └── envs/          # path_planning, autonomous_vehicle, vlm_finetuning
├── pyproject.toml
└── README.md
```
