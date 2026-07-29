---
title: "Ray Engineering"
subtitle: "A self-contained project that illustrates **every major Ray subsystem** through minimal, runnable Python examples.  Each module demonstrates real patterns used in production distributed ML systems—including the..."
category: technical
project: ray_engineering
project_title: "Ray Engineering"
date: 2025-09-18
reading_time: 3
tags:
  - ray-engineering
author: "Shreedhar Kodate"
output: "blogs/technical/posts/ray_engineering/index.html"
---
A self-contained project that illustrates **every major Ray subsystem** through
minimal, runnable Python examples.  Each module demonstrates real patterns
used in production distributed ML systems—including the corresponding
anti-patterns so you know what to avoid.

---

## Concepts Covered

| Module | Ray API | Pattern demonstrated |
|--------|---------|----------------------|
| `core_primitives/remote_functions.py` | `@ray.remote`, `ray.get`, `ray.wait` | Submit-all-then-get, DAG composition, streaming collection |
| `core_primitives/actors.py` | `@ray.remote class`, `ActorPool` | ParameterServer, stateful workers, load-balanced pool |
| `core_primitives/object_store.py` | `ray.put`, `ObjectRef` | Zero-copy fan-out, lifecycle management |
| `distributed_training/data_pipeline.py` | `ray.data.Dataset`, `map_batches` | Lazy ETL pipeline, train/val split |
| `distributed_training/torch_trainer.py` | `TorchTrainer`, `ScalingConfig`, `Checkpoint` | DDP training, per-epoch checkpointing |
| `hyperparameter_tuning/tune_experiment.py` | `Tuner`, `ASHAScheduler`, `OptunaSearch` | Bayesian HPO with early stopping |
| `model_serving/serve_deployment.py` | `@serve.deployment`, `@serve.batch` | Multi-model pipeline, request batching |

---

## Project Structure

```
ray_engineering/
├── config.yaml                    # ALL constants and hyperparameters
├── pyproject.toml
├── README.md
├── logs/                          # Rotating JSON log files written here
├── src/
│   ├── utils/
│   │   ├── config_loader.py       # YAML config reader
│   │   └── logging_setup.py      # JSON formatter + rotating file handler
│   ├── core_primitives/
│   │   ├── remote_functions.py
│   │   ├── actors.py
│   │   └── object_store.py
│   ├── distributed_training/
│   │   ├── data_pipeline.py
│   │   └── torch_trainer.py
│   ├── hyperparameter_tuning/
│   │   └── tune_experiment.py
│   ├── model_serving/
│   │   └── serve_deployment.py
│   └── notebooks/
│       └── ray_concepts_demo.ipynb
├── docs/
│   ├── architecture.md            # Mermaid cluster / serving diagrams
│   └── flow_diagrams.md           # Step-by-step execution flows
└── docker/
    ├── Dockerfile
    ├── docker-compose.yml
    └── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install ray[all]==2.9.0 torch pyyaml numpy pandas optuna matplotlib
```

### 2. Run individual modules

Each Python file is independently runnable.  All parameters come from
`config.yaml` – edit the file to change behaviour.

```bash
# From the project root (ray_engineering/)
PYTHONPATH=src python src/core_primitives/remote_functions.py
PYTHONPATH=src python src/core_primitives/actors.py
PYTHONPATH=src python src/core_primitives/object_store.py
PYTHONPATH=src python src/distributed_training/data_pipeline.py
PYTHONPATH=src python src/distributed_training/torch_trainer.py
PYTHONPATH=src python src/hyperparameter_tuning/tune_experiment.py
PYTHONPATH=src python src/model_serving/serve_deployment.py
```

### 3. Run the Jupyter notebook

```bash
cd src/notebooks
jupyter lab ray_concepts_demo.ipynb
```

The notebook walks through every concept step-by-step with inline
visualisations (label distributions, training curves, HPO scatter plots,
latency histograms).

---

## Docker

### Build and run locally (single container)

```bash
cd docker
docker build -t ray-engineering:latest -f Dockerfile ..
docker run --rm -v $(pwd)/../logs:/app/logs ray-engineering:latest
```

### Multi-node cluster with Docker Compose

```bash
cd docker
docker compose up --build

# Scale to 4 workers
docker compose up --scale ray-worker=4
```

The head node starts Ray and runs `remote_functions.py`.  Worker nodes join
the cluster and wait for tasks.  The Ray Dashboard is available at
`http://localhost:8265`.

---

## Configuration

All tunable parameters live in `config.yaml`.  **No values are hardcoded**
in the Python source files.

Key sections:

```yaml
logging:
  level: INFO        # Set to DEBUG for verbose output
  log_file: logs/ray_engineering.log

ray.init:
  num_cpus: 4        # Simulated local cluster size
  object_store_memory: 1073741824  # 1 GB Plasma store

distributed_training:
  num_workers: 2
  epochs: 5
  learning_rate: 0.001

hyperparameter_tuning:
  num_samples: 20    # Total Tune trials
  metric: accuracy
  mode: max
```

---

## Logging

Every module writes **JSON-formatted** logs to `logs/ray_engineering.log`
(rotating, max 100 MB, 5 backups) and human-readable text to stdout.

Log level is controlled by `logging.level` in `config.yaml`.  Set to
`DEBUG` to see per-batch loss values, object references, and scheduler
decisions.

---

## Anti-Patterns Demonstrated

The project explicitly codes and labels anti-patterns so learners can
recognise them:

| Anti-pattern | File | Correct alternative |
|-------------|------|---------------------|
| `ray.get()` inside submission loop | `remote_functions.py::demo_anti_pattern` | Collect all futures, then single `ray.get()` |
| Calling actor methods one-at-a-time | `actors.py::demo_anti_pattern_actor` | Batch futures, then `ray.get(futures)` |
| Passing raw large arrays to tasks | `object_store.py::demo_anti_pattern_vs_correct` | `ray.put()` once, pass `ObjectRef` |