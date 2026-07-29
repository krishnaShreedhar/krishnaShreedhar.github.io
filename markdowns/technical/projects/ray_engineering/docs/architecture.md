---
title: "Ray Engineering – Architecture Reference"
subtitle: "Comprehensive architectural diagrams and explanatory text covering every major Ray subsystem implemented in this project."
category: technical
project: ray_engineering
project_title: "Ray Engineering"
date: 2025-01-02
reading_time: 5
tags:
  - ray-engineering
  - docs
author: "Shreedhar Kodate"
output: "blogs/technical/posts/ray_engineering/docs/architecture.html"
---
Comprehensive architectural diagrams and explanatory text covering every
major Ray subsystem implemented in this project.

---

## 1. Ray Cluster Architecture

A Ray cluster consists of a **Head Node** and one or more **Worker Nodes**.
All coordination flows through the Global Control Service (GCS) on the head
node, while actual task and object data flows directly between workers.

```mermaid
graph TB
    subgraph HeadNode["Head Node"]
        GCS["Global Control Service (GCS)\n─ Node registry\n─ Actor table\n─ Job metadata"]
        Scheduler["Distributed Scheduler\n─ Resource matching\n─ Task placement"]
        Dashboard["Ray Dashboard\n:8265"]
        Raylet_H["Raylet\n─ Local task queue\n─ Object manager"]
        PlasmaH["Plasma Store\n(Object Store)\nShared Memory"]
        GCS --- Scheduler
        Raylet_H --- PlasmaH
    end

    subgraph Worker1["Worker Node 1"]
        Raylet_W1["Raylet"]
        Plasma_W1["Plasma Store"]
        WP1["Worker Processes\n(Task/Actor)"]
        Raylet_W1 --- Plasma_W1
        Raylet_W1 --- WP1
    end

    subgraph Worker2["Worker Node 2"]
        Raylet_W2["Raylet"]
        Plasma_W2["Plasma Store"]
        WP2["Worker Processes\n(Task/Actor)"]
        Raylet_W2 --- Plasma_W2
        Raylet_W2 --- WP2
    end

    GCS <-->|Node heartbeats| Raylet_W1
    GCS <-->|Node heartbeats| Raylet_W2
    Scheduler -->|Task assignment| Raylet_W1
    Scheduler -->|Task assignment| Raylet_W2
    Plasma_W1 <-->|Object transfer| Plasma_W2
    Plasma_W1 <-->|Object transfer| PlasmaH
    Dashboard -.->|Metrics / events| GCS

    Client["Driver (Python process)\nray.init()"] -->|Submit tasks / actors| GCS
    Client -->|ray.get / ray.put| PlasmaH
```

### Key components

| Component | Role |
|-----------|------|
| **GCS** | Single source of truth for cluster state. Stores actor handles, node liveness, job info. |
| **Raylet** | Per-node daemon. Maintains a local task queue, manages the local Plasma Store, and communicates with the GCS. |
| **Plasma Store** | Shared-memory object store (Apache Arrow format). Workers on the same node read objects via zero-copy mmap. Cross-node transfers go over the network. |
| **Distributed Scheduler** | Matches task resource requirements (`num_cpus`, `num_gpus`, custom resources) to available worker slots. |
| **Dashboard** | Web UI at `:8265` showing task throughput, actor states, object store usage, and logs. |

---

## 2. Ray Data Pipeline Architecture

Ray Data provides a lazy, parallel Dataset abstraction built on top of Ray
tasks.  Transformations are fused and executed in a streaming fashion to
avoid materialising large intermediate datasets.

```mermaid
graph LR
    subgraph Ingestion
        CSV["CSV / Parquet\nfiles"]
        PD["Pandas\nDataFrame"]
        HF["HuggingFace\nDataset"]
        CSV --> DS_Raw["ray.data.Dataset\n(partitioned into Blocks)"]
        PD --> DS_Raw
        HF --> DS_Raw
    end

    subgraph Transforms["Lazy Transforms (fused)"]
        DS_Raw -->|".map_batches(normalise)"| DS_Norm["Normalised\nDataset"]
        DS_Norm -->|".map_batches(features)"| DS_Feat["Feature-Engineered\nDataset"]
        DS_Feat -->|".filter()"| DS_Filt["Filtered\nDataset"]
    end

    subgraph Output
        DS_Filt -->|".split(n_workers)"| Shard0["Shard 0\n→ Worker 0"]
        DS_Filt -->|".split(n_workers)"| Shard1["Shard 1\n→ Worker 1"]
        DS_Filt -->|".take_batch()"| Sample["Sample batch\nfor inspection"]
    end

    Shard0 --> Trainer0["TorchTrainer\nWorker 0"]
    Shard1 --> Trainer1["TorchTrainer\nWorker 1"]
```

### Key concepts

- **Block**: the unit of parallelism in Ray Data.  Each block is a Pandas
  DataFrame stored as a plasma object.
- **`map_batches(fn)`**: applies `fn` to each block in parallel using Ray
  tasks.  The `batch_format="numpy"` option returns NumPy arrays for
  compute-intensive transforms.
- **Lazy execution**: transformations are not executed until `.take()`,
  `.iter_batches()`, or a trainer consumes the dataset.
- **Streaming ingestion**: during training, Ray Data streams blocks from
  disk / object store directly into the training loop, avoiding OOM.

---

## 3. Actor Communication Sequence

This diagram shows the message flow when multiple workers push gradients to
a centralised ParameterServer actor.

```mermaid
sequenceDiagram
    participant Driver as Driver Process
    participant GCS as Global Control Service
    participant PS as ParameterServer Actor
    participant W0 as Worker 0
    participant W1 as Worker 1

    Driver->>GCS: Create actor (ParameterServer)
    GCS-->>Driver: ActorHandle ref
    Driver->>GCS: Create actor (Worker 0)
    Driver->>GCS: Create actor (Worker 1)
    GCS-->>Driver: ActorHandle refs

    loop Training Iteration
        Driver->>PS: ps.get_params.remote()
        PS-->>Driver: ObjectRef[params]
        Driver->>Driver: params = ray.get(ref)

        par Parallel Gradient Computation
            Driver->>W0: w0.compute_gradient.remote(params, step)
            Driver->>W1: w1.compute_gradient.remote(params, step)
        end

        W0-->>Driver: ObjectRef[grad_0]
        W1-->>Driver: ObjectRef[grad_1]
        Driver->>Driver: grads = ray.get([ref0, ref1])

        Driver->>PS: ps.apply_gradients.remote(grad_0, grad_1)
        PS->>PS: average gradients, SGD step
        PS-->>Driver: ObjectRef[new_params]
        Driver->>Driver: new_params = ray.get(ref)
    end

    Driver->>GCS: ray.shutdown()
    GCS->>PS: terminate actor
    GCS->>W0: terminate actor
    GCS->>W1: terminate actor
```

### Why actors instead of tasks?

| | Remote Task | Actor |
|--|-------------|-------|
| **State** | Stateless – restarted fresh each call | Stateful – persists across calls |
| **Scheduling** | Re-scheduled each call (possible migration) | Pinned to a node/worker process |
| **Use case** | Pure functions, embarrassingly parallel work | Parameter servers, caches, counters, simulators |

---

## 4. Ray Train DDP Architecture

```mermaid
graph TB
    Driver["Driver\nTorchTrainer.fit()"] --> RC["RunConfig\n(checkpoint, storage)"]
    Driver --> SC["ScalingConfig\n(num_workers, use_gpu)"]

    Driver --> W0["Train Worker 0\n_train_loop_per_worker()"]
    Driver --> W1["Train Worker 1\n_train_loop_per_worker()"]

    subgraph Worker0["Worker 0 Process"]
        W0 --> M0["prepare_model()\n→ DDP wrapping"]
        W0 --> L0["prepare_data_loader()\n→ Shard 0"]
        M0 --> FWD0["Forward pass\nLoss computation"]
        FWD0 --> BWD0["Backward pass\nGradients"]
    end

    subgraph Worker1["Worker 1 Process"]
        W1 --> M1["prepare_model()\n→ DDP wrapping"]
        W1 --> L1["prepare_data_loader()\n→ Shard 1"]
        M1 --> FWD1["Forward pass\nLoss computation"]
        FWD1 --> BWD1["Backward pass\nGradients"]
    end

    BWD0 <-->|"AllReduce\n(gradient sync)"| BWD1

    BWD0 --> RPT0["ray.train.report()\nmetrics + Checkpoint"]
    BWD1 --> RPT1["ray.train.report()\nmetrics + Checkpoint"]

    RPT0 --> CKP["CheckpointManager\n(num_to_keep=3)"]
    RPT1 --> CKP
    CKP --> Storage["Local / Remote\nCheckpoint Storage"]
```

---

## 5. Ray Tune HPO Architecture

```mermaid
graph TB
    subgraph Tuner
        PS["Param Space\n(loguniform, choice, uniform)"]
        TC["TuneConfig\n(metric, mode, num_samples)"]
        ASHA["ASHAScheduler\n(early stopping)"]
        Optuna["OptunaSearch\n(Bayesian next-point)"]
    end

    PS --> Trial0["Trial 0\n(lr=0.01, hidden=64, ...)"]
    PS --> Trial1["Trial 1\n(lr=0.001, hidden=128, ...)"]
    PS --> TrialN["Trial N\n(sampled by Optuna)"]

    Trial0 --> TT0["TorchTrainer\n_trainable_loop"]
    Trial1 --> TT1["TorchTrainer\n_trainable_loop"]
    TrialN --> TTN["TorchTrainer\n_trainable_loop"]

    TT0 -->|"ray.train.report(accuracy)"| ASHA
    TT1 -->|"ray.train.report(accuracy)"| ASHA
    TTN -->|"ray.train.report(accuracy)"| ASHA

    ASHA -->|"Stopped (grace_period)"| Pruned["Pruned Trials\n(no further epochs)"]
    ASHA -->|"Promoted"| Optuna
    Optuna -->|"Suggest next config"| PS

    Tuner -->|"fit() → ResultGrid"| RG["ResultGrid\n.get_best_result()\n.get_dataframe()"]
```

---

## 6. Ray Serve Deployment Graph

```mermaid
graph LR
    Client["HTTP Client\nPOST /predict\n{features: [...]}"]

    subgraph RayServe["Ray Serve"]
        Ingress["InferencePipeline\n(num_replicas=1)\nIngress deployment"]

        subgraph PreprocessorPool["Preprocessor (num_replicas=2)"]
            P0["Preprocessor\nReplica 0"]
            P1["Preprocessor\nReplica 1"]
        end

        subgraph ClassifierPool["BatchClassifier (num_replicas=2)"]
            C0["BatchClassifier\nReplica 0"]
            C1["BatchClassifier\nReplica 1"]
        end
    end

    Client -->|HTTP| Ingress
    Ingress -->|DeploymentHandle| P0
    Ingress -->|DeploymentHandle| P1
    P0 -->|normalised features| C0
    P1 -->|normalised features| C1

    subgraph Batching["@serve.batch (max_batch_size=32)"]
        C0 -->|coalesced inference| Model["PyTorch\nModel"]
        C1 -->|coalesced inference| Model
    end

    Model -->|predictions| Ingress
    Ingress -->|JSON response| Client
```

### @serve.batch behaviour

When multiple requests arrive concurrently at `BatchClassifier`, Ray Serve
coalesces them into a single call to `_batched_predict()` up to
`max_batch_size`.  The `batch_wait_timeout_s` controls how long Serve waits
to fill a batch before dispatching a partial one.

| Config key | Value | Effect |
|------------|-------|--------|
| `max_batch_size` | 32 | Maximum samples per inference call |
| `batch_wait_timeout_s` | 0.05 | Max latency penalty for batching |
| `num_replicas` | 2 | Horizontal scale-out |
| `max_concurrent_queries` | 10 | Back-pressure limit per replica |