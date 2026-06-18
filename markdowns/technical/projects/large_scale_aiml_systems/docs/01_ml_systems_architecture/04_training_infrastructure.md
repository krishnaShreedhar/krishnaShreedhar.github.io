# Training Infrastructure

Training infrastructure encompasses the hardware, scheduling, and tooling needed to run ML training jobs efficiently at scale. Well-designed training infrastructure maximizes GPU utilization, minimizes queue wait times, provides experiment isolation, and enables rapid iteration from experimentation to production-scale training runs.

## GPU Cluster Architecture

```mermaid
graph TD
    subgraph Cluster[GPU Cluster]
        subgraph ControlPlane[Control Plane]
            Scheduler[Job Scheduler\nSLURM or Kubernetes\nwith GPU Operator]
            Registry[Experiment Registry\nMLflow or W&B\nartifact and metric storage]
            Storage[Shared Storage\nLustre or NFS\ntraining data and checkpoints]
        end

        subgraph NodePool[Worker Node Pool]
            subgraph Node1[GPU Node - 8x A100]
                GPU1[GPU 0]
                GPU2[GPU 1]
                GPU3[GPU 2-7]
                NVLink[NVLink / NVSwitch\nhigh-bandwidth GPU interconnect\n600 GB/s]
            end
            subgraph Node2[GPU Node - 8x A100]
                GPU4[GPU 0-7]
                NVLink2[NVLink / NVSwitch]
            end
            InfiniBand[InfiniBand Network\n200 Gb/s\ncross-node all-reduce]
            Node1 <--> InfiniBand
            Node2 <--> InfiniBand
        end
    end

    Researcher[Researcher\nsubmit job] --> Scheduler
    Scheduler --> Node1 & Node2
    Node1 & Node2 --> Registry

    style Cluster fill:#dbeafe,stroke:#2563eb,stroke-width:2px
    style ControlPlane fill:#f0fdf4,stroke:#16a34a
```

## Job Scheduling and Resource Management

```mermaid
graph TD
    subgraph JobLifecycle[Training Job Lifecycle]
        Submit[Job Submitted\nresource request: 8 GPUs\nmemory: 640 GB\ntime limit: 24h]
        Queue[Job Queue\nPending - waiting for resources\nFIFO or priority-based]
        Allocated[Resources Allocated\nGPU nodes reserved\nenvironment provisioned]
        Running[Job Running\ntraining in progress\ncheckpoints saved periodically]
        Done{Job Done?}
        Success[Success\nmodel artifacts saved\nmetrics logged]
        Failure[Failure\nauto-resume from checkpoint\nor error logged]

        Submit --> Queue --> Allocated --> Running --> Done
        Done -->|succeeded| Success
        Done -->|failed| Failure
        Failure -->|has checkpoint| Running
    end

    subgraph Schedulers[Scheduling Approaches]
        SLURM[SLURM\nHPC workloads\nbare metal\npriority queues]
        K8s[Kubernetes\ncontainerized\nmulti-tenant\nGPU Operator]
        Ray[Ray Cluster\nPython-native\ndynamic tasks\nActor model]
    end
```

## Hyperparameter Optimization

```mermaid
graph TD
    subgraph HPO[Hyperparameter Optimization]
        Config[Search Space Definition\nlearning_rate: loguniform 1e-5 to 1e-1\nbatch_size: choice 16 32 64 128\ndropout: uniform 0.1 to 0.5]

        subgraph Strategies[Search Strategies]
            Grid[Grid Search\nexhaustive\nexponential cost]
            Random[Random Search\nefficient for many dims\nno learning between trials]
            Bayes[Bayesian Optimization\nGaussian Process surrogate\nlearns from previous trials\nOptuna TPE sampler]
            ASHA[ASHA - Async Successive Halving\nearly stopping bad trials\nresource-efficient\nRay Tune]
        end

        Trial[Trial: train with sampled config\nevaluate validation metric]
        Best[Best Config Found\nfull training run\nfinal model]

        Config --> Strategies --> Trial --> Best
    end

    style Bayes fill:#dcfce7,stroke:#16a34a,stroke-width:2px
    style ASHA fill:#dcfce7,stroke:#16a34a,stroke-width:2px
```

## Checkpointing Strategy

```mermaid
graph TD
    subgraph Checkpointing[Checkpoint Management]
        Train[Training Loop\nstep N]
        SaveFreq{Every K steps\nor epoch end?}
        Save[Save Checkpoint\nmodel weights\noptimizer state\nscheduler state\nstep number\nRNG state]
        S3[Checkpoint Store\nS3 or shared filesystem]
        Keep[Keep Policy\nlast 3 checkpoints\nbest val loss checkpoint\nmilestone checkpoints]

        Train --> SaveFreq
        SaveFreq -->|Yes| Save --> S3 --> Keep
        SaveFreq -->|No| Train

        Crash[Job Crash\nor preemption]
        Resume[Resume from latest checkpoint\nno lost progress\nbeyond K steps]
        Crash --> Resume --> Train
    end

    style Save fill:#fef3c7,stroke:#d97706
    style Resume fill:#dcfce7,stroke:#16a34a
```

## Key Concepts

- **GPU Interconnect**: For multi-GPU training on a single node, NVLink provides high-bandwidth (600 GB/s) GPU-to-GPU communication, critical for all-reduce gradient synchronization. Across nodes, InfiniBand (200 Gb/s) or RoCE (RDMA over Converged Ethernet) provides low-latency interconnect. The interconnect bandwidth is often the bottleneck for distributed training efficiency.

- **SLURM**: The dominant workload manager for HPC GPU clusters. Jobs are submitted with resource requirements (GPUs, memory, time), placed in priority queues, and dispatched to reserved nodes. SLURM supports job arrays (HPO sweeps), preemption policies, and fair-share scheduling across research groups.

- **Kubernetes GPU Operator**: Extends Kubernetes to manage GPU resources in containerized environments. Automatically installs GPU drivers, CUDA toolkit, and device plugins on GPU nodes. Enables running training jobs as Kubernetes Jobs or via frameworks like Kubeflow Training Operator. Better for multi-tenant cloud environments than SLURM.

- **Checkpointing**: Periodically saving model weights, optimizer state, and training state to durable storage. Essential for long training runs on preemptible instances (cloud spot instances) or fault-tolerant clusters. Checkpoints must include optimizer state (momentum, Adam second moments) and RNG state for exact reproducibility. Async checkpointing (write in background) minimizes training interruption.

- **Hyperparameter Optimization (HPO)**: Systematically searching for the best hyperparameter configuration. Random search is surprisingly effective and embarrassingly parallel. Bayesian optimization (Optuna, Ray Tune) learns from previous trials using a surrogate model to propose promising configurations. ASHA enables early termination of clearly poor configurations, saving compute.

- **Experiment Isolation**: Each training run should be isolated — its own container/namespace, its own output directory, its own experiment entry in the tracking system. Prevents runs from interfering with each other and ensures reproducibility. Containers provide environment isolation; experiment IDs provide artifact isolation.

- **Spot/Preemptible Instances**: Cloud GPU instances available at 60-90% discount but can be reclaimed with 2-minute notice. Training code must be checkpoint-aware to resume from the latest checkpoint. Effective for HPO sweeps and fault-tolerant distributed training with checkpointing every 5-15 minutes.

- **GPU Utilization**: The key efficiency metric for training infrastructure — target >80% GPU compute utilization. Common causes of low utilization: data loading bottlenecks (CPU-bound preprocessing), small batch sizes, synchronization overhead in distributed training, I/O-bound checkpointing. Profile with `nvidia-smi`, PyTorch Profiler, or Nsight.

## Trade-offs

| Scheduling Approach | Multi-tenancy | Flexibility | Overhead | Best For |
|--------------------|--------------|------------|---------|----------|
| SLURM bare metal | Medium | Low | Very Low | HPC research clusters |
| Kubernetes + GPU Operator | High | High | Medium | Cloud-native, mixed workloads |
| Ray Cluster | High | Very High | Medium | Dynamic ML workloads, HPO |
| Cloud managed (SageMaker) | High | Low | Low | Teams without infra expertise |

## When to Use

- **SLURM**: Research institution or HPC center with dedicated GPU hardware and homogeneous workloads
- **Kubernetes GPU Operator**: Cloud-native teams needing multi-tenant isolation and mixed ML/serving workloads on the same cluster
- **Ray Tune for HPO**: When running many short trials (minutes to hours) — ASHA dramatically reduces total compute vs. grid or random search
- **Spot instances with checkpointing**: Any training run on cloud where cost matters and runs exceed 30 minutes — checkpointing overhead is justified above this threshold
- **Dedicated training clusters**: Production ML systems where training SLAs matter — shared clusters introduce queue delays that disrupt model refresh pipelines
