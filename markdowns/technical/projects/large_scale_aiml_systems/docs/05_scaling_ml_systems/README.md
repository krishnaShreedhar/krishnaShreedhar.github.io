# Scaling ML Systems

Scaling ML systems addresses the engineering challenges that emerge when models, datasets, and traffic grow beyond what a single machine or naive distributed approach can handle. Scaling in ML spans four dimensions: training (data and model parallelism for trillion-parameter models), inference (throughput optimization and KV-cache management), serving (handling millions of requests per second), and data (managing petabyte-scale datasets).

## Overview

```mermaid
mindmap
  root((Scaling\nML Systems))
    Distributed Training
      Data parallelism DDP
      FSDP fully sharded
      DeepSpeed ZeRO
      Gradient checkpointing
      Mixed precision BF16
    Model Parallelism
      Tensor parallelism
      Pipeline parallelism
      Sequence parallelism
      Expert parallelism MoE
    Inference Optimization
      Quantization INT8 INT4
      KV cache management
      Continuous batching
      vLLM PagedAttention
      Speculative decoding
    Serving at Scale
      Horizontal autoscaling
      Multi-region deployment
      Load balancing strategies
      Cost optimization
      Capacity planning
```

## Scaling Dimensions

```mermaid
graph TD
    subgraph ScalingDims[ML System Scaling Challenges]
        Training[Training Scale\nModel too large for single GPU\nDataset too large for single machine\nTraining too slow on single node]

        Inference[Inference Scale\nHigh latency - model too slow\nLow throughput - not enough GPU\nMemory bound - KV cache too large]

        Serving[Serving Scale\nHigh RPS - need many replicas\nGlobal users - need multi-region\nBurst traffic - need autoscaling]

        Data[Data Scale\nPetabytes of training data\nCannot fit in RAM\nneed distributed data loading]
    end
```

## Topics in This Section

| File | Topic | Key Concepts |
|------|-------|--------------|
| [01_distributed_training.md](01_distributed_training.md) | Distributed Training | DDP, FSDP, DeepSpeed ZeRO |
| [02_model_parallelism.md](02_model_parallelism.md) | Model Parallelism | Tensor, pipeline, MoE parallelism |
| [03_inference_optimization.md](03_inference_optimization.md) | Inference Optimization | Quantization, KV cache, vLLM |
| [04_serving_at_scale.md](04_serving_at_scale.md) | Serving at Scale | Autoscaling, multi-region, cost |
