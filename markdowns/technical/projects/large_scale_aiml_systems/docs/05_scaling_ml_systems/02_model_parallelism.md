---
title: "Model Parallelism"
subtitle: "Model parallelism distributes different parts of a neural network across multiple GPUs when the model is too large to fit on a single device even with parameter sharding. While data parallelism scales by processing..."
category: technical
project: large_scale_aiml_systems
project_title: "Large Scale AI/ML Systems"
date: 2025-12-24
reading_time: 5
tags:
  - large-scale-aiml-systems
  - docs
author: "Shreedhar Kodate"
output: "blogs/technical/posts/large_scale_aiml_systems/docs/05_scaling_ml_systems/02_model_parallelism.html"
---
Model parallelism distributes different parts of a neural network across multiple GPUs when the model is too large to fit on a single device even with parameter sharding. While data parallelism scales by processing more data simultaneously, model parallelism scales by partitioning the model itself — dividing layers, splitting weight matrices across devices, or routing to specialized expert subnetworks.

## Model Parallelism Strategies

```mermaid
graph TD
    subgraph MPStrategies[Model Parallelism Overview]
        subgraph TensorP[Tensor Parallelism - Intra-Layer]
            TP1[Split individual weight matrices\ncolumn-wise or row-wise\nacross N GPUs\nEach GPU computes a slice of each layer]
            TP2[Requires all-reduce within each layer\nhigh communication frequency\nbest with NVLink not cross-node\nMegatron-LM]
        end

        subgraph PipeP[Pipeline Parallelism - Inter-Layer]
            PP1[Assign consecutive layers to GPUs\nGPU 0: layers 1-8\nGPU 1: layers 9-16\nGPU 2: layers 17-24\nGPU 3: layers 25-32]
            PP2[Micro-batching to hide pipeline bubble\nGPX PipeDream schedule\nGPipe schedule]
        end

        subgraph SeqP[Sequence Parallelism]
            SP1[Split sequence dimension\nacross GPUs\nReduces activation memory\nfor long context training\nUsed with tensor parallelism]
        end

        subgraph MoE[Mixture of Experts]
            MOE1[Sparse gating: route each token\nto top-2 of N expert networks\nOnly activated experts compute\nScales parameters without\nscaling FLOPs linearly]
        end
    end
```

## Tensor Parallelism in Transformers

```mermaid
graph TD
    subgraph TensorParallel[Tensor Parallelism for MLP Block]
        Input[Input X\ndim: seq_len x hidden_dim]

        subgraph GPU0[GPU 0]
            W0[W1 column slice 0\nGELU activation\nW2 row slice 0\npartial output Y0]
        end

        subgraph GPU1[GPU 1]
            W1M[W1 column slice 1\nGELU activation\nW2 row slice 1\npartial output Y1]
        end

        AllReduce2[All-Reduce\nsum partial outputs\nY = Y0 plus Y1\nfull output restored]

        Output[Full Output Y\ndim: seq_len x hidden_dim]

        Input --> GPU0 & GPU1
        GPU0 --> AllReduce2
        GPU1 --> AllReduce2
        AllReduce2 --> Output
    end

    style AllReduce2 fill:#fef3c7,stroke:#d97706
```

## Pipeline Parallelism with Micro-batching

```mermaid
graph TD
    subgraph PipelineSchedule[Pipeline Schedule - GPipe Style]
        subgraph Devices[Device Assignment]
            D0[GPU 0\nLayers 1-8\nEmbedding + early transformer blocks]
            D1[GPU 1\nLayers 9-16\nMiddle transformer blocks]
            D2[GPU 2\nLayers 17-24\nLater transformer blocks]
            D3[GPU 3\nLayers 25-32\nFinal blocks + LM head]
        end

        subgraph Bubble[Pipeline Bubble Problem]
            Prob[Without micro-batching:\nGPU1 idle while GPU0 runs forward\nGPU2 idle while GPU0 and GPU1 run\nGPU utilization: low]
        end

        subgraph Solution[Micro-batch Solution]
            Sol[Split mini-batch into M micro-batches\nGPU1 processes micro-batch 1\nwhile GPU0 processes micro-batch 2\nPipeline bubble: 1 divided by M fraction of time\nM=8 gives 87.5% efficiency]
            style Sol fill:#dcfce7,stroke:#16a34a,stroke-width:2px
        end
    end
```

## Mixture of Experts (MoE) Architecture

```mermaid
graph TD
    subgraph MoEArch[Mixture of Experts Architecture]
        Input2[Token Representation\nafter attention layer]

        Router[Gating Network\nlearned softmax router\nselects top-2 experts\nfor each token]

        subgraph Experts[Expert Pool - 64 experts]
            E1[Expert 1\nFeed-Forward Network]
            E2[Expert 2]
            Edots[...]
            E8[Expert 8]
            Emore[...]
        end

        Combine[Weighted Sum\nactivation weighted\nby router softmax scores]

        Output2[Token Output\nnext layer input]

        Input2 --> Router
        Router -->|route token to top-2| E1 & E2 & E8
        E1 & E2 & E8 --> Combine --> Output2
    end

    style Router fill:#fef3c7,stroke:#d97706,stroke-width:2px
```

## Key Concepts

- **Tensor Parallelism (TP)**: Splitting individual weight matrices within a single transformer layer across multiple GPUs. For a linear layer Y=XW, the weight matrix W is split column-wise — each GPU computes a partial output, then an all-reduce sums the partial outputs to produce the full result. Requires communication on every layer's output, making it communication-intensive — best suited for NVLink-connected GPUs on the same node. Megatron-LM pioneered this approach for training GPT-3 scale models.

- **Pipeline Parallelism (PP)**: Assigning consecutive groups of transformer layers to different GPUs. The forward pass produces activations at GPU boundary points, which are transmitted to the next GPU. Without scheduling tricks, this creates a "pipeline bubble" where most GPUs are idle. Micro-batching (GPipe) and interleaved schedules (PipeDream) reduce the bubble to less than 10% of total time with M=8+ micro-batches.

- **3D Parallelism**: Combining data parallelism, tensor parallelism, and pipeline parallelism simultaneously — the approach used by Megatron-LM and DeepSpeed to train the largest models. Tensor parallel within a node (high-bandwidth NVLink), pipeline parallel across nodes, data parallel across pipeline replicas. Each dimension is tuned to maximize hardware utilization given the specific network topology.

- **Mixture of Experts (MoE)**: A sparse scaling approach where a transformer's feed-forward layers are replaced by N experts (separate FFN networks) plus a learned router. Each token is routed to the top-2 experts for computation. The total parameter count scales with N, but the FLOPs per token only increase by ~2 (routing adds overhead, but only 2 of N experts are activated). Enables building models with 100B+ parameters that compute at the cost of a much smaller dense model. Used by Mixtral, GPT-4 (reported), and Switch Transformer.

- **Expert Parallelism**: Placing different MoE experts on different GPUs for distributed serving of MoE models. Each GPU specializes in a subset of experts. Requires routing tokens across GPUs (all-to-all collective) to reach the correct expert. The all-to-all communication overhead is the main constraint on scaling expert parallelism.

- **Load Balancing in MoE**: A key challenge — the router must distribute tokens roughly equally across experts to avoid some GPUs being overloaded while others are idle. Auxiliary load balancing loss terms encourage the router to maintain balanced routing. Without load balancing, MoE models can collapse to using only 2-3 experts ("expert collapse").

- **Activation Memory in Pipeline Parallelism**: Pipeline parallel stages must retain activations for the backward pass until their gradient is computed. With many micro-batches in flight, activation memory can become the bottleneck. Gradient checkpointing (recomputing activations during backward) trades compute for memory and is commonly combined with pipeline parallelism.

## Trade-offs

| Strategy | Communication Volume | GPU Utilization | Memory Efficiency | Implementation |
|---------|---------------------|----------------|------------------|---------------|
| Tensor Parallel | High (per layer) | High | High | Complex |
| Pipeline Parallel | Low | Medium (bubble) | Very High | Very Complex |
| MoE | Medium (all-to-all) | High | Excellent | Very Complex |
| FSDP alone | Medium (per step) | High | Very High | Medium |

## When to Use

- **Tensor parallelism**: Training models where a single layer's weight matrix exceeds GPU memory, or to reduce per-GPU memory with fast NVLink interconnect
- **Pipeline parallelism**: Very deep models (100+ layers) spread across multiple nodes where tensor parallelism's communication overhead is too high for cross-node links
- **3D parallelism**: Training frontier LLMs (100B+ parameters) — the only practical approach at this scale, combining all three strategies
- **MoE architecture**: When parameter count must scale beyond compute budget — MoE provides capacity at dense model compute cost. Excellent for LLM pretraining at scale
- **Sequence parallelism**: Long-context training (32K+ tokens) where attention activations become the memory bottleneck, combined with tensor parallelism