---
title: "Serving at Scale"
subtitle: "Serving ML models at scale means maintaining low latency, high availability, and cost efficiency while handling highly variable traffic — from zero to millions of requests per hour. Large-scale ML serving combines..."
category: technical
project: large_scale_aiml_systems
project_title: "Large Scale AI/ML Systems"
date: 2025-06-02
reading_time: 5
tags:
  - large-scale-aiml-systems
  - docs
author: "Shreedhar Kodate"
output: "blogs/technical/posts/large_scale_aiml_systems/docs/05_scaling_ml_systems/04_serving_at_scale.html"
---
Serving ML models at scale means maintaining low latency, high availability, and cost efficiency while handling highly variable traffic — from zero to millions of requests per hour. Large-scale ML serving combines infrastructure patterns (autoscaling, load balancing, multi-region deployment) with ML-specific optimizations (batching, caching, model routing) to meet production SLAs at sustainable cost.

## Large-Scale Serving Architecture

```mermaid
graph TD
    subgraph GlobalArch[Global ML Serving Architecture]
        DNS[DNS\ngeographic routing\nreturn nearest region]

        subgraph RegionUS[US Region - Primary]
            LBUS[Load Balancer\nL7 - request routing]
            subgraph ServingFleet[Model Serving Fleet]
                S1[Inference Server 1\nGPU instance\nTriton or vLLM]
                S2[Inference Server 2]
                S3[Inference Server 3\n...]
            end
            ModelCacheUS[Model Cache\nweights loaded in GPU memory\nno cold start]
            FeatureStoreUS[Feature Store\nRedis cluster\nonline feature serving]

            LBUS --> S1 & S2 & S3
            S1 & S2 & S3 --> ModelCacheUS & FeatureStoreUS
        end

        subgraph RegionEU[EU Region - Secondary]
            LBEU[Load Balancer]
            ServingFleetEU[Inference Servers]
            LBEU --> ServingFleetEU
        end

        DNS --> LBUS & LBEU
    end

    Monitoring[Prometheus + Grafana\nlatency p50 p99\nthroughput tokens per second\nGPU utilization\nerror rate]

    LBUS --> Monitoring
    LBEU --> Monitoring
```

## Autoscaling Strategy

```mermaid
graph TD
    subgraph Autoscaling[GPU Serving Autoscaling]
        Metrics[Custom Metrics\nrequests per GPU per second\nGPU memory utilization\nqueue depth - pending requests]

        HPA[Horizontal Pod Autoscaler\nKubernetes HPA with custom metrics\nscale out when queue depth exceeds N\nscale in when utilization below threshold]

        subgraph ScaleOut[Scale Out Logic]
            Trigger[Trigger: queue_depth greater than 10\nor GPU_util greater than 85%]
            Provision[Provision new GPU instances\nload model weights from S3\nwarm up with test request\nadd to load balancer pool]
            Cooldown[Cooldown period 5 min\nprevent flapping]
        end

        subgraph ScaleIn[Scale In Logic]
            SI[Trigger: GPU_util less than 20%\nfor 10 consecutive minutes]
            Drain[Drain: stop sending new requests\nwait for in-flight requests to complete\nthen terminate instance]
        end

        Metrics --> HPA --> ScaleOut & ScaleIn
    end
```

## Model Routing and Cascading

```mermaid
graph TD
    subgraph ModelCascade[Model Cascade - Cost Optimization]
        Request[Inference Request]

        Router2[Routing Logic\nClassify request complexity]

        SmallModel[Small Fast Model\n7B params INT4\ncost: 0.001 USD per request\nlatency: 50ms]

        Confidence{Confidence\nthreshold\nmet?}

        LargeModel[Large Accurate Model\n70B params FP16\ncost: 0.05 USD per request\nlatency: 500ms]

        Response[Return Prediction]

        Request --> Router2 --> SmallModel --> Confidence
        Confidence -->|High confidence| Response
        Confidence -->|Low confidence| LargeModel --> Response

        Savings[80% of requests served\nby small model\n90% cost reduction\nvs always using large model]
        style Savings fill:#dcfce7,stroke:#16a34a,stroke-width:2px
    end
```

## Cost Optimization Strategies

```mermaid
graph TD
    subgraph CostOpt[Serving Cost Optimization Levers]
        subgraph Hardware[Hardware Selection]
            H1[Spot GPU instances\n60-90% discount\nrequires checkpoint-resume\nor stateless serving]
            H2[Reserved instances\ncommit 1-3 years\n30-60% discount vs on-demand\nfor baseline load]
            H3[Right-size GPU type\nA10G for smaller models\nA100 or H100 for large LLMs\nT4 for low-throughput tasks]
        end

        subgraph ModelOpt[Model Optimization]
            M1[Quantization\nINT4 for 4x cost reduction\nINT8 for 2x cost reduction]
            M2[Model distillation\ntrain small student from large teacher\n10x smaller model\n5-10% quality loss]
            M3[Caching\ncache common prefixes\ncache identical requests]
        end

        subgraph Traffic[Traffic Management]
            T1[Request batching\ngroup requests for batch inference\nhigher GPU utilization]
            T2[Rate limiting\nprevent cost overruns\nprotect GPU capacity]
            T3[Priority queuing\nSLA requests first\nbatch jobs during low traffic]
        end
    end
```

## Key Concepts

- **Cold Start**: The latency penalty incurred when a new serving instance starts up and must load model weights from storage (S3/GCS) into GPU memory before serving the first request. A 70B model at FP16 requires loading ~140GB, which at 10 GB/s network bandwidth takes 14 seconds. Mitigations: keep warm instances (scale-to-zero is infeasible for latency-sensitive serving), pre-load models on instance initialization, use instance storage for faster loading.

- **Request Batching**: Grouping multiple inference requests into a single batch maximizes GPU utilization by keeping more compute units busy simultaneously. Dynamic batching (waiting up to N milliseconds for more requests to accumulate before processing) trades a small latency increase for significant throughput improvement. Triton Inference Server, vLLM, and TensorRT-LLM all implement dynamic batching.

- **Prefix Caching**: For LLMs, if multiple requests share a common prefix (e.g., all start with the same system prompt), the KV cache for that shared prefix can be computed once and reused. vLLM implements automatic prefix caching. For applications with long system prompts (RAG context, few-shot examples), prefix caching can reduce time-to-first-token by 50-80%.

- **Model Distillation**: Training a smaller "student" model to mimic the output distribution of a larger "teacher" model. The student is trained on the teacher's soft probability outputs (not just hard labels), which carry richer information. A well-distilled 7B model can approach the quality of a 70B model at 10x lower inference cost. DistilBERT, TinyLlama, and Phi-3 demonstrate the power of distillation.

- **Capacity Planning**: Estimating GPU resource requirements for a given traffic target. Key calculation: throughput (tokens/second per GPU) from benchmarks, target RPS and average output length → required GPUs per region. Add 30-50% headroom for traffic spikes. Plan for model upgrade overhead (old and new versions run simultaneously during rollout).

- **Traffic Shaping**: Managing inbound request traffic to protect serving infrastructure. Rate limiting prevents individual clients from consuming all GPU capacity. Priority queuing ensures latency-sensitive synchronous requests (user-facing) are served before batch jobs. Circuit breakers reject requests during overload to prevent cascade failure (better to return an error than a 30-second timeout).

- **Multi-Region Serving**: Deploying serving infrastructure in multiple geographic regions to reduce latency for globally distributed users and improve availability (one region can handle traffic if another fails). DNS-based geographic routing directs users to the nearest region. Model weights and configuration must be replicated to all regions.

## Trade-offs

| Serving Strategy | Latency | Throughput | Cost | Complexity |
|----------------|---------|-----------|------|-----------|
| Single GPU, no batching | Lowest | Very Low | High | Low |
| Dynamic batching | Low | High | Medium | Low |
| Model cascade | Variable | High | Low | Medium |
| Multi-region | Lowest (geo) | High | High | Very High |
| Spot instances + stateless | Low | High | Low | High |

## When to Use

- **Dynamic batching**: Always for latency-tolerant API endpoints — 5-10ms batching window typically yields 3-5x throughput improvement with negligible latency impact
- **Model cascade**: Cost-sensitive applications where a fraction of requests require high-accuracy — route to expensive model only for uncertain cases
- **Prefix caching**: Any LLM application with consistent system prompts or few-shot examples — free latency improvement with no quality tradeoff
- **Spot instances for serving**: Background/batch inference where occasional interruptions are acceptable — not recommended for user-facing real-time serving below 99.9% availability requirements
- **Multi-region**: User-facing global applications where regional latency matters, or when availability requirements (99.99%) exceed single-region fault tolerance