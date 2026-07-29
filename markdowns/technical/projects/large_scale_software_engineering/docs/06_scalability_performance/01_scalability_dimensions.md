---
title: "Scalability Dimensions"
subtitle: "Scalability has multiple dimensions — systems can scale along compute, storage, network, and geographic axes. Understanding which dimension is the bottleneck determines the appropriate scaling strategy. Premature..."
category: technical
project: large_scale_software_engineering
project_title: "Large Scale Software Engineering"
date: 2025-11-06
reading_time: 4
tags:
  - large-scale-software-engineering
  - docs
author: "Shreedhar Kodate"
output: "blogs/technical/posts/large_scale_software_engineering/docs/06_scalability_performance/01_scalability_dimensions.html"
---
Scalability has multiple dimensions — systems can scale along compute, storage, network, and geographic axes. Understanding which dimension is the bottleneck determines the appropriate scaling strategy. Premature scaling optimizations are expensive; late scaling decisions cause outages.

## Scaling Dimensions

```mermaid
graph TD
    subgraph Vertical[Vertical Scaling - Scale Up]
        VS[Bigger single machine\nMore CPU, RAM, faster storage\nNo code changes required\nEasiest but has ceiling\nSingle point of failure]
        style VS fill:#dbeafe,stroke:#2563eb
    end

    subgraph Horizontal[Horizontal Scaling - Scale Out]
        HS[More instances\nStateless required\nLoad balancer needed\nNo ceiling\nFault tolerant]
        style HS fill:#dcfce7,stroke:#16a34a
    end

    subgraph Functional[Functional Scaling - Decomposition]
        FS[Split by function\nMicroservices\nScale hot services independently\nPer-component resource tuning]
        style FS fill:#fef3c7,stroke:#d97706
    end

    subgraph Geographic[Geographic Scaling - Distribution]
        GS[Multi-region deployment\nReduce latency for global users\nData residency compliance\nDisaster recovery]
        style GS fill:#ffe4e6,stroke:#be123c
    end
```

## Amdahl's Law

```mermaid
graph TD
    subgraph Amdahl[Amdahl's Law - Limits of Parallel Scaling]
        Formula[Speedup = 1 divided by\n S + 1-S divided by N\nwhere S = serial fraction\nN = number of processors]
        P10[S=0.10 - 10% serial\nMax speedup: 10x\neven with infinite processors]
        P20[S=0.20 - 20% serial\nMax speedup: 5x]
        P50[S=0.50 - 50% serial\nMax speedup: 2x]
        Formula --> P10 & P20 & P50
    end

    Implication[Implication:\nReduce the serial fraction first\nbefore adding more nodes.\nFind and eliminate sequential bottlenecks.]

    style Implication fill:#fef3c7,stroke:#d97706
```

## Stateless vs Stateful Scaling

```mermaid
graph LR
    subgraph Stateless[Stateless Service - Easy to Scale]
        LBS[Load Balancer] --> S1[Instance 1]
        LBS --> S2[Instance 2]
        LBS --> S3[Instance 3]
        S1 & S2 & S3 --> SharedDB[(Shared Database\nShared Cache)]
        Note1[Any instance can handle\nany request\nSimply add instances]
        style Note1 fill:#dcfce7,stroke:#16a34a
    end

    subgraph Stateful[Stateful Service - Hard to Scale]
        LBF[Load Balancer\nsticky sessions] --> SF1[Instance 1\nUser A state]
        LBF --> SF2[Instance 2\nUser B state]
        LBF --> SF3[Instance 3\nUser C state]
        Note2[User must reach same instance\nFailover loses state\nRebalancing complex]
        style Note2 fill:#fee2e2,stroke:#dc2626
    end
```

## Multi-Region Architecture

```mermaid
graph TD
    DNS[Global DNS / Traffic Manager\nGeo-proximity routing]

    DNS -->|US users| USRegion[US-East Region]
    DNS -->|EU users| EURegion[EU-West Region]
    DNS -->|AP users| APRegion[AP-Southeast Region]

    subgraph USRegion[US-East]
        USLB[Load Balancer]
        USApp[App Cluster]
        USDB[(Primary DB)]
        USLB --> USApp --> USDB
    end

    subgraph EURegion[EU-West]
        EULB[Load Balancer]
        EUApp[App Cluster]
        EUDB[(Replica DB)]
        EULB --> EUApp --> EUDB
    end

    USDB -->|async replication| EUDB

    style USRegion fill:#dbeafe,stroke:#2563eb
    style EURegion fill:#dcfce7,stroke:#16a34a
```

## Key Concepts

- **Vertical Scaling**: Adding more resources (CPU, RAM, disk, network bandwidth) to an existing machine. The path of least resistance — no application changes required. Hardware limits eventually cap vertical scaling. Modern cloud instances (AWS x1e.32xlarge: 3.9TB RAM, 128 vCPUs) push the ceiling very high before horizontal scaling is forced.

- **Horizontal Scaling**: Adding more instances of a service, distributing load via a load balancer. Requires stateless service design — each instance must be able to handle any request independently. Theoretically unlimited but introduces distributed systems complexity.

- **Functional Decomposition**: Splitting a monolith into services that can be scaled independently based on their specific resource needs. A payment service (CPU-bound, low volume) and a search service (IO-bound, high volume) have different optimal instance types.

- **Amdahl's Law**: The theoretical maximum speedup from parallelization is bounded by the sequential fraction of the program. If 20% of execution is sequential, the maximum possible speedup is 5x regardless of how many parallel processors are added. The takeaway: identify and eliminate sequential bottlenecks before scaling horizontally.

- **Gustafson's Law**: Counter to Amdahl's Law — as the problem size grows, the serial fraction typically stays constant while the parallel portion grows. Adding more processors allows solving larger problems in the same time, even if per-problem speedup is bounded.

- **Stateless Design**: Services that store no session state locally can be horizontally scaled trivially — any instance can serve any request. State must be externalised to databases, caches, or object storage. The twelve-factor app principle: store state in backing services, not in process memory.

- **Database Bottleneck**: The database is typically the first horizontal scaling bottleneck in web applications. Mitigation strategies: read replicas (scale reads), caching (reduce load), connection pooling (reduce connection overhead), partitioning/sharding (scale writes and storage).

## Trade-offs

| Approach | Benefit | Cost |
|----------|---------|------|
| Vertical | Simple, no code changes | Ceiling, expensive at top end, SPOF |
| Horizontal | Theoretically unlimited, fault tolerant | Statelessness required, LB needed |
| Functional decomposition | Right-sized resources per service | Microservices complexity |
| Multi-region | Global performance, DR | Data consistency, cost |
| Read replicas | Scale reads cheaply | Stale reads, replication lag |

## When to Use

- **Vertical first**: For most systems, vertical scaling buys time cheaply — scale vertically until the cost or ceiling forces horizontal
- **Horizontal**: When vertical ceiling is reached or when fault tolerance requires multiple instances
- **Functional decomposition**: When clear bottlenecks exist in specific capabilities that need independent scaling
- **Multi-region**: When user latency to a single region is unacceptable, or when disaster recovery and data residency requirements mandate it