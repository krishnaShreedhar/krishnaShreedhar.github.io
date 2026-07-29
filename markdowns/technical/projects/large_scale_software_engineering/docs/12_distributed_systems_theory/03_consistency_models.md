---
title: "Consistency Models"
subtitle: "Consistency models define the guarantees a distributed system makes about the order in which operations appear to execute. Different models offer different trade-offs between correctness and performance. Choosing the..."
category: technical
project: large_scale_software_engineering
project_title: "Large Scale Software Engineering"
date: 2025-12-10
reading_time: 3
tags:
  - large-scale-software-engineering
  - docs
author: "Shreedhar Kodate"
output: "blogs/technical/posts/large_scale_software_engineering/docs/12_distributed_systems_theory/03_consistency_models.html"
---
Consistency models define the guarantees a distributed system makes about the order in which operations appear to execute. Different models offer different trade-offs between correctness and performance. Choosing the right model requires understanding what ordering guarantees the application actually needs.

## Consistency Model Hierarchy

```mermaid
graph TD
    subgraph ConsistencyHierarchy[Consistency Strength - Strongest to Weakest]
        Linear[Linearizability\nStrongest - single copy illusion\nAll operations appear instantaneous\nReal-time ordering respected]
        Sequential[Sequential Consistency\nAll operations ordered\nbut not necessarily real-time]
        Causal[Causal Consistency\nCausally related operations ordered\nConcurrent operations may differ]
        PRAM[PRAM - Pipeline RAM\nPer-process ordering only\nWrites in one process ordered]
        Eventual[Eventual Consistency\nWeakest\nConverges eventually\nNo ordering guarantees]

        Linear --> Sequential --> Causal --> PRAM --> Eventual
    end

    subgraph PerformanceCost[Performance Cost]
        P1[Linearizability: highest latency\nneed consensus for every op]
        P2[Eventual: lowest latency\nno coordination needed]
    end
```

## Linearizability vs Sequential Consistency

```mermaid
graph LR
    subgraph LinearizableOK[Linearizable - Real-time order]
        T[Timeline]
        WA[Write A=1 at T=1] --> RA[Read A must return 1\nfor all subsequent reads\nin wall clock time]
    end

    subgraph SeqConsistent[Sequential but not Linearizable]
        P1Op[Process 1:\nWrite X=1\nWrite Y=1]
        P2Op[Process 2:\nRead Y=1\nRead X=0\nLegal in Seq Consistency\nIllegal in Linearizability]
        P1Op --> P2Op
    end
```

## Causal Consistency

```mermaid
graph TD
    subgraph CausalExample[Causal Consistency Example]
        A[Alice posts: Going to lunch]
        B[Bob reads post\nReplies: Enjoy!]
        C[Carol sees Bobs reply\nbefore Alices original post]
        D[Causal Consistency Violation!\nThe reply causally depends on the post\nbut Carol sees them out of order]

        A --> B --> C
        C --> D
        style D fill:#fee2e2,stroke:#dc2626
    end

    subgraph CausalFix[With Causal Consistency Guaranteed]
        Fix[System ensures:\nAnyone who sees Bobs reply\nalso sees Alices post first\nbecause reply CAUSED BY post]
        style Fix fill:#dcfce7,stroke:#16a34a
    end
```

## Eventual Consistency in Practice

```mermaid
sequenceDiagram
    participant Client1
    participant Node1
    participant Node2
    participant Client2

    Client1->>Node1: Write: likes=100
    Node1-->>Client1: ACK
    Note over Node1,Node2: Async replication lag ~100ms

    Client2->>Node2: Read likes
    Node2-->>Client2: Returns: likes=99 (stale!)

    Note over Node1,Node2: Replication completes
    Client2->>Node2: Read likes again
    Node2-->>Client2: Returns: likes=100 (converged)

    Note over Client2: System is eventually consistent\nAll replicas converge to same value
```

## Key Concepts

- **Linearizability (Atomicity)**: The strongest consistency model. Every operation appears to execute instantaneously at some point between its invocation and completion, and the global ordering respects real-time ordering. Reads always see the latest committed write. Requires coordination on every operation. Implemented by: single-leader databases with synchronous replication, etcd, ZooKeeper.

- **Sequential Consistency**: All operations from all processes appear in a single global order that is consistent with the order within each individual process. Does not need to respect real-time ordering across processes. Weaker than linearizability — two concurrent writes could appear in different orders to different readers, but each reader sees a consistent ordering.

- **Causal Consistency**: Operations that are causally related (write-then-read, read-then-write) appear in causal order to all nodes. Concurrent operations (no causal relationship) may appear in different orders. Causally consistent reads never see a reply before the original message. More available than linearizability with fewer anomalies than eventual consistency.

- **Eventual Consistency**: Given sufficient time with no new updates, all replicas will converge to the same value. No ordering guarantees — reads may return stale values. Very high availability. Used by DNS, Cassandra, DynamoDB (default), Amazon S3. Requires conflict resolution for concurrent writes (last-write-wins, CRDTs, application-level merge).

- **Read Your Writes (Session Consistency)**: A client always reads its own most recent writes. Weaker than linearizability but stronger than eventual consistency. Can be implemented by routing each client's reads to the same replica or by tracking write timestamps.

- **Monotonic Reads**: Once a process reads a value, subsequent reads will not return older values. Prevents a process from seeing value `v`, then later seeing an older value. Simple to implement with sticky routing.

- **CRDT (Conflict-free Replicated Data Types)**: Data structures that can be concurrently updated and deterministically merged without coordination. Examples: G-Counter (grow-only counter), OR-Set (observed-remove set). Enable eventually consistent systems to maintain correct semantics without coordination.

## Trade-offs

| Model | Correctness | Latency | Availability | Use Case |
|-------|------------|---------|-------------|---------|
| Linearizability | Strongest | Highest | Lower | Financial transactions, config |
| Sequential | Strong | High | Lower | Chat messages |
| Causal | Good | Medium | Good | Social media, collaboration |
| Eventual | Weakest | Lowest | Highest | DNS, shopping cart, CDN |

## When to Apply

- **Linearizability**: Financial data, inventory, any "exactly-once" counting, leader election
- **Causal**: Social media feeds, collaborative editing, comment systems
- **Eventual**: Product ratings, analytics counters, DNS, content delivery — where approximate correctness is acceptable
- **Read Your Writes**: Any system where a user makes a change and needs to see it reflected immediately in the UI