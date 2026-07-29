---
title: "Time and Ordering in Distributed Systems"
subtitle: "Time and ordering are fundamental challenges in distributed systems. Clocks drift, messages are delayed arbitrarily, and there is no global \"now\". Logical clocks provide ordering without relying on synchronized..."
category: technical
project: large_scale_software_engineering
project_title: "Large Scale Software Engineering"
date: 2025-03-03
reading_time: 4
tags:
  - large-scale-software-engineering
  - docs
author: "Shreedhar Kodate"
output: "blogs/technical/posts/large_scale_software_engineering/docs/12_distributed_systems_theory/04_time_ordering.html"
---
Time and ordering are fundamental challenges in distributed systems. Clocks drift, messages are delayed arbitrarily, and there is no global "now". Logical clocks provide ordering without relying on synchronized physical clocks, enabling correct reasoning about causality and event ordering.

## Physical Clock Problems

```mermaid
graph TD
    subgraph ClockDrift[Physical Clock Drift Problem]
        Node1[Node 1 Clock: 10:00:00.000]
        Node2[Node 2 Clock: 10:00:00.050]
        Node3[Node 3 Clock: 09:59:59.900]

        Problem[Problem:\nNode 3 event at 09:59:59.900\nappears to predate Node 1 event at 10:00:00.000\nbut in reality happened AFTER it\n\nLast-write-wins based on timestamp\nproduces incorrect results!]
        style Problem fill:#fee2e2,stroke:#dc2626
    end

    NTP[NTP synchronizes clocks\nbut accuracy is 1-100ms\nnot sufficient for ordering distributed events]
```

## Lamport Logical Clocks

```mermaid
sequenceDiagram
    participant P1 as Process 1
    participant P2 as Process 2
    participant P3 as Process 3

    Note over P1: time=1
    P1->>P1: Event a (send) time=1
    P1->>P2: Message with timestamp=1

    Note over P2: time=1
    P2->>P2: Event b time=2
    P2->>P2: Receive from P1: max(2,1)+1=3

    Note over P3: time=1
    P3->>P3: Event c time=2
    P3->>P1: Message with timestamp=2

    Note over P1: Receive from P3: max(1,2)+1=3
    P1->>P1: Event d time=4

    Note over P1,P3: Rule: receive(msg) → clock = max(local, msg.clock) + 1
    Note over P1,P3: If A happened-before B, then clock(A) < clock(B)
    Note over P1,P3: Converse is NOT true - concurrent events can have any order
```

## Vector Clocks

```mermaid
graph TD
    subgraph VectorClockExample[Vector Clock Example - 3 Nodes]
        Init[Initial state:\nN1: 0,0,0\nN2: 0,0,0\nN3: 0,0,0]

        E1[N1 local event\nN1: 1,0,0]
        E2[N2 local event\nN2: 0,1,0]
        E3[N1 sends to N2\nN1: 2,0,0\nN2 receives: max = 2,1,0 + N2 inc = 2,2,0]
        E4[N3 local event\nN3: 0,0,1]

        Init --> E1 & E2 & E4
        E1 & E2 --> E3

        Comparison[Comparing vectors:\nA happens-before B if A[i] <= B[i] for all i\nand A[i] < B[i] for at least one i\nConcurrent if neither A < B nor B < A]
    end
```

## Hybrid Logical Clocks (HLC)

```mermaid
graph LR
    subgraph HLC[Hybrid Logical Clock Design]
        Physical[Physical Time Component\nNTP-synchronized wall clock\nHuman-readable timestamps\nGood for TTL and time-based queries]
        Logical[Logical Counter Component\nIncremented on causally related events\nOrders events within same millisecond]
        Combined[HLC = max physical, received HLC\nplus logical counter\nPreserves causality\nApproximates physical time]
        Physical --> Combined
        Logical --> Combined
    end

    Usage[Used by:\nCockroachDB\nYugabyteDB\nMongoDBatlas\nFor distributed transactions\nwith time-based ordering]
    style Combined fill:#dcfce7,stroke:#16a34a,stroke-width:2px
```

## Key Concepts

- **Clock Drift**: Physical clocks on different machines drift over time and are not perfectly synchronized even with NTP (Network Time Protocol). NTP accuracy is typically 1-100ms, insufficient for ordering events in a distributed system where message delivery is measured in microseconds.

- **Lamport Logical Clocks**: A monotonically increasing integer counter attached to every event and message. Rules: (1) increment counter before each local event, (2) when sending a message, attach current counter, (3) when receiving a message, set counter to max(local, received) + 1. If event A happens-before event B, then L(A) < L(B). The converse is false — concurrent events can have any Lamport order.

- **Happens-Before Relation**: Event A happens-before event B if: (1) A and B are in the same process and A comes first, (2) A is the sending and B is the receiving of the same message, or (3) there is a transitive chain. Events that are not connected by happens-before are **concurrent** — their order is undefined and unimportant.

- **Vector Clocks**: An array of counters, one per process. Capture causal relationships completely — vector clocks can determine whether any two events are causally related or concurrent. A happens-before B iff A[i] ≤ B[i] for all i and A[j] < B[j] for some j. Used by Dynamo-style databases for conflict detection.

- **Version Vectors**: Similar to vector clocks but track the version of each replica's latest update. Used by Riak, Voldemort, and DynamoDB to detect conflicting writes in multi-master replication.

- **Hybrid Logical Clocks (HLC)**: Combines physical time (wall clock) with a logical counter. The physical component keeps HLC close to NTP time (for human-readable timestamps and TTL-based operations); the logical component provides causality ordering. Used by CockroachDB and YugabyteDB for distributed transaction timestamps.

- **TrueTime (Google Spanner)**: Google's GPS and atomic clock-based API that provides a time interval [earliest, latest] within which the current time falls, with a bounded uncertainty of a few milliseconds. Spanner uses this to implement external consistency (linearizability) globally by waiting out the uncertainty interval before committing.

## Trade-offs

| Clock Type | Causality Tracking | Space | Physical Time | Systems |
|-----------|------------------|-------|--------------|---------|
| Physical (NTP) | No | O(1) | Approximate | Logs, TTL |
| Lamport | Partial (no concurrent detection) | O(1) | No | Event ordering |
| Vector Clocks | Full | O(N) | No | Conflict detection |
| HLC | Full | O(1) | Approximate | Distributed DBs |
| TrueTime | Full | O(1) | Exact (bounded) | Spanner |

## When to Apply

- **Lamport clocks**: When you need total event ordering and don't need to distinguish concurrent events
- **Vector clocks**: When you need to detect causally unrelated (concurrent) operations — required for CRDT and conflict detection in eventually consistent systems
- **HLC**: When you need both causality and approximate real-time ordering in a distributed database
- **Physical time only**: Never for distributed event ordering — only for human-readable timestamps and approximate TTL-based expiry