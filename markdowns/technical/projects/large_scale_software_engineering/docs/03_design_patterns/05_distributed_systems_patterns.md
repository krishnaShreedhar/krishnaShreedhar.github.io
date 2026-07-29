---
title: "Distributed Systems Patterns"
subtitle: "Distributed systems patterns solve the unique challenges of building reliable, consistent, and resilient software across multiple processes and machines — where networks are unreliable, clocks are unsynchronized, and..."
category: technical
project: large_scale_software_engineering
project_title: "Large Scale Software Engineering"
date: 2025-10-11
reading_time: 4
tags:
  - large-scale-software-engineering
  - docs
author: "Shreedhar Kodate"
output: "blogs/technical/posts/large_scale_software_engineering/docs/03_design_patterns/05_distributed_systems_patterns.html"
---
Distributed systems patterns solve the unique challenges of building reliable, consistent, and resilient software across multiple processes and machines — where networks are unreliable, clocks are unsynchronized, and failures are partial and silent.

## Circuit Breaker Pattern

```mermaid
stateDiagram-v2
    [*] --> Closed: Initial state - requests pass through

    Closed --> Open: Failure threshold exceeded\n(e.g., 50% failures in 10s window)
    Open --> HalfOpen: Probe timeout expires\n(e.g., after 30s)
    HalfOpen --> Closed: Probe request succeeds
    HalfOpen --> Open: Probe request fails

    note right of Closed
        Requests flow normally
        Failure count monitored
    end note
    note right of Open
        Requests fail immediately
        No downstream calls made
        Fast failure to caller
    end note
    note right of HalfOpen
        Limited requests allowed
        Testing if downstream recovered
    end note
```

## Saga Pattern

```mermaid
sequenceDiagram
    participant Saga as Saga Orchestrator
    participant Order as Order Service
    participant Inventory as Inventory Service
    participant Payment as Payment Service
    participant Shipping as Shipping Service

    Saga->>Order: create_order
    Order-->>Saga: OrderCreated

    Saga->>Inventory: reserve_inventory
    Inventory-->>Saga: InventoryReserved

    Saga->>Payment: charge_payment
    Payment--xSaga: FAILURE: CardDeclined

    Note over Saga: Compensation begins

    Saga->>Inventory: release_reservation (compensate)
    Inventory-->>Saga: ReservationReleased

    Saga->>Order: cancel_order (compensate)
    Order-->>Saga: OrderCancelled
```

## Outbox Pattern

```mermaid
graph TD
    subgraph Service[Order Service - Single Transaction]
        AppLogic[Application Logic]
        DB[(Orders Table\nstatus: placed)]
        Outbox[(Outbox Table\nevent: OrderPlaced\nstatus: pending)]
        AppLogic -->|ACID transaction| DB
        AppLogic -->|same transaction| Outbox
    end

    Relay[Outbox Relay / CDC\ncontinuously polls outbox] -->|reads pending events| Outbox
    Relay -->|publishes| Broker[Message Broker\nKafka / RabbitMQ]
    Relay -->|marks published| Outbox
    Broker --> Consumers[Downstream Consumers]

    style Outbox fill:#fef3c7,stroke:#d97706,stroke-width:2px
    style Relay fill:#dcfce7,stroke:#16a34a
```

## Sidecar Pattern

```mermaid
graph TD
    subgraph Pod[Kubernetes Pod]
        App[Application Container\nbusiness logic\nno networking concerns]
        Sidecar[Sidecar Container\nEnvoy Proxy]
        App <-->|localhost| Sidecar
    end

    subgraph SidecarCapabilities[Sidecar Provides]
        mTLS[Mutual TLS]
        Retry[Retry / Timeout]
        Circuit[Circuit Breaking]
        Trace[Distributed Tracing]
        Metrics[Metrics Collection]
    end

    Sidecar --> mTLS & Retry & Circuit & Trace & Metrics
    Sidecar <-->|network| OtherServices[Other Services]

    style Sidecar fill:#fef3c7,stroke:#d97706,stroke-width:2px
    style App fill:#dcfce7,stroke:#16a34a
```

## Leader Election Pattern

```mermaid
graph TD
    subgraph Nodes[Distributed Nodes]
        N1[Node 1]
        N2[Node 2 - LEADER]
        N3[Node 3]
        N4[Node 4]
    end

    CoordSvc[Coordination Service\nZooKeeper / etcd]

    N1 & N2 & N3 & N4 -->|try to acquire leader lock| CoordSvc
    CoordSvc -->|grants ephemeral lock to| N2

    N2 -->|performs leader duties| Work[Singleton Work\nscheduled jobs\npartition assignment\nconsensus decisions]

    N2 -->|heartbeat| CoordSvc
    CoordSvc -->|lock expires if heartbeat stops| NewElection[Re-election\nremaining nodes compete]

    style N2 fill:#fef3c7,stroke:#d97706,stroke-width:3px
```

## Key Concepts

- **Circuit Breaker**: Wraps a remote call and monitors for failures. When failures exceed a threshold, the circuit "opens" and subsequent calls fail immediately without attempting the downstream call. After a timeout, the circuit enters "half-open" and probes whether the downstream has recovered. Prevents cascading failures — a slow or failing downstream service drains the calling service's thread pool.

- **Saga**: A sequence of local transactions, each publishing an event or message that triggers the next step. If any step fails, the saga executes compensating transactions (undo actions) for all preceding steps. Replaces distributed ACID transactions in microservices with eventual consistency. Two flavours: choreography (event-driven, decentralized) and orchestration (coordinator controls flow).

- **Outbox Pattern**: Solves the "dual write" problem — the impossibility of atomically writing to a database and publishing a message. Instead, the application writes both the business data and the event to the same database in a single ACID transaction. A separate relay process (or CDC pipeline) reads the outbox table and publishes events. Guarantees at-least-once event delivery.

- **Sidecar**: Deploys a helper process alongside the application in the same execution unit (pod, VM). The sidecar handles cross-cutting concerns (service mesh proxying, logging, configuration) without modifying the application. The application uses the sidecar via localhost, treating it as a transparent intermediary.

- **Ambassador**: A specialized sidecar that acts as a proxy to external services, abstracting connection details, retry logic, and circuit breaking from the application. The application talks to the ambassador on localhost; the ambassador handles the complexities of reaching the actual service.

- **Bulkhead**: Isolates different parts of a system into pools so that failure in one pool does not exhaust resources for another. Named after ship compartments that contain flooding. Implemented as separate thread pools per downstream dependency — if one dependency is slow, only its pool fills, not the entire service's thread pool.

- **Leader Election**: Ensures only one node in a cluster performs a particular task at a time (singleton work). Uses a distributed coordination service (ZooKeeper, etcd, Consul) where nodes compete for an ephemeral lock. If the leader fails to renew its heartbeat, the lock expires and a new election occurs.

## Trade-offs

| Pattern | Problem Solved | Complexity Introduced |
|---------|---------------|----------------------|
| Circuit Breaker | Cascading failures | State management, tuning thresholds |
| Saga | Cross-service consistency | Compensation logic, partial failure |
| Outbox | Dual write atomicity | Relay infrastructure, eventual delivery |
| Sidecar | Cross-cutting concerns without code change | More containers, sidecar lifecycle |
| Bulkhead | Resource isolation between dependencies | More pools to configure and monitor |
| Leader Election | Singleton work in distributed cluster | Coordination service dependency |

## When to Use

- **Circuit Breaker**: Any synchronous call to an external service or database
- **Saga**: Multi-step business processes spanning multiple services where all-or-nothing semantics are required
- **Outbox**: Any microservice that must publish events and persist data atomically
- **Sidecar**: When adding cross-cutting capabilities (observability, security) to services without modifying them
- **Bulkhead**: When a service depends on multiple external systems with different reliability profiles