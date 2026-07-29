---
title: "Distributed Coordination"
subtitle: "Distributed coordination solves the problem of multiple nodes agreeing on shared state and coordinating actions — tasks that are trivial with a single thread but require careful design in distributed systems where..."
category: technical
project: large_scale_software_engineering
project_title: "Large Scale Software Engineering"
date: 2025-06-07
reading_time: 4
tags:
  - large-scale-software-engineering
  - docs
author: "Shreedhar Kodate"
output: "blogs/technical/posts/large_scale_software_engineering/docs/12_distributed_systems_theory/05_distributed_coordination.html"
---
Distributed coordination solves the problem of multiple nodes agreeing on shared state and coordinating actions — tasks that are trivial with a single thread but require careful design in distributed systems where nodes can fail independently.

## Distributed Locking

```mermaid
sequenceDiagram
    participant SvcA as Service A
    participant SvcB as Service B
    participant Redis as Redis (Lock Service)

    SvcA->>Redis: SET lock:resource1 svcA-uuid NX PX 30000
    Redis-->>SvcA: OK (lock acquired)

    SvcB->>Redis: SET lock:resource1 svcB-uuid NX PX 30000
    Redis-->>SvcB: nil (lock NOT acquired)

    SvcA->>SvcA: Do exclusive work...
    SvcA->>Redis: DEL lock:resource1 (only if value = svcA-uuid)
    Redis-->>SvcA: OK (lock released)

    SvcB->>Redis: SET lock:resource1 svcB-uuid NX PX 30000
    Redis-->>SvcB: OK (lock acquired now)

    Note over SvcA,Redis: Key insight: NX = set only if not exists\nPX = expire in milliseconds\nValue is unique ID to prevent releasing another's lock
```

## Distributed Lock Failure Modes

```mermaid
graph TD
    subgraph LockFailures[Distributed Lock Failure Scenarios]
        F1[Lock holder crashes\nbefore releasing\nMitigation: TTL ensures\nlock auto-expires]

        F2[Lock holder is slow\nTTL expires\nAnother acquires lock\nBoth think they hold lock!\nMitigation: fencing tokens\nor extend lock before expiry]

        F3[Redis node fails\nafter granting lock\nbut before replication\nMitigation: Redlock algorithm\nrequires majority of N Redis nodes]

        F4[Network partition between\nlock holder and lock service\nholder can no longer extend\nMitigation: accept that exclusive\naccess may be briefly broken]
    end

    style F2 fill:#fee2e2,stroke:#dc2626,stroke-width:2px
```

## Fencing Tokens

```mermaid
sequenceDiagram
    participant Client
    participant LockSvc as Lock Service
    participant Storage

    Client->>LockSvc: Acquire lock
    LockSvc-->>Client: Lock granted, fencing token: 33

    Client->>Storage: Write data, token: 33
    Storage->>Storage: Accept (33 > last seen token)

    Note over Client: Client GC pause - lock may expire
    Note over LockSvc: Lock expires, new client acquires

    Note over Client: Client resumes
    Client->>Storage: Write data, token: 33
    Storage->>Storage: REJECT (lock service now at token 34)
    Storage-->>Client: Rejected - stale token

    Note over Storage: Monotonically increasing tokens\nprevent stale lock holders from corrupting data
```

## Service Registry and Discovery

```mermaid
graph TD
    subgraph Registry[Service Registry - Consul / etcd]
        ServiceReg[Service Registry\nstores: service name\ninstance IP and port\nhealth status\nmetadata]
    end

    subgraph Registration[Service Registration]
        SvcA[Service A instance\non startup: register self\nperiodic: send heartbeat\non shutdown: deregister]
        SvcA -->|register + heartbeat| ServiceReg
    end

    subgraph Discovery[Service Discovery]
        SvcB[Service B] -->|lookup: service-a| ServiceReg
        ServiceReg -->|return healthy instances| SvcB
        SvcB -->|pick one, send request| SvcA
    end

    HealthCheck[Health Checking\nConsul probes each instance\nRemoves unhealthy instances\nfrom registry]
    ServiceReg --> HealthCheck
```

## Key Concepts

- **Distributed Lock**: A mutual exclusion mechanism that spans multiple processes. Implemented using a shared coordination service (Redis, ZooKeeper, etcd). Critical properties: (1) Mutual exclusion — only one holder at a time, (2) No deadlock — lock expires automatically via TTL if holder crashes, (3) Fault tolerance — survives failure of non-majority nodes.

- **Fencing Tokens**: A monotonically increasing token issued with each lock grant. The storage system accepts only writes with tokens greater than the last seen token. Prevents a slow/GC-paused lock holder from writing after the lock has expired and been re-granted to another client. Solves the "process resumes after pause thinks it still holds the lock" problem.

- **Redlock Algorithm**: A distributed lock algorithm using multiple independent Redis nodes. A client acquires the lock on a majority (N/2+1) of nodes. If it succeeds within a validity time, the lock is held. Provides stronger guarantees than single-node Redis lock but has known correctness concerns under certain timing scenarios (Martin Kleppmann's critique).

- **Service Registry**: A centralized directory of available service instances with their network addresses and health status. Enables dynamic service discovery — services register on startup and deregister on shutdown. Health checks remove unhealthy instances automatically.

- **Gossip Protocol**: A method for nodes to exchange state information peer-to-peer (like gossip spreading through a social network). Each node periodically selects random peers and exchanges information. Eventually, all nodes converge to the same state. Used by Cassandra, Consul, and Redis Cluster for cluster membership and failure detection.

- **Phi Accrual Failure Detector**: A probabilistic failure detection algorithm that outputs a continuously growing suspicion value (phi) rather than a binary alive/dead judgment. Allows applications to tune their failure detection sensitivity. Used by Akka and Cassandra.

- **Distributed Coordination Services**: ZooKeeper, etcd, and Consul provide primitives (leader election, distributed locks, service discovery, configuration management) built on top of consensus algorithms. These are the coordination backbone of distributed systems.

## Trade-offs

| Mechanism | Correctness | Availability | Complexity |
|-----------|------------|-------------|-----------|
| Single Redis lock | Good (single point failure) | Medium | Low |
| Redlock | Better | Medium | Medium |
| ZooKeeper lock | Strong | Depends on quorum | Medium |
| etcd lock | Strong | Depends on quorum | Medium |
| Gossip (no lock) | Eventual | High | Medium |

## When to Apply

- **Distributed locks**: Any resource that must not be concurrently modified (scheduled job, file write, database record) across multiple processes
- **Always use fencing tokens**: When the locked resource supports a conditional write mechanism
- **Service registry**: All microservices deployments — enables dynamic discovery and health-based routing
- **ZooKeeper/etcd**: When strong consistency is required for coordination — leader election, config management