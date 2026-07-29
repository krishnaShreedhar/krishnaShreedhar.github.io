---
title: "Consensus Algorithms"
subtitle: "Consensus algorithms enable a distributed cluster of nodes to agree on a single value or sequence of values, even in the presence of node failures. Consensus is the foundation of replicated state machines,..."
category: technical
project: large_scale_software_engineering
project_title: "Large Scale Software Engineering"
date: 2025-03-18
reading_time: 3
tags:
  - large-scale-software-engineering
  - docs
author: "Shreedhar Kodate"
output: "blogs/technical/posts/large_scale_software_engineering/docs/12_distributed_systems_theory/02_consensus_algorithms.html"
---
Consensus algorithms enable a distributed cluster of nodes to agree on a single value or sequence of values, even in the presence of node failures. Consensus is the foundation of replicated state machines, distributed locks, leader election, and linearizable databases.

## Raft Consensus Overview

```mermaid
stateDiagram-v2
    [*] --> Follower: Node starts

    Follower --> Candidate: Election timeout expires\nno heartbeat from leader
    Candidate --> Leader: Receives majority votes
    Candidate --> Follower: Another candidate wins\nor split vote
    Leader --> Follower: Higher term discovered\nor network partition heals

    note right of Follower
        Receives log entries from leader
        Votes for candidates
        Resets election timer on heartbeat
    end note

    note right of Candidate
        Increments term
        Votes for self
        Requests votes from peers
    end note

    note right of Leader
        Accepts client requests
        Replicates log entries
        Sends heartbeats
    end note
```

## Raft Log Replication

```mermaid
sequenceDiagram
    participant Client
    participant Leader
    participant Follower1
    participant Follower2

    Client->>Leader: Write: Set x=5

    Leader->>Leader: Append to log\nEntry: term=3, index=7, cmd=set_x_5

    Leader->>Follower1: AppendEntries(term=3, entry=set_x_5)
    Leader->>Follower2: AppendEntries(term=3, entry=set_x_5)

    Follower1-->>Leader: OK
    Follower2-->>Leader: OK

    Note over Leader: Majority achieved (2 of 3 nodes)
    Leader->>Leader: Commit entry
    Leader->>Follower1: Commit entry 7
    Leader->>Follower2: Commit entry 7

    Leader-->>Client: Success: x=5 committed
```

## Paxos Overview

```mermaid
sequenceDiagram
    participant Proposer
    participant Acceptor1
    participant Acceptor2
    participant Acceptor3

    Note over Proposer: Phase 1: Prepare
    Proposer->>Acceptor1: Prepare(n=5)
    Proposer->>Acceptor2: Prepare(n=5)
    Proposer->>Acceptor3: Prepare(n=5)

    Acceptor1-->>Proposer: Promise(n=5, no prior accepted)
    Acceptor2-->>Proposer: Promise(n=5, no prior accepted)
    Acceptor3-->>Proposer: Promise(n=5, no prior accepted)

    Note over Proposer: Phase 2: Accept
    Proposer->>Acceptor1: Accept(n=5, v=order123)
    Proposer->>Acceptor2: Accept(n=5, v=order123)
    Proposer->>Acceptor3: Accept(n=5, v=order123)

    Acceptor1-->>Proposer: Accepted
    Acceptor2-->>Proposer: Accepted
    Note over Proposer: Majority: value committed
```

## Key Concepts

- **Consensus Problem**: Given a set of N nodes that may fail independently, reach agreement on a single value. Requirements: validity (agreed value must have been proposed), agreement (all non-faulty nodes agree on the same value), termination (all non-faulty nodes eventually decide).

- **Raft**: A consensus algorithm designed for understandability, providing the same safety guarantees as Paxos with a simpler conceptual model. Raft decomposes consensus into three problems: leader election, log replication, and safety. A strong leader replicates all entries to followers; entries are committed once a majority have acknowledged them.

- **Leader Election in Raft**: Each node has an election timeout (random 150-300ms). If a follower doesn't receive a heartbeat before its timeout, it becomes a candidate and requests votes. A candidate wins if it receives votes from a majority of nodes. Randomized timeouts prevent split votes from persisting.

- **Raft Log Commitment**: A log entry is committed when the leader has replicated it to a majority (quorum) of nodes. Only committed entries are applied to the state machine. A crash before commitment means the entry may be lost — this is correct behaviour.

- **Paxos**: The original consensus algorithm (Lamport, 1989), proven correct but notoriously difficult to understand and implement. Multi-Paxos extends single-decree Paxos to a replicated log. Most practical Paxos implementations (Google Chubby) are closer to the Paxos Made Live paper than the original.

- **Quorum**: A majority of nodes (N/2 + 1). Writes must be acknowledged by a quorum before being considered committed. Reads must query a quorum to guarantee seeing the latest committed write. A quorum of write nodes and a quorum of read nodes are guaranteed to overlap, ensuring reads see committed writes.

- **Byzantine Fault Tolerance (PBFT)**: Consensus in the presence of malicious nodes (Byzantine faults). Requires 3f+1 nodes to tolerate f Byzantine nodes. Used in permissioned blockchains and high-security coordination systems. Orders of magnitude more expensive than crash-failure-only consensus.

## Trade-offs

| Algorithm | Safety | Understandability | Performance | Byzantine |
|-----------|--------|-------------------|-------------|-----------|
| Raft | Yes | High | Good | No |
| Multi-Paxos | Yes | Low | Good | No |
| PBFT | Yes | Medium | Poor | Yes |
| Viewstamped Replication | Yes | Medium | Good | No |

## When to Apply

- Raft is implemented in: etcd (Kubernetes), CockroachDB, TiKV, Consul
- Use these systems when you need linearizable distributed state, leader election, or distributed locks
- Do not implement consensus algorithms yourself — use battle-tested implementations