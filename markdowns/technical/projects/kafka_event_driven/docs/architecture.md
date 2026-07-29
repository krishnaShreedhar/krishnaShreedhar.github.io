---
title: "Kafka Event-Driven Architecture — System Architecture"
subtitle: "This project demonstrates Kafka and event-driven architecture concepts using a fully in-memory mock broker.  The mock broker faithfully reproduces the partition model, consumer group offsets, and lag mechanics of a..."
category: technical
project: kafka_event_driven
project_title: "Kafka Event-Driven Architecture — Demonstration Project"
date: 2025-04-02
reading_time: 4
tags:
  - kafka-event-driven
  - docs
author: "Shreedhar Kodate"
output: "blogs/technical/posts/kafka_event_driven/docs/architecture.html"
---
## Overview

This project demonstrates Kafka and event-driven architecture concepts using
a fully in-memory mock broker.  The mock broker faithfully reproduces the
partition model, consumer group offsets, and lag mechanics of a real Apache
Kafka cluster, allowing all patterns to run without an external service.

---

## Core Kafka Concepts

### Topics, Partitions, and Replication

A Kafka **topic** is a named, ordered, immutable log.  Each topic is divided
into **partitions** — the unit of parallelism and ordering.  Within a
partition, messages are totally ordered by offset.  A **replication factor**
determines how many broker replicas store each partition.

**In-Sync Replicas (ISR)** is the set of replicas that are fully caught up to
the leader.  `acks=all` means the producer waits for all ISR members to
acknowledge the write before declaring success, giving the strongest durability
guarantee.

### Delivery Semantics

| Semantic        | Producer config              | Consumer behaviour        |
|-----------------|------------------------------|---------------------------|
| At-most-once    | `acks=0`, fire-and-forget    | Auto-commit before process|
| At-least-once   | `acks=all` + retries         | Manual commit after process|
| Exactly-once    | Idempotent + transactions    | Transactional read-process|

This project implements **at-least-once** delivery on the consumer side
(manual commit after processing) combined with DLQ routing for failed messages.

### Offset Management

```
Partition log:
  offset: 0  1  2  3  4  5  6  7  8  9
              ↑                    ↑
         consumer group        high-water
         committed offset      mark (LEO)

lag = high-water-mark − committed-offset = 9 − 2 = 7
```

Consumer groups maintain committed offsets per (topic, partition).  The
`MockKafkaBroker` stores these in `_consumer_offsets[group_id][topic][partition]`.

---

## System Architecture Diagram

```mermaid
graph TB
    subgraph Producers
        P1[OrderService Producer]
        P2[UserService Producer]
        P3[OutboxRelay Producer]
    end

    subgraph Kafka Broker ["MockKafkaBroker (in-memory)"]
        T1["Topic: user_events\nPartitions: 4\nRetention: 7d"]
        T2["Topic: predictions\nPartitions: 4"]
        T3["Topic: monitoring\nPartitions: 2"]
        T4["Topic: dlq\nPartitions: 1"]
    end

    subgraph ConsumerGroups ["Consumer Groups"]
        CG1["feature-pipeline\n(FeaturePipeline)"]
        CG2["inference-pipeline\n(InferencePipeline)"]
        CG3["monitoring-pipeline\n(MonitoringPipeline)"]
        CG4["dlq-handler\n(DLQHandler)"]
    end

    subgraph Storage
        FS[(FeatureStore\nin-memory)]
        ES[(EventStore\nappend-only log)]
        OB[(OutboxTable\ndomain DB)]
    end

    P1 --> T1
    P2 --> T1
    P3 --> T1
    P3 --> T2

    T1 --> CG1
    T1 --> CG2
    T1 --> T4

    T2 --> CG3
    T4 --> CG4

    CG1 --> FS
    CG2 --> T2
    CG3 --> T3

    style T4 fill:#ff9999
    style CG4 fill:#ff9999
```

---

## Component Interaction

```mermaid
sequenceDiagram
    participant App as Application
    participant OB as OutboxTable
    participant Relay as OutboxRelay
    participant Kafka as MockKafkaBroker
    participant FP as FeaturePipeline
    participant FS as FeatureStore
    participant IP as InferencePipeline
    participant MP as MonitoringPipeline

    App->>OB: atomic_write(domain_record, outbox_entry)
    Note over OB: Single "transaction"

    loop Every poll_interval_s
        Relay->>OB: get_unpublished()
        OB-->>Relay: [entry1, entry2, ...]
        Relay->>Kafka: produce(topic, key, value)
        Relay->>OB: mark_published(entry_id)
    end

    loop Poll loop
        FP->>Kafka: poll(user_events)
        Kafka-->>FP: MockMessage
        FP->>FS: update(user_id, event)
        FP->>Kafka: commit()
    end

    loop Poll loop
        IP->>Kafka: poll(user_events)
        Kafka-->>IP: MockMessage
        IP->>FS: get(user_id)
        FS-->>IP: UserFeatures
        IP->>Kafka: produce(predictions, score)
        IP->>Kafka: commit()
    end

    loop Every N predictions
        MP->>Kafka: poll(predictions)
        Kafka-->>MP: prediction record
        MP->>MP: compute_psi(reference, current)
        MP->>Kafka: produce(monitoring, psi_event)
    end
```

---

## Saga Pattern — Sequence Diagram

```mermaid
sequenceDiagram
    participant Client
    participant Saga as OrderSaga
    participant Kafka as MockKafkaBroker

    Client->>Saga: execute(order_context)

    rect rgb(200, 255, 200)
        Note over Saga: Forward pass
        Saga->>Saga: PlaceOrder (local tx)
        Saga->>Kafka: produce(OrderPlaced)
        Saga->>Saga: ReserveInventory (local tx)
        Saga->>Kafka: produce(InventoryReserved)
        Saga->>Saga: ProcessPayment (local tx)
        Saga->>Kafka: produce(PaymentProcessed)
        Saga->>Kafka: produce(OrderCompleted)
    end

    Saga-->>Client: SagaExecution(status=COMPLETED)

    Note over Client,Kafka: ---- Failure scenario ----

    Client->>Saga: execute(order_context_with_bad_item)

    rect rgb(255, 220, 220)
        Note over Saga: Forward pass — fails at step 2
        Saga->>Saga: PlaceOrder ✓
        Saga->>Kafka: produce(OrderPlaced)
        Saga->>Saga: ReserveInventory ✗ FAIL
    end

    rect rgb(255, 240, 200)
        Note over Saga: Compensation pass (reverse)
        Saga->>Saga: CancelOrder (compensate PlaceOrder)
        Saga->>Kafka: produce(OrderCancelled)
        Saga->>Kafka: produce(OrderFailed)
    end

    Saga-->>Client: SagaExecution(status=FAILED)
```

---

## DLQ Lifecycle

```mermaid
stateDiagram-v2
    [*] --> MainTopic : produce

    MainTopic --> Consumer : poll
    Consumer --> Processing : message received
    Processing --> Commit : success
    Processing --> DLQ : failure (route_to_dlq)
    Commit --> [*]

    DLQ --> RetryScheduled : add_to_dlq (backoff)
    RetryScheduled --> RetryAttempt : next_retry_at reached

    RetryAttempt --> Commit : success
    RetryAttempt --> RetryScheduled : failure, retry_count < max_retries
    RetryAttempt --> Quarantine : retry_count >= max_retries

    Quarantine --> ManualIntervention : human review required
```

---

## A/B Routing Decision

```mermaid
graph LR
    Event[User Event] --> Hash["hash(user_id) % 100"]
    Hash --> Bucket{bucket < v2_pct?}
    Bucket -- Yes --> ModelV2["Model v2 (20% traffic)"]
    Bucket -- No --> ModelV1["Model v1 (80% traffic)"]
    ModelV2 --> Score[Prediction Score]
    ModelV1 --> Score
    Score --> PredictionsTopic["Topic: predictions"]
```

---

## Configuration Reference

All parameters are driven by `config.yaml`.  Key sections:

| Section | Key | Description |
|---------|-----|-------------|
| `kafka` | `use_mock` | `true` = MockKafkaBroker, `false` = real Kafka |
| `producer` | `acks` | Delivery acknowledgement level (`all` = strongest) |
| `producer` | `enable_idempotence` | Prevents duplicate messages on retry |
| `consumer` | `enable_auto_commit` | Always `false` — manual commit only |
| `consumer` | `auto_offset_reset` | `earliest` = start from oldest message |
| `topics` | `user_events.partitions` | Parallelism level (number of consumer threads) |
| `event_patterns` | `dlq_max_retries` | Maximum retries before quarantine |
| `event_patterns` | `dlq_backoff_base_s` | Exponential backoff base (seconds) |
| `ml_pipeline` | `model_v2_traffic_pct` | % of users routed to challenger model |
| `streaming` | `lag_alert_threshold` | Consumer lag count triggering a warning |