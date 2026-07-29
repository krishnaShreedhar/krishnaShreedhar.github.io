---
title: "Kafka Event-Driven Architecture — Flow Diagrams"
subtitle: "flowchart TD subgraph EventStore [\"EventStore (append-only log)\"] E1[\"v1: UserRegistered\n{email: alice@example.com}\"] E2[\"v2: EmailVerified\n{verified_by: email_link}\"] E3[\"v3: ProfileUpdated\n{display_name: Alice..."
category: technical
project: kafka_event_driven
project_title: "Kafka Event-Driven Architecture — Demonstration Project"
date: 2025-01-24
reading_time: 4
tags:
  - kafka-event-driven
  - docs
author: "Shreedhar Kodate"
output: "blogs/technical/posts/kafka_event_driven/docs/flow_diagrams.html"
---
## 1. Event Sourcing — State Reconstruction

```mermaid
flowchart TD
    subgraph EventStore ["EventStore (append-only log)"]
        E1["v1: UserRegistered\n{email: alice@example.com}"]
        E2["v2: EmailVerified\n{verified_by: email_link}"]
        E3["v3: ProfileUpdated\n{display_name: Alice Smith}"]
        E4["v4: ProfileUpdated\n{bio: Senior ML engineer}"]
        E1 --> E2 --> E3 --> E4
    end

    subgraph Replay ["replay_state(aggregate_id)"]
        S0["initial_state = {}"]
        S1["state after v1:\n{email: alice, status: pending}"]
        S2["state after v2:\n{email_verified: true, status: active}"]
        S3["state after v3:\n{display_name: Alice Smith}"]
        S4["state after v4:\n{bio: Senior ML engineer}"]
        S0 --> S1 --> S2 --> S3 --> S4
    end

    E1 -.apply handler.-> S1
    E2 -.apply handler.-> S2
    E3 -.apply handler.-> S3
    E4 -.apply handler.-> S4

    subgraph PointInTime ["Point-in-Time Restore"]
        PIT["replay_state(up_to_version=2)\nreturns state after EmailVerified"]
    end

    E2 -.-> PIT
```

---

## 2. Outbox Relay — Guaranteed At-Least-Once Delivery

```mermaid
flowchart LR
    subgraph Application ["Application (single transaction)"]
        DomainWrite["Write domain record\ne.g. UPDATE orders SET status='created'"]
        OutboxWrite["INSERT INTO outbox\n(topic, key, value, published=false)"]
        DomainWrite -.same TX.-> OutboxWrite
    end

    subgraph OutboxRelay ["OutboxRelay (background thread)"]
        Poll["get_unpublished(limit=100)"]
        Publish["producer.produce(topic, key, value)"]
        Mark["mark_published(entry_id)"]
        Poll --> Publish --> Mark
        Mark -.loop every poll_interval_s.-> Poll
    end

    subgraph Kafka
        Topic["Kafka Topic"]
    end

    OutboxWrite --> Poll
    Publish --> Topic

    style DomainWrite fill:#d4edda
    style OutboxWrite fill:#d4edda
    style Publish fill:#cce5ff
    style Topic fill:#cce5ff
```

**Crash scenarios handled:**

| Crash point | Outcome |
|-------------|---------|
| Before OutboxWrite | Domain TX rolled back → no event |
| After OutboxWrite, before Publish | Relay picks up on restart → event published |
| After Publish, before Mark | Relay republishes → at-least-once (idempotent consumers handle dedup) |
| After Mark | Happy path completed |

---

## 3. DLQ Retry with Exponential Backoff

```mermaid
flowchart TD
    Consume["Consumer.poll()"] --> Process{"Process\nMessage"}
    Process -- success --> Commit["commit()"]
    Process -- failure --> DLQ["route_to_dlq(msg, error)"]
    Commit --> NextMsg["next message"]

    DLQ --> AddToDLQ["DLQHandler.add_to_dlq()\nretry_count=0\nnext_retry_at = now + 2^0"]

    subgraph DLQLoop ["DLQHandler.process_dlq() — periodic"]
        Ready{"is_ready_for_retry?"}
        Retry["Re-process message"]
        Success["Remove from DLQ ✓"]
        ReQueue["Increment retry_count\nnext_retry_at = now + 2^retry_count\nRe-queue"]
        Quarantine["Quarantine\n(retry_count >= max_retries)"]

        Ready -- Yes --> Retry
        Ready -- No --> SkipForNow["Skip (still in backoff)"]
        Retry -- success --> Success
        Retry -- failure, retries remaining --> ReQueue
        Retry -- failure, max retries hit --> Quarantine
    end

    AddToDLQ --> Ready

    style DLQ fill:#fff3cd
    style Quarantine fill:#f8d7da
    style Success fill:#d4edda
```

**Backoff schedule (backoff_base=2.0, max_retries=3):**

| Retry | Delay before retry | Total wait |
|-------|--------------------|-----------|
| 0 → 1 | 2^0 = 1s | 1s |
| 1 → 2 | 2^1 = 2s | 3s |
| 2 → 3 | 2^2 = 4s | 7s |
| 3     | QUARANTINE | — |

---

## 4. ML Inference Pipeline

```mermaid
flowchart LR
    subgraph Events ["Event Stream"]
        UE["Topic: user_events\nclick / purchase / view"]
    end

    subgraph FeaturePipeline
        FP_C["MockKafkaConsumer\ngroup: feature-pipeline"]
        FP_U["FeatureStore.update()\nclick_count, purchase_count,\nsession_count, last_event_ts"]
        FP_FS["FeatureStore\n(in-memory dict)"]
        FP_C --> FP_U --> FP_FS
    end

    subgraph InferencePipeline
        IP_C["MockKafkaConsumer\ngroup: inference-pipeline"]
        IP_F["FeatureStore.get(user_id)"]
        IP_M["MockModel.predict(features)\nscore = sigmoid(w·x + b)"]
        IP_P["MockKafkaProducer\nproduce(predictions)"]
        IP_C --> IP_F --> IP_M --> IP_P
    end

    subgraph MonitoringPipeline
        MP_C["MockKafkaConsumer\ngroup: monitoring-pipeline"]
        MP_PSI["DriftDetector.compute_psi()\nreference vs current scores"]
        MP_A["PSI Alert if PSI ≥ 0.10"]
        MP_C --> MP_PSI --> MP_A
    end

    UE --> FP_C
    UE --> IP_C
    FP_FS --> IP_F
    IP_P --> MP_C

    style UE fill:#e2f0fb
    style MP_A fill:#fff3cd
```

---

## 5. A/B Model Routing Decision Flow

```mermaid
flowchart TD
    Event["Incoming Event\n{user_id: 'user-042'}"] --> Hash

    subgraph ABRouter
        Hash["MD5(user_id.encode())\ntake first 4 bytes as uint32\nbucket = uint32 % 100"]
        Compare{"bucket < model_v2_traffic_pct\n(e.g. 20)?"}
        Hash --> Compare
    end

    Compare -- Yes: bucket 0–19 --> V2["Model v2\n(challenger, 20%)"]
    Compare -- No: bucket 20–99 --> V1["Model v1\n(control, 80%)"]

    V1 --> Score["Prediction Score"]
    V2 --> Score

    Score --> Produce["produce(predictions, user_id, score, model_id)"]

    subgraph Properties
        P1["Deterministic:\nsame user_id → same model every time"]
        P2["Configurable:\nchange model_v2_traffic_pct in config.yaml"]
        P3["Stateless:\nno session storage or sticky routing required"]
    end
```

---

## 6. Consumer Group and Offset Flow

```mermaid
flowchart LR
    subgraph Topic ["Topic: user_events (4 partitions)"]
        P0["P0: [0,1,2,3,4]\nHWM=5"]
        P1["P1: [0,1,2]\nHWM=3"]
        P2["P2: [0,1,2,3]\nHWM=4"]
        P3["P3: [0,1]\nHWM=2"]
    end

    subgraph ConsumerGroup ["Consumer Group: feature-pipeline"]
        CG_O["Committed offsets:\nP0→3, P1→3, P2→2, P3→0"]
        LAG["Lag:\nP0=2, P1=0, P2=2, P3=2\nTotal=6"]
        CG_O --> LAG
    end

    subgraph PollLoop
        Poll["poll() → round-robin\nacross assigned partitions"]
        Process["process message"]
        Commit["commit() → advance\ncommitted offset"]
        Poll --> Process --> Commit --> Poll
    end

    P0 --> CG_O
    P1 --> CG_O
    P2 --> CG_O
    P3 --> CG_O

    style LAG fill:#fff3cd
```

---

## 7. PSI Drift Detection Computation

```mermaid
flowchart TD
    Ref["Reference Scores\n(training distribution)\nn=200 samples"]
    Cur["Current Scores\n(production window)\nn=100 samples"]

    Ref --> Bin1["Bin into 10 equal buckets\n[0.0, 0.1), [0.1, 0.2), ..., [0.9, 1.0]"]
    Cur --> Bin2["Bin into same 10 buckets"]

    Bin1 --> Pct1["ref_pct_i = count_i / total_ref"]
    Bin2 --> Pct2["cur_pct_i = count_i / total_cur"]

    Pct1 --> PSI_Formula["PSI = Σ (cur_pct_i - ref_pct_i)\n     × ln(cur_pct_i / ref_pct_i)"]
    Pct2 --> PSI_Formula

    PSI_Formula --> Classify{"PSI value?"}
    Classify -- "< 0.10" --> Stable["Stable ✓\nNo action needed"]
    Classify -- "0.10 – 0.25" --> Moderate["Moderate Drift ⚠\nInvestigate"]
    Classify -- "> 0.25" --> Major["Major Drift ✗\nRetrain model"]

    style Stable fill:#d4edda
    style Moderate fill:#fff3cd
    style Major fill:#f8d7da
```

---

## 8. Windowing — Tumbling vs Session

```mermaid
gantt
    title Event Timeline: Tumbling (300s) vs Session (30s gap)
    dateFormat X
    axisFormat %s

    section User A
    Tumbling Window 0   :0, 300
    Tumbling Window 1   :300, 300

    section Session Windows (User A)
    Session 1 (3 events)  :active, 10, 60
    Session 2 (2 events)  :active, 200, 40
    Session 3 (4 events)  :active, 350, 100
```

**Key differences:**

| Property | Tumbling Window | Session Window |
|----------|----------------|----------------|
| Duration | Fixed (e.g. 300s) | Variable (activity-based) |
| Boundaries | Clock-aligned | Inactivity-driven |
| Use case | Time-series metrics | User behaviour grouping |
| Memory | O(window_size) | O(active_sessions) |
| Ordering | Requires event timestamps | Requires event timestamps |