# Event-Driven Architecture

Event-Driven Architecture (EDA) structures a system around the production, detection, and reaction to events. Components communicate by publishing and subscribing to events rather than invoking each other directly, enabling loose temporal and spatial coupling between producers and consumers.

## Architecture Diagrams

### Event-Driven System Overview

```mermaid
graph TD
    subgraph Producers
        OrderSvc[Order Service]
        UserSvc[User Service]
        PaySvc[Payment Service]
    end

    subgraph Broker[Event Broker - Kafka]
        OT[orders topic]
        UT[users topic]
        PT[payments topic]
    end

    subgraph Consumers
        NotifSvc[Notification Service]
        AnalyticsSvc[Analytics Service]
        FraudSvc[Fraud Detection]
        SearchSvc[Search Indexer]
    end

    OrderSvc -->|OrderPlaced event| OT
    UserSvc -->|UserRegistered event| UT
    PaySvc -->|PaymentProcessed event| PT

    OT --> NotifSvc
    OT --> AnalyticsSvc
    OT --> FraudSvc
    PT --> NotifSvc
    UT --> SearchSvc
    PT --> AnalyticsSvc

    style Broker fill:#fef3c7,stroke:#d97706,stroke-width:2px
```

### Event Sourcing Pattern

```mermaid
graph LR
    subgraph Commands
        C1[PlaceOrder]
        C2[CancelOrder]
        C3[UpdateQuantity]
    end

    CH[Command Handler] -->|validates & processes| ES[Event Store\nAppend-Only Log]

    ES --> E1[OrderPlaced v1]
    ES --> E2[OrderPlaced v2]
    ES --> E3[OrderCancelled v1]

    ES -->|replay events| PM[Projection Manager]
    PM --> RM1[Order Read Model\nPostgres View]
    PM --> RM2[Analytics Read Model\nClickhouse]
    PM --> RM3[Search Index\nElasticsearch]

    style ES fill:#dcfce7,stroke:#16a34a,stroke-width:3px
```

### CQRS Architecture

```mermaid
graph TD
    Client[Client]

    Client -->|Commands| CH[Command Handler]
    Client -->|Queries| QH[Query Handler]

    subgraph WriteModel[Write Model]
        CH --> Agg[Domain Aggregate]
        Agg --> ES2[Event Store]
    end

    subgraph ReadModel[Read Model - Projections]
        ES2 -->|publish events| PB[Projection Builder]
        PB --> RM[Denormalized Read DB]
        QH --> RM
    end

    style WriteModel fill:#eff6ff,stroke:#3b82f6
    style ReadModel fill:#f0fdf4,stroke:#16a34a
```

### Choreography vs Orchestration

```mermaid
graph TD
    subgraph Choreography[Choreography - Decentralized]
        E1[OrderSvc publishes OrderPlaced]
        E2[InventorySvc reacts, publishes InventoryReserved]
        E3[PaymentSvc reacts, publishes PaymentCharged]
        E4[ShippingSvc reacts, publishes OrderShipped]
        E1 --> E2 --> E3 --> E4
    end

    subgraph Orchestration[Orchestration - Centralized Saga]
        Orch[Saga Orchestrator]
        Orch -->|1. ReserveInventory| InvSvc[Inventory Service]
        Orch -->|2. ChargePayment| PmtSvc[Payment Service]
        Orch -->|3. CreateShipment| ShpSvc[Shipping Service]
        InvSvc -->|result| Orch
        PmtSvc -->|result| Orch
        ShpSvc -->|result| Orch
    end
```

## Key Concepts

- **Event**: An immutable record of something that happened in the past, named in past tense (OrderPlaced, PaymentFailed). Events carry the data needed for consumers to react without calling back the producer.

- **Event Broker**: The durable, ordered, distributed log through which events flow. Apache Kafka, AWS EventBridge, and Google Pub/Sub are common implementations. The broker decouples producers from consumers temporally — consumers can be offline and catch up later.

- **Event Sourcing**: Storing the complete sequence of domain events as the system of record, rather than storing current state. The current state is derived by replaying events. Enables time travel, audit logs, and new projections without data migration.

- **CQRS (Command Query Responsibility Segregation)**: Separates the write model (command-side, normalised for writes) from the read model (query-side, denormalised for reads). Often paired with event sourcing — events from the write side populate read-side projections.

- **Saga Pattern**: A sequence of local transactions across services, each publishing an event that triggers the next step. If a step fails, compensating transactions undo prior steps. Sagas replace distributed ACID transactions in EDA systems.

- **Choreography**: Each service independently listens for relevant events and reacts. No central coordinator. Highly decoupled but workflow is implicit and hard to observe.

- **Orchestration**: A central saga orchestrator directs each step explicitly. More observable and easier to reason about complex flows, but introduces a coordination bottleneck.

- **Outbox Pattern**: Solves the dual-write problem — instead of writing to the database and publishing an event separately (two-phase commit), services write both to a local outbox table atomically. A relay process then reads the outbox and publishes events reliably.

## Trade-offs

| Aspect | Event-Driven | Synchronous Request-Response |
|--------|-------------|------------------------------|
| Temporal coupling | None | High |
| Complexity | High | Low |
| Consistency | Eventual | Can be strong |
| Auditability | Excellent (event log) | Requires extra work |
| Debugging | Challenging (async) | Straightforward |
| Throughput | Very high (buffered) | Rate-limited by slowest service |
| Schema evolution | Requires careful versioning | Easier to version |

## When to Use

**Use event-driven when:**
- Multiple consumers need to react to the same business event independently
- Temporal decoupling is critical (downstream services may be unavailable)
- Audit trail and event replay capabilities are needed
- Throughput needs to be very high and producers should not block on consumers

**Avoid when:**
- Strong consistency is required across the entire workflow
- The team lacks experience with async debugging and distributed tracing
- Latency requirements are tight and the overhead of message serialization/broker is unacceptable
