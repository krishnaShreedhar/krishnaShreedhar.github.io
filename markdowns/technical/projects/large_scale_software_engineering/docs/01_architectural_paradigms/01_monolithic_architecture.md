---
title: "Monolithic Architecture"
subtitle: "A monolithic architecture packages all application functionality into a single deployable unit. All components — UI, business logic, and data access — share the same process space and are deployed together, making..."
category: technical
project: large_scale_software_engineering
project_title: "Large Scale Software Engineering"
date: 2025-07-08
reading_time: 3
tags:
  - large-scale-software-engineering
  - docs
author: "Shreedhar Kodate"
output: "blogs/technical/posts/large_scale_software_engineering/docs/01_architectural_paradigms/01_monolithic_architecture.html"
---
A monolithic architecture packages all application functionality into a single deployable unit. All components — UI, business logic, and data access — share the same process space and are deployed together, making development straightforward but creating coupling that complicates independent scaling and evolution.

## Architecture Diagrams

### Traditional Monolith

```mermaid
graph TD
    Client[Client Browser / Mobile] --> LB[Load Balancer]
    LB --> App1[App Instance 1]
    LB --> App2[App Instance 2]
    LB --> App3[App Instance 3]

    subgraph App1[Monolithic Application Instance]
        UI[Presentation Layer]
        BL[Business Logic Layer]
        DAL[Data Access Layer]
        UI --> BL --> DAL
    end

    DAL --> DB[(Shared Database)]
    DAL --> Cache[(Redis Cache)]
    DAL --> FS[File Storage]

    style App1 fill:#f0e6ff,stroke:#7c3aed,stroke-width:3px
```

### Modular Monolith

```mermaid
graph TD
    Entry[API Entry Point] --> Router[Request Router]

    Router --> OrderMod[Order Module]
    Router --> UserMod[User Module]
    Router --> PayMod[Payment Module]
    Router --> InventMod[Inventory Module]

    subgraph Kernel[Shared Kernel]
        Events[Domain Events Bus]
        Auth[Auth Context]
        Config[Config Service]
    end

    OrderMod --> Events
    PayMod --> Events
    InventMod --> Events
    UserMod --> Auth

    OrderMod --> DB[(Database)]
    UserMod --> DB
    PayMod --> DB
    InventMod --> DB

    style Kernel fill:#fff7ed,stroke:#ea580c,stroke-width:2px
```

### Strangler Fig Migration Pattern

```mermaid
graph LR
    subgraph Phase1[Phase 1: Facade]
        C1[Client] --> F[Facade / Proxy]
        F --> M[Monolith]
    end

    subgraph Phase2[Phase 2: Extract]
        C2[Client] --> F2[Facade / Proxy]
        F2 --> M2[Monolith - reduced]
        F2 --> S1[New Service A]
    end

    subgraph Phase3[Phase 3: Retire]
        C3[Client] --> GW[API Gateway]
        GW --> S2[Service A]
        GW --> S3[Service B]
        GW --> S4[Service C]
    end

    Phase1 --> Phase2 --> Phase3
```

## Key Concepts

- **Single Deployable Unit**: Everything compiles and deploys together. One artifact (JAR, WAR, binary, container image) contains all functionality. This simplifies deployment logistics but creates a monolithic blast radius — a bug in one module can crash the entire process.

- **Shared Memory Space**: All modules share the same heap, which enables zero-cost in-process function calls and avoids the latency and serialization overhead of network calls. This is a genuine performance advantage over microservices for tightly coupled workflows.

- **Modular Monolith**: A disciplined variant where internal modules have explicit contracts, no direct cross-module database access, and communicate via internal events or interfaces. Provides most of the development simplicity of a monolith while preparing for future service extraction.

- **Vertical Scaling**: Monoliths scale by deploying more instances of the entire application. This wastes resources for unevenly loaded modules but is operationally simple.

- **Strangler Fig Pattern**: A migration strategy where a façade intercepts all traffic; new functionality is redirected to new services incrementally until the monolith is fully replaced (or reduced to a manageable residue).

- **Shared Database Anti-pattern**: The classic monolith failure mode: all modules directly access shared tables, creating implicit coupling that makes it impossible to extract services later without extensive refactoring.

## Trade-offs

| Aspect | Traditional Monolith | Modular Monolith | Microservices |
|--------|---------------------|-----------------|---------------|
| Development speed (early) | Fastest | Fast | Slow |
| Operational complexity | Low | Low | High |
| Independent scaling | Not possible | Not possible | Per-service |
| Fault isolation | None | Limited | Strong |
| Deployment risk | High (all-or-nothing) | High | Low per service |
| Latency (internal calls) | Near-zero | Near-zero | Network + serialization |
| Team autonomy | Low | Moderate | High |
| Database flexibility | One schema | One schema | Per-service schema |

## When to Use

**Use a monolith when:**
- Team is small (fewer than ~10 engineers)
- Domain is not yet well understood — premature decomposition is expensive
- Time-to-market is the primary constraint
- Throughput requirements are moderate and vertically scalable

**Use a modular monolith when:**
- You expect to eventually extract services but aren't sure where the boundaries lie
- You want monolith simplicity with architectural discipline enforced by tooling (ArchUnit, module system boundaries)

**Avoid when:**
- Multiple teams need independent deployment cadences
- Different modules have vastly different scaling needs (e.g., batch processing vs. real-time API)
- Regulatory requirements mandate strict data isolation between capabilities