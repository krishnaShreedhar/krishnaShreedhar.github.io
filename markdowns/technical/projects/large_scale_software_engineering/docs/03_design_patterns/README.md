# Design Patterns

Design patterns are reusable solutions to commonly occurring problems in software design. They are not code templates but rather blueprints for how to structure code to solve a specific class of problem. Patterns exist at multiple levels: object-oriented (GoF), concurrency, distributed systems, and cloud infrastructure.

## Overview

```mermaid
mindmap
  root((Design\nPatterns))
    Creational
      Singleton
      Factory Method
      Abstract Factory
      Builder
      Prototype
    Structural
      Adapter
      Bridge
      Composite
      Decorator
      Facade
      Flyweight
      Proxy
    Behavioral
      Chain of Responsibility
      Command
      Iterator
      Mediator
      Memento
      Observer
      State
      Strategy
      Template Method
      Visitor
    Concurrency
      Thread Pool
      Producer-Consumer
      Read-Write Lock
      Active Object
      Reactor
    Distributed Systems
      Circuit Breaker
      Saga
      Outbox
      Sidecar
      Ambassador
      Strangler Fig
    Cloud Infrastructure
      CQRS
      Event Sourcing
      Bulkhead
      Retry with Backoff
      Health Endpoint
```

## Pattern Classification

```mermaid
graph LR
    subgraph Purpose
        C[Creational\nHow objects are created]
        S[Structural\nHow objects are composed]
        B[Behavioral\nHow objects communicate]
    end

    subgraph Scope
        Class[Class-level\nUse inheritance]
        Object[Object-level\nUse composition]
    end

    C --> Class & Object
    S --> Class & Object
    B --> Class & Object
```

## Topics in This Section

| File | Topic | Key Patterns |
|------|-------|-------------|
| [01_creational_patterns.md](01_creational_patterns.md) | Creational | Singleton, Factory, Builder, Prototype |
| [02_structural_patterns.md](02_structural_patterns.md) | Structural | Adapter, Decorator, Facade, Proxy, Composite |
| [03_behavioral_patterns.md](03_behavioral_patterns.md) | Behavioral | Strategy, Observer, Command, State, Chain |
| [04_concurrency_patterns.md](04_concurrency_patterns.md) | Concurrency | Thread Pool, Producer-Consumer, Reactor |
| [05_distributed_systems_patterns.md](05_distributed_systems_patterns.md) | Distributed | Circuit Breaker, Saga, Outbox, Sidecar |
| [06_cloud_infrastructure_patterns.md](06_cloud_infrastructure_patterns.md) | Cloud | CQRS, Event Sourcing, Bulkhead, Retry |
