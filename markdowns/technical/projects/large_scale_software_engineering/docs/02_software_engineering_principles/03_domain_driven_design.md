---
title: "Domain-Driven Design (DDD)"
subtitle: "Domain-Driven Design is an approach to software development that places the business domain model at the center of the design process. DDD provides a set of strategic and tactical patterns that help align software..."
category: technical
project: large_scale_software_engineering
project_title: "Large Scale Software Engineering"
date: 2025-03-25
reading_time: 4
tags:
  - large-scale-software-engineering
  - docs
author: "Shreedhar Kodate"
output: "blogs/technical/posts/large_scale_software_engineering/docs/02_software_engineering_principles/03_domain_driven_design.html"
---
Domain-Driven Design is an approach to software development that places the business domain model at the center of the design process. DDD provides a set of strategic and tactical patterns that help align software structure with business concepts, enabling complex domains to be modeled accurately and evolved safely.

## Strategic DDD

```mermaid
graph TD
    subgraph Domain[Business Domain: E-Commerce]
        subgraph CoreDomain[Core Domain - Competitive Advantage]
            OrderBC[Order Management\nBounded Context]
            PricingBC[Dynamic Pricing\nBounded Context]
        end

        subgraph SupportingDomain[Supporting Subdomain]
            InventoryBC[Inventory\nBounded Context]
            ShippingBC[Shipping\nBounded Context]
        end

        subgraph GenericDomain[Generic Subdomain - Buy or Build]
            AuthBC[Authentication\nBounded Context]
            NotifBC[Notifications\nBounded Context]
        end
    end

    OrderBC -->|Conformist| InventoryBC
    OrderBC -->|Partnership| PricingBC
    OrderBC -->|Customer-Supplier| ShippingBC
    OrderBC -->|Anticorruption Layer| AuthBC

    style CoreDomain fill:#dcfce7,stroke:#16a34a,stroke-width:2px
    style SupportingDomain fill:#dbeafe,stroke:#2563eb
    style GenericDomain fill:#f5f5f4,stroke:#78716c
```

## Bounded Context Map

```mermaid
graph LR
    subgraph OrderContext[Order Context]
        OrdModel[Order\nOrderLine\nCustomer reference]
    end

    subgraph CatalogContext[Catalog Context]
        CatModel[Product\nSKU\nCategory]
    end

    subgraph InventoryContext[Inventory Context]
        InvModel[StockItem\nWarehouse\nReservation]
    end

    ACL1[Anti-Corruption Layer] --- OrderContext
    ACL1 --- CatalogContext
    OFC[Open Host Service] --- InventoryContext
    OFC --- OrderContext

    Note1[Order calls Catalog\nvia ACL to translate\nProduct -> OrderLine]
    Note2[Inventory exposes\npublished language\nfor all consumers]
```

## Tactical DDD Building Blocks

```mermaid
graph TD
    subgraph AggregateRoot[Order Aggregate]
        OrderAR[Order - Aggregate Root\nID: OrderId\nenforce invariants]
        OrderLine[OrderLine - Entity\nID: LineId]
        Money[Money - Value Object\nimmutable, no ID]
        Address[ShippingAddress - Value Object]

        OrderAR --> OrderLine
        OrderAR --> Money
        OrderAR --> Address
    end

    subgraph DomainEvents[Domain Events]
        OPlaced[OrderPlaced]
        OShipped[OrderShipped]
        OCancelled[OrderCancelled]
    end

    OrderAR -->|emits| OPlaced & OShipped & OCancelled

    subgraph Repository[Repository]
        OrderRepo[OrderRepository\nfind_by_id\nsave\ndelete]
    end

    OrderRepo -->|reconstructs| OrderAR

    style OrderAR fill:#fef3c7,stroke:#d97706,stroke-width:3px
```

## Ubiquitous Language in Practice

```mermaid
sequenceDiagram
    participant BA as Business Analyst
    participant Dev as Developer
    participant Code as Codebase

    BA->>Dev: "When a customer places an order,\nwe need to reserve inventory"
    Dev->>Code: class Order { place_order() }
    Dev->>Code: class InventoryService { reserve(order_items) }
    Dev->>Code: event: OrderPlaced

    Note over BA,Code: Language in meetings = Language in code

    BA->>Dev: "Orders expire if not confirmed\nwithin 30 minutes"
    Dev->>Code: class Order { expire() }
    Dev->>Code: OrderExpired domain event
    Dev->>Code: OrderExpirationPolicy value object
```

## Key Concepts

- **Ubiquitous Language**: A shared vocabulary developed between domain experts and developers that is used consistently in conversations, documentation, and code. Method names, class names, and event names should use the same words a domain expert would use. Prevents the translation tax between business intent and implementation.

- **Bounded Context**: An explicit boundary within which a particular domain model applies and has a consistent meaning. The word "Order" in the Order Context and the Shipping Context may refer to completely different objects with different attributes and behaviours. Bounded contexts make this explicit rather than forcing a single unified model.

- **Context Map**: A diagram showing all bounded contexts in a system and the relationships between them. Relationship types include: Partnership (two teams coordinate), Customer-Supplier (upstream/downstream dependency), Conformist (downstream adopts upstream's model), Anti-Corruption Layer (downstream translates), Open Host Service (published language for many consumers), and Published Language (shared schema/API specification).

- **Aggregate**: A cluster of domain objects treated as a single unit of consistency. The Aggregate Root is the only object that external code can reference directly — access to internal entities goes through the root. The aggregate enforces all invariants for the objects within its boundary.

- **Entity**: An object with a distinct identity that persists over time. Two entities with the same attribute values are not the same entity — they have different identities. Example: two Customer objects with the same name are different customers.

- **Value Object**: An object defined entirely by its attributes with no conceptual identity. Two value objects with the same attributes are interchangeable. Value objects are immutable. Example: Money(100, USD) is equivalent to any other Money(100, USD).

- **Domain Event**: An immutable record of something significant that happened in the domain, named in past tense. Events are the mechanism through which aggregates communicate side effects to the rest of the system without tight coupling.

- **Repository**: An abstraction over the persistence mechanism that provides collection-like semantics for aggregates. The domain layer depends on the repository interface; the infrastructure layer provides the implementation.

- **Domain Service**: Business logic that doesn't naturally belong to a single entity or value object. A TransferService that moves money between two Accounts is a domain service — the logic belongs to neither Account alone.

## Trade-offs

| Aspect | DDD | Simpler Approach |
|--------|-----|-----------------|
| Model accuracy | High — matches domain reality | Lower — often DB-driven |
| Learning curve | Steep | Low |
| Team alignment | Excellent (ubiquitous language) | Varies |
| Upfront design cost | High | Low |
| Maintenance cost | Lower long-term | Higher as complexity grows |
| Suitable domain | Complex, core business logic | Simple CRUD, generic subdomains |

## When to Use

**Apply DDD when:**
- The domain is complex with intricate business rules and invariants
- The system is expected to evolve significantly over time
- Multiple teams need to work on different parts of the domain independently
- Domain experts are accessible and engaged in the design process

**Avoid full DDD for:**
- Generic subdomains (auth, notifications, storage) — buy or use off-the-shelf
- Simple CRUD applications with minimal business logic
- Short-lived prototypes or throwaway systems
- Teams without domain expert access — DDD without domain knowledge produces inaccurate models