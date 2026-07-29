---
title: "Behavioral Patterns"
subtitle: "Behavioral design patterns describe how objects interact and distribute responsibilities. They focus on algorithms, responsibility assignment, and communication patterns between objects — making the collaboration..."
category: technical
project: large_scale_software_engineering
project_title: "Large Scale Software Engineering"
date: 2025-06-02
reading_time: 4
tags:
  - large-scale-software-engineering
  - docs
author: "Shreedhar Kodate"
output: "blogs/technical/posts/large_scale_software_engineering/docs/03_design_patterns/03_behavioral_patterns.html"
---
Behavioral design patterns describe how objects interact and distribute responsibilities. They focus on algorithms, responsibility assignment, and communication patterns between objects — making the collaboration between objects flexible and extensible.

## Strategy Pattern

```mermaid
graph TD
    Context[ShippingCalculator\ncontext]
    Strategy[ShippingStrategy\ncalculate_cost]

    Standard[StandardShipping\ncalculate_cost]
    Express[ExpressShipping\ncalculate_cost]
    Overnight[OvernightShipping\ncalculate_cost]
    International[InternationalShipping\ncalculate_cost]

    Context -->|uses| Strategy
    Strategy --> Standard & Express & Overnight & International

    Client[Client] -->|configure strategy| Context
    Client -->|call calculate_shipping| Context

    style Strategy fill:#fef3c7,stroke:#d97706
    style Context fill:#dbeafe,stroke:#2563eb
```

## Observer Pattern

```mermaid
graph TD
    Subject[EventEmitter / Subject\nsubscribe\nunsubscribe\nnotify]

    ObsA[EmailNotifier\nupdate - send email]
    ObsB[SMSNotifier\nupdate - send SMS]
    ObsC[AuditLogger\nupdate - write log]
    ObsD[WebhookEmitter\nupdate - call webhook]

    Subject -->|notifies| ObsA & ObsB & ObsC & ObsD

    Event[OrderPlaced Event] -->|triggers| Subject

    style Subject fill:#dcfce7,stroke:#16a34a,stroke-width:2px
```

## Command Pattern

```mermaid
graph LR
    Invoker[CommandInvoker\nexecute\nundo\nqueue]
    Command[Command interface\nexecute\nundo]
    Receiver[TextEditor\ninsert_text\ndelete_text]

    InsertCmd[InsertTextCommand\nexecute: insert\nundo: delete]
    DeleteCmd[DeleteTextCommand\nexecute: delete\nundo: insert]

    Invoker --> Command
    Command --> InsertCmd & DeleteCmd
    InsertCmd -->|calls| Receiver
    DeleteCmd -->|calls| Receiver

    History[Command History\nstack for undo/redo]
    Invoker --> History

    style Command fill:#fef3c7,stroke:#d97706
```

## State Pattern

```mermaid
stateDiagram-v2
    [*] --> Draft: Order created

    Draft --> Pending: submit()
    Draft --> Cancelled: cancel()

    Pending --> Processing: payment_received()
    Pending --> Cancelled: cancel()
    Pending --> Draft: reject()

    Processing --> Shipped: ship()
    Processing --> Refunding: cancel()

    Shipped --> Delivered: confirm_delivery()
    Shipped --> Returning: return_requested()

    Delivered --> Returning: return_requested()
    Refunding --> Refunded: refund_processed()
    Returning --> Refunded: return_received()
    Refunded --> [*]
    Cancelled --> [*]
```

## Chain of Responsibility Pattern

```mermaid
graph LR
    Request[HTTP Request] --> Auth[AuthHandler\ncheck JWT token]
    Auth -->|authenticated| RateLimit[RateLimitHandler\ncheck quotas]
    RateLimit -->|within limits| Validate[ValidationHandler\nschema validation]
    Validate -->|valid| Biz[BusinessHandler\nprocess request]
    Auth -->|rejected| R1[401 Unauthorized]
    RateLimit -->|exceeded| R2[429 Too Many Requests]
    Validate -->|invalid| R3[400 Bad Request]
    Biz --> R4[200 OK Response]

    style Auth fill:#dbeafe,stroke:#2563eb
    style RateLimit fill:#dbeafe,stroke:#2563eb
    style Validate fill:#dbeafe,stroke:#2563eb
    style Biz fill:#dcfce7,stroke:#16a34a
```

## Template Method Pattern

```mermaid
graph TD
    AbstractClass[DataProcessor\nprocess - template method\n1. read_data - abstract\n2. transform_data - abstract\n3. validate_data - hook\n4. write_data - abstract]

    CSVProcessor[CSVDataProcessor\nread_data: parse CSV\ntransform_data: map rows\nwrite_data: insert to DB]

    JSONProcessor[JSONDataProcessor\nread_data: parse JSON\ntransform_data: flatten nested\nwrite_data: stream to S3]

    AbstractClass --> CSVProcessor & JSONProcessor

    style AbstractClass fill:#fef3c7,stroke:#d97706,stroke-width:2px
```

## Key Concepts

- **Strategy**: Defines a family of algorithms, encapsulates each one, and makes them interchangeable. The strategy pattern lets the algorithm vary independently from clients that use it. Used when you have multiple ways of doing something (sorting, pricing, shipping calculation) and need to switch between them at runtime.

- **Observer**: Defines a one-to-many dependency — when the subject changes state, all dependents (observers) are notified and updated automatically. Enables loose coupling between the event source and its handlers. The foundation of event-driven programming and reactive systems.

- **Command**: Encapsulates a request as an object, enabling parameterization, queuing, logging, and undo/redo operations. The command object contains the receiver, the action, and its parameters. Command queues enable job processing systems; command history enables undo stacks.

- **State**: Allows an object to alter its behaviour when its internal state changes. The object will appear to change its class. State encapsulates state-specific behaviour in separate state objects, eliminating large conditional blocks based on state flags.

- **Chain of Responsibility**: Passes a request along a chain of handlers, where each handler decides whether to process the request or pass it to the next handler. Decouples the sender of a request from its receivers. Used for middleware pipelines, event handling chains, and approval workflows.

- **Template Method**: Defines the skeleton of an algorithm in a base class, deferring some steps to subclasses. Subclasses can override specific steps without changing the algorithm's structure. The Hollywood Principle: "Don't call us, we'll call you" — the abstract class calls subclass methods.

- **Mediator**: Defines how a set of objects interact. Promotes loose coupling by keeping objects from referring to each other explicitly. A mediator centralises complex communications between objects. Used in air traffic control metaphor — planes communicate via tower, not directly.

- **Iterator**: Provides a way to access elements of an aggregate object sequentially without exposing its underlying representation. Enables uniform traversal of different data structures (arrays, trees, graphs) via a common interface.

## Trade-offs

| Pattern | Benefit | Cost |
|---------|---------|------|
| Strategy | Runtime algorithm switching | Clients must know all strategies |
| Observer | Loose coupling between events and handlers | Unexpected order of notifications |
| Command | Undo/redo, queuing, logging | Proliferation of command classes |
| State | Eliminates state-based conditionals | Many state classes |
| Chain of Responsibility | Flexible handler assignment | Request may go unhandled |
| Template Method | Code reuse, algorithmic consistency | Tight coupling to base class |

## When to Use

- **Strategy**: When you have multiple conditional branches selecting algorithm variants — replace them with polymorphism
- **Observer**: When changes in one object require updating others and you don't know how many objects need to change
- **Command**: When you need transactional behaviour, undo/redo, command queuing, or remote execution
- **State**: When an object's behaviour depends on its state and must change at runtime
- **Chain of Responsibility**: When more than one handler may handle a request and the handler isn't known a priori