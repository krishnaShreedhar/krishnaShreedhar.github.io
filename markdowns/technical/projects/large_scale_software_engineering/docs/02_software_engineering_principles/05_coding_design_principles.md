# Coding Design Principles

Beyond SOLID and package design, a set of lower-level coding principles guides the design of individual functions, classes, and interactions. These principles — Separation of Concerns, Law of Demeter, Command-Query Separation, Fail Fast, and Composition over Inheritance — are the everyday vocabulary of clean code.

## Overview

```mermaid
mindmap
  root((Coding\nDesign Principles))
    Separation of Concerns
      Each module handles one concern
      Horizontal slicing
      Vertical feature slicing
    Law of Demeter
      Talk only to close friends
      No method chaining across objects
      Avoid deep object navigation
    Command-Query Separation
      Commands - change state return void
      Queries - return value no side effects
      CQRS at code level
    Fail Fast
      Validate at the boundary
      Throw early, loudly
      Defensive programming
    Composition Over Inheritance
      Prefer delegation to extension
      Avoid inheritance hierarchies
      Mixins and traits
    Tell Dont Ask
      Objects should act on their own data
      Avoid extracting state to decide externally
```

## Separation of Concerns (SoC)

```mermaid
graph TD
    subgraph Violation[SoC Violation - Mixed Concerns]
        Handler[HTTP Handler\nparses HTTP\nvalidates input\nexecutes business logic\nformats response\nlogs request\nhandles errors]
        style Handler fill:#fee2e2,stroke:#dc2626
    end

    subgraph Correct[SoC Correct - Layered Concerns]
        MW[Middleware Layer\nlogging, auth, error handling]
        HC[HTTP Controller\nparse request, format response]
        VS[Validation Service\ninput validation]
        BL[Business Logic Service\ndomain operations]
        RP[Repository\ndata persistence]

        MW --> HC --> VS --> BL --> RP
        style BL fill:#dcfce7,stroke:#16a34a,stroke-width:2px
    end
```

## Law of Demeter

```mermaid
graph LR
    subgraph Violation[LoD Violation - Train Wreck]
        Code1["order.getCustomer()\n  .getAddress()\n  .getCity()\n  .getPostalCode()"]
        Problem[Couples OrderService to\nCustomer, Address, City internals.\nAny intermediate change breaks this.]
        style Code1 fill:#fee2e2,stroke:#dc2626
        style Problem fill:#fee2e2,stroke:#dc2626
    end

    subgraph Correct[LoD Correct - Tell, Don't Ask]
        Code2["order.getShippingPostalCode()"]
        Deleg[Order delegates internally\nto Customer to get postal code.\nCaller knows nothing about\ninternal structure.]
        style Code2 fill:#dcfce7,stroke:#16a34a
        style Deleg fill:#dcfce7,stroke:#16a34a
    end
```

## Command-Query Separation (CQS)

```mermaid
graph TD
    subgraph CQS[Command-Query Separation]
        Commands[Commands\nVoid return type\nModify state\nMay have side effects\nExamples: place_order, cancel_order, update_price]
        Queries[Queries\nReturn a value\nNo side effects\nIdempotent - safe to retry\nExamples: get_order, find_by_id, calculate_total]
    end

    subgraph Violation[CQS Violation]
        pop["stack.pop()\nReturns value AND\nmodifies state simultaneously"]
        style pop fill:#fee2e2,stroke:#dc2626
    end

    subgraph Better[CQS Compliant]
        peek["stack.peek() → value\nstack.remove() → void"]
        style peek fill:#dcfce7,stroke:#16a34a
    end
```

## Fail Fast Pattern

```mermaid
flowchart TD
    Input[External Input\nAPI request, config, user data] --> Guard[Guard Clauses\nat entry points]
    Guard -->|Invalid: fail immediately| Error[Throw Exception\nReturn error response\nDo not proceed]
    Guard -->|Valid| Core[Execute Core Logic\nno defensive checks needed]
    Core --> Output[Return result]

    subgraph GuardExamples[Guard Clause Examples]
        G1[if user is None: raise ValueError]
        G2[if amount <= 0: raise InvalidAmountError]
        G3[if len str > MAX: raise TooLongError]
    end
```

## Composition Over Inheritance

```mermaid
graph TD
    subgraph Inheritance[Inheritance - Fragile]
        Animal[Animal\nmove, speak]
        Dog[Dog extends Animal\nspeak = bark]
        RobotDog[RobotDog extends Dog?\nDoesn't bark, shouldn't inherit speak]
        Animal --> Dog --> RobotDog
        style RobotDog fill:#fee2e2,stroke:#dc2626
    end

    subgraph Composition[Composition - Flexible]
        Mover[Mover interface\nmove]
        Speaker[Speaker interface\nspeak]
        RealDog[Dog\nhas Mover\nhas Speaker]
        Robot[RobotDog\nhas Mover only]
        Mover --> RealDog & Robot
        Speaker --> RealDog
        style RealDog fill:#dcfce7,stroke:#16a34a
        style Robot fill:#dcfce7,stroke:#16a34a
    end
```

## Key Concepts

- **Separation of Concerns (SoC)**: Each module, class, or function should address a single concern — a distinct dimension of a problem. Concerns can be horizontal (layers: UI, business logic, data) or vertical (features: ordering, billing, shipping). Mixing concerns (business logic in the HTTP handler, SQL in the domain object) creates tight coupling and makes each concern harder to change, test, or replace independently.

- **Law of Demeter (LoD)**: A method should only call methods on: itself, objects passed as parameters, objects it creates, and its direct component objects. This prevents "train wreck" chains (a.b().c().d()) that couple a class to the internal structure of objects several hops away. Violations create fragile code where changing any intermediate object breaks distant callers.

- **Command-Query Separation (CQS)**: Functions should be either commands (they change state, return void) or queries (they return a value, have no side effects) — never both. This makes code easier to reason about: you can call a query any number of times without side effects, and commands have predictable boundaries of effect.

- **Fail Fast**: Validate inputs at the earliest possible point and throw an exception immediately when invalid inputs are detected. Don't propagate invalid state deep into the system where the error becomes confusing. Guard clauses at the boundaries of functions and services eliminate the need for defensive checks throughout the core logic.

- **Composition Over Inheritance**: Prefer building complex behaviour by composing simple objects (delegation) rather than by creating deep inheritance hierarchies. Inheritance is a strong form of coupling — subclasses depend on parent internals and changes to the parent break subclasses. Composition is more flexible, easier to test, and avoids the fragile base class problem.

- **Tell, Don't Ask**: Rather than asking an object for its data and making decisions externally, tell the object what to do and let it decide internally. This keeps behaviour with the data it operates on, preventing the data/behaviour split that leads to anaemic domain models.

## Trade-offs

| Principle | Benefit | Cost |
|-----------|---------|------|
| SoC | Independent changeability of concerns | More files, more abstraction |
| LoD | Reduced coupling, easier refactoring | More wrapper methods needed |
| CQS | Predictable side effects | Cannot use return value of mutating operations |
| Fail Fast | Earlier error detection, cleaner core logic | More upfront validation code |
| Composition | Flexibility, testability | More objects, delegation boilerplate |
| Tell Don't Ask | Encapsulation, coherent objects | Requires rich domain model |

## When to Apply

- Apply these principles consistently in production codebases, starting with the most impactful: Fail Fast (immediate debugging benefit) and SoC (testability)
- LoD is especially important in deeply nested object graphs — enforce with linting tools where possible
- CQS becomes critical when building concurrent systems where interleaved reads and writes cause race conditions
- Composition is almost always preferable to inheritance beyond one level of hierarchy
