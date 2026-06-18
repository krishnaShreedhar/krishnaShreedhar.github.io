# Creational Patterns

Creational design patterns abstract the object creation process, making it more flexible and decoupled from the concrete classes being instantiated. They allow systems to be independent of how objects are created, composed, and represented.

## Pattern Overview

```mermaid
graph TD
    Creational[Creational Patterns]
    Creational --> Singleton[Singleton\nOne instance globally]
    Creational --> Factory[Factory Method\nSubclasses decide instantiation]
    Creational --> AbsFactory[Abstract Factory\nFamily of related objects]
    Creational --> Builder[Builder\nStep-by-step complex construction]
    Creational --> Prototype[Prototype\nClone existing objects]
```

## Singleton Pattern

```mermaid
graph LR
    subgraph Singleton
        Class[ConfigService]
        Instance[Single Instance\nshared global state]
        Class -->|getInstance| Instance
        ClientA[Client A] --> Instance
        ClientB[Client B] --> Instance
        ClientC[Client C] --> Instance
    end

    style Instance fill:#fef3c7,stroke:#d97706,stroke-width:3px
```

## Factory Method Pattern

```mermaid
graph TD
    Creator[PaymentProcessorFactory\nabstract create_processor]
    ConcreteA[StripeFactory\ncreate_processor -> StripeProcessor]
    ConcreteB[PayPalFactory\ncreate_processor -> PayPalProcessor]
    ConcreteC[BraintreeFactory\ncreate_processor -> BraintreeProcessor]

    Product[PaymentProcessor\nprocess_payment\nrefund]
    StripeProc[StripeProcessor]
    PayPalProc[PayPalProcessor]
    BraintreeProc[BraintreeProcessor]

    Creator --> ConcreteA & ConcreteB & ConcreteC
    ConcreteA -->|creates| StripeProc
    ConcreteB -->|creates| PayPalProc
    ConcreteC -->|creates| BraintreeProc
    StripeProc & PayPalProc & BraintreeProc -.->|implements| Product

    style Product fill:#dbeafe,stroke:#2563eb
```

## Builder Pattern

```mermaid
graph TD
    Director[QueryBuilder\nDirector]
    Builder[SQLQueryBuilder\nselect\nwhere\norder_by\nlimit\nbuild]

    Director -->|calls step by step| Builder

    subgraph Construction
        S1[select fields] --> S2[from table] --> S3[where conditions] --> S4[order_by clause] --> S5[limit rows] --> S6[build - returns Query]
    end

    Builder --> Construction
    S6 --> Result[SQL Query object]

    style Builder fill:#dcfce7,stroke:#16a34a
    style Result fill:#fef3c7,stroke:#d97706
```

## Abstract Factory Pattern

```mermaid
graph TD
    AbstractFactory[UIComponentFactory\nabstract interface]
    LightTheme[LightThemeFactory]
    DarkTheme[DarkThemeFactory]
    MaterialTheme[MaterialThemeFactory]

    AbstractFactory --> LightTheme & DarkTheme & MaterialTheme

    LightTheme -->|creates| LB[LightButton]
    LightTheme -->|creates| LI[LightInput]
    LightTheme -->|creates| LD[LightDialog]

    DarkTheme -->|creates| DB[DarkButton]
    DarkTheme -->|creates| DI[DarkInput]
    DarkTheme -->|creates| DD[DarkDialog]

    style AbstractFactory fill:#fef3c7,stroke:#d97706,stroke-width:2px
```

## Key Concepts

- **Singleton**: Ensures only one instance of a class exists in the process, providing a global access point. Used for shared resources like configuration, connection pools, and logging services. The main risk is hidden global state that makes testing difficult. Thread-safe implementation requires double-checked locking or class-level initialization.

- **Factory Method**: Defines an interface for creating an object but lets subclasses decide which class to instantiate. Defers instantiation to subclasses while the creator works with the product through an interface. Useful when the exact type of object to create is determined by subclass context.

- **Abstract Factory**: Provides an interface for creating families of related objects without specifying their concrete classes. All objects produced by a factory are designed to work together. Commonly used for cross-platform UI components, database driver families, or cloud provider SDK abstraction.

- **Builder**: Separates the construction of a complex object from its representation, allowing the same construction process to create different representations. Useful when an object has many optional parameters (avoids telescoping constructors) or when construction involves many steps that must be executed in sequence.

- **Prototype**: Creates new objects by copying (cloning) an existing object. Useful when creating a new object is expensive (requires database lookup, network call, heavy computation) and an existing instance can serve as a template. Requires careful handling of deep vs. shallow copy semantics.

## Trade-offs

| Pattern | Use Case | Risk |
|---------|----------|------|
| Singleton | Shared global resource | Hidden coupling, testability issues |
| Factory Method | Subclass-driven instantiation | Proliferation of subclasses |
| Abstract Factory | Family of related objects | Adding new products requires changing all factories |
| Builder | Complex objects with many optional fields | Verbose builder class to maintain |
| Prototype | Expensive object creation | Deep copy complexity, clone semantics |

## When to Use

- **Singleton**: Logging, configuration, thread pools, registry services — where exactly one instance is semantically correct and global access is required
- **Factory Method**: When a class cannot anticipate the class of objects it must create, or when subclasses should specify the objects they create
- **Abstract Factory**: When the system must be independent of product creation, composition, and representation — common in platform-abstraction layers
- **Builder**: When constructing complex objects step-by-step, especially when optional parameters would create many constructor overloads
- **Prototype**: When instantiation is more expensive than copying, or when you need to copy an object's state without coupling to its class
