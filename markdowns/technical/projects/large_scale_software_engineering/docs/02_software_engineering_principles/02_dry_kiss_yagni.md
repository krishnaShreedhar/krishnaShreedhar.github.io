---
title: "DRY, KISS, and YAGNI"
subtitle: "DRY, KISS, and YAGNI are three complementary code quality heuristics that guide engineers toward simpler, more maintainable code. Each addresses a common failure mode in software development: unnecessary duplication..."
category: technical
project: large_scale_software_engineering
project_title: "Large Scale Software Engineering"
date: 2025-09-19
reading_time: 4
tags:
  - large-scale-software-engineering
  - docs
author: "Shreedhar Kodate"
output: "blogs/technical/posts/large_scale_software_engineering/docs/02_software_engineering_principles/02_dry_kiss_yagni.html"
---
DRY, KISS, and YAGNI are three complementary code quality heuristics that guide engineers toward simpler, more maintainable code. Each addresses a common failure mode in software development: unnecessary duplication (DRY), accidental complexity (KISS), and premature scope expansion (YAGNI).

## The Three Heuristics

```mermaid
mindmap
  root((Code Quality\nHeuristics))
    DRY - Don't Repeat Yourself
      Every piece of knowledge must have a single authoritative representation
      Duplication of logic not just syntax
      WET code - Write Everything Twice
      Rule of Three - duplicate twice, abstract on third
    KISS - Keep It Simple Stupid
      Simplest solution that works
      Avoid clever code
      Complexity as a liability
      Simple code is secure code
    YAGNI - You Aren't Gonna Need It
      Implement things when needed
      Avoid speculative generality
      Cost of unused features
      XP principle
```

## DRY - The Duplication Problem

```mermaid
graph TD
    subgraph WET[WET Code - Duplication]
        OrderSvc[OrderService\nvalidate_email\ncalculate_tax\nformat_currency]
        UserSvc[UserService\nvalidate_email\nformat_currency]
        InvSvc[InvoiceService\nvalidate_email\ncalculate_tax\nformat_currency]
        Note1[Bug in validate_email?\nFix in 3 places!]
        style OrderSvc fill:#fee2e2,stroke:#dc2626
        style Note1 fill:#fee2e2,stroke:#dc2626
    end

    subgraph DRY[DRY Code - Single Source of Truth]
        EmailUtil[EmailValidator\nvalidate_email]
        TaxCalc[TaxCalculator\ncalculate_tax]
        CurrFmt[CurrencyFormatter\nformat_currency]
        OrderSvc2[OrderService] --> EmailUtil & TaxCalc & CurrFmt
        UserSvc2[UserService] --> EmailUtil & CurrFmt
        InvSvc2[InvoiceService] --> EmailUtil & TaxCalc & CurrFmt
        Note2[Bug in validate_email?\nFix in 1 place!]
        style Note2 fill:#dcfce7,stroke:#16a34a
    end
```

## KISS - Complexity Spectrum

```mermaid
graph LR
    subgraph TooSimple[Too Simple - Under-engineered]
        God[God Class\nall logic in one 5000-line file]
    end

    subgraph JustRight[Just Right]
        Clear[Clear separation of concerns\nSmall focused functions\nObvious naming\nMinimal indirection]
        style Clear fill:#dcfce7,stroke:#16a34a,stroke-width:2px
    end

    subgraph TooComplex[Too Complex - Over-engineered]
        Abstract[7 layers of abstraction\nFactory of factories\nDynamic plugin framework\nfor a CRUD app]
        style Abstract fill:#fee2e2,stroke:#dc2626
    end

    TooSimple -->|Refactor| JustRight
    TooComplex -->|Simplify| JustRight
```

## YAGNI - The Speculative Feature Problem

```mermaid
flowchart TD
    A[Feature Request / Design Decision] --> B{Is this needed NOW\nfor a current requirement?}
    B -->|Yes| C[Implement it]
    B -->|No| D{Is this needed in\nthe next sprint?}
    D -->|Yes - confirmed in backlog| E[Design for it, defer implementation]
    D -->|No - just a future idea| F[YAGNI - Don't build it]
    F --> G[Costs Avoided:\nDevelopment time\nTest coverage\nDocumentation\nMaintenance burden\nCode complexity]
    C --> H[Deliver value now]

    style F fill:#fef3c7,stroke:#d97706
    style G fill:#dcfce7,stroke:#16a34a
```

## Key Concepts

- **DRY (Don't Repeat Yourself)**: Every piece of knowledge must have a single, unambiguous, authoritative representation in the system. DRY applies to logic, not just syntax — two pieces of code that happen to look alike but represent different business concepts are NOT a DRY violation. The "Rule of Three" is a practical guide: duplicate once (accepting the copy), abstract on the third occurrence.

- **WET Code**: The opposite of DRY — "Write Everything Twice" or "We Enjoy Typing". WET code scatters business rules across many locations, making bugs harder to fix (you must find and update all copies) and rules harder to understand (no single authoritative source).

- **KISS (Keep It Simple, Stupid)**: Prefer the simplest solution that correctly solves the problem. Complexity is a liability — every layer of abstraction, every design pattern, every configuration option has a cognitive cost for future maintainers. Simple code is also more secure (fewer edge cases, less attack surface) and more testable.

- **Accidental vs Essential Complexity**: Essential complexity is inherent to the problem domain (unavoidable). Accidental complexity is introduced by the solution — over-engineering, premature optimization, unnecessary abstraction. KISS is primarily about eliminating accidental complexity.

- **YAGNI (You Aren't Gonna Need It)**: Don't implement functionality until it is actually needed. Speculative generality — building extensibility points "just in case" — adds code that must be maintained, tested, and documented while delivering no current value. The cost of adding a feature when needed is almost always lower than the cost of maintaining a feature that's never used.

- **The Duplication vs Coupling Trade-off**: Sometimes the choice is between duplication (WET) and tight coupling (pulling two unrelated things into a shared abstraction). Prefer duplication when the two pieces of code evolve independently and only happen to look similar today.

## Trade-offs

| Principle | Over-application Risk | Under-application Risk |
|-----------|----------------------|------------------------|
| DRY | Wrong abstractions couple unrelated concepts | Bug-fixing requires updating many places |
| KISS | Under-engineered, non-scalable solutions | Unmaintainable complexity, clever hacks |
| YAGNI | Missing extensibility causes expensive refactors | Dead code, unnecessary maintenance burden |

## When to Apply

- **DRY**: Apply when the same business rule or calculation appears in more than two places. Do not apply mechanically to any code that happens to look similar.
- **KISS**: Apply always — question every layer of indirection and ask "what does this complexity buy us?"
- **YAGNI**: Apply during feature planning and code review — regularly delete unused code paths and configuration options that were added speculatively.
- The three principles are most powerful together: KISS prevents over-engineering, YAGNI prevents premature features, and DRY prevents knowledge fragmentation.