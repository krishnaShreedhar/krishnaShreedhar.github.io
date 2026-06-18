# Software Engineering Principles

Software engineering principles are the foundational rules and heuristics that guide the design of code at all levels — from individual functions to entire system modules. Mastery of these principles separates software that is easy to change, test, and understand from code that accrues technical debt.

## Overview

```mermaid
mindmap
  root((Engineering\nPrinciples))
    SOLID
      Single Responsibility
      Open-Closed
      Liskov Substitution
      Interface Segregation
      Dependency Inversion
    DRY / KISS / YAGNI
      Don't Repeat Yourself
      Keep It Simple
      You Aren't Gonna Need It
    Domain-Driven Design
      Ubiquitous Language
      Bounded Contexts
      Aggregates
      Domain Events
      Repositories
    Package Design
      REP - Reuse
      CCP - Common Closure
      CRP - Common Reuse
      ADP - Acyclic Dependencies
      SDP - Stable Dependencies
    Coding Principles
      Separation of Concerns
      Law of Demeter
      Command-Query Separation
      Fail Fast
      Composition Over Inheritance
```

## How Principles Relate

```mermaid
graph TD
    HLPrinciples[High-Level Design Principles]
    HLPrinciples --> DDD[Domain-Driven Design\nDefines system structure]
    HLPrinciples --> PKG[Package Design Principles\nDefines module boundaries]

    ModPrinciples[Module-Level Principles]
    ModPrinciples --> SOLID[SOLID\nDefines class/component design]
    ModPrinciples --> DRY[DRY / KISS / YAGNI\nDefines code quality heuristics]

    ImplPrinciples[Implementation Principles]
    ImplPrinciples --> Code[Coding Principles\nSoC, LoD, CQS, Fail Fast]

    DDD --> ModPrinciples
    PKG --> ModPrinciples
    SOLID --> ImplPrinciples
    DRY --> ImplPrinciples
```

## Topics in This Section

| File | Topic | Key Concepts |
|------|-------|--------------|
| [01_solid_principles.md](01_solid_principles.md) | SOLID | SRP, OCP, LSP, ISP, DIP |
| [02_dry_kiss_yagni.md](02_dry_kiss_yagni.md) | DRY / KISS / YAGNI | Code quality heuristics |
| [03_domain_driven_design.md](03_domain_driven_design.md) | DDD | Bounded contexts, aggregates, ubiquitous language |
| [04_package_module_design.md](04_package_module_design.md) | Package Design | REP, CCP, CRP, ADP, SDP, SAP |
| [05_coding_design_principles.md](05_coding_design_principles.md) | Coding Principles | SoC, LoD, CQS, fail fast, composition |
