---
title: "Package and Module Design Principles"
subtitle: "Package and module design principles, formalized by Robert C. Martin, govern how to group classes and components into cohesive, stable, and reusable packages. Poor package structure causes ripple-effect changes,..."
category: technical
project: large_scale_software_engineering
project_title: "Large Scale Software Engineering"
date: 2025-11-03
reading_time: 4
tags:
  - large-scale-software-engineering
  - docs
author: "Shreedhar Kodate"
output: "blogs/technical/posts/large_scale_software_engineering/docs/02_software_engineering_principles/04_package_module_design.html"
---
Package and module design principles, formalized by Robert C. Martin, govern how to group classes and components into cohesive, stable, and reusable packages. Poor package structure causes ripple-effect changes, makes testing difficult, and prevents independent deployment of components.

## The Six Principles

```mermaid
mindmap
  root((Package Design\nPrinciples))
    Cohesion Principles
      REP - Reuse-Release Equivalence
        Unit of reuse = unit of release
        Semantic cohesion required
      CCP - Common Closure Principle
        Classes that change together stay together
        Package-level SRP
      CRP - Common Reuse Principle
        Classes used together stay together
        Package-level ISP
    Coupling Principles
      ADP - Acyclic Dependencies
        No cycles in dependency graph
        Breaking cycles with DIP
      SDP - Stable Dependencies
        Depend in direction of stability
        Stability metrics
      SAP - Stable Abstractions
        Stable packages should be abstract
        Instability vs abstractness
```

## Cohesion: REP, CCP, CRP

```mermaid
graph TD
    subgraph REP[REP - Reuse-Release Equivalence]
        R1[Classes released together\nmust be semantically cohesive]
        R2[Users of a package\naccept all or nothing]
        R1 --> R2
    end

    subgraph CCP[CCP - Common Closure]
        C1[If classes change\nfor the same reason...]
        C2[...they belong in\nthe same package]
        C1 --> C2
    end

    subgraph CRP[CRP - Common Reuse]
        CR1[If classes are used together,\nthey belong together]
        CR2[Don't force users to depend\non classes they don't use]
        CR1 --> CR2
    end

    subgraph Tension[Tension Triangle]
        T_REP[REP] --- T_CCP[CCP]
        T_CCP --- T_CRP[CRP]
        T_CRP --- T_REP
        Note[Optimizing for one\npenalizes the others]
    end
```

## Acyclic Dependencies Principle (ADP)

```mermaid
graph TD
    subgraph Cyclic[ADP Violation - Cycle]
        A[package.auth] -->|depends on| B[package.user]
        B -->|depends on| C[package.session]
        C -->|depends on| A
        style A fill:#fee2e2,stroke:#dc2626
        style B fill:#fee2e2,stroke:#dc2626
        style C fill:#fee2e2,stroke:#dc2626
    end

    subgraph BreakCycle[Breaking the Cycle with DIP]
        A2[package.auth] -->|depends on| Interface[package.contracts\nUserProvider interface]
        B2[package.user] -->|implements| Interface
        C2[package.session] -->|depends on| Interface
        style Interface fill:#fef3c7,stroke:#d97706,stroke-width:2px
    end
```

## Stability Metrics and SAP

```mermaid
graph LR
    subgraph Unstable[Unstable Package\nI=1.0, A=0.0]
        Leaf[Concrete implementations\nMany dependents, no dependees\nFree to change]
    end

    subgraph Stable[Stable Package\nI=0.0, A=1.0]
        Core[Abstract interfaces\nMany dependents, many dependers\nHard to change - must be abstract]
    end

    subgraph Main[main package\nI=0.0, A=0.0\nZone of Pain]
        DB[Concrete DB Impl\nStable but not abstract\nRigid, hard to change]
        style DB fill:#fee2e2,stroke:#dc2626
    end

    Leaf -->|depends on| Core
    Core -.-|ideally abstract| Stable

    style Stable fill:#dcfce7,stroke:#16a34a
    style Leaf fill:#dbeafe,stroke:#2563eb
```

## Key Concepts

- **REP (Reuse-Release Equivalence Principle)**: The granule of reuse is the granule of release. A package must be released as a versioned unit — everything in it is released together. This forces semantic cohesion: classes that don't belong together by meaning shouldn't share a package just for convenience.

- **CCP (Common Closure Principle)**: Classes that change for the same reasons should be in the same package. This is the package-level equivalent of SRP. When a requirement changes, ideally only one package needs to change — reducing the number of packages that need to be re-released, re-tested, and re-deployed.

- **CRP (Common Reuse Principle)**: Don't force users of a package to depend on things they don't need. When you depend on a package, you depend on all of it — so put only classes that are always used together into the same package. This is the package-level ISP.

- **ADP (Acyclic Dependencies Principle)**: The dependency graph of packages must have no cycles. Cycles prevent independent builds and releases — to build package A you need B which needs C which needs A. Break cycles using DIP (extract an interface package that both sides depend on) or by merging the cyclic packages.

- **SDP (Stable Dependencies Principle)**: A package should only depend on packages that are more stable than itself. Stability is measured as I = Fan-out / (Fan-in + Fan-out) where I=0 means maximally stable (many things depend on it, it depends on nothing) and I=1 means maximally unstable. Depending on something more changeable than yourself makes your package fragile.

- **SAP (Stable Abstractions Principle)**: A package's abstractness should be proportional to its stability. Stable packages (many dependents) should be abstract (interfaces, abstract classes) so that they can be extended without modification. Concrete stable packages (the Zone of Pain) are rigid and hard to change without breaking everything.

## Trade-offs

| Principle | Benefit | Cost of Violation |
|-----------|---------|-------------------|
| REP | Clear release units | Semantic incoherence, confusing APIs |
| CCP | Fewer packages to change per requirement | Large packages with mixed concerns |
| CRP | Minimal transitive dependencies | Many small, hard-to-discover packages |
| ADP | Independent builds and deployments | Circular compile-time dependencies |
| SDP | Stable dependency hierarchy | Fragile packages depending on unstable ones |
| SAP | Stable packages can be extended | Concrete stable packages cannot evolve |

## When to Apply

- Apply package design principles when a codebase grows beyond a single team or has multiple independently deployable components
- ADP is non-negotiable for large codebases — cycles must be detected and broken by CI
- Use SDP and SAP to guide where interfaces and abstractions should live versus concrete implementations
- The tension between REP, CCP, and CRP shifts over a project's lifetime — early projects optimise for CCP (change together), mature projects shift toward REP (independent release) and CRP (minimal dependencies)