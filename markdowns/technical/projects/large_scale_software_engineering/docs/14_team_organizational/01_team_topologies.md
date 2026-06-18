# Team Topologies

Team Topologies (Matthew Skelton and Manuel Pais) provides a framework for organizing teams to optimize software delivery flow. It identifies four fundamental team types and three interaction modes, with the goal of reducing cognitive load and enabling fast, independent delivery.

## Four Team Types

```mermaid
graph TD
    subgraph StreamAligned[Stream-Aligned Team]
        SA[Aligned to a flow of business change\nCross-functional: dev ops product design\nFull ownership: code to production\nMost common team type\nExamples: checkout team orders team]
        style SA fill:#dcfce7,stroke:#16a34a
    end

    subgraph Platform[Platform Team]
        PT[Provides self-service capabilities\nto stream-aligned teams\nReduces cognitive load\nExamples: infra team CI/CD team\ndeveloper experience team]
        style PT fill:#dbeafe,stroke:#2563eb
    end

    subgraph Enabling[Enabling Team]
        ET[Time-limited specialists\nHelp stream-aligned teams\nacquire missing capabilities\nExamples: security champions\nperformance specialists]
        style ET fill:#fef3c7,stroke:#d97706
    end

    subgraph ComplexSubsystem[Complicated Subsystem Team]
        CS[Specialist knowledge required\nHeavy cognitive load of subsystem\nExamples: ML model team\ngraphics engine team\ncryptography team]
        style CS fill:#ffe4e6,stroke:#be123c
    end
```

## Three Interaction Modes

```mermaid
graph LR
    subgraph Collaboration[Collaboration Mode]
        C[Two teams work closely together\nfor a defined period\nHigh bandwidth, high cost\nUsed for new capabilities\nor solving complex unknowns]
        style C fill:#dcfce7,stroke:#16a34a
    end

    subgraph XaaS[X-as-a-Service Mode]
        X[One team provides a service\nother consumes it\nLow interaction, self-service\nClear API contract\nUsed for platform services]
        style X fill:#dbeafe,stroke:#2563eb
    end

    subgraph Facilitating[Facilitating Mode]
        F[Enabling team helps\nstream-aligned team\nlearn and adopt capability\nFades out after transfer]
        style F fill:#fef3c7,stroke:#d97706
    end
```

## Conway's Law and Inverse Conway Maneuver

```mermaid
graph TD
    subgraph ConwaysLaw[Conway's Law]
        Law[Any organization that designs a system\nwill produce a design whose structure\nis a copy of the organization's\ncommunication structure]
        Example[Monolith org → Monolithic software\nSiloed teams → Siloed services\nStream-aligned teams → Independent microservices]
        Law --> Example
    end

    subgraph InverseConway[Inverse Conway Maneuver]
        ICM[First design the desired software architecture\nThen structure teams to match\nTeam communication patterns\nwill produce the desired system structure]
        style ICM fill:#dcfce7,stroke:#16a34a
    end
```

## Cognitive Load and Team Size

```mermaid
graph TD
    subgraph CognitiveLoad[Managing Cognitive Load]
        Intrinsic[Intrinsic Load\nInherent complexity of the domain\nCannot be reduced\nMinimize by good abstractions]
        Extraneous[Extraneous Load\nEnvironment complexity\nPoor tooling, process overhead\nMinimize by platform teams]
        Germane[Germane Load\nLearning and innovation\nDesirable cognitive load]

        Intrinsic & Extraneous --> TotalLoad[Total Cognitive Load\nMust fit within team capacity]
        TotalLoad --> TeamSize[Team sizing: 5-8 engineers\nBrooks Law: adding people to a late project\nmakes it later]
    end
```

## Key Concepts

- **Stream-Aligned Team**: A cross-functional team (engineering, product, design, operations) aligned to a stream of business change — a product, feature, or user journey. The team has full end-to-end ownership from code to production metrics. This is the most common and most important team type.

- **Platform Team**: Builds and maintains internal platforms (Kubernetes infrastructure, CI/CD pipelines, shared libraries, developer portals) that reduce the cognitive load on stream-aligned teams. The platform team's customers are other engineering teams. The platform must be self-service — stream-aligned teams should not need to contact the platform team for every infrastructure change.

- **Enabling Team**: A temporary team of specialists that helps stream-aligned teams acquire capabilities they currently lack (Kubernetes expertise, security best practices, performance optimization). Unlike a centre of excellence, enabling teams intend to transfer knowledge and make themselves unnecessary for any given team.

- **Cognitive Load**: The mental effort required to understand a system and perform a task. Brook's Law describes the maximum cognitive load a team can sustainably maintain. Platform teams reduce cognitive load for stream-aligned teams by abstracting away infrastructure complexity.

- **Conway's Law**: Mel Conway (1967) observed that organizations tend to produce systems that mirror their own communication structure. A siloed organization with separate frontend, backend, and DBA teams tends to produce a three-tier monolith. The implication: if you want a certain architecture, design the team structure first.

- **Inverse Conway Maneuver**: Deliberately structure teams to match the desired target architecture. If the goal is microservices, create stream-aligned teams with end-to-end ownership of services before the code is decomposed — the architecture will follow.

- **Team Cognitive Load Limit**: A team can sustainably own a bounded set of services and domains. When a team's cognitive load exceeds their capacity, quality, velocity, and wellbeing suffer. The solution is to split the team or invest in platform tooling that reduces incidental complexity.

## Trade-offs

| Structure | Delivery Speed | Cognitive Load | Coordination |
|-----------|--------------|----------------|-------------|
| Feature teams | High | High | Low |
| Functional teams | Low | Low | High |
| Stream-aligned (Topologies) | High | Managed | Medium |
| Monolithic team | Variable | High | Low initially |

## When to Apply

- Team Topologies is most valuable when an organization grows beyond 30-50 engineers and informal coordination breaks down
- Invest in platform teams when stream-aligned teams spend more than 30% of time on infrastructure and toil
- Use enabling teams for time-limited capability transfer, not for permanent gatekeeping
- Measure team cognitive load through surveys and team health checks
