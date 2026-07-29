---
title: "Edge and Distributed Computing"
subtitle: "Edge computing moves computation closer to where data is generated and consumed — end users, IoT devices, and geographically distributed systems. This reduces latency dramatically and enables new application patterns..."
category: technical
project: large_scale_software_engineering
project_title: "Large Scale Software Engineering"
date: 2025-11-01
reading_time: 3
tags:
  - large-scale-software-engineering
  - docs
author: "Shreedhar Kodate"
output: "blogs/technical/posts/large_scale_software_engineering/docs/15_emerging_paradigms/02_edge_distributed_computing.html"
---
Edge computing moves computation closer to where data is generated and consumed — end users, IoT devices, and geographically distributed systems. This reduces latency dramatically and enables new application patterns that were impossible when all computation was centralized.

## Edge Compute Hierarchy

```mermaid
graph TD
    subgraph Device[Device Edge]
        IoT[IoT Sensors\nEmbedded ML\nLocal inference]
        Mobile[Mobile Devices\nOn-device AI\nOffline capability]
    end

    subgraph NetworkEdge[Network Edge - MEC]
        CellTower[5G Cell Tower\nMulti-Access Edge Computing\nUltra-low latency\n1-5ms to device]
        CPE[Customer Premise\nEquipment]
    end

    subgraph CDNEdge[CDN Edge - PoPs]
        CloudflareEdge[Cloudflare Workers\nVercel Edge Functions\nFastly Compute\n50-100ms to global users]
    end

    subgraph Cloud[Cloud Region]
        AWS[AWS Region\nFull compute and data\n100-300ms globally]
    end

    Device --> NetworkEdge --> CDNEdge --> Cloud

    style Device fill:#dcfce7,stroke:#16a34a
    style CDNEdge fill:#dbeafe,stroke:#2563eb
    style Cloud fill:#fef3c7,stroke:#d97706
```

## CDN Edge Computing (Cloudflare Workers)

```mermaid
graph TD
    User[User in Singapore] --> CF[Cloudflare PoP\nSingapore\nEdge Function executes HERE]
    CF --> Origin[Origin Server\nUS-East\ncalled only when needed]

    subgraph EdgeCapabilities[What Runs at Edge]
        Auth[JWT Validation\nno origin round-trip]
        Routing[A/B Test Routing\nCanary traffic splitting]
        Transform[Response Transformation\nPersonalization]
        Cache[Edge Cache\nContent serving]
        Rate2[Rate Limiting\nper IP globally]
    end

    CF --> Auth & Routing & Transform & Cache & Rate2

    Latency[Result:\n10ms vs 200ms\nfor authenticated edge response]
    style Latency fill:#dcfce7,stroke:#16a34a,stroke-width:2px
```

## WebAssembly (WASM) at the Edge

```mermaid
graph LR
    subgraph WASM[WebAssembly Value Proposition]
        Lang[Write in: Rust Go C Python]
        Compile[Compile to WASM\nportable binary]
        Run[Run anywhere:\nBrowser\nEdge functions\nServer-side\nEmbedded]
        Lang --> Compile --> Run
    end

    subgraph Advantages[WASM Advantages]
        Fast[Near-native performance\nno interpreter overhead]
        Safe[Sandboxed execution\nno system access by default]
        Portable[Write once run anywhere\ncross-platform]
        Small[Small binary size\nfast startup]
    end

    style Fast fill:#dcfce7,stroke:#16a34a
    style Safe fill:#dbeafe,stroke:#2563eb
```

## Key Concepts

- **Edge Computing**: Processing data at or near the source of data generation rather than in a centralized cloud region. Reduces latency by bringing compute closer to users, reduces bandwidth costs by processing data locally before sending to the cloud, and enables offline-capable applications.

- **CDN Edge Functions**: Serverless functions that run within CDN points of presence (PoPs) globally. Cloudflare Workers, Vercel Edge Functions, and Fastly Compute at Edge execute JavaScript, TypeScript, and WASM at hundreds of locations worldwide. Enables sub-10ms response times for authenticated, personalized responses.

- **Multi-Access Edge Computing (MEC)**: ETSI standard for deploying compute within cellular network infrastructure (at or near cell towers). Enables ultra-low latency (1-5ms) applications for autonomous vehicles, augmented reality, and real-time industrial control that require more than CDN edge can provide.

- **WebAssembly (WASM)**: A binary instruction format for a stack-based virtual machine. Designed as a portable compilation target for high-level languages (Rust, C++, Go). Runs at near-native speed in browsers and server-side runtimes. WASM + WASI (system interface) enables portable server-side compute without containers.

- **Offline-First**: Application design where the application is fully functional without internet connectivity. Data is stored locally (IndexedDB, SQLite via SQLite WASM) and synchronized when connectivity is available. Conflict resolution is required. CRDTs enable conflict-free offline-first sync.

- **Durable Objects (Cloudflare)**: Stateful compute at the edge. Single-instance JavaScript objects that run at the closest edge location to the client, maintaining state across requests. Enables real-time collaboration (shared cursors, chat), game servers, and rate limiting without a central data store.

## Trade-offs

| Approach | Latency | Compute Power | Data Consistency | Cost |
|----------|---------|--------------|-----------------|------|
| Cloud-only | High | Unlimited | Strong | Low (scale) |
| CDN edge | Low | Limited | Eventual | Medium |
| MEC | Very low | Moderate | Eventual | High |
| On-device | Near-zero | Very limited | Local | Hardware |

## When to Use

- **CDN edge functions**: Authentication, A/B testing, personalization, bot protection — request-scoped logic that doesn't need database writes
- **Full edge compute**: Real-time applications requiring ultra-low latency (gaming, AR/VR, industrial IoT)
- **WASM at edge**: Compute-intensive tasks (image processing, ML inference) that exceed JavaScript performance limits
- **Offline-first**: Consumer mobile apps where poor connectivity is expected (developing regions, mobile use cases)