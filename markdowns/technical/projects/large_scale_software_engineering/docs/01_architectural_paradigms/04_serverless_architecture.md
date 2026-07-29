---
title: "Serverless Architecture"
subtitle: "Serverless architecture abstracts away server management entirely — developers deploy functions or backend services without provisioning infrastructure. Execution is triggered by events and billed per invocation,..."
category: technical
project: large_scale_software_engineering
project_title: "Large Scale Software Engineering"
date: 2025-04-27
reading_time: 3
tags:
  - large-scale-software-engineering
  - docs
author: "Shreedhar Kodate"
output: "blogs/technical/posts/large_scale_software_engineering/docs/01_architectural_paradigms/04_serverless_architecture.html"
---
Serverless architecture abstracts away server management entirely — developers deploy functions or backend services without provisioning infrastructure. Execution is triggered by events and billed per invocation, enabling near-infinite scale with zero idle cost, at the expense of runtime constraints and cold-start latency.

## Architecture Diagrams

### Serverless Application Architecture

```mermaid
graph TD
    subgraph Triggers
        HTTP[HTTP / API Gateway]
        S3E[S3 Bucket Event]
        Queue[SQS Queue]
        Schedule[EventBridge Schedule]
        Stream[Kinesis Stream]
    end

    subgraph Functions[Lambda Functions / FaaS]
        FnAPI[API Handler Fn]
        FnProc[Image Processor Fn]
        FnWorker[Queue Worker Fn]
        FnCron[Scheduled Job Fn]
        FnStream[Stream Processor Fn]
    end

    subgraph Backends[Backend Services - BaaS]
        DDB[(DynamoDB)]
        S3[S3 Storage]
        RDS[RDS Proxy]
        SNS[SNS Notifications]
        CF[CloudFront CDN]
    end

    HTTP --> FnAPI
    S3E --> FnProc
    Queue --> FnWorker
    Schedule --> FnCron
    Stream --> FnStream

    FnAPI --> DDB
    FnAPI --> S3
    FnProc --> S3
    FnWorker --> RDS
    FnCron --> SNS
    FnStream --> DDB

    style Functions fill:#fef3c7,stroke:#d97706,stroke-width:2px
    style Backends fill:#eff6ff,stroke:#3b82f6
```

### Cold Start Lifecycle

```mermaid
stateDiagram-v2
    [*] --> Cold: First invocation or after idle
    Cold --> Initializing: Runtime download
    Initializing --> HandlerInit: Container bootstrap
    HandlerInit --> Warm: Handler code loaded
    Warm --> Executing: Request arrives
    Executing --> Warm: Request complete, container retained
    Warm --> Frozen: Idle timeout
    Frozen --> Warm: Invocation arrives (warm start)
    Frozen --> [*]: Garbage collected

    note right of Cold: 100ms - 10s penalty
    note right of Warm: <10ms overhead
```

### Serverless Event Processing Pipeline

```mermaid
graph LR
    API[API Gateway] -->|HTTP POST| AuthFn[Auth Function]
    AuthFn -->|Authorized| ValidFn[Validation Function]
    ValidFn -->|Valid| Queue[SQS FIFO Queue]
    Queue --> ProcessFn[Process Function]
    ProcessFn --> DB[(DynamoDB)]
    ProcessFn --> Notify[SNS Topic]
    Notify --> Email[Email Lambda]
    Notify --> Push[Push Notification Lambda]

    ProcessFn -->|DLQ on failure| DLQ[Dead Letter Queue]
    DLQ --> AlertFn[Alert Function]
```

## Key Concepts

- **Function as a Service (FaaS)**: The compute unit in serverless. A function is a single-purpose, stateless code unit that executes in response to a trigger. AWS Lambda, Google Cloud Functions, and Azure Functions are the major implementations. Each invocation gets its own isolated ephemeral execution environment.

- **Backend as a Service (BaaS)**: Managed cloud services (databases, auth, storage, queues) that replace self-managed backend infrastructure. Serverless applications combine FaaS for compute with BaaS for persistence and cross-cutting concerns.

- **Cold Start**: When a function hasn't been invoked recently, the cloud provider must allocate a container, download the runtime, and initialize the application. This adds 100ms–10s of latency. Mitigation strategies include provisioned concurrency, minimising dependency bundle size, and using lightweight runtimes.

- **Stateless Execution**: Each function invocation must treat execution context as ephemeral. Persistent state must be externalised to databases, object storage, or caches. The `/tmp` filesystem is available within an invocation but not guaranteed between invocations.

- **Event Triggers**: Functions are invoked by events from API gateways, object storage events, queue messages, stream records, scheduled events, or other cloud service triggers. The trigger defines the concurrency model and retry behaviour.

- **Concurrency Model**: Cloud providers scale function instances automatically — each concurrent request gets its own instance. This enables massive parallelism but can overwhelm downstream databases (connection storm anti-pattern).

- **Dead Letter Queues (DLQ)**: Failed async invocations that exhaust retries are sent to a DLQ for inspection and reprocessing. Essential for event-driven serverless reliability.

- **Step Functions / Durable Orchestration**: For multi-step workflows requiring state, orchestration engines (AWS Step Functions, Azure Durable Functions) coordinate function execution with retry, branching, and parallel execution — externalising workflow state from functions.

## Trade-offs

| Aspect | Serverless | Containers (Always-on) |
|--------|-----------|----------------------|
| Idle cost | Zero | Full cost |
| Cold start latency | Present | None |
| Maximum execution time | 15 min (AWS Lambda) | Unlimited |
| Operational overhead | Minimal | Significant |
| Memory/CPU control | Limited | Full |
| Local dev experience | Complex | Straightforward |
| Vendor lock-in | High | Moderate |
| Concurrency control | Automatic | Manual |
| Database connections | Can exhaust pools | Predictable |

## When to Use

**Use serverless when:**
- Workloads are spiky, intermittent, or unpredictable
- Event-driven processing pipelines (file uploads, queue workers, webhooks)
- Rapid prototyping where operational simplicity is prioritised
- Tasks with clear execution boundaries (image resize, PDF generation, data transformation)

**Avoid when:**
- Long-running processes exceed FaaS time limits
- Cold start latency is unacceptable (real-time user-facing APIs with strict p99 SLAs)
- High-throughput persistent connections (WebSockets, streaming) are needed
- Tight control over runtime, networking, or hardware is required