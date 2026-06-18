# Resilience Patterns

Resilience patterns make systems robust against partial failures, resource exhaustion, and cascade effects. A resilient system degrades gracefully under load and recovers automatically — without requiring human intervention for every failure mode.

## Resilience Pattern Overview

```mermaid
graph TD
    FailureScenario[Service A calls Service B\nService B becomes slow or unavailable]

    FailureScenario --> CB[Circuit Breaker\nStop calling failing service\nFail fast instead]
    FailureScenario --> Retry[Retry with Backoff\nRetry transient failures\nwith exponential wait]
    FailureScenario --> Timeout[Timeout\nDon't wait forever\nFail after N ms]
    FailureScenario --> BH[Bulkhead\nIsolate failures\nSeparate thread pools]
    FailureScenario --> Fallback[Fallback\nReturn cached data\nor degraded response]

    style CB fill:#fef3c7,stroke:#d97706
    style BH fill:#dbeafe,stroke:#2563eb
    style Fallback fill:#dcfce7,stroke:#16a34a
```

## Bulkhead Pattern

```mermaid
graph TD
    subgraph NoBulkhead[Without Bulkhead - Cascade Failure]
        SharedPool[Shared Thread Pool: 100 threads]
        PaymentSlow[Payment Service - slow\nHolds 90 threads waiting]
        CatalogOK[Catalog Service - healthy\nOnly 10 threads available\nDegrades too]
        SharedPool --> PaymentSlow & CatalogOK
        style SharedPool fill:#fee2e2,stroke:#dc2626
    end

    subgraph WithBulkhead[With Bulkhead - Isolated Failure]
        PayPool[Payment Thread Pool: 50 threads]
        CatalogPool[Catalog Thread Pool: 30 threads]
        DefaultPool[Default Thread Pool: 20 threads]
        PaymentSlow2[Payment Service - slow\nFills its own pool\nOther services unaffected]
        CatalogOK2[Catalog Service - healthy\nFull pool available]
        PayPool --> PaymentSlow2
        CatalogPool --> CatalogOK2
        style PayPool fill:#fef3c7,stroke:#d97706
        style CatalogPool fill:#dcfce7,stroke:#16a34a
    end
```

## Timeout Cascade Prevention

```mermaid
sequenceDiagram
    participant Client
    participant SvcA
    participant SvcB
    participant SvcC

    Note over Client: Client timeout: 1000ms
    Client->>SvcA: Request (deadline: 1000ms)
    Note over SvcA: SvcA timeout to SvcB: 500ms
    SvcA->>SvcB: Request (deadline: 500ms remaining)
    Note over SvcB: SvcB timeout to SvcC: 200ms
    SvcB->>SvcC: Request (deadline: 200ms remaining)
    SvcC--xSvcB: Timeout after 200ms
    SvcB-->>SvcA: 503 error with remaining time context
    SvcA-->>Client: 503 error (within 700ms total)

    Note over Client: Timeout budgets flow downstream
    Note over Client: Prevents one slow service from\nblocking the entire call chain
```

## Chaos Engineering

```mermaid
flowchart TD
    A[Define Steady State\nWhat does normal look like?\ne.g. p99 latency under 200ms\nerror rate under 0.1%] --> B[Hypothesize\nIf we kill 1 instance\nthe system should remain\nwithin steady state]
    B --> C[Inject Chaos\nin non-production first\ne.g. terminate random pod\ndelay network calls 500ms\nexhaust CPU on one node]
    C --> D[Observe\nDoes system stay within\nsteady state metrics?]
    D -->|Yes| E[Expand scope\nRun in production\nor increase failure severity]
    D -->|No| F[Fix the discovered weakness\nAdd circuit breaker\nImprove monitoring\nFix recovery procedure]
    F --> A
    E --> F
```

## Key Concepts

- **Timeout**: Every outbound call must have an explicit timeout. Without timeouts, a hanging downstream service causes threads to wait indefinitely, eventually exhausting the thread pool. Timeouts should be cascaded: the outermost timeout must be shorter than the sum of all downstream timeouts to prevent the caller from timing out before getting a response.

- **Retry with Exponential Backoff**: Automatically retry failed operations with increasing wait times. Only appropriate for idempotent operations and transient errors (network blip, temporary overload). Always add jitter to prevent synchronized retry storms. Set a maximum retry count to avoid infinite retries.

- **Circuit Breaker**: Monitors failure rates for a service dependency. When failures exceed a threshold, the circuit "opens" — subsequent calls fail immediately without attempting the downstream call. After a cooldown, the circuit "half-opens" to probe recovery. Prevents cascading failures and allows failing services to recover without being overwhelmed.

- **Bulkhead**: Allocates separate resource pools for different dependencies. If one dependency becomes slow or unresponsive, only its bulkhead fills — other dependencies retain their own pools. Named after ship compartments that prevent sinking by containing flooding to a section.

- **Fallback**: When a dependency fails, return a degraded but acceptable response. Examples: return cached data, return an empty list, return a default recommendation, or return a user-facing message explaining partial functionality. Fallbacks must be designed at the feature level — what is "good enough" when the real answer is unavailable?

- **Chaos Engineering**: The discipline of intentionally injecting failures into a system to find weaknesses before they manifest as incidents. Based on the scientific method: define steady state, hypothesize, experiment, observe, improve. Netflix Chaos Monkey is the original implementation.

- **Graceful Degradation**: A system under partial failure continues to serve core functionality while degrading non-critical features. Example: a product page still shows the product without reviews if the review service is down.

- **Hedging (Speculative Retry)**: Send the same request to multiple backends simultaneously and use the first response. Reduces tail latency at the cost of increased resource utilisation. Used when p99 latency is critical and extra load on backends is acceptable.

## Trade-offs

| Pattern | Benefit | Cost |
|---------|---------|------|
| Timeout | Prevents indefinite blocking | May fail valid slow operations |
| Retry | Hides transient failures | Can amplify load on struggling service |
| Circuit Breaker | Prevents cascade failures | Delayed recovery detection |
| Bulkhead | Failure isolation | Resource underutilisation |
| Fallback | User experience preserved | May serve stale data |
| Chaos Engineering | Finds weaknesses before incidents | Risk (even in controlled environments) |

## When to Use

- **Timeout**: Always — on every outbound call without exception
- **Retry**: Idempotent operations with transient failure patterns (network blips, rate limits)
- **Circuit Breaker**: All synchronous calls to external services
- **Bulkhead**: When multiple downstream dependencies have different reliability characteristics
- **Fallback**: For non-critical features where a degraded response is better than an error
- **Chaos Engineering**: After basic observability and on-call processes are mature
