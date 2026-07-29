---
title: "The Three Pillars of Observability"
subtitle: "The three pillars of observability — metrics, traces, and logs — provide complementary views into system behaviour. Metrics answer \"what is wrong\"; traces answer \"where is it wrong\"; logs answer \"why is it wrong\"...."
category: technical
project: large_scale_software_engineering
project_title: "Large Scale Software Engineering"
date: 2025-10-28
reading_time: 4
tags:
  - large-scale-software-engineering
  - docs
author: "Shreedhar Kodate"
output: "blogs/technical/posts/large_scale_software_engineering/docs/09_observability_monitoring/01_three_pillars.html"
---
The three pillars of observability — metrics, traces, and logs — provide complementary views into system behaviour. Metrics answer "what is wrong"; traces answer "where is it wrong"; logs answer "why is it wrong". Together they enable effective production debugging and monitoring.

## Three Pillars Overview

```mermaid
graph TD
    subgraph Metrics[Metrics - What is happening]
        M1[Numeric time-series data]
        M2[Aggregated - not per request]
        M3[Low cardinality labels]
        M4[Prometheus Datadog CloudWatch]
        M5[Use for: alerting dashboards capacity]
    end

    subgraph Traces[Traces - Where is it happening]
        T1[Per-request call trees]
        T2[Spans with timing and metadata]
        T3[Distributed across services]
        T4[Jaeger Zipkin Tempo DatadogAPM]
        T5[Use for: latency debugging bottlenecks]
    end

    subgraph Logs[Logs - Why it happened]
        L1[Timestamped event records]
        L2[High cardinality details]
        L3[Structured JSON preferred]
        L4[Elasticsearch Splunk Loki CloudWatch]
        L5[Use for: root cause debugging errors]
    end

    Alert[Alert fires:\np99 latency elevated] --> Metrics
    Metrics --> Traces
    Traces --> Logs

    style Metrics fill:#dbeafe,stroke:#2563eb
    style Traces fill:#fef3c7,stroke:#d97706
    style Logs fill:#dcfce7,stroke:#16a34a
```

## Metrics Types

```mermaid
graph TD
    subgraph MetricTypes[Prometheus Metric Types]
        Counter[Counter\nMonotonically increasing\nHTTP requests total\nErrors total\nBytes processed]
        Gauge[Gauge\nCan go up or down\nActive connections\nMemory usage\nQueue depth]
        Histogram[Histogram\nObservation distribution\nRequest duration in buckets\nle 0.1 le 0.5 le 1.0 le inf]
        Summary[Summary\nPre-computed percentiles\nLess flexible than histogram]
    end

    Counter --> Use1[Rate queries: rate in 5m]
    Gauge --> Use2[Current value: memory_used_bytes]
    Histogram --> Use3[Percentile queries: histogram_quantile 0.99]
```

## Distributed Tracing

```mermaid
graph TD
    subgraph Trace[Distributed Trace: order placement]
        Root[HTTP POST /orders\nSpan: 350ms total]

        subgraph AuthSpan[Auth Validation: 15ms]
            AuthDetail[Validate JWT\nCheck permissions]
        end

        subgraph DBSpan[Database Write: 45ms]
            DBDetail[INSERT INTO orders\nPostgres primary]
        end

        subgraph PaySpan[Payment Service: 280ms]
            PayDetail[gRPC call\nCharge credit card]
            subgraph ExtSpan[Stripe API: 240ms]
                ExtDetail[External HTTP call\nP99 bottleneck here]
            end
        end

        Root --> AuthSpan & DBSpan & PaySpan
        PaySpan --> ExtSpan
    end

    style ExtSpan fill:#fee2e2,stroke:#dc2626,stroke-width:2px
```

## Context Propagation

```mermaid
sequenceDiagram
    participant Client
    participant SvcA
    participant SvcB
    participant SvcC

    Client->>SvcA: HTTP Request\nno trace context
    SvcA->>SvcA: Generate trace_id: abc123\nspan_id: span1
    SvcA->>SvcB: HTTP Request\ntraceparent: 00-abc123-span1-01
    SvcB->>SvcB: New span: span2\nparent: span1
    SvcB->>SvcC: HTTP Request\ntraceparent: 00-abc123-span2-01
    SvcC->>SvcC: New span: span3\nparent: span2
    SvcC-->>SvcB: Response
    SvcB-->>SvcA: Response
    SvcA-->>Client: Response

    Note over SvcA,SvcC: All spans share trace_id: abc123\nLinked by parent-child span relationships
```

## Key Concepts

- **Metrics**: Numeric measurements aggregated over time, stored as time-series. Low storage cost — aggregation discards individual request details. Excellent for alerting (threshold-based and SLO burn rate), capacity planning, and dashboard visualization. Prometheus is the standard open-source system; the OpenMetrics format enables interoperability.

- **Counter**: A metric that only increases. Represents cumulative totals (total requests, total errors, total bytes). To find the rate, use the rate() or irate() function. Counters reset to zero on service restart.

- **Gauge**: A metric that can increase or decrease. Represents current state (active connections, memory usage, queue depth, temperature). Use directly for current-value dashboards.

- **Histogram**: Records observations (request durations, response sizes) in configurable buckets. Enables server-side percentile calculation via histogram_quantile. More flexible than summaries for cross-dimensional aggregation. The recommended metric type for latency measurement.

- **Distributed Trace**: A collection of spans representing a single request's journey through a distributed system. Each span records an operation (HTTP call, DB query, cache lookup) with its start time, duration, and metadata. Spans are linked by parent-child relationships, forming a tree rooted at the entry point.

- **Trace Sampling**: Tracing every request at high throughput is expensive. Sampling strategies: head-based (decide to trace at entry point based on probability), tail-based (decide after the request completes, keeping traces that show errors or high latency). Tail-based sampling with OpenTelemetry Collector is the most powerful approach.

- **OpenTelemetry (OTel)**: A CNCF project providing vendor-neutral APIs, SDKs, and collector for metrics, traces, and logs. Instrument code once with OTel; route telemetry to any backend (Jaeger, Prometheus, Datadog, Honeycomb). The standard for modern observability instrumentation.

- **Correlation ID**: A unique identifier attached to every request at the entry point (API gateway or first service). Propagated in HTTP headers to all downstream services. Stored in every log line. Enables correlating logs across services for a single request.

## Trade-offs

| Pillar | Detail Level | Storage Cost | Query Speed | Best For |
|--------|------------|-------------|-------------|---------|
| Metrics | Low (aggregated) | Low | Fast | Alerting, dashboards |
| Traces | Medium (per request) | Medium | Medium | Latency debugging |
| Logs | High (per event) | High | Slow | Root cause analysis |

## When to Use

- **Start with metrics**: Set up Prometheus + Grafana for all services before anything else
- **Add traces**: When debugging "why is this endpoint slow" becomes frequent — traces show where time is spent
- **Improve logs**: Move from unstructured text to structured JSON with correlation IDs for searchable logs
- **Use all three**: Production incidents typically require all three: metrics detect the problem, traces localize it, logs explain it