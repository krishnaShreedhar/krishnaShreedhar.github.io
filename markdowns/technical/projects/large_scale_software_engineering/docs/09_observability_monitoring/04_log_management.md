---
title: "Log Management"
subtitle: "Log management covers the production, collection, aggregation, storage, and analysis of application and system logs. Effective logging provides the detailed narrative needed to diagnose production incidents and audit..."
category: technical
project: large_scale_software_engineering
project_title: "Large Scale Software Engineering"
date: 2025-12-28
reading_time: 3
tags:
  - large-scale-software-engineering
  - docs
author: "Shreedhar Kodate"
output: "blogs/technical/posts/large_scale_software_engineering/docs/09_observability_monitoring/04_log_management.html"
---
Log management covers the production, collection, aggregation, storage, and analysis of application and system logs. Effective logging provides the detailed narrative needed to diagnose production incidents and audit security events.

## Structured Logging

```mermaid
graph LR
    subgraph Unstructured[Unstructured Logs - Hard to Parse]
        ULog[2024-01-15 10:23:45 ERROR Payment failed for user 12345 amount 99.99 error Card declined]
        UProblem[Cannot filter by user_id\nCannot aggregate error types\nCannot correlate with traces]
        style ULog fill:#fee2e2,stroke:#dc2626
        style UProblem fill:#fee2e2,stroke:#dc2626
    end

    subgraph Structured[Structured JSON Logs - Queryable]
        SLog[JSON object:\ntimestamp: 2024-01-15T10:23:45Z\nlevel: ERROR\nservice: payment-service\ntrace_id: abc123\nspan_id: xyz789\nuser_id: 12345\namount: 99.99\nerror: card_declined\nerror_code: 4001\nduration_ms: 234]
        SGood[Filter by user_id=12345\nAggregate by error_code\nJoin with trace abc123]
        style SLog fill:#dcfce7,stroke:#16a34a
        style SGood fill:#dcfce7,stroke:#16a34a
    end
```

## Log Aggregation Pipeline

```mermaid
graph TD
    subgraph Sources[Log Sources]
        AppLogs[Application Logs\nJSON to stdout]
        K8sLogs[Kubernetes Pod Logs]
        SysLogs[System Logs\nsyslog journald]
        AuditLogs[Security Audit Logs\nCloudTrail K8s audit]
    end

    subgraph Collection[Collection]
        FluentBit[Fluent Bit\nDaemonSet collector\nlow overhead]
        Fluentd[Fluentd\naggregator]
        FluentBit --> Fluentd
    end

    subgraph Storage[Storage and Indexing]
        Kafka2[Kafka\nbuffer and fan-out]
        ES[Elasticsearch\nfull-text search\nJSON indexing]
        Loki[Grafana Loki\nlog aggregation\nchunked storage]
        S3[S3 Glacier\nlong-term archive]
    end

    subgraph Analysis[Analysis]
        Kibana[Kibana\nElasticsearch UI]
        GrafanaUI[Grafana\nLoki queries]
        Athena[AWS Athena\nSQL on S3 logs]
    end

    Sources --> FluentBit
    Fluentd --> Kafka2
    Kafka2 --> ES & Loki & S3
    ES --> Kibana
    Loki --> GrafanaUI
    S3 --> Athena

    style FluentBit fill:#dbeafe,stroke:#2563eb
    style ES fill:#fef3c7,stroke:#d97706
```

## Log Levels

```mermaid
graph TD
    subgraph LogLevels[Log Level Hierarchy]
        TRACE[TRACE\nVery detailed execution flow\nDisabled in production\nPerformance cost]
        DEBUG[DEBUG\nDiagnostic information\nEnabled in dev and staging\nHigh volume]
        INFO[INFO\nBusiness-level events\nRequest received payment processed\nAlways enabled in production]
        WARN[WARN\nUnexpected but recoverable\nRetry succeeded slow query\nAlways enabled]
        ERROR[ERROR\nFailure requiring attention\nException stack traces\nAlways enabled - should alert]
        FATAL[FATAL\nApplication cannot continue\nLog then exit\nAlways enabled]
    end

    TRACE --> DEBUG --> INFO --> WARN --> ERROR --> FATAL

    Production[Production recommendation:\nINFO and above always\nDEBUG on-demand with dynamic config\nNever TRACE in production]
    style Production fill:#dcfce7,stroke:#16a34a
```

## Key Concepts

- **Structured Logging**: Write logs as JSON (or another parseable format) instead of free-form strings. Every log event should include: timestamp (ISO 8601), level, service name, trace_id, span_id, user_id (if applicable), and relevant domain fields. Structured logs can be queried, aggregated, and correlated with traces.

- **Correlation IDs**: A unique ID generated at the request entry point and propagated through all downstream calls and stored in all log lines. Enables finding all logs related to a single request across multiple services. Typically the same as the OpenTelemetry trace ID.

- **Log Levels**: Use levels appropriately — INFO for business events (request processed, user logged in), WARN for unexpected but handled conditions (retry succeeded, fallback used), ERROR for failures requiring investigation (exception thrown, external call failed), DEBUG for developer information.

- **Log Aggregation**: Centralize logs from all instances and services into a single searchable system. Without aggregation, debugging requires SSH-ing into individual instances and grepping log files. Fluent Bit (lightweight) collects from each host; Fluentd or Kafka buffers and routes; Elasticsearch or Loki stores.

- **Retention Policy**: Different log types need different retention. Security audit logs: 12-24 months (compliance). Application error logs: 30-90 days (operational). Debug logs: 7 days maximum. Archive to cold storage (S3 Glacier) for compliance requirements beyond operational retention.

- **Log Sampling**: High-throughput services can generate terabytes of logs per day. For INFO-level request logs, sample at 1-10% in production. Always log ERROR and WARN at 100%. Use the trace sampling decision to ensure sampled traces have full logs.

- **Security Audit Logs**: A separate, tamper-evident log stream of security-relevant events: logins (successful and failed), privilege escalations, data access, configuration changes, and secrets access. These logs must be protected from tampering (append-only storage) and retained for compliance.

## Trade-offs

| Approach | Queryability | Storage Cost | Setup Complexity |
|---------|------------|-------------|-----------------|
| Unstructured logs | Low | Low | Low |
| Structured JSON logs | High | Medium | Low |
| Elasticsearch | Very high | High | Medium |
| Grafana Loki | Medium | Low | Medium |
| S3 + Athena | High (SQL) | Very low | Low |

## When to Apply

- **Structured logging**: Adopt in every new service; retrofit in existing services where log analysis is painful
- **Correlation IDs**: Add at the gateway/entry point; propagate via OpenTelemetry context to all downstream calls
- **Centralized aggregation**: Any system with more than one service instance — local log files do not scale
- **Log sampling**: Only for very high-volume INFO-level logs; never sample ERROR/WARN
- **Separate audit log stream**: Any service handling user data, financial transactions, or security decisions