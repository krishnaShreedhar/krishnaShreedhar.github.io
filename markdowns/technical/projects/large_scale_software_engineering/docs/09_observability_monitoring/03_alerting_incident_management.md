---
title: "Alerting and Incident Management"
subtitle: "Effective alerting pages engineers only when human action is required, with enough context to act quickly. Poor alerting causes alert fatigue — too many false-positive pages result in engineers ignoring alerts,..."
category: technical
project: large_scale_software_engineering
project_title: "Large Scale Software Engineering"
date: 2025-05-23
reading_time: 4
tags:
  - large-scale-software-engineering
  - docs
author: "Shreedhar Kodate"
output: "blogs/technical/posts/large_scale_software_engineering/docs/09_observability_monitoring/03_alerting_incident_management.html"
---
Effective alerting pages engineers only when human action is required, with enough context to act quickly. Poor alerting causes alert fatigue — too many false-positive pages result in engineers ignoring alerts, including real ones. Incident management provides a structured process for responding to and learning from production failures.

## Alert Design Principles

```mermaid
graph TD
    subgraph BadAlert[Bad Alert Design]
        BA1[CPU above 80 percent\nNot actionable - maybe fine]
        BA2[Server down\nToo late - users already impacted]
        BA3[HTTP 5xx spike\nNo context - what service what user impact]
        BA4[Memory above 70 percent\nAlways fires]
        style BA1 fill:#fee2e2,stroke:#dc2626
        style BA2 fill:#fee2e2,stroke:#dc2626
        style BA3 fill:#fee2e2,stroke:#dc2626
        style BA4 fill:#fee2e2,stroke:#dc2626
    end

    subgraph GoodAlert[Good Alert Design - Symptom Based]
        GA1[SLO burn rate high\nUser experience impacted right now]
        GA2[Error rate above 1 percent for 5 min\nClear threshold with duration]
        GA3[Payment success rate below 99 percent\nBusiness metric directly impacted]
        GA4[p99 latency above 2s for 10 min\nUsers experiencing slowness]
        style GA1 fill:#dcfce7,stroke:#16a34a
        style GA2 fill:#dcfce7,stroke:#16a34a
        style GA3 fill:#dcfce7,stroke:#16a34a
        style GA4 fill:#dcfce7,stroke:#16a34a
    end
```

## SLO Burn Rate Alerting

```mermaid
graph TD
    subgraph BurnRate[SLO Burn Rate Alert - Multi-Window]
        SLO[SLO: 99.9 percent success rate\nError Budget: 0.1 percent = 43.2 min per month]

        Fast[Fast Burn Alert\n5 minute window AND 1 hour window\nBurn rate above 14.4x\nConsumes 1 hour of budget in 5 min\nPage immediately - severe incident]

        Slow[Slow Burn Alert\n30 minute window AND 6 hour window\nBurn rate above 1x\nOn track to exhaust budget\nTicket alert - investigate soon]

        SLO --> Fast & Slow
        style Fast fill:#fee2e2,stroke:#dc2626,stroke-width:2px
        style Slow fill:#fef3c7,stroke:#d97706
    end
```

## Incident Response Flow

```mermaid
stateDiagram-v2
    [*] --> Detected: Alert fires / user report

    Detected --> Acknowledged: On-call acknowledges\nwithin SLO window

    Acknowledged --> Triaged: Assess scope and severity\nDeclare severity level

    Triaged --> Mitigating: Apply immediate fixes\nRollback, scale, redirect

    Mitigating --> Resolved: Service restored to normal\nClose incident

    Resolved --> Postmortem: Schedule postmortem\nwithin 48 hours

    Postmortem --> ActionItems: Create tracked action items

    ActionItems --> [*]: Complete actions\nprevent recurrence
```

## Alert Routing

```mermaid
graph TD
    Alert[Alert Fires] --> AlertManager[AlertManager\nPrometheus AlertManager\nor Grafana OnCall]

    AlertManager --> Route{Route by\nLabel and Severity}

    Route -->|severity:critical\nteam:platform| P1[PagerDuty P1\nImmediate page\nPhone call]
    Route -->|severity:warning\nteam:backend| P2[PagerDuty P2\nSMS notification]
    Route -->|severity:info| Slack[Slack channel\nNo page]

    P1 --> Esc[Escalation Policy\nNo response in 5min\nEscalate to manager]

    style P1 fill:#fee2e2,stroke:#dc2626,stroke-width:2px
    style Slack fill:#dcfce7,stroke:#16a34a
```

## Key Concepts

- **Alert Fatigue**: The phenomenon where engineers stop responding carefully to alerts because there are too many false positives. Alert fatigue is one of the most dangerous operational conditions — it causes real incidents to be missed. Every alert must be: actionable (requires human intervention), urgent (cannot wait until morning), and accurate (fires only when the condition is truly present).

- **Symptom-Based Alerting**: Alert on what users experience (high error rate, slow responses, service unavailable) rather than what the system is doing internally (high CPU, high memory). Symptom-based alerts have fewer false positives and directly indicate user impact.

- **SLO Burn Rate Alerting**: Instead of alerting on raw error rates, alert on how fast the error budget is being consumed. Multi-window burn rate alerting (fast alert for severe incidents, slow alert for gradual degradation) provides early warning while minimizing false positives. The Google SRE workbook provides the specific burn rate thresholds.

- **Incident Severity Levels**: A common classification: P1 (complete service unavailability or data loss), P2 (major degradation affecting many users), P3 (minor degradation or internal-only issue). Severity drives response urgency, escalation paths, and stakeholder notification.

- **Incident Command Structure**: For P1 incidents, assign clear roles: Incident Commander (coordinates response, owns communication), Operations Lead (implements technical fixes), Communications Lead (updates stakeholders). Avoid everyone trying to fix the problem simultaneously without coordination.

- **MTTD and MTTR**: Mean Time To Detect and Mean Time To Recover. Better alerting reduces MTTD. Better runbooks, automation, and on-call training reduce MTTR. Both are key operational metrics.

- **Runbooks**: Documented procedures for responding to specific alerts. Should answer: what is this alert about, who is impacted, how to verify the issue, what immediate mitigation steps to take, who to escalate to. Runbooks linked directly from alert notifications reduce MTTR dramatically.

## Trade-offs

| Alert Type | False Positive Rate | MTTD | Actionability |
|-----------|-------------------|------|--------------|
| Threshold-based (CPU>80%) | High | Fast | Low |
| Symptom-based (error rate) | Medium | Medium | High |
| SLO burn rate | Low | Slightly slower | Highest |
| Anomaly detection | Variable | Fast | Medium |

## When to Apply

- **SLO burn rate alerts**: Primary alerting strategy for all user-facing services
- **Symptom-based alerts first**: Start with error rate and latency alerts; add cause-based alerts only after symptoms are covered
- **Review alert quality monthly**: Remove alerts that fire without requiring action; fix alerts that trigger false positives
- **Runbooks for every alert**: No alert should fire without a linked runbook explaining how to respond