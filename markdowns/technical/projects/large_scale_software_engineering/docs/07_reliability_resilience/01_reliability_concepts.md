---
title: "Reliability Concepts"
subtitle: "Reliability engineering provides a structured vocabulary and measurement framework for reasoning about system dependability. SLIs, SLOs, and error budgets transform vague reliability aspirations into measurable,..."
category: technical
project: large_scale_software_engineering
project_title: "Large Scale Software Engineering"
date: 2025-12-26
reading_time: 3
tags:
  - large-scale-software-engineering
  - docs
author: "Shreedhar Kodate"
output: "blogs/technical/posts/large_scale_software_engineering/docs/07_reliability_resilience/01_reliability_concepts.html"
---
Reliability engineering provides a structured vocabulary and measurement framework for reasoning about system dependability. SLIs, SLOs, and error budgets transform vague reliability aspirations into measurable, actionable engineering targets.

## SLI, SLO, SLA Hierarchy

```mermaid
graph TD
    SLI[SLI - Service Level Indicator\nMeasured metric\nWhat we observe\nExamples: request success rate\nlatency p99 availability]
    SLO[SLO - Service Level Objective\nTarget for the SLI\nInternal engineering goal\nExample: 99.9% success rate\nmeasured over 30 days]
    SLA[SLA - Service Level Agreement\nExternal contractual commitment\nConsequences for breach\nUsually more conservative than SLO\nExample: 99.5% uptime]
    ErrorBudget[Error Budget\n100% minus SLO\nAllowable failure budget\nBalances reliability vs velocity]

    SLI --> SLO --> SLA
    SLO --> ErrorBudget

    style SLO fill:#dcfce7,stroke:#16a34a,stroke-width:2px
    style ErrorBudget fill:#fef3c7,stroke:#d97706,stroke-width:2px
```

## Error Budget Consumption

```mermaid
graph LR
    subgraph ErrorBudget[Monthly Error Budget - 99.9% SLO]
        Total[Total Budget\n43.2 minutes downtime\nper month]
        Incident1[Incident 1 - DB outage\n15 minutes consumed]
        Incident2[Incident 2 - Deploy gone wrong\n20 minutes consumed]
        Incident3[Incident 3 - DNS issue\n5 minutes consumed]
        Remaining[Remaining: 3.2 minutes\n7% of budget left]

        Total --> Incident1 & Incident2 & Incident3
        Incident1 & Incident2 & Incident3 --> Remaining
    end

    Policy[Error Budget Policy:\n- Budget healthy: deploy freely\n- Budget 50% consumed: review risks\n- Budget exhausted: freeze releases\n  focus on reliability work]

    style Remaining fill:#fee2e2,stroke:#dc2626
    style Policy fill:#fef3c7,stroke:#d97706
```

## MTBF, MTTR, and Availability

```mermaid
graph LR
    subgraph Timeline[System Failure Timeline]
        UP1[UP] --> F1[FAILURE]
        F1 --> UP2[UP]
        UP2 --> F2[FAILURE]
        F2 --> UP3[UP]
    end

    subgraph Calculations[Calculations]
        MTBF[MTBF = Mean Time Between Failures\nAverage time system operates before failing\nHigher MTBF = more reliable]
        MTTR[MTTR = Mean Time To Recovery\nAverage time to restore service\nLower MTTR = more resilient]
        Avail[Availability = MTBF divided by MTBF + MTTR\nThe key metric for uptime nines]
    end

    style MTBF fill:#dcfce7,stroke:#16a34a
    style MTTR fill:#dbeafe,stroke:#2563eb
```

## Common SLIs

```mermaid
graph TD
    subgraph SLITypes[Common SLI Categories]
        Avail[Availability\nFraction of successful requests\nsuccessful divided by total\nTarget: 99.9%]

        Latency[Latency\nFraction of requests below threshold\nrequests under 200ms divided by total\nTarget: 95% under 200ms]

        Throughput[Throughput\nActual processed vs expected\nfor batch pipelines\nTarget: 99% of jobs complete]

        Error[Error Rate\nFailed requests divided by total\nHTTP 5xx rate\nTarget: less than 0.1%]

        Durability[Durability\nData survived vs total stored\nfor storage systems\nTarget: 99.999999999% 11 nines]
    end
```

## Key Concepts

- **SLI (Service Level Indicator)**: A measurable property of a service used to assess its reliability. Must be: quantifiable (a number), representative of user experience, and meaningful to the business. Good SLIs: request success rate (not "is the server up" but "are users getting successful responses"), p99 latency.

- **SLO (Service Level Objective)**: A target value or range for an SLI, measured over a specific time window. Internal — not a customer commitment. The SLO defines the engineering quality goal. Example: "99.9% of API requests return 2xx over a rolling 30-day window."

- **SLA (Service Level Agreement)**: A contractual commitment to customers with financial or legal consequences for breach. SLAs are typically more conservative than SLOs (e.g., SLO is 99.9%, SLA is 99.5%) — the gap provides a buffer.

- **Error Budget**: (100% - SLO%) × time. The amount of reliability the service is allowed to "spend" on failures, deployments, and experiments. If the SLO is 99.9%, the error budget is 0.1% = 43.2 minutes per month. Error budgets align engineering effort — when budget is abundant, teams can ship faster; when exhausted, reliability work takes priority over features.

- **MTBF (Mean Time Between Failures)**: Average time the system operates between failures. A higher MTBF means fewer failures. Improving MTBF requires better hardware, better code quality, and better testing.

- **MTTR (Mean Time To Recovery)**: Average time from failure detection to full service restoration. Improving MTTR requires better alerting (detect faster), runbooks, on-call training, and automation. MTTR improvement often has more ROI than MTBF improvement.

- **Availability = MTBF / (MTBF + MTTR)**: Even with frequent failures (low MTBF), high availability is achievable with very fast recovery (low MTTR). For critical services, invest in MTTR reduction first.

## Trade-offs

| Target | Benefit | Cost |
|--------|---------|------|
| Higher SLO (99.99%) | Better user experience | 10x engineering effort per nine |
| Lower SLO (99%) | Development velocity | Users experience more downtime |
| Tight error budget | Forces reliability investment | Can block feature deployments |
| Loose error budget | More deployment freedom | Reliability debt accumulates |

## When to Apply

- Define SLOs before incidents, not during — reactive reliability targets are set under pressure
- Set SLOs slightly above what you can currently achieve — aspirational but reachable
- Review SLOs quarterly as the system and user expectations evolve
- Always set an SLO for your most critical user journeys first