# Site Reliability Engineering (SRE)

Site Reliability Engineering is the discipline of applying software engineering to operations problems — automating toil, defining reliability standards through SLOs, and building systems that are easy to operate. SRE was formalized at Google and has since been adopted broadly as the operational counterpart to DevOps.

## SRE Principles

```mermaid
mindmap
  root((SRE\nPrinciples))
    Error Budget
      Balance reliability vs velocity
      Shared ownership of risk
      Data-driven release decisions
    SLOs as Guiding Metric
      Engineering goal not marketing
      Measured not assumed
      Drives engineering priorities
    Toil Reduction
      Manual repetitive work
      Automate or eliminate
      Cap toil at 50 percent of time
    Postmortem Culture
      Blameless
      Learning-focused
      Corrective actions required
    Automation
      Eliminate human intervention
      Runbooks become code
      Self-healing systems
    Production Readiness Reviews
      Criteria before launch
      Observability requirements
      Capacity planning
```

## On-Call Process

```mermaid
graph TD
    Alert[Alert Fires\nPagerDuty / OpsGenie] --> OncallEng[On-call Engineer\nAcknowledges within SLO]
    OncallEng --> Triage[Triage\nWhat is impacted?\nWhat is the severity?]
    Triage --> Sev1{Severity?}
    Sev1 -->|P1 - User-facing outage| Incident[Declare Incident\nNotify stakeholders\nOpen war room]
    Sev1 -->|P2 - Degraded performance| Investigate[Investigate\nCheck dashboards\nCheck recent deploys]
    Sev1 -->|P3 - Internal issue| Investigate

    Incident --> Mitigation[Mitigation First\nRollback deploy\nScale up\nRedirect traffic]
    Investigate --> Mitigation
    Mitigation --> Resolved[Service Restored]
    Resolved --> Postmortem[Postmortem\nWithin 48 hours]
    Postmortem --> ActionItems[Action Items\nTracked to completion]
```

## Blameless Postmortem Structure

```mermaid
graph TD
    subgraph PostmortemTemplate[Blameless Postmortem Template]
        Impact[Impact\nWhat failed, duration, user impact]
        Timeline[Timeline\nDetailed sequence of events\nwhat happened when]
        RootCause[Root Cause\n5 Whys analysis\nNot who but what failed]
        Contributing[Contributing Factors\nsystem and process factors\nnot individual blame]
        Actions[Action Items\nSpecific preventive measures\nOwner and deadline for each]
        Lessons[Lessons Learned\nWhat did the system\ndo well what can improve]
    end

    Impact --> Timeline --> RootCause --> Contributing --> Actions --> Lessons
```

## Toil vs Engineering Work

```mermaid
graph LR
    subgraph Toil[Toil Characteristics]
        T1[Manual - no automation possible today]
        T2[Repetitive - done over and over]
        T3[Reactive - triggered by external event]
        T4[No lasting value - does not improve system]
        T5[Scales with traffic - grows linearly with load]
        style T1 fill:#fee2e2,stroke:#dc2626
        style T2 fill:#fee2e2,stroke:#dc2626
        style T3 fill:#fee2e2,stroke:#dc2626
        style T4 fill:#fee2e2,stroke:#dc2626
        style T5 fill:#fee2e2,stroke:#dc2626
    end

    subgraph Engineering[Engineering Work]
        E1[Permanent improvement]
        E2[Reduces future toil]
        E3[Builds capability]
        E4[Automation that runs itself]
        style E1 fill:#dcfce7,stroke:#16a34a
        style E2 fill:#dcfce7,stroke:#16a34a
        style E3 fill:#dcfce7,stroke:#16a34a
        style E4 fill:#dcfce7,stroke:#16a34a
    end

    SREPrinciple[SRE Principle:\nCap toil at 50 percent\nof SRE team time.\nAbove 50 percent: stop feature work\nuntil toil is reduced.]
```

## Production Readiness Review (PRR)

```mermaid
graph TD
    NewService[New Service / Major Feature] --> PRR[Production Readiness Review]

    subgraph PRRCriteria[PRR Criteria Checklist]
        Obs[Observability\nMetrics, traces, structured logs\nDashboard created]
        Alerts[Alerting\nSLO-based alerts configured\nRunbook for each alert]
        Capacity[Capacity\nLoad tested to 2x expected\nAutoscaling configured]
        Deploy[Deployment\nRollback procedure documented\nFeature flags for risky changes]
        DR[Disaster Recovery\nBackup and restore tested\nRTO and RPO defined]
        Deps[Dependency Analysis\nAll dependencies have circuit breakers\nFallbacks for critical dependencies]
    end

    PRR --> Obs & Alerts & Capacity & Deploy & DR & Deps
    Obs & Alerts & Capacity & Deploy & DR & Deps --> Launch[Service Approved for Production]

    style PRR fill:#fef3c7,stroke:#d97706,stroke-width:2px
    style Launch fill:#dcfce7,stroke:#16a34a,stroke-width:2px
```

## Key Concepts

- **Error Budget**: The 1 - SLO reliability budget that the service is allowed to "spend" on failures, experiments, and deployments. The error budget creates a shared incentive between SRE and product teams — when budget is plentiful, both teams want to ship features; when exhausted, reliability investment takes priority.

- **Toil**: Manual, repetitive, reactive operational work that scales linearly with service growth and provides no lasting value. Examples: manually restarting crashed pods, manually clearing a queue, manually running a database cleanup script. SRE teams should cap toil at 50% of their time; excess toil triggers automation investment.

- **Blameless Postmortem**: A structured retrospective after an incident focused on understanding what failed in the system (not who was at fault) and creating action items to prevent recurrence. Blamelessness is essential — blame-focused postmortems cause engineers to hide problems and discourage honest reporting.

- **Toil Automation**: Every piece of toil should be a candidate for automation. The process: document the runbook for the manual process, then convert the runbook into code. Self-healing systems (automated remediation) are the goal — the system detects and fixes its own problems without human intervention.

- **Production Readiness Review (PRR)**: A structured checklist-based review that a service must pass before being allowed to serve production traffic. Ensures observability, alerting, capacity, deployment safety, and disaster recovery are all addressed before launch.

- **Error Budget Policy**: The documented policy for what happens when the error budget is exhausted. Typically: freeze non-emergency deployments, redirect engineering capacity to reliability improvements, require higher testing standards for next releases.

- **Capacity Planning**: Forecasting resource requirements based on growth projections and traffic patterns. Load tests verify that the service can handle 2-5x current peak traffic before a deployment that might drive increased load.

## Trade-offs

| SRE Practice | Benefit | Cost |
|---|---|---|
| Error budget policy | Aligns reliability/velocity incentives | Can block feature development |
| Blameless postmortems | Learning culture, honest reporting | Requires leadership buy-in |
| Toil automation | Reduces operational burden | Upfront engineering time |
| PRR | Prevents production fires | Slows initial deployment |
| Chaos engineering | Finds weaknesses early | Operational risk |

## When to Apply

- Start with SLOs and basic observability before anything else — you cannot improve what you cannot measure
- Introduce PRRs when the service is nearing production — too early wastes effort on changing systems
- Error budget policy requires organizational maturity — start with lighter-weight versions (SLO dashboards, weekly reviews)
- SRE practices scale: small teams should apply the principles without the full ceremony
