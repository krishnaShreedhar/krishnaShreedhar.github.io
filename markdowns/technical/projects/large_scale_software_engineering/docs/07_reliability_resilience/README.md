# Reliability and Resilience

Reliability is a system's ability to function correctly over time. Resilience is its ability to maintain acceptable behaviour in the face of failures. In distributed systems, failures are not exceptional events — they are the norm. The question is not whether components will fail, but how gracefully the system handles those failures.

## Overview

```mermaid
mindmap
  root((Reliability and\nResilience))
    Reliability Concepts
      SLI - Service Level Indicators
      SLO - Service Level Objectives
      SLA - Service Level Agreements
      Error Budgets
      MTTR and MTBF
      Availability Nines
    Resilience Patterns
      Circuit Breaker
      Retry with Backoff
      Bulkhead Isolation
      Timeout and Deadline
      Fallback and Degradation
      Chaos Engineering
    Site Reliability Engineering
      SRE Principles
      Toil Reduction
      Postmortem Culture
      Error Budget Policy
      On-call Practices
      Production Readiness Reviews
```

## Availability Calculation

```mermaid
graph TD
    subgraph AvailabilityNines[Availability Nines - Annual Downtime]
        N2[99 percent - 2 nines\n87.6 hours downtime per year]
        N3[99.9 percent - 3 nines\n8.76 hours downtime per year]
        N4[99.99 percent - 4 nines\n52.6 minutes downtime per year]
        N5[99.999 percent - 5 nines\n5.26 minutes downtime per year]
        N6[99.9999 percent - 6 nines\n31.5 seconds downtime per year]

        N2 --> N3 --> N4 --> N5 --> N6
        Cost[Each additional 9 is\n10x harder and more expensive]
    end
```

## Topics in This Section

| File | Topic | Key Concepts |
|------|-------|--------------|
| [01_reliability_concepts.md](01_reliability_concepts.md) | Reliability Concepts | SLI/SLO/SLA, error budgets, MTTR |
| [02_resilience_patterns.md](02_resilience_patterns.md) | Resilience Patterns | Circuit breaker, bulkhead, chaos |
| [03_sre.md](03_sre.md) | SRE | Error budget policy, toil, postmortems |
