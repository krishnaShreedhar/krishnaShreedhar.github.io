---
title: "Retraining Pipelines"
subtitle: "Retraining pipelines automate the process of updating ML models with fresh data to maintain prediction quality as the world changes. Unlike traditional software that remains correct until explicitly changed, ML..."
category: technical
project: large_scale_aiml_systems
project_title: "Large Scale AI/ML Systems"
date: 2025-06-18
reading_time: 4
tags:
  - large-scale-aiml-systems
  - docs
author: "Shreedhar Kodate"
output: "blogs/technical/posts/large_scale_aiml_systems/docs/02_mlops/04_retraining_pipelines.html"
---
Retraining pipelines automate the process of updating ML models with fresh data to maintain prediction quality as the world changes. Unlike traditional software that remains correct until explicitly changed, ML models are correct only relative to the data distribution they were trained on — systematic retraining is required to keep them accurate over time.

## Retraining Pipeline Architecture

```mermaid
graph TD
    subgraph Triggers[Retraining Triggers]
        Schedule[Schedule Trigger\ncron: every Monday 2am\ntime-based retraining]
        DriftTrigger[Drift Trigger\nPSI greater than 0.2\nstatistical alert from monitoring]
        PerfTrigger[Performance Trigger\nAUC dropped below 0.82\nbusiness metric alert]
        DataTrigger[Data Volume Trigger\n1M new labeled examples available\ndata-driven retrain]
    end

    subgraph Pipeline[Automated Retraining Pipeline]
        DataPrep[Data Preparation\nfetch recent N months of data\nrun feature computation\nvalidate schema and stats]
        Train[Model Training\nsame architecture as champion\nhyperparams from registry or HPO]
        Evaluate[Model Evaluation\ntest set metrics\ncompare to champion baseline]
        Gate{Champion\nChallenger\nComparison}
        Register[Register to Model Registry\nstage: Staging]
        Deploy[Promote to Production\nupdate serving alias]
        Reject[Reject New Model\nkeep champion in production\nalert team]

        DataPrep --> Train --> Evaluate --> Gate
        Gate -->|challenger wins| Register --> Deploy
        Gate -->|champion wins| Reject
    end

    Triggers --> DataPrep

    style Deploy fill:#dcfce7,stroke:#16a34a,stroke-width:2px
    style Reject fill:#fee2e2,stroke:#dc2626
```

## Champion-Challenger Evaluation Framework

```mermaid
graph TD
    subgraph Evaluation[Evaluation Gate Logic]
        ChampionMetrics[Champion Metrics\nAUC: 0.847\nF1: 0.821\nPrecision: 0.834\nRecall: 0.808]

        ChallengerMetrics[Challenger Metrics\nnew model trained on recent data\nAUC: 0.851\nF1: 0.828\nPrecision: 0.841\nRecall: 0.815]

        Rules[Promotion Rules\nAUC improvement greater than 0.005\nNo single metric degrades more than 1%\nLatency within serving SLA\nPasses bias and fairness checks\nPasses integration tests]

        Decision{All rules\npassed?}
        Promote[Promote Challenger\nnew champion]
        Keep[Keep Champion\nlog reason for rejection]

        ChampionMetrics --> Rules
        ChallengerMetrics --> Rules
        Rules --> Decision
        Decision -->|Yes| Promote
        Decision -->|No| Keep
    end
```

## Continuous Training vs Scheduled Retraining

```mermaid
graph LR
    subgraph Strategies[Retraining Strategy Spectrum]
        Manual[Manual\nEngineer triggers\nwhen needed\nmonths between retrains]

        Scheduled[Scheduled\nWeekly or monthly\npredictable overhead\nno data event sensitivity]

        DriftBased[Drift-Based\nTrigger on statistical\nalert\nresponsive to shifts]

        Continuous[Continuous Training\nOnline learning\nmodel updates on\neach new batch\nor example]

        Manual -->|more automation| Scheduled -->|more responsiveness| DriftBased -->|maximum freshness| Continuous
    end
```

## Data Window Strategy

```mermaid
graph TD
    subgraph DataWindows[Training Data Window Strategies]
        subgraph AllHistory[All Historical Data]
            Full[Use all data from start\nPros: maximum data volume\nCons: old patterns may hurt\nexponentially growing compute]
        end

        subgraph SlidingWindow[Sliding Window]
            Sliding[Use last N months only\nPros: captures recent patterns\nCons: forgets long-term patterns\nConstant compute]
        end

        subgraph Weighted[Weighted Recency]
            Weight[All data but weight recent higher\nPros: balance history and recency\nCons: complex to tune]
        end

        subgraph Hybrid[Hybrid]
            Mix[Recent data full weight\nOld data subsampled\nPros: practical balance\nCommon in fraud and ads]
        end
    end
```

## Key Concepts

- **Trigger Strategy**: What causes a retraining run to start. Schedule-based triggers (cron) are simple and predictable but may retrain unnecessarily (wasting compute) or too infrequently (missing rapid drift). Event-based triggers (drift threshold exceeded, performance degradation) are more responsive but require reliable monitoring infrastructure as prerequisites.

- **Champion-Challenger Evaluation**: The new model (challenger) must outperform the current production model (champion) on a held-out evaluation set before promotion. The evaluation set should be recent, representative, and use the same time-based split logic as the training set. Champion-challenger comparison prevents regressions from being automatically deployed.

- **Evaluation Gates**: Automated checks the challenger must pass before promotion. Include: metric thresholds (AUC, F1, business KPIs), latency checks (serving must stay within SLA), bias checks (performance across subgroups), integration tests (model loads and handles edge cases correctly). Gates are the quality assurance layer for automated ML deployments.

- **Data Window**: Which time period of historical data to use for retraining. A fixed sliding window (last 6 months) captures recent patterns but requires decisions about what recent means for the business. Too short a window risks forgetting rare but important patterns (fraud spikes, seasonal events). Too long a window dilutes the signal from recent shifts.

- **Online Learning**: Updating model parameters incrementally as each new example arrives, without full retraining. Enables very fresh models but is harder to validate (no batch evaluation), sensitive to outliers, and not supported by all model types. Used for recommendation and ad click prediction where recency matters enormously and data volume is very high.

- **Continual Learning Problem**: When retraining on new data causes the model to forget what it learned from old data (catastrophic forgetting). Particularly relevant for neural networks. Mitigations include experience replay (include samples from old data), regularization methods (EWC), and hybrid window strategies.

- **Monitoring as Prerequisite**: Automated retraining on drift triggers requires that monitoring infrastructure is reliable — false drift alerts cause unnecessary retraining. Building monitoring before automated retraining is the correct order.

## Trade-offs

| Strategy | Freshness | Compute Cost | Complexity | Risk |
|---------|---------|-------------|-----------|------|
| Manual retraining | Low | Low | Very Low | Model staleness |
| Scheduled weekly | Medium | Predictable | Low | Unnecessary retrains |
| Drift-triggered | High | Variable | Medium | False trigger risk |
| Continuous online | Very High | Continuous | Very High | Catastrophic forgetting |

## When to Use

- **Scheduled retraining**: Stable domains where drift is slow (credit risk, churn prediction) — weekly or monthly retraining with champion-challenger gate
- **Drift-triggered retraining**: Rapidly evolving domains (fraud, social media content) where drift is unpredictable and timeliness matters more than compute cost
- **Online learning**: Real-time personalization (news feeds, recommendations) where model freshness within minutes matters and data volume justifies the engineering complexity
- **Skip automated retraining**: Avoid automation until monitoring is robust — automated retraining without good monitoring can deploy bad models without human review