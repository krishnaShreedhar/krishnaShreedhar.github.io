---
title: "CI/CD Pipelines"
subtitle: "CI/CD (Continuous Integration / Continuous Delivery) pipelines automate the path from code commit to production deployment, providing fast feedback on code quality and enabling safe, frequent releases. Well-designed..."
category: technical
project: large_scale_software_engineering
project_title: "Large Scale Software Engineering"
date: 2025-12-11
reading_time: 3
tags:
  - large-scale-software-engineering
  - docs
author: "Shreedhar Kodate"
output: "blogs/technical/posts/large_scale_software_engineering/docs/10_devops_infrastructure/01_cicd_pipelines.html"
---
CI/CD (Continuous Integration / Continuous Delivery) pipelines automate the path from code commit to production deployment, providing fast feedback on code quality and enabling safe, frequent releases. Well-designed pipelines are the foundation of high-velocity software delivery.

## CI/CD Pipeline Stages

```mermaid
graph LR
    Commit[Code Commit\ngit push] --> CI[Continuous Integration]

    subgraph CI[CI - Validate Code]
        Lint[Lint and\nFormat Check]
        UnitTest[Unit Tests]
        IntTest[Integration Tests]
        Security[Security Scan\nSAST SCA]
        Build[Build Artifact\nDocker image]
        Lint --> UnitTest --> IntTest --> Security --> Build
    end

    CI --> CD[Continuous Delivery]

    subgraph CD[CD - Deploy Code]
        Dev[Deploy to Dev\nautomatically]
        Staging[Deploy to Staging\nautomatically]
        ProdApprove[Manual Approval\nfor Production]
        Prod[Deploy to Production\ncanary or blue-green]
        Dev --> Staging --> ProdApprove --> Prod
    end

    style CI fill:#dbeafe,stroke:#2563eb
    style CD fill:#dcfce7,stroke:#16a34a
```

## GitHub Actions Pipeline

```mermaid
graph TD
    subgraph Triggers[Trigger Events]
        PR[Pull Request\nrun CI]
        Main[Push to main\nrun CI + deploy staging]
        Tag[Git Tag v1.x.x\ndeploy production]
    end

    subgraph Jobs[Pipeline Jobs]
        Test[test job\nmatrix: py3.10 py3.11 py3.12\nunit and integration tests]
        Scan[security job\nSemgrep Snyk trivy]
        Build[build job\nDocker build\npush to ECR]
        DeployStaging[deploy-staging job\nhelm upgrade staging]
        DeployProd[deploy-prod job\nrequires: manual approval\nhelm upgrade production]
    end

    PR --> Test & Scan
    Main --> Test & Scan --> Build --> DeployStaging
    Tag --> Build --> DeployProd

    style Test fill:#dbeafe,stroke:#2563eb
    style DeployProd fill:#dcfce7,stroke:#16a34a
```

## Deployment Strategies

```mermaid
graph TD
    subgraph Recreate[Recreate - Downtime]
        R1[Stop all v1] --> R2[Start all v2]
        RNote[Simplest\nDowntime required\nFast]
        style RNote fill:#fee2e2,stroke:#dc2626
    end

    subgraph RollingUpdate[Rolling Update - No Downtime]
        RU1[v1 x4] --> RU2[v1 x3 v2 x1] --> RU3[v1 x2 v2 x2] --> RU4[v2 x4]
        RUNote[Kubernetes default\nNo downtime\nRollback possible]
        style RUNote fill:#dbeafe,stroke:#2563eb
    end

    subgraph BlueGreen[Blue-Green - Instant Switch]
        BG1[Blue v1: 100% traffic]
        BG2[Green v2: 0% traffic, fully deployed]
        BG3[Switch: Green v2: 100%\nBlue v1: standby for rollback]
        BG1 --> BG2 --> BG3
        BGNote[Zero downtime\nInstant rollback\nDouble resources during switch]
        style BGNote fill:#fef3c7,stroke:#d97706
    end

    subgraph Canary[Canary - Gradual]
        C1[v1: 95%, v2: 5%]
        C2[v1: 80%, v2: 20%]
        C3[v1: 0%, v2: 100%]
        C1 --> C2 --> C3
        CNote[Low blast radius\nMetrics-gated rollout\nSlow]
        style CNote fill:#dcfce7,stroke:#16a34a
    end
```

## Feature Flags

```mermaid
graph TD
    Code[Feature Code\nif flag.enabled checkout_v2\n  new_checkout_flow\nelse\n  old_checkout_flow]

    FlagService[Feature Flag Service\nLaunchDarkly Split.io\nor self-hosted Unleash]

    subgraph Rules[Flag Targeting Rules]
        R1[Employees only: 100%]
        R2[Beta users: 20%]
        R3[US region: 50%]
        R4[Everyone: 0%]
    end

    Code --> FlagService
    FlagService --> Rules

    Benefits[Benefits:\nDeploy without releasing\nA/B testing\nInstant kill switch\nGradual rollout\nPermission-based access]

    style FlagService fill:#fef3c7,stroke:#d97706,stroke-width:2px
```

## Key Concepts

- **Continuous Integration (CI)**: Every code change is automatically built, tested, and validated. The goal: detect integration problems early when they are cheap to fix. CI requires: fast tests (under 10 minutes for PR builds), high test coverage, and a single main branch as the source of truth.

- **Continuous Delivery (CD)**: Every validated code change is automatically deployable to production at any time. In continuous deployment (a stricter form), every change that passes CI is automatically deployed to production without human approval.

- **Pipeline as Code**: CI/CD pipeline configuration is stored in the repository alongside the code it builds (e.g., `.github/workflows/`, `Jenkinsfile`, `.gitlab-ci.yml`). Changes to the pipeline go through code review and version control.

- **Build Matrix**: Running tests across multiple configurations simultaneously (Python 3.10, 3.11, 3.12; Ubuntu and macOS). Parallelizes testing to find compatibility issues without sequential overhead.

- **Rolling Deployment**: The default Kubernetes deployment strategy. New pods are started gradually while old pods are terminated, maintaining a minimum number of ready pods throughout. Requires the application to be backward-compatible (new code running alongside old code during the transition).

- **Feature Flags**: Decouple deployment from release. Code is deployed to production but hidden behind a flag. The flag is then gradually enabled for subsets of users (internal users, beta users, geographic segments) independent of code deployments. Enables instant rollback by disabling the flag without a code change.

- **Deployment Verification**: After deployment, automated checks verify the deployment succeeded: health endpoint responds, error rate hasn't increased, key metrics remain within SLO. If checks fail, automatic rollback triggers. This reduces MTTR for bad deployments.

## Trade-offs

| Strategy | Downtime | Rollback Speed | Resource Cost | Risk |
|----------|---------|--------------|--------------|------|
| Recreate | Yes | Fast | Low | High |
| Rolling | No | Medium (re-roll) | Low | Medium |
| Blue-Green | No | Instant | Double | Low |
| Canary | No | Instant (flag off) | Low | Lowest |

## When to Use

- **Rolling**: Default for stateless services — zero downtime with minimal resource overhead
- **Blue-Green**: Database schema migrations, major version upgrades where rollback must be instant
- **Canary**: High-risk changes to large user bases — algorithmic changes, pricing changes, major UX changes
- **Feature flags**: Always — for any change that can be wrapped in a flag, decouple deployment from release