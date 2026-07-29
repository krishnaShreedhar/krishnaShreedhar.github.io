---
title: "Infrastructure as Code (IaC)"
subtitle: "Infrastructure as Code manages and provisions computing infrastructure through machine-readable configuration files rather than manual processes. IaC enables version-controlled, repeatable, auditable infrastructure..."
category: technical
project: large_scale_software_engineering
project_title: "Large Scale Software Engineering"
date: 2025-02-07
reading_time: 3
tags:
  - large-scale-software-engineering
  - docs
author: "Shreedhar Kodate"
output: "blogs/technical/posts/large_scale_software_engineering/docs/10_devops_infrastructure/04_iac.html"
---
Infrastructure as Code manages and provisions computing infrastructure through machine-readable configuration files rather than manual processes. IaC enables version-controlled, repeatable, auditable infrastructure that can be treated with the same engineering discipline as application code.

## IaC Approaches

```mermaid
graph TD
    subgraph Declarative[Declarative - Define desired state]
        Terraform[Terraform / OpenTofu\nHCL language\nPlan then Apply\nState file management\nMulti-cloud]
        Pulumi[Pulumi\nReal programming languages\nPython TypeScript Go\nStronger abstractions]
        CloudFormation[AWS CloudFormation\nAWS-native\nYAML or JSON\nNo external state]
    end

    subgraph Imperative[Imperative - Define steps]
        Ansible[Ansible\nYAML playbooks\nAgentless - SSH\nConfiguration management\nIdempotent when designed well]
        Chef[Chef / Puppet\nRuby DSL\nAgent-based\nComplex but powerful]
    end

    subgraph GitOps[GitOps - Git as source of truth]
        ArgoCD[ArgoCD\nKubernetes-native\nPull-based deployment\nDrift detection]
        Flux[FluxCD\nKubernetes GitOps\nHelm support]
    end
```

## Terraform Workflow

```mermaid
graph LR
    Code[Write Terraform HCL\nresource VPC\nresource RDS\nmodule k8s_cluster]

    Init[terraform init\nDownload providers\nInitialize state backend]

    Plan[terraform plan\nShow what will change\nReview before applying\nSafe - read-only]

    Apply[terraform apply\nCreate or modify resources\nUpdate state file]

    Destroy[terraform destroy\nRemove all resources]

    Code --> Init --> Plan --> Apply --> Destroy

    StateBackend[Remote State Backend\nS3 + DynamoDB lock\nor Terraform Cloud\nShared team state]
    Apply --> StateBackend

    style Plan fill:#dcfce7,stroke:#16a34a,stroke-width:2px
    style Apply fill:#fef3c7,stroke:#d97706,stroke-width:2px
```

## GitOps Workflow

```mermaid
graph TD
    Dev[Developer] -->|code change| AppRepo[Application Repo]
    Dev -->|infra change| InfraRepo[Infrastructure Repo\nHelm charts, K8s manifests]

    AppCI[App CI Pipeline] -->|build and push image| Registry[Container Registry]
    AppCI -->|update image tag| InfraRepo

    subgraph GitOpsLoop[GitOps Reconciliation Loop]
        ArgoCD[ArgoCD / FluxCD] -->|watch| InfraRepo
        ArgoCD -->|detect drift| Cluster[Kubernetes Cluster]
        ArgoCD -->|apply desired state| Cluster
        Cluster -->|actual state| ArgoCD
    end

    InfraRepo --> ArgoCD

    style GitOpsLoop fill:#dbeafe,stroke:#2563eb,stroke-width:2px
    style ArgoCD fill:#fef3c7,stroke:#d97706
```

## Key Concepts

- **Infrastructure as Code (IaC)**: Managing infrastructure through version-controlled, machine-readable definitions. Benefits: reproducible environments (dev, staging, prod created from same code), auditable changes (who changed what and when via git history), disaster recovery (recreate entire infrastructure from code), peer review for infrastructure changes.

- **Declarative vs Imperative**: Declarative (Terraform, CloudFormation) specifies the desired end state — the tool figures out how to reach it. Imperative (Ansible, Chef) specifies the steps to execute. Declarative is preferred for infrastructure provisioning; imperative is preferred for configuration management.

- **Terraform State**: Terraform maintains a state file mapping resource definitions to real cloud resources. The state must be stored in a shared backend (S3 + DynamoDB for locking) when used by a team. State drift occurs when infrastructure is changed outside Terraform — `terraform plan` shows what needs to change to reconcile.

- **Terraform Modules**: Reusable, parameterized collections of Terraform resources. A `k8s_cluster` module encapsulates all resources needed to create a Kubernetes cluster with standard configuration. Modules are the primary abstraction mechanism in Terraform.

- **GitOps**: An operational framework where Git is the single source of truth for declarative infrastructure and application configuration. All changes go through pull requests. An automated agent (ArgoCD, FluxCD) continuously reconciles the cluster state with the git repository, automatically applying changes and detecting drift.

- **Drift Detection**: GitOps controllers continuously compare actual cluster state with the git repository. If someone applies changes directly to the cluster (bypassing git), the controller alerts and/or automatically reverts the change. Enforces that git is the only path to change production.

- **Immutable Infrastructure**: Instead of modifying running servers, replace them with new ones based on updated images. Avoids configuration drift and makes rollback simple — redeploy the previous image. Containers make this natural.

## Trade-offs

| Tool | Strength | Limitation |
|------|---------|-----------|
| Terraform | Multi-cloud, mature ecosystem | State management complexity |
| Pulumi | Real languages, strong abstractions | Smaller community |
| CloudFormation | AWS-native, no external state | AWS only, verbose |
| Ansible | Agentless, flexible | Not purely declarative |
| GitOps | Audit trail, drift detection | Requires Kubernetes |

## When to Use

- **Terraform**: Primary IaC for cloud infrastructure (VPCs, databases, Kubernetes clusters, IAM)
- **Ansible**: OS-level configuration management, application deployment to VMs, secrets distribution
- **GitOps (ArgoCD)**: Kubernetes application deployment — excellent for managing many services across multiple clusters
- **IaC for everything**: All production infrastructure should be in code — no manual console changes