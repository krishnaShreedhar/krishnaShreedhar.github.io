---
title: "Containerization and Orchestration"
subtitle: "Containers package applications with their dependencies into portable, isolated units. Container orchestration (Kubernetes) manages the scheduling, scaling, networking, and lifecycle of containers at scale."
category: technical
project: large_scale_software_engineering
project_title: "Large Scale Software Engineering"
date: 2025-10-24
reading_time: 3
tags:
  - large-scale-software-engineering
  - docs
author: "Shreedhar Kodate"
output: "blogs/technical/posts/large_scale_software_engineering/docs/10_devops_infrastructure/02_containerization_orchestration.html"
---
Containers package applications with their dependencies into portable, isolated units. Container orchestration (Kubernetes) manages the scheduling, scaling, networking, and lifecycle of containers at scale.

## Docker Architecture

```mermaid
graph TD
    subgraph DockerBuild[Docker Build]
        Dockerfile[Dockerfile\nmulti-stage build]
        Image[Container Image\nlayer cache optimized]
        Registry[Container Registry\nECR GCR Artifact Registry]
        Dockerfile --> Image --> Registry
    end

    subgraph DockerRun[Docker Runtime]
        Container[Container\nisolated process]
        Volumes[Volumes\npersistent storage]
        Network[Docker Network\nbridge overlay]
        Container --> Volumes & Network
    end

    Registry --> Container
```

## Kubernetes Architecture

```mermaid
graph TD
    subgraph ControlPlane[Control Plane]
        APIServer[kube-apiserver\nAll kubectl commands go here]
        Etcd[etcd\nCluster state storage]
        Scheduler[kube-scheduler\nAssign pods to nodes]
        Controller[controller-manager\nDeployment ReplicaSet jobs]
        APIServer --> Etcd
        APIServer --> Scheduler
        APIServer --> Controller
    end

    subgraph Node1[Worker Node 1]
        Kubelet1[kubelet\nManage pods on node]
        Proxy1[kube-proxy\nNetwork routing]
        PodA[Pod: api-server]
        PodB[Pod: worker]
        Kubelet1 --> PodA & PodB
    end

    subgraph Node2[Worker Node 2]
        Kubelet2[kubelet]
        PodC[Pod: api-server]
        Kubelet2 --> PodC
    end

    APIServer --> Kubelet1 & Kubelet2

    style ControlPlane fill:#dbeafe,stroke:#2563eb
    style Node1 fill:#dcfce7,stroke:#16a34a
    style Node2 fill:#dcfce7,stroke:#16a34a
```

## Kubernetes Object Hierarchy

```mermaid
graph TD
    Deployment[Deployment\nDeclared desired state\nrollingUpdate strategy] --> RS[ReplicaSet\nMaintains N replicas] --> Pod1[Pod 1] & Pod2[Pod 2] & Pod3[Pod 3]

    Service[Service\nStable virtual IP\nLoad balances to pods] --> Pod1 & Pod2 & Pod3

    HPA[HorizontalPodAutoscaler\nScale pods by CPU/memory\nor custom metrics] --> Deployment

    Ingress[Ingress\nHTTP routing rules\nTLS termination] --> Service

    ConfigMap[ConfigMap\nnon-sensitive config] --> Pod1
    Secret[Secret\nencrypted sensitive data] --> Pod1

    style HPA fill:#fef3c7,stroke:#d97706
    style Ingress fill:#dbeafe,stroke:#2563eb
```

## Multi-Stage Dockerfile

```mermaid
graph LR
    subgraph BuildStage[Stage 1: Build]
        BS[FROM python:3.12 AS builder\nCOPY requirements.txt\nRUN pip install requirements\nCOPY src\nRUN python -m compileall]
    end

    subgraph FinalStage[Stage 2: Runtime]
        FS[FROM python:3.12-slim\nCOPY --from=builder dist\nCopy only compiled artifacts\nNo build tools in final image\nSmaller attack surface\nSmaller image size]
        style FS fill:#dcfce7,stroke:#16a34a
    end

    BuildStage --> FinalStage
```

## Key Concepts

- **Container**: An isolated process running in a namespaced environment. Shares the host OS kernel (unlike VMs) but has isolated filesystem (overlay), process namespace, network namespace, and resource limits (cgroups). Docker is the most popular container runtime; containerd and CRI-O are used by Kubernetes directly.

- **Container Image**: An immutable, layered filesystem snapshot. Each Dockerfile instruction creates a layer. Layers are cached — unchanged layers are reused in subsequent builds, dramatically speeding up CI. Multi-stage builds keep final images small by discarding build-time dependencies.

- **Pod**: The smallest deployable unit in Kubernetes. A pod contains one or more containers that share a network namespace (same IP) and storage volumes. Sidecar containers (logging agent, service mesh proxy) run in the same pod as the application container.

- **Deployment**: A Kubernetes controller that manages the desired state for pods — how many replicas, which image version, and what update strategy. Deployments handle rolling updates, rollbacks, and scaling.

- **HorizontalPodAutoscaler (HPA)**: Automatically scales the number of pod replicas based on CPU utilization, memory utilization, or custom metrics (queue depth, RPS). Enables elastic scaling — the cluster responds to load changes automatically.

- **Kubernetes Service**: Provides a stable virtual IP and DNS name for a set of pods. Pods are ephemeral and their IPs change on restart; the Service provides a stable endpoint. Service types: ClusterIP (internal), NodePort (exposes on each node's port), LoadBalancer (provisions cloud LB), ExternalName (DNS alias).

- **Helm**: Kubernetes package manager. Charts are templates for Kubernetes objects with configurable values. Enables deploying complex multi-object applications with a single command and managing upgrades and rollbacks.

- **Resource Limits and Requests**: Pod resource requests (minimum guaranteed) and limits (maximum allowed) allow the Kubernetes scheduler to place pods appropriately and prevent one pod from starving others. Always set both — pods without limits can consume unbounded resources.

## Trade-offs

| Approach | Benefit | Cost |
|----------|---------|------|
| Kubernetes | Full orchestration, self-healing | High operational complexity |
| Docker Compose | Simple multi-container local dev | Not production-grade |
| ECS / Cloud Run | Managed orchestration | Less control, vendor lock-in |
| Serverless containers | No cluster to manage | Slower cold starts |

## When to Use

- **Kubernetes**: Any microservices deployment with 5+ services needing independent scaling
- **Managed Kubernetes (EKS/GKE/AKS)**: Almost always — managing the control plane yourself is significant operational overhead
- **HPA**: All stateless services — always configure autoscaling
- **Helm**: Package and version-manage all Kubernetes deployments