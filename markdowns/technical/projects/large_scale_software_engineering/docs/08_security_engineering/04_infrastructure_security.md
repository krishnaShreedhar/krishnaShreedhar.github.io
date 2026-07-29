---
title: "Infrastructure Security"
subtitle: "Infrastructure security protects the compute, network, and storage layers that run applications. Cloud environments require explicit security configuration — the shared responsibility model means the cloud provider..."
category: technical
project: large_scale_software_engineering
project_title: "Large Scale Software Engineering"
date: 2025-06-09
reading_time: 3
tags:
  - large-scale-software-engineering
  - docs
author: "Shreedhar Kodate"
output: "blogs/technical/posts/large_scale_software_engineering/docs/08_security_engineering/04_infrastructure_security.html"
---
Infrastructure security protects the compute, network, and storage layers that run applications. Cloud environments require explicit security configuration — the shared responsibility model means the cloud provider secures the infrastructure, but tenants must secure everything they deploy on it.

## Network Security Architecture

```mermaid
graph TD
    Internet[Internet]

    subgraph VPC[VPC - Virtual Private Cloud]
        subgraph PublicSubnet[Public Subnet]
            WAF2[WAF / Shield]
            ALB[Application Load Balancer]
            Bastion[Bastion Host\nfor SSH access]
            WAF2 --> ALB
        end

        subgraph PrivateSubnet[Private App Subnet - No Public IP]
            EC2A[App Server A]
            EC2B[App Server B]
            ALB --> EC2A & EC2B
        end

        subgraph DataSubnet[Data Subnet - Isolated]
            RDS[(RDS Database\nNo internet access)]
            EC2A & EC2B --> RDS
        end

        subgraph SecurityGroups[Security Groups - Stateful Firewall]
            SG1[ALB SG: 443 from 0.0.0.0]
            SG2[App SG: 8080 from ALB SG only]
            SG3[DB SG: 5432 from App SG only]
        end
    end

    Internet --> WAF2
    style PrivateSubnet fill:#fef3c7,stroke:#d97706
    style DataSubnet fill:#dcfce7,stroke:#16a34a
```

## Secrets Management

```mermaid
graph TD
    subgraph BadPractice[Bad: Secrets in Code / Config]
        Env[DB_PASSWORD=mypassword123\nin environment or .env file\nor hardcoded in source]
        Risk[Committed to git\nVisible in logs\nExposed in crash dumps]
        style Env fill:#fee2e2,stroke:#dc2626
        style Risk fill:#fee2e2,stroke:#dc2626
    end

    subgraph GoodPractice[Good: Secrets Manager]
        SM[HashiCorp Vault\nAWS Secrets Manager\nGCP Secret Manager]
        App[Application] -->|authenticate with IAM/service account| SM
        SM -->|dynamic short-lived credentials| App
        App -->|use credential| DB[(Database)]
        SM -->|automatic rotation| DB

        Features[Features:\nAudit log of all access\nAutomatic rotation\nFine-grained policies\nLease-based credentials expire]
        style SM fill:#dcfce7,stroke:#16a34a,stroke-width:2px
    end
```

## Container Security

```mermaid
graph TD
    subgraph ContainerSecurity[Container Security Layers]
        Image[Image Security\nnon-root user\nminimal base image\nno secrets in image\nimage signing]

        Registry[Registry Security\nimage scanning - Trivy\nprivate registry\npull policy]

        Runtime[Runtime Security\nread-only filesystem\nno privileged containers\nseccomp profiles\nno host network]

        Network[Network Policies\npod-to-pod traffic rules\ndefault deny all\nexplicit allow rules]

        Secrets2[Secrets\nmounted as volumes\nnot environment variables\nexternal secrets operator]
    end

    Image --> Registry --> Runtime --> Network --> Secrets2
```

## Key Concepts

- **VPC and Network Segmentation**: Isolate resources into subnets by trust level. Public subnets contain internet-facing resources (load balancers). Private subnets contain application servers with no public IPs. Data subnets contain databases with no internet access. Security groups (stateful firewalls) enforce which subnets can communicate with which.

- **Security Groups**: Virtual firewalls that control inbound and outbound traffic for resources. Rules reference other security groups rather than IP ranges — application servers allow traffic from the load balancer's security group, databases allow traffic from the application server's security group. This ensures only expected traffic flows regardless of IP changes.

- **Secrets Management**: Never store secrets (database passwords, API keys, TLS certificates) in environment variables, configuration files, or source code. Use a secrets manager that: encrypts secrets at rest and in transit, provides fine-grained access control per service, maintains an audit log of all secret accesses, and supports automatic rotation.

- **Dynamic Secrets (Vault)**: HashiCorp Vault can generate short-lived database credentials on demand (dynamic secrets). Applications request credentials from Vault, receive credentials valid for 1 hour, use them, then discard them. A breach of those credentials is limited to the lease duration.

- **WAF (Web Application Firewall)**: Inspects HTTP traffic and blocks malicious requests based on rules. Provides protection against OWASP Top 10 attacks, bot traffic, and DDoS at the application layer. AWS WAF, Cloudflare WAF, and ModSecurity are common implementations.

- **Container Security**: Run containers with non-root users, read-only root filesystems, and minimal capabilities (seccomp profiles drop Linux capabilities not needed). Never run privileged containers. Scan images for CVEs before deployment. Sign images with Cosign/Notary for supply chain integrity.

- **IAM Least Privilege**: Cloud IAM roles should be scoped to the minimum actions on the minimum resources. Prefer IAM roles over access keys. Rotate access keys regularly. Use IAM conditions to restrict access to specific resource tags, regions, or time windows.

- **mTLS Between Services**: All internal service-to-service communication should use mTLS to authenticate service identities and encrypt traffic. In a service mesh (Istio, Linkerd), this is automated — certificates are issued by the mesh CA and rotated automatically.

## Trade-offs

| Control | Security Benefit | Operational Cost |
|---------|----------------|-----------------|
| Network segmentation | Limits lateral movement | VPC design complexity |
| Secrets manager | No hardcoded secrets | App must authenticate to secrets manager |
| Dynamic secrets | Short credential lifetime | Vault operational complexity |
| WAF | Blocks known attacks | False positive management |
| Container security hardening | Reduces container escape risk | More restrictive runtime |

## When to Apply

- Network segmentation: from day one — retrofitting VPC structure is painful
- Secrets management: replace all environment-variable secrets before production launch
- WAF: all internet-facing services
- Container hardening: enabled by default in production; relaxed only with explicit justification
- mTLS: all production internal service traffic in microservices architectures