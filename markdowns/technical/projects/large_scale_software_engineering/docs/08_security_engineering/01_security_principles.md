---
title: "Security Engineering Principles"
subtitle: "Security principles are foundational guidelines that shape the design of secure systems. Applying these principles during design is orders of magnitude cheaper than remediating vulnerabilities after deployment...."
category: technical
project: large_scale_software_engineering
project_title: "Large Scale Software Engineering"
date: 2025-01-03
reading_time: 3
tags:
  - large-scale-software-engineering
  - docs
author: "Shreedhar Kodate"
output: "blogs/technical/posts/large_scale_software_engineering/docs/08_security_engineering/01_security_principles.html"
---
Security principles are foundational guidelines that shape the design of secure systems. Applying these principles during design is orders of magnitude cheaper than remediating vulnerabilities after deployment. Security is not a feature — it is a property of the system.

## Core Security Principles

```mermaid
mindmap
  root((Security\nPrinciples))
    Defense in Depth
      Multiple security layers
      No single point of failure
      Compensating controls
    Least Privilege
      Minimal permissions
      JIT access
      Service accounts
    Zero Trust
      Never trust always verify
      Network location is not identity
      Continuous verification
    Secure by Default
      No open ports by default
      Deny all then allow
      Strong defaults
    Fail Secure
      On error deny not allow
      Fail closed
    Economy of Mechanism
      Simple designs
      Smaller attack surface
```

## Defense in Depth

```mermaid
graph TD
    Internet[Internet - Untrusted]

    subgraph Layer1[Layer 1: Network Perimeter]
        WAF[Web Application Firewall\nDDoS Protection]
        DDOS[CDN - Cloudflare / Akamai]
    end

    subgraph Layer2[Layer 2: Network Segmentation]
        PublicSubnet[Public Subnet\nLoad Balancers only]
        PrivateSubnet[Private Subnet\nApp Servers]
        DataSubnet[Data Subnet\nDatabases - no internet]
    end

    subgraph Layer3[Layer 3: Application Security]
        AuthN[Authentication]
        AuthZ[Authorization]
        InputVal[Input Validation]
        Encryption[Encryption at Rest and in Transit]
    end

    subgraph Layer4[Layer 4: Data Security]
        FieldEncrypt[Field-level Encryption]
        DLP[Data Loss Prevention]
        Audit[Audit Logging]
    end

    Internet --> Layer1 --> Layer2 --> Layer3 --> Layer4

    style Layer1 fill:#fee2e2,stroke:#dc2626
    style Layer2 fill:#fef3c7,stroke:#d97706
    style Layer3 fill:#dbeafe,stroke:#2563eb
    style Layer4 fill:#dcfce7,stroke:#16a34a
```

## Zero Trust Architecture

```mermaid
graph TD
    subgraph Traditional[Traditional - Castle and Moat]
        TrustsNetwork[Trust the network perimeter\nInside = trusted\nOutside = untrusted]
        TFlaw[Flaw: Breach inside perimeter\ngrants full access]
        style TFlaw fill:#fee2e2,stroke:#dc2626
    end

    subgraph ZeroTrust[Zero Trust - Never Trust Always Verify]
        ZTIdentity[Verify identity\nfor every request]
        ZTDevice[Verify device\nhealth and posture]
        ZTContext[Evaluate context\nlocation, time, behavior]
        ZTLeastPriv[Grant minimal access\nfor minimal duration]

        ZTIdentity --> ZTDevice --> ZTContext --> ZTLeastPriv
    end

    subgraph ZTComponents[Zero Trust Components]
        IdP[Identity Provider\nOkta, Azure AD]
        PAM[Privileged Access Management]
        SIEM[SIEM - continuous monitoring]
        MicroSeg[Micro-segmentation]
    end
```

## Key Concepts

- **Defense in Depth**: Layer multiple independent security controls so that if one layer is bypassed, others still protect the asset. No single security control should be the only barrier between an attacker and sensitive data. Layers include: network perimeter, network segmentation, host security, application security, data encryption, audit logging.

- **Least Privilege**: Every process, user, and service should have the minimum permissions necessary to perform its function. A read-only service should have read-only database credentials. An application should not run as root. Just-In-Time (JIT) access grants elevated permissions only for the duration needed.

- **Zero Trust**: Security model that assumes no implicit trust based on network location. Every request must be authenticated and authorized, regardless of whether it originates from inside or outside the corporate network. "Never trust, always verify" — the replacement for castle-and-moat perimeter security.

- **Secure by Default**: Systems should be configured securely out of the box. Default configurations should deny all, require authentication, use strong encryption, and expose minimal functionality. Users who need more permissive configurations should explicitly enable it.

- **Fail Secure**: When a security check fails (error, timeout, unavailable), the system should deny access rather than allow it. An authentication service that crashes should prevent access, not grant it. Fail-open is a common and dangerous anti-pattern.

- **Economy of Mechanism**: Keep security mechanisms as simple as possible. Complex security systems are harder to reason about, more likely to have subtle bugs, and more expensive to audit. Simplicity reduces attack surface.

- **Complete Mediation**: Every access to every object must be checked for authorization, every time. Caching authorization decisions must account for permission changes between cache population and access.

## Trade-offs

| Principle | Security Benefit | Operational Cost |
|-----------|----------------|-----------------|
| Least privilege | Limits breach scope | Access management overhead |
| Zero trust | Eliminates lateral movement | Higher authentication overhead |
| Defense in depth | No single failure is fatal | More infrastructure to maintain |
| Secure by default | Reduces accidental exposure | May block legitimate use |
| Fail secure | Prevents access on error | May cause availability impact |

## When to Apply

- Apply least privilege to all service accounts, IAM roles, and database users from day one
- Adopt zero trust when moving workloads to cloud or when remote work eliminates network perimeter as a security boundary
- Defense in depth should guide all architecture reviews — ask "what happens if this control fails?"