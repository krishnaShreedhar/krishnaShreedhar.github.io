# Cloud Architecture

Cloud architecture leverages managed cloud services to build scalable, resilient, and cost-efficient systems without managing physical hardware. The cloud provider's shared responsibility model, global infrastructure, and broad service catalog enable capabilities that were previously only available to large enterprises.

## Cloud Service Models

```mermaid
graph TD
    subgraph IaaS[IaaS - Infrastructure as a Service]
        I1[You manage: OS Apps Data Runtime]
        I2[Provider manages: Virtualization Servers Storage Network]
        I3[Examples: EC2 GCE Azure VMs]
        style I3 fill:#fef3c7,stroke:#d97706
    end

    subgraph PaaS[PaaS - Platform as a Service]
        P1[You manage: Applications Data]
        P2[Provider manages: Runtime OS Infra]
        P3[Examples: Heroku App Engine Cloud Run]
        style P3 fill:#dbeafe,stroke:#2563eb
    end

    subgraph SaaS[SaaS - Software as a Service]
        S1[You manage: Data and configuration]
        S2[Provider manages: Everything]
        S3[Examples: Salesforce Gmail Datadog]
        style S3 fill:#dcfce7,stroke:#16a34a
    end

    IaaS --> PaaS --> SaaS
```

## AWS Well-Architected Framework

```mermaid
mindmap
  root((Well-Architected\nFramework))
    Operational Excellence
      Infrastructure as code
      Frequent small reversible changes
      Anticipate failure
      Learn from operational events
    Security
      Strong identity foundation
      Enable traceability
      Apply security at all layers
      Protect data in transit and at rest
    Reliability
      Automatic recovery from failure
      Scale horizontally
      Stop guessing capacity
      Manage change with automation
    Performance Efficiency
      Democratize advanced technologies
      Go global in minutes
      Use serverless architectures
      Experiment more often
    Cost Optimization
      Adopt consumption model
      Measure overall efficiency
      Analyze and attribute expenditure
      Use managed services
    Sustainability
      Understand your impact
      Maximize utilization
      Use efficient hardware
```

## Multi-Region Architecture

```mermaid
graph TD
    subgraph Global[Global Services]
        Route53[Route 53\nGlobal DNS\nHealth-based routing]
        CF[CloudFront CDN\nGlobal edge network\n300+ PoPs]
        IAM[IAM\nGlobal identity]
    end

    subgraph Primary[Primary Region: us-east-1]
        ALB1[Application Load Balancer]
        ECS1[ECS / EKS Cluster]
        RDS1[(RDS Primary\nMulti-AZ)]
        ALB1 --> ECS1 --> RDS1
    end

    subgraph DR[DR Region: us-west-2]
        ALB2[Application Load Balancer]
        ECS2[ECS / EKS Cluster\nstale or cold]
        RDS2[(RDS Read Replica\nor Aurora Global)]
        ALB2 --> ECS2 --> RDS2
    end

    Route53 -->|primary| ALB1
    Route53 -->|failover on health check fail| ALB2
    RDS1 -->|async replication| RDS2
    CF --> ECS1 & ECS2

    style Primary fill:#dcfce7,stroke:#16a34a
    style DR fill:#fef3c7,stroke:#d97706
```

## Key Concepts

- **IaaS (Infrastructure as a Service)**: Cloud provider offers virtualized compute, storage, and networking. Customer manages OS, runtime, and applications. Maximum control, maximum management responsibility. EC2 is the canonical example.

- **PaaS (Platform as a Service)**: Cloud provider manages the runtime environment; customers deploy applications. Lower operational overhead — no OS patching, no runtime management. Less control. Examples: Google App Engine, AWS Elastic Beanstalk, Cloud Run.

- **Serverless**: Cloud provider manages all infrastructure including scaling. Pay per invocation. Zero idle cost. Maximum operational simplicity for stateless event-driven workloads.

- **Shared Responsibility Model**: The cloud provider secures the infrastructure (physical datacenters, hypervisors, global network). The customer secures what they deploy (OS configuration, network security groups, IAM policies, application code, data encryption).

- **Availability Zones (AZs)**: Physically separate datacenters within a region, connected by low-latency high-bandwidth links. Deploying across multiple AZs provides resilience against single-facility failures. Most cloud services offer Multi-AZ deployment as a standard option.

- **Cost Optimization**: Cloud costs can grow unexpectedly. Strategies: right-size instances (don't over-provision), use Spot/Preemptible instances for fault-tolerant workloads (60-90% discount), use Reserved Instances for predictable base load (30-60% discount), implement lifecycle policies to delete old snapshots/logs, use Savings Plans for committed usage.

- **Cloud Native**: Designing systems that exploit cloud capabilities — auto-scaling, managed services, pay-per-use, global distribution — rather than lifting-and-shifting on-premises architectures to the cloud.

## Trade-offs

| Approach | Control | Operational Cost | Portability |
|----------|---------|----------------|------------|
| IaaS (EC2) | High | High | Medium |
| PaaS (Cloud Run) | Medium | Low | Medium |
| Serverless | Low | Very Low | Lower |
| On-premises | Highest | Highest | Full |

## When to Use

- **IaaS**: When you need specific hardware, custom OS configuration, or capabilities not available as managed services
- **PaaS**: Stateless web applications, API backends — significant reduction in operational overhead
- **Serverless**: Event-driven processing, variable/spiky workloads, rapid prototyping
- **Multi-region**: When latency to a single region is unacceptable for global users, or when disaster recovery requirements mandate geographic redundancy
