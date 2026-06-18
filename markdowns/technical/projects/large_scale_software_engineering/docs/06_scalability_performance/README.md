# Scalability and Performance

Scalability is the ability of a system to handle growing amounts of work by adding resources. Performance is the efficiency with which a system uses those resources. At scale, these two properties must be co-designed — a system can be scalable but slow, or fast but brittle under load.

## Overview

```mermaid
mindmap
  root((Scalability and\nPerformance))
    Scalability Dimensions
      Vertical Scaling
      Horizontal Scaling
      Load Distribution
      Stateless Design
      Data Partitioning
    Load Balancing
      Round Robin
      Least Connections
      IP Hash
      Consistent Hashing
      Weighted Routing
      Layer 4 vs Layer 7
    Performance Engineering
      Profiling and Bottlenecks
      Latency vs Throughput
      Amdahls Law
      Latency Percentiles
      Database Query Optimization
      Connection Pooling
    Rate Limiting
      Token Bucket
      Leaky Bucket
      Fixed Window Counter
      Sliding Window Log
      Sliding Window Counter
```

## Topics in This Section

| File | Topic | Key Concepts |
|------|-------|--------------|
| [01_scalability_dimensions.md](01_scalability_dimensions.md) | Scalability Dimensions | Vertical, horizontal, functional, geographic |
| [02_load_balancing.md](02_load_balancing.md) | Load Balancing | Algorithms, L4 vs L7, health checks |
| [03_performance_engineering.md](03_performance_engineering.md) | Performance Engineering | Profiling, latency, Amdahl's law |
| [04_rate_limiting.md](04_rate_limiting.md) | Rate Limiting | Algorithms, distributed rate limiting |
