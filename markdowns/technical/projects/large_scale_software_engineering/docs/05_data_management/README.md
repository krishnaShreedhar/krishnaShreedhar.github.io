---
title: "Data Management"
subtitle: "Data management covers the storage, retrieval, transformation, and movement of data at scale. The explosion of database paradigms, caching systems, and streaming platforms over the past decade means engineers must..."
category: technical
project: large_scale_software_engineering
project_title: "Large Scale Software Engineering"
date: 2025-06-13
reading_time: 1
tags:
  - large-scale-software-engineering
  - docs
author: "Shreedhar Kodate"
output: "blogs/technical/posts/large_scale_software_engineering/docs/05_data_management/index.html"
---
Data management covers the storage, retrieval, transformation, and movement of data at scale. The explosion of database paradigms, caching systems, and streaming platforms over the past decade means engineers must understand a broad landscape to make effective choices for their workloads.

## Overview

```mermaid
mindmap
  root((Data\nManagement))
    Database Paradigms
      Relational SQL
      Document NoSQL
      Key-Value
      Wide Column
      Graph
      Time-Series
      Vector
    Consistency and Transactions
      ACID Properties
      BASE Properties
      Transaction Isolation Levels
      Distributed Transactions
      Two-Phase Commit
    Data Modeling
      ER Modeling
      Normalization
      Denormalization
      Schema Design
      Polyglot Persistence
    Caching
      In-Process Cache
      Distributed Cache
      Cache Aside
      Write-Through
      Write-Behind
      Read-Through
    Replication and Partitioning
      Leader-Follower
      Multi-Leader
      Leaderless
      Range Partitioning
      Hash Partitioning
    Data Pipelines
      ETL vs ELT
      Batch Processing
      Stream Processing
      Change Data Capture
      Data Lakehouse
```

## Topics in This Section

| File | Topic | Key Concepts |
|------|-------|--------------|
| [01_database_paradigms.md](01_database_paradigms.md) | Database Paradigms | SQL, NoSQL types, vector DBs |
| [02_consistency_transactions.md](02_consistency_transactions.md) | Consistency & Transactions | ACID, BASE, isolation levels |
| [03_data_modeling.md](03_data_modeling.md) | Data Modeling | ER, normalization, schema design |
| [04_caching_strategies.md](04_caching_strategies.md) | Caching | Cache patterns, eviction, invalidation |
| [05_replication_partitioning.md](05_replication_partitioning.md) | Replication & Partitioning | Replication modes, sharding strategies |
| [06_data_pipelines.md](06_data_pipelines.md) | Data Pipelines | ETL/ELT, CDC, streaming, lakehouse |