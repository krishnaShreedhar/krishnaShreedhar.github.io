---
title: "Caching Strategies"
subtitle: "Caching stores copies of frequently accessed data in a faster storage layer (memory) to reduce latency and database load. The fundamental trade-off is between data freshness (how up-to-date cached data is) and..."
category: technical
project: large_scale_software_engineering
project_title: "Large Scale Software Engineering"
date: 2025-06-14
reading_time: 4
tags:
  - large-scale-software-engineering
  - docs
author: "Shreedhar Kodate"
output: "blogs/technical/posts/large_scale_software_engineering/docs/05_data_management/04_caching_strategies.html"
---
Caching stores copies of frequently accessed data in a faster storage layer (memory) to reduce latency and database load. The fundamental trade-off is between data freshness (how up-to-date cached data is) and performance (how much work is saved by serving from cache).

## Cache Architecture Layers

```mermaid
graph TD
    Client[Client Request]

    Client --> L1[L1: CPU Cache\nnanoseconds\nper-process]
    L1 -->|miss| L2[L2: In-Process Cache\nmicroseconds\nlocal HashMap]
    L2 -->|miss| L3[L3: Distributed Cache\nmilliseconds\nRedis / Memcached]
    L3 -->|miss| L4[L4: Database / Origin\n10-100 milliseconds\nPostgres / S3]

    style L1 fill:#dcfce7,stroke:#16a34a
    style L2 fill:#dbeafe,stroke:#2563eb
    style L3 fill:#fef3c7,stroke:#d97706
    style L4 fill:#fee2e2,stroke:#dc2626
```

## Cache-Aside (Lazy Loading)

```mermaid
sequenceDiagram
    participant App
    participant Cache as Cache (Redis)
    participant DB as Database

    App->>Cache: GET user:123
    Cache-->>App: MISS (null)

    App->>DB: SELECT * FROM users WHERE id=123
    DB-->>App: {id:123, name:Alice}

    App->>Cache: SET user:123 {name:Alice} EX 300
    App-->>App: Use data

    Note over App: Next request
    App->>Cache: GET user:123
    Cache-->>App: HIT {name:Alice}
```

## Write-Through vs Write-Behind

```mermaid
graph LR
    subgraph WriteThrough[Write-Through]
        WTA[App] -->|1. write| WTC[Cache]
        WTC -->|2. synchronously write| WTDB[Database]
        WTC -->|3. confirm| WTA
        Note1[Consistent but\nhigher write latency]
    end

    subgraph WriteBehind[Write-Behind - Write-Back]
        WBA[App] -->|1. write| WBC[Cache]
        WBC -->|2. confirm immediately| WBA
        WBC -->|3. async batch write| WBDB[Database]
        Note2[Low write latency\nbut risk of data loss\nif cache fails before flush]
    end
```

## Cache Eviction Policies

```mermaid
graph TD
    Cache[Cache at Capacity] --> Evict[Eviction Decision]

    Evict --> LRU[LRU - Least Recently Used\nEvict item not accessed longest\nGood for temporal locality]
    Evict --> LFU[LFU - Least Frequently Used\nEvict item accessed fewest times\nGood for frequency locality]
    Evict --> TTL[TTL - Time To Live\nEvict after fixed time\nGood for time-sensitive data]
    Evict --> FIFO[FIFO - First In First Out\nEvict oldest inserted item\nSimple, poor hit rate]
    Evict --> Random[Random\nEvict random item\nLow overhead, unpredictable]

    LRU --> Note[Most common default\nRedis uses approximation\nO1 with doubly-linked list]
```

## Cache Invalidation Patterns

```mermaid
graph TD
    subgraph InvalidationStrategies[Cache Invalidation Strategies]
        TTL2[TTL Expiration\nPassive - data expires automatically\nSimple, stale data window = TTL]
        EventBased[Event-Based Invalidation\nActive - invalidate on data change\nLower staleness, more complex]
        WriteThrough2[Write-Through\nAlways write to cache and DB\nAlways fresh, write overhead]
        Versioned[Cache Versioning\nChange cache key on data update\nuser:123:v2 vs user:123:v1\nNo invalidation needed]
    end

    subgraph InvalidationPitfalls[Common Pitfalls]
        ThunderingHerd[Thundering Herd\nMany cache misses simultaneously\non expiry - use jitter on TTL]
        StaleData[Stale Data\nCache not invalidated on update\nuse short TTL or event-based]
        CacheStampede[Cache Stampede\nMany threads try to repopulate\nsame cache key simultaneously\nuse mutex/lock-on-miss]
    end
```

## Key Concepts

- **Cache Hit and Miss**: A cache hit occurs when the requested data is found in the cache. A cache miss requires fetching from the slower origin (database). Cache hit rate is the primary metric — most applications target 90%+ for hot data.

- **Cache-Aside (Lazy Loading)**: The application is responsible for reading from and writing to the cache. On a miss, the application reads from the database and populates the cache. Simple to implement, but the first request after expiry always goes to the database (cold start). Most common pattern.

- **Read-Through**: The cache sits in front of the database. On a miss, the cache itself fetches from the database and stores the result. The application always talks to the cache. Simplifies application code but requires cache support for this pattern.

- **Write-Through**: Every write goes to the cache first, then synchronously to the database. Keeps cache and database always in sync. Write latency is the sum of both writes. Eliminates stale data at the cost of write performance.

- **Write-Behind (Write-Back)**: Writes go to the cache immediately (fast), and the cache asynchronously flushes to the database in batches. Excellent write throughput but risks data loss if the cache fails before flushing. Good for high-write workloads where durability is less critical.

- **Cache Eviction**: When the cache is full, the eviction policy determines what to remove. LRU (Least Recently Used) is the most common — it evicts the item that hasn't been accessed for the longest time. LFU evicts the least frequently accessed. TTL is time-based expiry.

- **Thundering Herd**: When a popular cache entry expires, many concurrent requests hit the database simultaneously to repopulate it, causing a spike. Mitigation: add random jitter to TTL, use probabilistic early expiration, or use a mutex to allow only one thread to repopulate while others wait.

- **Cache Warming**: Pre-populating the cache before a service launch or after a cache flush, to avoid a cold-start thundering herd.

## Trade-offs

| Strategy | Consistency | Write Performance | Read Performance | Complexity |
|----------|------------|-------------------|-----------------|------------|
| Cache-Aside | Stale for TTL period | Normal (DB write) | High (after warm) | Low |
| Read-Through | Stale for TTL period | Normal | High | Low |
| Write-Through | Always fresh | Lower (double write) | High | Medium |
| Write-Behind | Risk of loss on failure | Highest | High | High |

## When to Use

- **Cache-Aside**: Most common choice — use for any read-heavy workload with tolerable staleness
- **Write-Through**: When read performance AND freshness are both required
- **Write-Behind**: Write-heavy workloads (counters, analytics) where some data loss is acceptable
- **Short TTL**: When data changes frequently and staleness matters (prices, inventory levels)
- **Long TTL + Event Invalidation**: When data rarely changes but must be fresh when it does (user profiles)