---
title: "Rate Limiting"
subtitle: "Rate limiting controls the rate at which clients can make requests to a service, protecting it from abuse, overload, and denial-of-service attacks. A well-designed rate limiter is fair, accurate at scale, and adds..."
category: technical
project: large_scale_software_engineering
project_title: "Large Scale Software Engineering"
date: 2025-07-08
reading_time: 4
tags:
  - large-scale-software-engineering
  - docs
author: "Shreedhar Kodate"
output: "blogs/technical/posts/large_scale_software_engineering/docs/06_scalability_performance/04_rate_limiting.html"
---
Rate limiting controls the rate at which clients can make requests to a service, protecting it from abuse, overload, and denial-of-service attacks. A well-designed rate limiter is fair, accurate at scale, and adds minimal latency to the request path.

## Rate Limiting Algorithms

```mermaid
graph TD
    subgraph TokenBucket[Token Bucket]
        TB[Bucket capacity: 100 tokens\nRefill rate: 10 tokens/sec]
        TBFlow[Request arrives\nToken available? Consume 1 and allow\nNo token? Reject 429]
        TB --> TBFlow
        TBNote[Allows bursts up to capacity\nSmooths traffic long-term\nMost common algorithm]
    end

    subgraph LeakyBucket[Leaky Bucket]
        LB[Fixed-size queue]
        LBFlow[Requests enqueue\nProcess at fixed rate\nQueue full? Reject]
        LB --> LBFlow
        LBNote[Strictly smooth output rate\nNo bursts at all\nGood for downstream protection]
    end

    subgraph FixedWindow[Fixed Window Counter]
        FW[Window: 1 minute\nCounter resets each window]
        FWFlow[Increment counter\nCounter less than limit? Allow\nElse: reject]
        FW --> FWFlow
        FWNote[Simple, low memory\nBoundary burst attack:\n2x limit in 2s at window boundary]
    end

    subgraph SlidingWindow[Sliding Window Log]
        SW[Log of timestamps for each request]
        SWFlow[Remove entries older than window\nCount remaining\nLess than limit? Allow and log]
        SW --> SWFlow
        SWNote[Precise, no boundary issues\nHigh memory - stores all timestamps\nNot suitable for high-volume]
    end
```

## Token Bucket Algorithm

```mermaid
sequenceDiagram
    participant Client
    participant RateLimiter
    participant Bucket as Token Bucket
    participant Service

    Note over Bucket: capacity=10, refill=2/sec

    Client->>RateLimiter: Request 1
    RateLimiter->>Bucket: consume 1 token (9 remaining)
    RateLimiter->>Service: Allow

    Client->>RateLimiter: Request 2 (0.1s later)
    RateLimiter->>Bucket: consume 1 token (8 remaining)
    RateLimiter->>Service: Allow

    Note over Bucket: 5 seconds pass - refilled to 10

    Client->>RateLimiter: Burst: 10 requests
    RateLimiter->>Bucket: consume 10 tokens (0 remaining)
    RateLimiter->>Service: All 10 allowed (burst)

    Client->>RateLimiter: Request 11 (immediately)
    RateLimiter->>Bucket: 0 tokens available
    RateLimiter-->>Client: 429 Too Many Requests
```

## Distributed Rate Limiting

```mermaid
graph TD
    subgraph LocalRateLimit[Local Rate Limiting - Per Instance]
        C1[Client] --> I1[Instance 1\nlocal counter: 40/100]
        C2[Client] --> I2[Instance 2\nlocal counter: 30/100]
        C3[Client] --> I3[Instance 3\nlocal counter: 35/100]

        Problem[Problem:\nSame client can make 300 req\nif spread across instances]
        style Problem fill:#fee2e2,stroke:#dc2626
    end

    subgraph CentralizedRL[Centralized Rate Limiting - Redis]
        Clients[All Clients] --> RL[Rate Limiter\nMiddleware]
        RL --> Redis[(Redis\nuser:123:counter\nTTL: 60s)]
        RL -->|allowed| Service[Service]
        RL -->|429| Rejected[Client]

        Note[Accurate across all instances\nRedis adds ~1ms latency\nRedis failure = full pass or full block]
        style Note fill:#dcfce7,stroke:#16a34a
    end
```

## Rate Limit Response Design

```mermaid
graph TD
    RateLimited[Rate Limited Request] --> Response[HTTP 429 Too Many Requests]
    Response --> Headers[Response Headers]

    Headers --> H1[X-RateLimit-Limit: 100\nTotal allowed per window]
    Headers --> H2[X-RateLimit-Remaining: 0\nTokens remaining this window]
    Headers --> H3[X-RateLimit-Reset: 1717236000\nUnix timestamp when limit resets]
    Headers --> H4[Retry-After: 30\nSeconds until client can retry]

    Client[Well-behaved Client] -->|reads Retry-After| Wait[Wait 30s then retry]
    Client -->|exponential backoff| Backoff[1s 2s 4s 8s with jitter]
```

## Key Concepts

- **Token Bucket**: Tokens are added to a bucket at a constant rate (refill rate). Each request consumes one token. If the bucket is empty, the request is rejected. The bucket capacity defines the burst allowance. Most popular algorithm — it allows short bursts while controlling average rate.

- **Leaky Bucket**: Requests are added to a queue (the "bucket"). Requests are processed from the queue at a constant rate (the "leak rate"). If the queue is full, new requests are rejected. Unlike token bucket, this strictly limits the output rate — no bursts. Used when downstream services must receive a smooth request rate.

- **Fixed Window Counter**: Divide time into fixed windows (e.g., 1 minute). Maintain a counter per client per window. Reset the counter at the window boundary. Simple to implement (Redis INCR with EXPIRE), but vulnerable to boundary bursts: a client can make 2x the limit by making requests at the end of one window and the start of the next.

- **Sliding Window Log**: Maintain a log of request timestamps for each client. On each request, remove timestamps older than the window, count remaining, and allow if count < limit. Precise but memory-intensive — not suitable for millions of clients.

- **Sliding Window Counter**: A hybrid — track the current window counter and the previous window counter. Estimate the count in the sliding window as: prev_count * ((window_length - time_in_current_window) / window_length) + current_count. Low memory, accurate approximation, widely used.

- **Rate Limit Keys**: The entity being rate limited can be: IP address (simple, easily spoofed), API key/client ID (per application), user ID (per user), or endpoint (per route). A layered approach combines multiple keys.

- **Distributed Rate Limiting**: In a multi-instance deployment, local counters are inaccurate because requests are spread across instances. Centralized rate limiting uses Redis with atomic operations (INCR + EXPIRE, or Lua scripts for token bucket) to maintain accurate counts. Redis Cluster provides high availability.

## Trade-offs

| Algorithm | Accuracy | Memory | Burst Handling | Complexity |
|-----------|---------|--------|--------------|------------|
| Fixed Window | Low (boundary attack) | Very Low | Allows 2x limit at boundary | Low |
| Sliding Window Log | Exact | High | Exact | Medium |
| Sliding Window Counter | ~99% accurate | Low | Smooth | Medium |
| Token Bucket | High | Low | Explicit burst capacity | Medium |
| Leaky Bucket | High | Low | No bursts | Low |

## When to Use

- **Token Bucket**: Default for API rate limiting — allows natural bursts while controlling average rate
- **Leaky Bucket**: When downstream services need a strictly metered input rate
- **Sliding Window Counter**: When accuracy and memory efficiency are both required at high scale
- **Fixed Window**: Acceptable only for low-value rate limits where boundary attacks don't matter
- **Centralized (Redis-based)**: All production rate limiters with more than one instance