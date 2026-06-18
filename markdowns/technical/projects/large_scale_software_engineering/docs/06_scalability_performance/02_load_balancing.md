# Load Balancing

Load balancing distributes incoming network traffic across multiple backend instances to maximize throughput, minimize latency, and ensure no single server becomes a bottleneck. Load balancers also provide health checking and automatic failover.

## Load Balancer Architecture

```mermaid
graph TD
    Internet[Internet Traffic] --> DNS[DNS / Global Load Balancer\nGeographic routing]
    DNS --> L4LB[L4 Network Load Balancer\nTCP/UDP level\nexample: AWS NLB]
    L4LB --> L7LB[L7 Application Load Balancer\nHTTP level\nexample: AWS ALB, nginx]

    L7LB -->|/api/*| APICluster[API Server Pool]
    L7LB -->|/static/*| CDN[CDN / Static Assets]
    L7LB -->|/ws/*| WSCluster[WebSocket Server Pool]

    APICluster --> DB[(Database)]
    APICluster --> Cache[(Cache)]

    style L4LB fill:#dbeafe,stroke:#2563eb
    style L7LB fill:#dcfce7,stroke:#16a34a
```

## Load Balancing Algorithms

```mermaid
graph TD
    subgraph Algorithms[Load Balancing Algorithms]
        RR[Round Robin\nA B C A B C\nEqual weight rotation\nSimplest, uniform load assumption]

        WRR[Weighted Round Robin\nA A A B B C\nWeight by server capacity\nGood for heterogeneous servers]

        LC[Least Connections\nRoute to server with\nfewest active connections\nGood for variable request duration]

        WLC[Weighted Least Connections\nLeast connections divided by weight\nBest for heterogeneous servers\nwith variable workload]

        IPH[IP Hash\nhash(client_ip) mod N\nSame client = same server\nSession affinity without cookies]

        RandP2[Random Pick of 2\nSample 2 random servers\nPick least loaded of the 2\nO1 approximates optimal]
    end
```

## Health Checking

```mermaid
sequenceDiagram
    participant LB as Load Balancer
    participant H1 as Healthy Instance
    participant H2 as Unhealthy Instance

    loop Every 10 seconds
        LB->>H1: GET /health
        H1-->>LB: 200 OK
        LB->>H2: GET /health
        H2--xLB: Timeout (>2s)
    end

    Note over LB: H2 failed 3 consecutive checks
    LB->>LB: Mark H2 as UNHEALTHY
    LB->>LB: Remove H2 from rotation

    Note over LB,H2: H2 recovers
    LB->>H2: GET /health
    H2-->>LB: 200 OK
    LB->>LB: Mark H2 as HEALTHY (after 2 passes)
    LB->>LB: Add H2 back to rotation
```

## Layer 4 vs Layer 7

```mermaid
graph LR
    subgraph L4[Layer 4 - Transport]
        L4In[Packet in]
        L4Decision[Route by\nSrc/Dst IP + Port\nTCP session affinity]
        L4Out[Forward packet]
        L4In --> L4Decision --> L4Out
        L4Note[Very fast - line rate\nNo TLS termination by default\nNo content inspection\nLower latency]
        style L4Note fill:#dcfce7,stroke:#16a34a
    end

    subgraph L7[Layer 7 - Application]
        L7In[HTTP request in]
        L7Decision[Route by\nHTTP headers\nURL path\nCookies\nRequest body]
        L7Out[Forward request]
        L7In --> L7Decision --> L7Out
        L7Note[Full HTTP inspection\nTLS termination\nContent-based routing\nSlower but more capable]
        style L7Note fill:#dbeafe,stroke:#2563eb
    end
```

## Key Concepts

- **L4 Load Balancer**: Operates at the transport layer, routing TCP/UDP connections based on IP addresses and ports. Does not inspect packet contents. Extremely fast (often hardware-accelerated). Cannot route based on HTTP paths, headers, or cookies. Used for very high-throughput TCP traffic. Examples: AWS Network Load Balancer, HAProxy in TCP mode.

- **L7 Load Balancer**: Operates at the application layer, inspecting HTTP requests. Can route based on URL path (`/api/` vs `/static/`), headers, cookies, query parameters, or request body. Enables advanced features: SSL termination, request transformation, rate limiting, WAF, canary deployments. Examples: NGINX, AWS ALB, Envoy.

- **Round Robin**: Distributes requests in rotation. Assumes equal server capacity and equal request cost. Simplest algorithm. Works well when requests have similar resource consumption and servers are homogeneous.

- **Least Connections**: Routes each new request to the server with the fewest active connections. Better than round robin for long-lived connections (WebSockets, file uploads) where connection count correlates with load. Requires centralized state tracking.

- **Weighted Round Robin**: Assigns a weight to each server proportional to its capacity. A 4-core server gets 4x the requests of a 1-core server. Useful when backend instances have different hardware specifications.

- **Power of Two Choices (P2C)**: Pick two servers at random, send to the one with fewer active connections. Provides near-optimal load distribution in O(1) without centralized coordination. Used in Envoy and modern service meshes.

- **Session Affinity (Sticky Sessions)**: Routes a client's requests to the same backend instance. Necessary for stateful sessions but undermines horizontal scaling. Prefer externalising session state to eliminate stickiness requirements.

- **Health Checking**: Load balancers continuously probe backends with health checks. Failing backends are removed from rotation. Passive health checking (detecting errors on real traffic) is faster but harsher; active health checking (synthetic probes) is more conservative but catches failures before users are impacted.

## Trade-offs

| Algorithm | Use Case | Limitation |
|-----------|----------|-----------|
| Round Robin | Homogeneous servers, uniform requests | Poor for variable request cost |
| Least Connections | Variable request duration | State tracking overhead |
| IP Hash | Session affinity | Uneven distribution if few clients |
| Weighted Round Robin | Heterogeneous servers | Requires manual weight configuration |
| Power of Two Choices | Best general-purpose | Requires connection count from instances |

## When to Use

- **L4 LB**: When raw throughput and minimal latency overhead are the primary requirements (high-volume TCP services, UDP-based protocols)
- **L7 LB**: Default for HTTP services — enables content-based routing, SSL termination, health checking at the application level
- **Least Connections**: Long-lived connections, WebSockets, streaming services
- **Round Robin**: Stateless HTTP services with uniform request profiles
- **No sticky sessions**: Design stateless services to eliminate the stickiness requirement entirely
