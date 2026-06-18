# Synchronous Communication Protocols

Synchronous protocols require the caller to wait for a response before proceeding. They provide immediate feedback, simplify error handling, and are easy to reason about — at the cost of temporal coupling between caller and callee.

## REST vs gRPC vs GraphQL Comparison

```mermaid
graph TD
    subgraph REST[REST / HTTP]
        R_Transport[HTTP/1.1 or HTTP/2]
        R_Format[JSON text]
        R_Contract[OpenAPI spec]
        R_Style[Resource-oriented\nGET POST PUT DELETE PATCH]
    end

    subgraph gRPC[gRPC]
        G_Transport[HTTP/2 mandatory]
        G_Format[Protocol Buffers binary]
        G_Contract[.proto schema mandatory]
        G_Style[Procedure-oriented\nUnary, Server Stream, Client Stream, Bidi]
    end

    subgraph GraphQL[GraphQL]
        Q_Transport[HTTP/1.1 or HTTP/2]
        Q_Format[JSON text]
        Q_Contract[GraphQL Schema]
        Q_Style[Query - read\nMutation - write\nSubscription - stream]
    end
```

## HTTP Request-Response Flow

```mermaid
sequenceDiagram
    participant Client
    participant LB as Load Balancer
    participant Service
    participant DB as Database

    Client->>LB: POST /api/orders HTTP/2
    LB->>Service: Forward request
    Service->>Service: Validate request body
    Service->>DB: INSERT order
    DB-->>Service: order_id: 12345
    Service->>Service: Build response
    Service-->>LB: 201 Created {id: 12345}
    LB-->>Client: 201 Created {id: 12345}

    Note over Client,Service: Total RTT: 10-50ms typical
```

## gRPC Streaming Modes

```mermaid
graph LR
    subgraph Unary[Unary RPC]
        UC[Client] -->|single request| US[Server]
        US -->|single response| UC
    end

    subgraph ServerStream[Server Streaming]
        SSC[Client] -->|single request| SSS[Server]
        SSS -->|stream of responses| SSC
    end

    subgraph ClientStream[Client Streaming]
        CSC[Client] -->|stream of requests| CSS[Server]
        CSS -->|single response| CSC
    end

    subgraph Bidi[Bidirectional Streaming]
        BC[Client] <-->|concurrent streams| BS[Server]
    end
```

## WebSocket Protocol

```mermaid
sequenceDiagram
    participant Client
    participant Server

    Client->>Server: HTTP GET /ws\nUpgrade: websocket\nConnection: Upgrade

    Server-->>Client: 101 Switching Protocols

    Note over Client,Server: WebSocket connection established - full duplex

    Client->>Server: text frame: {type: subscribe, channel: prices}
    Server-->>Client: text frame: {type: price, symbol: AAPL, price: 189.50}
    Server-->>Client: text frame: {type: price, symbol: AAPL, price: 189.55}
    Client->>Server: ping frame
    Server-->>Client: pong frame
    Client->>Server: close frame
    Server-->>Client: close frame
```

## Key Concepts

- **REST (Representational State Transfer)**: An architectural style for distributed hypermedia systems. Uses standard HTTP verbs (GET, POST, PUT, PATCH, DELETE) to operate on resources identified by URIs. Stateless — each request contains all information needed to process it. JSON is the de facto standard format. REST's simplicity and ubiquity make it the default choice for public APIs.

- **gRPC**: A high-performance, open-source RPC framework using HTTP/2 and Protocol Buffers. Provides strongly-typed contracts via `.proto` schema files, enabling code generation in multiple languages. Supports four communication modes: unary (request/response), server streaming, client streaming, and bidirectional streaming. ~5-10x more efficient than JSON REST for serialization.

- **GraphQL**: A query language and runtime for APIs. Clients specify exactly what data they need in a query, eliminating over-fetching (too much data) and under-fetching (too many round trips). The schema is the contract. Mutations modify data; subscriptions stream real-time updates. Requires schema design discipline to avoid N+1 query problems.

- **WebSocket**: Full-duplex communication channel over a single TCP connection, established via HTTP upgrade handshake. Enables server-to-client push without polling. Suitable for real-time features: live charts, collaborative editing, gaming, chat. Managing WebSocket connections at scale requires sticky sessions or a shared pub/sub layer.

- **Server-Sent Events (SSE)**: One-directional server-to-client streaming over plain HTTP. Simpler than WebSockets for use cases where only the server pushes data (activity feeds, live logs, progress notifications). SSE automatically reconnects on disconnection and handles event IDs for resumption.

- **HTTP/2**: Binary protocol with header compression (HPACK), multiplexing (multiple streams over one TCP connection), server push, and stream prioritization. Eliminates head-of-line blocking at the HTTP layer (but not at the TCP layer). Required by gRPC.

## Trade-offs

| Aspect | REST | gRPC | GraphQL | WebSocket |
|--------|------|------|---------|-----------|
| Performance | Moderate | High | Moderate | Low overhead after connect |
| Human readable | Yes | No (binary) | Yes | Yes/No |
| Streaming | Limited (SSE) | Native (4 modes) | Subscriptions | Native full-duplex |
| Browser support | Full | Limited | Full | Full |
| Code generation | Optional | Mandatory | Optional | Manual |
| Schema contract | Optional (OpenAPI) | Mandatory (.proto) | Mandatory (schema) | Manual |
| Caching | HTTP cache friendly | Not cache friendly | Complex | Not cacheable |

## When to Use

- **REST**: Public APIs, browser-facing services, when simplicity and HTTP caching are priorities
- **gRPC**: Internal service-to-service communication, polyglot environments, high-throughput data transfer
- **GraphQL**: Client-driven data fetching (mobile apps, SPAs with varying data needs per view)
- **WebSocket**: Real-time bidirectional communication (chat, gaming, collaborative tools)
- **SSE**: Server-to-client push where client never sends data after initial connection
