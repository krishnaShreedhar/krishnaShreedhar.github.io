---
title: "Communication Protocols"
subtitle: "Communication protocols define how services exchange data — the encoding, transport, sequencing, and error-handling rules that govern inter-service interactions. Choosing the right protocol involves trade-offs..."
category: technical
project: large_scale_software_engineering
project_title: "Large Scale Software Engineering"
date: 2025-01-23
reading_time: 1
tags:
  - large-scale-software-engineering
  - docs
author: "Shreedhar Kodate"
output: "blogs/technical/posts/large_scale_software_engineering/docs/04_communication_protocols/index.html"
---
Communication protocols define how services exchange data — the encoding, transport, sequencing, and error-handling rules that govern inter-service interactions. Choosing the right protocol involves trade-offs between latency, reliability, coupling, and operational complexity.

## Overview

```mermaid
mindmap
  root((Communication\nProtocols))
    Synchronous
      REST HTTP/1.1 HTTP/2
      gRPC Protocol Buffers
      GraphQL
      WebSocket
      Server-Sent Events
    Async Messaging
      Message Queues FIFO
      Pub-Sub Topics
      Event Streaming Kafka
      Dead Letter Queues
      Competing Consumers
    Serialization
      JSON
      Protocol Buffers protobuf
      Apache Avro
      MessagePack
      Apache Thrift
    Service-to-Service
      Service Discovery
      Load Balancing
      mTLS
      API Gateway
      Service Mesh
    Network Protocols
      TCP UDP
      HTTP/1.1 HTTP/2 HTTP/3 QUIC
      TLS/SSL
      DNS
      BGP
```

## Topics in This Section

| File | Topic | Key Concepts |
|------|-------|--------------|
| [01_synchronous_protocols.md](01_synchronous_protocols.md) | Synchronous | REST, gRPC, WebSocket, SSE |
| [02_async_messaging_patterns.md](02_async_messaging_patterns.md) | Async Messaging | Queues, pub/sub, streaming, DLQ |
| [03_serialization_formats.md](03_serialization_formats.md) | Serialization | JSON, Protobuf, Avro, schema evolution |
| [04_service_to_service_patterns.md](04_service_to_service_patterns.md) | Service-to-Service | Discovery, mTLS, service mesh, gateways |
| [05_network_protocols.md](05_network_protocols.md) | Network | TCP, HTTP versions, TLS, DNS, QUIC |