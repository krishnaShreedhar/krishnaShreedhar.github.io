# Network Protocols

Network protocols are the foundational rules governing how data is transmitted across networks. Understanding protocol behaviour — connection establishment, flow control, encryption negotiation, and congestion management — enables engineers to diagnose performance problems and make informed infrastructure decisions.

## Protocol Stack

```mermaid
graph TD
    subgraph ApplicationLayer[Application Layer L7]
        HTTP[HTTP/1.1 / HTTP/2 / HTTP/3]
        gRPC[gRPC over HTTP/2]
        DNS[DNS]
        SMTP[SMTP / IMAP]
    end

    subgraph TransportLayer[Transport Layer L4]
        TCP[TCP\nReliable, ordered, connection-oriented]
        UDP[UDP\nUnreliable, connectionless, low overhead]
        QUIC[QUIC\nUDP-based, built-in TLS, multiplexing]
    end

    subgraph NetworkLayer[Network Layer L3]
        IP[IP - IPv4 / IPv6\nRouting, fragmentation]
        ICMP[ICMP\nPing, Traceroute]
        BGP[BGP\nInter-AS routing]
    end

    subgraph DataLink[Data Link Layer L2]
        ETH[Ethernet]
        WiFi[802.11 WiFi]
    end

    HTTP & gRPC & DNS --> TCP & QUIC
    TCP & UDP & QUIC --> IP
    IP --> ETH & WiFi
```

## TCP Connection Lifecycle

```mermaid
sequenceDiagram
    participant Client
    participant Server

    Note over Client,Server: Three-Way Handshake (Connection Establishment)
    Client->>Server: SYN (seq=x)
    Server-->>Client: SYN-ACK (seq=y, ack=x+1)
    Client->>Server: ACK (ack=y+1)
    Note over Client,Server: Connection Established - 1.5 RTT cost

    Client->>Server: HTTP Request (data)
    Server-->>Client: HTTP Response (data)

    Note over Client,Server: Four-Way Handshake (Teardown)
    Client->>Server: FIN
    Server-->>Client: ACK
    Server-->>Client: FIN
    Client->>Server: ACK
```

## HTTP Version Evolution

```mermaid
graph LR
    subgraph HTTP11[HTTP/1.1]
        H11[Text protocol\nOne request per connection\nHead-of-line blocking\nKeep-alive: limited pipelining]
        style H11 fill:#fee2e2,stroke:#dc2626
    end

    subgraph HTTP2[HTTP/2]
        H2[Binary protocol\nMultiplexed streams\nHeader compression - HPACK\nServer push\nSingle TCP connection\nTCP-level HOL still present]
        style H2 fill:#fef3c7,stroke:#d97706
    end

    subgraph HTTP3[HTTP/3 - QUIC]
        H3[QUIC over UDP\n0-RTT resumption\nStream-level loss recovery\nNo TCP HOL blocking\nBuilt-in TLS 1.3\nConnection migration]
        style H3 fill:#dcfce7,stroke:#16a34a
    end

    HTTP11 -->|2015| HTTP2 -->|2022| HTTP3
```

## TLS Handshake

```mermaid
sequenceDiagram
    participant Client
    participant Server

    Note over Client,Server: TLS 1.3 Handshake (1-RTT)
    Client->>Server: ClientHello\n(supported ciphers, key_share)
    Server-->>Client: ServerHello + Certificate\n(selected cipher, server key_share, finished)
    Note over Client: Verify certificate\nDerive session keys
    Client->>Server: Finished + HTTP Request
    Server-->>Client: HTTP Response
    Note over Client,Server: Total: 1 RTT after TCP connect
```

## DNS Resolution Flow

```mermaid
graph TD
    Browser[Browser] -->|1. cache miss| Resolver[Local DNS Resolver\n/etc/resolv.conf]
    Resolver -->|2. recursive query| Root[Root Nameserver\n13 root servers]
    Root -->|3. referral: .com TLD| Resolver
    Resolver -->|4. query .com TLD| TLD[TLD Nameserver\n.com servers]
    TLD -->|5. referral: example.com NS| Resolver
    Resolver -->|6. query authoritative| Auth[Authoritative Nameserver\nexample.com]
    Auth -->|7. A record: 93.184.216.34| Resolver
    Resolver -->|8. cached + returned| Browser

    style Auth fill:#dcfce7,stroke:#16a34a
    style Resolver fill:#dbeafe,stroke:#2563eb
```

## Key Concepts

- **TCP (Transmission Control Protocol)**: Provides reliable, ordered, error-checked delivery of bytes over a network. Uses a three-way handshake for connection establishment (1.5 RTT overhead), sliding window flow control, and congestion control algorithms (CUBIC, BBR). TCP guarantees ordering within a connection but causes head-of-line blocking — a lost packet stalls all subsequent data.

- **UDP (User Datagram Protocol)**: Connectionless, unreliable protocol with minimal overhead. No handshake, no ordering guarantees, no flow control. Suitable for latency-sensitive applications (gaming, VoIP, DNS queries) where an occasional lost packet is preferable to the latency of retransmission.

- **QUIC**: A transport protocol built over UDP, developed by Google and standardized by IETF as the foundation for HTTP/3. Provides: 0-RTT connection resumption, stream-level multiplexing (no TCP head-of-line blocking), built-in TLS 1.3, and connection migration (changing IP without reconnecting). Solves the fundamental limitations of HTTP/2 over TCP.

- **TLS (Transport Layer Security)**: Cryptographic protocol providing encryption, authentication, and integrity for network communications. TLS 1.3 (current standard) simplified the handshake to 1-RTT, removed weak cipher suites, and made forward secrecy mandatory. TLS 1.2 is still widely deployed but deprecated.

- **HTTP/2 Multiplexing**: Multiple HTTP streams share a single TCP connection, eliminating the HTTP/1.1 requirement to open 6+ parallel connections per domain. Each stream is independent — a request for `/api/users` and `/api/orders` can be in-flight simultaneously. However, a lost TCP packet blocks all HTTP/2 streams (TCP head-of-line blocking).

- **DNS (Domain Name System)**: The hierarchical, distributed naming system that maps domain names to IP addresses. Resolution traverses root nameservers, TLD nameservers, and authoritative nameservers. DNS caching (controlled by TTL) is critical for performance — a TTL of 60s means every hostname lookup may add 60s to propagation time for changes.

- **BGP (Border Gateway Protocol)**: The routing protocol that exchanges reachability information between autonomous systems (ISPs, cloud providers). BGP route advertisements determine how traffic flows across the internet. BGP misconfigurations (route leaks, hijacks) can cause large-scale internet outages.

## Trade-offs

| Protocol | Reliability | Latency | Throughput | Use Case |
|---------|------------|---------|-----------|---------|
| TCP | Guaranteed | Higher (handshake) | Limited by window | Web, databases, file transfer |
| UDP | None | Lowest | Unlimited | Gaming, VoIP, DNS, streaming |
| QUIC/HTTP3 | Guaranteed | Low (0-RTT) | High (no HOL) | Modern web, mobile |
| TLS 1.3 | Encrypted | 1-RTT overhead | Minimal impact | All secure communications |

## When to Use

- **TCP**: All reliable data transfer where ordering and delivery guarantees are required
- **UDP**: Latency-critical applications where some loss is tolerable and retransmission adds more cost than benefit
- **QUIC/HTTP3**: New client-server applications targeting modern browsers and mobile clients
- **TLS 1.3**: All production communication — there is no valid reason to use older TLS versions in new systems
