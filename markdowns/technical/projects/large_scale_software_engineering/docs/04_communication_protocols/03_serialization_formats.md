# Serialization Formats

Serialization converts in-memory data structures into a format suitable for storage or transmission. The choice of format affects payload size, serialization speed, schema evolution safety, human readability, and cross-language compatibility.

## Format Comparison Overview

```mermaid
graph TD
    subgraph TextFormats[Text-Based Formats]
        JSON[JSON\nHuman readable\nNo schema required\nUniversal support]
        YAML[YAML\nHuman readable\nConfiguration focused\nJSON superset]
        XML[XML\nVerbose\nEnterprise legacy\nXPath / XSLT]
        CSV[CSV\nTabular data\nSimplest possible\nNo nesting]
    end

    subgraph BinaryFormats[Binary Formats]
        Protobuf[Protocol Buffers\nFastest + smallest\nStrong schema\ngRPC standard]
        Avro[Apache Avro\nSchema in data\nHadoop ecosystem\nKafka standard]
        Thrift[Apache Thrift\nMulti-language\nFacebook origin]
        MsgPack[MessagePack\nBinary JSON\nNo schema needed\nMinimal overhead]
        Parquet[Apache Parquet\nColumnar storage\nAnalytics optimized\nS3 / data lakes]
    end

    TextFormats --> Performance[Performance: Binary outperforms by 5-10x]
    BinaryFormats --> Performance
```

## Schema Evolution

```mermaid
graph LR
    subgraph Protobuf[Protocol Buffers - Schema Evolution]
        V1[Version 1\nfield 1: name\nfield 2: email]
        V2[Version 2\nfield 1: name\nfield 2: email\nfield 3: phone - optional new]
        V3[Version 3\nfield 1: name\nfield 2: email - deprecated\nfield 3: phone\nfield 4: contact_info - new]
        V1 --> V2 --> V3
        Note[Old readers ignore unknown fields\nNew fields have defaults\nField numbers never change]
    end

    subgraph Avro[Avro - Schema Registry]
        SchemaReg[Schema Registry\nversioned schemas]
        Producer[Producer\nwrites with schema v2]
        Consumer[Consumer\nreads with schema v1]
        SchemaReg --> Producer & Consumer
        Note2[Schema resolution resolves\nproducer schema to reader schema\nvia field matching]
    end
```

## Protobuf vs JSON Performance

```mermaid
graph LR
    subgraph JSON_Example[JSON - User object]
        J[open brace\n  name: John Smith\n  age: 30\n  email: john@example.com\n  active: true\nclosed brace\n107 bytes]
        style J fill:#fee2e2,stroke:#dc2626
    end

    subgraph PB_Example[Protobuf - Same User]
        P[Binary encoded\n~40 bytes\n2.7x smaller\n5-10x faster to serialize]
        style P fill:#dcfce7,stroke:#16a34a
    end
```

## Avro with Schema Registry

```mermaid
sequenceDiagram
    participant Producer
    participant Registry as Schema Registry
    participant Kafka
    participant Consumer

    Producer->>Registry: Register schema v1
    Registry-->>Producer: schema_id: 42

    Producer->>Kafka: magic byte + schema_id(42) + avro_payload
    Kafka-->>Consumer: raw bytes

    Consumer->>Registry: GET schema by id=42
    Registry-->>Consumer: schema definition

    Consumer->>Consumer: Deserialize using schema
```

## Key Concepts

- **JSON (JavaScript Object Notation)**: Human-readable, schema-optional, universally supported. Self-describing — field names are part of every message. This verbosity is its primary cost: name repetition across millions of messages wastes bandwidth and storage. No native binary type (base64 required). No schema enforcement by default (though JSON Schema provides optional validation).

- **Protocol Buffers (Protobuf)**: Google's binary serialization format used in gRPC. Fields are identified by integer field numbers (not names), making encoded messages extremely compact. Strongly typed with a mandatory `.proto` schema. Excellent forward/backward compatibility: unknown fields are preserved, new optional fields default gracefully.

- **Apache Avro**: Binary format used heavily in the Hadoop/Kafka ecosystem. The schema is included with the data or stored in a schema registry. Writer and reader schemas can differ — Avro resolves them using field names. Natural fit for event-driven systems with Confluent Schema Registry.

- **MessagePack**: Binary equivalent of JSON — uses the same data model (strings, numbers, arrays, maps) but encodes to binary. Significantly smaller and faster than JSON with no schema required. Good drop-in replacement for JSON in performance-sensitive contexts.

- **Apache Parquet**: Columnar binary format optimised for analytics workloads. Stores each column's data contiguously, enabling efficient columnar compression and predicate pushdown. Not suitable for row-by-row operations. Standard format for data lakes on S3.

- **Schema Registry**: A service (e.g., Confluent Schema Registry) that stores, versions, and enforces compatibility of schemas. Producers register schemas before publishing; consumers fetch schemas by ID. Enables schema evolution without breaking existing consumers.

- **Schema Evolution Rules**: For safe evolution: add new fields as optional (with defaults), never remove or rename fields (deprecate instead), never change a field's type, never reuse field numbers (Protobuf). Avro rules differ — use field names for resolution, so renaming requires aliases.

## Trade-offs

| Format | Size | Speed | Human Readable | Schema Required | Evolution Safety |
|--------|------|-------|---------------|-----------------|-----------------|
| JSON | Large | Slow | Yes | No | Low |
| XML | Very Large | Very Slow | Yes | Optional (XSD) | Low |
| Protobuf | Small | Very Fast | No | Yes (.proto) | High |
| Avro | Small | Fast | No | Yes (JSON schema) | High |
| MessagePack | Medium | Fast | No | No | Low |
| Parquet | Small (columnar) | Very Fast (analytics) | No | Yes | Moderate |

## When to Use

- **JSON**: Public APIs, configuration files, debugging contexts where human readability matters
- **Protobuf**: gRPC services, internal high-throughput APIs, mobile data transfer where bandwidth matters
- **Avro**: Kafka event streaming with Confluent Schema Registry, Hadoop/Spark data pipelines
- **Parquet**: Data lake storage, analytics queries on S3/GCS, Spark/Athena workloads
- **MessagePack**: Drop-in replacement for JSON when binary efficiency is needed without schema overhead
