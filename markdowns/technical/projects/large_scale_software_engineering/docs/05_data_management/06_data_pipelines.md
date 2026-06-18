# Data Pipelines

Data pipelines move data from source systems to destinations — transforming, enriching, and validating it along the way. Pipelines range from nightly batch ETL jobs to real-time streaming architectures processing millions of events per second.

## Pipeline Architecture Spectrum

```mermaid
graph LR
    subgraph Batch[Batch Processing - ETL]
        BSource[Source Systems\nDatabases, Files]
        BExtract[Extract\nFull or incremental]
        BTransform[Transform\nSpark, dbt, SQL]
        BLoad[Load\nData Warehouse]
        BSource --> BExtract --> BTransform --> BLoad
        BFreq[Frequency: hourly to daily]
    end

    subgraph Micro[Micro-Batch - ELT]
        MSrc[Sources] --> MLoad[Load Raw\nto Data Lake]
        MLoad --> MTransform[Transform in-place\ndbt, Spark]
        MFreq[Frequency: minutes]
    end

    subgraph Stream[Stream Processing]
        SSrc[Event Sources\nKafka topics] --> SProcess[Stream Processor\nFlink, Spark Streaming]
        SProcess --> SSink[Sinks\nDBs, caches, dashboards]
        SFreq[Frequency: milliseconds]
    end
```

## Change Data Capture (CDC)

```mermaid
graph TD
    subgraph SourceDB[Source Database - Postgres]
        Tables[Application Tables]
        WAL[Write-Ahead Log\nAll changes captured]
        Tables -->|all inserts updates deletes| WAL
    end

    CDC[CDC Tool\nDebezium / AWS DMS\nReads WAL stream]
    WAL --> CDC

    subgraph Destinations[Downstream Consumers]
        Kafka[Kafka Topic\neach change = event]
        ES[Elasticsearch\nsearch index sync]
        Cache[Redis Cache\ninvalidate on change]
        Warehouse[Data Warehouse\nnear-realtime sync]
    end

    CDC --> Kafka --> ES & Cache & Warehouse

    style WAL fill:#fef3c7,stroke:#d97706,stroke-width:2px
    style CDC fill:#dcfce7,stroke:#16a34a,stroke-width:2px
```

## Data Lakehouse Architecture

```mermaid
graph TD
    subgraph Sources[Data Sources]
        OLTP[OLTP Databases]
        Logs[Application Logs]
        Events[Event Streams]
        External[External APIs]
    end

    subgraph Landing[Landing Zone - Raw]
        S3Raw[S3 Raw Layer\nuntouched source data\nretained indefinitely]
    end

    subgraph Lakehouse[Data Lakehouse - Delta Lake / Iceberg]
        Bronze[Bronze Layer\nRaw + schema\nminimal cleaning]
        Silver[Silver Layer\nCleaned, deduplicated\njoined, validated]
        Gold[Gold Layer\nBusiness aggregations\nML features, reports]
        Bronze --> Silver --> Gold
    end

    subgraph Consumers[Consumers]
        BI[BI Tools\nTableau, Looker]
        DS[Data Scientists\nJupyter, Spark]
        ML[ML Training\nFeature Store]
        API[Serving APIs\nquery Gold layer]
    end

    Sources --> S3Raw --> Bronze
    Gold --> BI & DS & ML & API

    style Bronze fill:#cd7f32,stroke:#7c4000,color:#fff
    style Silver fill:#c0c0c0,stroke:#606060
    style Gold fill:#ffd700,stroke:#b8860b
```

## Stream Processing Architecture

```mermaid
graph LR
    Kafka[Kafka Topics\norder-events\npayment-events] --> Flink[Apache Flink\nStream Processor]

    subgraph FlinkOperators[Flink Processing]
        Filter[Filter: status=completed]
        Enrich[Enrich: join with user data]
        Window[Tumbling Window: 1min\naggregate revenue per category]
        Dedupe[Deduplication: idempotency key]
        Filter --> Enrich --> Window --> Dedupe
    end

    Flink --> Sinks[Output Sinks]
    Sinks --> DDB[(DynamoDB\nreal-time counters)]
    Sinks --> Redshift[(Redshift\nanalytics)]
    Sinks --> Kafka2[Kafka Topic\nderived events]
```

## Key Concepts

- **ETL (Extract, Transform, Load)**: Traditional batch pipeline — data is extracted from sources, transformed in a processing engine, and loaded into the target. The transformation is done before loading, meaning the data in the warehouse is always clean. Tight coupling between source and destination schemas.

- **ELT (Extract, Load, Transform)**: Data is loaded raw into the data lake/warehouse first, then transformed in-place using SQL or Spark. Enables iterative refinement of transformations without re-extracting. The modern approach with data lakehouses (dbt + BigQuery, dbt + Snowflake).

- **Change Data Capture (CDC)**: Captures row-level changes (inserts, updates, deletes) from a database's transaction log (WAL in Postgres, binlog in MySQL) as a stream of events. Enables real-time data synchronisation between systems without polling. Debezium is the leading open-source CDC tool.

- **Data Lakehouse**: Combines the low-cost storage of data lakes (S3/GCS) with the transactional features of data warehouses (ACID, schema enforcement, time travel) using open table formats (Delta Lake, Apache Iceberg, Apache Hudi). Eliminates the traditional lake/warehouse separation.

- **Bronze/Silver/Gold Layers (Medallion Architecture)**: A data organisation pattern where Bronze = raw data as-landed, Silver = cleaned and enriched, Gold = business-level aggregations ready for reporting. Each layer builds on the previous, with increasing data quality and specificity.

- **Stream Processing**: Continuous processing of unbounded data streams, typically with millisecond latency. Apache Flink provides stateful stream processing with exactly-once semantics, windowing (tumbling, sliding, session windows), and event-time processing with watermarks.

- **Watermarks**: In stream processing, a watermark is a signal to the processor indicating that all events with a timestamp up to a certain point have been observed. Watermarks allow the processor to correctly handle out-of-order events while limiting how long it waits for late data.

- **Idempotency in Pipelines**: Processing the same event multiple times produces the same result. Essential because at-least-once delivery (Kafka, SQS) can deliver duplicates. Achieved via idempotency keys, upsert operations, or exactly-once processing semantics.

## Trade-offs

| Approach | Latency | Complexity | Cost | Best For |
|---------|---------|-----------|------|---------|
| Batch ETL | Hours | Low | Low | Reporting, nightly refresh |
| Micro-batch | Minutes | Medium | Medium | Near-real-time dashboards |
| CDC | Seconds | Medium | Medium | Database sync, cache invalidation |
| Stream (Flink) | Milliseconds | High | Higher | Real-time features, fraud detection |

## When to Use

- **Batch ETL**: Historical reporting, compliance exports, when data freshness of hours is acceptable
- **ELT with dbt**: Modern analytics warehouse workloads with SQL-native transformations
- **CDC**: Whenever you need to keep downstream systems in sync with database changes without polling
- **Stream Processing**: Real-time fraud detection, live dashboards, real-time personalization, event-driven ML scoring
