# Feature Stores

A feature store is a centralized platform for storing, computing, and serving ML features. It solves two critical problems in ML systems: feature reuse (preventing different teams from recomputing the same features independently) and training-serving consistency (ensuring the same feature values are used in training and production serving).

## Feature Store Architecture

```mermaid
graph TD
    subgraph Sources[Data Sources]
        OLTP[OLTP Databases\ntransactional data]
        Warehouse[Data Warehouse\nhistorical aggregations]
        Streams[Event Streams\nKafka, Kinesis]
        External[External APIs\nenrichment data]
    end

    subgraph Computation[Feature Computation]
        BatchComp[Batch Computation\nSpark, dbt, SQL\nScheduled or triggered]
        StreamComp[Stream Computation\nFlink, Spark Streaming\nReal-time features]
    end

    subgraph Storage[Dual Storage]
        OfflineStore[Offline Store\nFeature values over time\nParquet on S3 or DeltaLake\nPoint-in-time correct joins]
        OnlineStore[Online Store\nLatest feature values\nRedis, DynamoDB, Cassandra\nLow latency serving]
    end

    subgraph Consumers[Feature Consumers]
        TrainingJob[ML Training Jobs\nbatch retrieve historical features\npoint-in-time correct]
        ServingInf[Online Inference\nreal-time feature lookup\nlow latency required]
        Analytics[Analytics\nfeature statistics\ndrift monitoring]
    end

    Sources --> BatchComp & StreamComp
    BatchComp --> OfflineStore & OnlineStore
    StreamComp --> OnlineStore
    OfflineStore --> TrainingJob & Analytics
    OnlineStore --> ServingInf

    style OfflineStore fill:#dbeafe,stroke:#2563eb
    style OnlineStore fill:#dcfce7,stroke:#16a34a
    style OfflineStore fill:#dbeafe,stroke:#2563eb,stroke-width:2px
```

## Point-in-Time Correct Joins

```mermaid
graph TD
    subgraph PTCJoin[Point-in-Time Correct Join]
        Problem[Problem: Leakage\nTraining query that joins user features\nas-of TODAY for all historical orders\nuses future information!\nModel sees data it wouldn't have at inference time]
        style Problem fill:#fee2e2,stroke:#dc2626

        Solution[Point-in-Time Join:\nFor each order at timestamp T\njoin user features as-of timestamp T\nnot the current value\nPrevents future data leakage]
        style Solution fill:#dcfce7,stroke:#16a34a,stroke-width:2px

        Problem --> Solution
    end

    subgraph Timeline[Example Timeline]
        T1[Order placed: 2024-01-15\nUser feature at 2024-01-15:\npurchase_count_30d = 5]
        T2[Current date: 2024-03-01\nUser feature today:\npurchase_count_30d = 47]
        
        BadJoin[Wrong join: uses 47\nModel trains on future data!]
        GoodJoin[Correct PIT join: uses 5\nModel trains on data available at order time]

        T1 --> BadJoin & GoodJoin
        T2 --> BadJoin
    end
```

## Feature Reuse Across Teams

```mermaid
graph TD
    subgraph WithoutFS[Without Feature Store - Duplication]
        Team1[Team 1 - Fraud\ncomputes: user_spend_7d\ncustom SQL pipeline]
        Team2[Team 2 - Recommendations\ncomputes: user_spend_7d\ndifferent SQL - subtle difference!]
        Team3[Team 3 - Marketing\ncomputes: user_spend_7d\nthird implementation]
        Problem2[3 teams, 3 implementations\nInconsistent values\nWasted compute]
        style Problem2 fill:#fee2e2,stroke:#dc2626
    end

    subgraph WithFS[With Feature Store - Shared Features]
        FS[Feature Store\nuser_spend_7d: canonical definition\ncomputed once, served to all]
        FTeam1[Fraud Team] --> FS
        FTeam2[Recommendations Team] --> FS
        FTeam3[Marketing Team] --> FS
        Benefit[Single implementation\nConsistent values\nCompute once]
        style Benefit fill:#dcfce7,stroke:#16a34a
    end
```

## Key Concepts

- **Offline Store**: Stores the historical time series of feature values. Used by training jobs to retrieve features as they existed at any point in time. Typically implemented on S3/GCS with Parquet files or Delta Lake format. Enables point-in-time correct joins to prevent label leakage.

- **Online Store**: Stores the latest (current) feature values for low-latency serving. Used by the inference serving layer to enrich incoming requests with precomputed features. Implemented with Redis, DynamoDB, or Bigtable for sub-10ms lookups. Only stores the current value, not the history.

- **Point-in-Time Correct Join**: When creating a training dataset, features must be joined to labels using the feature values that were available at the time of the label event — not the current values. Without this, the model trains on data it would not have had access to at prediction time (data leakage), producing overly optimistic evaluation metrics.

- **Feature Groups**: Features are organized into logical groups based on the entity they describe (user features, item features, session features) and the computation frequency (real-time, hourly, daily). Feature groups share a common source and computation logic.

- **Feature Backfilling**: Computing historical values for new features from historical raw data. Required when adding a new feature to the offline store. Can be computationally expensive for long history periods.

- **Feature Freshness**: The staleness of feature values in the online store. Some features (user lifetime value) can be stale for days; others (current session cart value) must be fresh within seconds. Feature freshness requirements drive online store update frequency.

- **Feast, Tecton, Hopsworks**: The leading feature store platforms. Feast is open-source; Tecton and Hopsworks offer managed cloud services. Most large companies build proprietary feature stores optimized for their specific data infrastructure.

## Trade-offs

| Approach | Training-Serving Consistency | Engineering Overhead | Feature Reuse |
|---------|------------------------------|---------------------|--------------|
| No feature store | Low (manual consistency) | Low | Very Low |
| Shared feature library | Medium | Medium | Medium |
| Full feature store | High | High | High |

## When to Use

- **Feature store**: When multiple models share features, when training-serving skew is causing production degradation, or when the ML team has 5+ data scientists with overlapping feature needs
- **Offline store only**: Start here — add the online store when online inference latency becomes a constraint
- **Skip for simple models**: If the model uses only request-time features (no historical context) and there's only one model, a full feature store may be premature
