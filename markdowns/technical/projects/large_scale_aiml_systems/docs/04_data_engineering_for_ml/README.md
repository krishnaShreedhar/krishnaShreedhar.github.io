# Data Engineering for ML

Data engineering for ML is the practice of designing, building, and maintaining the data infrastructure that supplies ML models with clean, consistent, and well-governed training data. The quality of training data is the single largest determinant of model quality — garbage in, garbage out — making robust data pipelines a foundational investment for any serious ML system.

## Overview

```mermaid
mindmap
  root((Data Engineering\nfor ML))
    Data Collection
      Web scraping
      APIs and feeds
      User telemetry
      Data labeling
      Synthetic data generation
    Data Preprocessing
      Cleaning and deduplication
      Missing value handling
      Normalization and scaling
      Encoding categoricals
      Train-val-test splitting
    Feature Engineering
      Domain-specific transforms
      Aggregation features
      Time-series features
      Embeddings
      Feature selection
    Data Validation
      Schema validation
      Statistical tests
      Great Expectations
      Anomaly detection
      Data contracts
```

## Data Quality Dimensions

```mermaid
graph TD
    subgraph Quality[Data Quality Dimensions]
        Complete[Completeness\nAll required fields present?\nMissing rate below threshold?]
        Accurate[Accuracy\nValues are correct?\nNo systematic errors in collection?]
        Consistent[Consistency\nSame entity has same values\nacross tables and time?]
        Fresh[Timeliness\nData is up to date?\nNo stale records in training set?]
        Valid[Validity\nValues within expected ranges?\nDates are dates? IDs are IDs?]
        Unique[Uniqueness\nNo duplicate records?\nDeduplication applied?]

        Complete & Accurate & Consistent & Fresh & Valid & Unique --> Quality2[High-Quality ML Training Data]
        style Quality2 fill:#dcfce7,stroke:#16a34a,stroke-width:2px
    end
```

## Topics in This Section

| File | Topic | Key Concepts |
|------|-------|--------------|
| [01_data_collection.md](01_data_collection.md) | Data Collection | Sources, labeling, synthetic data |
| [02_data_preprocessing.md](02_data_preprocessing.md) | Data Preprocessing | Cleaning, transforms, splitting |
| [03_feature_engineering.md](03_feature_engineering.md) | Feature Engineering | Domain features, aggregations |
| [04_data_validation.md](04_data_validation.md) | Data Validation | Schema, statistics, Great Expectations |
