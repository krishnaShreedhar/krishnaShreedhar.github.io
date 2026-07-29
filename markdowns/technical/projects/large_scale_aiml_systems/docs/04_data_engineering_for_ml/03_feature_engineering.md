---
title: "Feature Engineering"
subtitle: "Feature engineering is the process of transforming raw data into informative representations — features — that capture the signals most predictive of the target variable. Good feature engineering encodes domain..."
category: technical
project: large_scale_aiml_systems
project_title: "Large Scale AI/ML Systems"
date: 2025-10-22
reading_time: 4
tags:
  - large-scale-aiml-systems
  - docs
author: "Shreedhar Kodate"
output: "blogs/technical/posts/large_scale_aiml_systems/docs/04_data_engineering_for_ml/03_feature_engineering.html"
---
Feature engineering is the process of transforming raw data into informative representations — features — that capture the signals most predictive of the target variable. Good feature engineering encodes domain knowledge about what makes examples similar or different, enabling models to learn faster and generalize better. In tabular ML, feature engineering often has a larger impact on model performance than model architecture selection.

## Feature Engineering Taxonomy

```mermaid
graph TD
    subgraph Taxonomy[Feature Engineering Categories]
        subgraph Aggregation[Aggregation Features - Window Functions]
            Agg1[User spend in last 7 days\nCount of orders in last 30 days\nAverage session duration in last 7 days\nStd of transaction amounts in last month]
            Agg2[Require time-window aggregation\nComputed at specific point in time\nMost powerful for behavioral ML]
        end

        subgraph Interaction[Interaction Features]
            Int1[Feature crosses\nprice_per_unit = price divided by quantity\nclick_through_rate = clicks divided by impressions\ndays_since_last_purchase\ndistance_to_nearest_store]
            Int2[Encode domain relationships\nbetween pairs or groups of features]
        end

        subgraph Temporal[Temporal and Cyclical Features]
            Temp1[Time since last event\nhour of day sin and cos\nday of week sin and cos\nmonth of year sin and cos\nis_weekend is_holiday]
            Temp2[Cyclical encoding with sin and cos\nprevents discontinuity at midnight\nor year boundaries]
        end

        subgraph Embedding[Embedding Features]
            Emb1[Entity embeddings\nfor categorical IDs\nuser_id mapped to 32-dim vector\nlearned from collaborative filtering\nor LLM embeddings of text fields]
        end
    end
```

## Time-Window Aggregation Design

```mermaid
graph TD
    subgraph WindowFeatures[Window Aggregation Feature Engineering]
        subgraph RawEvents[Raw Event Log]
            E1[2024-01-15 12:00 user_123 purchase amount=45.99]
            E2[2024-01-14 09:30 user_123 purchase amount=12.50]
            E3[2024-01-10 15:45 user_123 purchase amount=89.00]
            E4[2023-12-20 11:00 user_123 purchase amount=200.00]
        end

        subgraph WindowDefs[Feature Window Definitions]
            W7d[7-day window\nsum: 58.49\ncount: 2\navg: 29.24\nmax: 45.99]
            W30d[30-day window\nsum: 147.49\ncount: 3\navg: 49.16\nmax: 89.00]
            W90d[90-day window\nsum: 347.49\ncount: 4\navg: 86.87\nmax: 200.00]
        end

        AsOfDate[As-of: 2024-01-15 13:00\nPoint-in-time correct\nno future data included]

        RawEvents --> AsOfDate --> W7d & W30d & W90d
    end

    style AsOfDate fill:#fef3c7,stroke:#d97706,stroke-width:2px
```

## Feature Selection Methods

```mermaid
graph TD
    subgraph FeatureSelection[Feature Selection Approaches]
        subgraph Filter[Filter Methods]
            F1[Univariate Statistical Tests\ncorrelation with target\nchi-squared for categoricals\nspearman rank correlation\nfast - no model required]
        end

        subgraph Wrapper[Wrapper Methods]
            W1[Recursive Feature Elimination\nRFE with cross-validation\ntrain model iteratively\nremove least important features\nExpensive but accurate]
        end

        subgraph Embedded[Embedded Methods - During Training]
            E1[Tree Feature Importance\nXGBoost SHAP values\nL1 Lasso regularization\nAttention weights in transformers]
            style E1 fill:#dcfce7,stroke:#16a34a,stroke-width:2px
        end

        subgraph Ablation[Ablation Studies]
            A1[Remove feature\ntrain without it\nmeasure performance delta\nquantifies each feature's value]
        end
    end
```

## Key Concepts

- **Window Aggregations**: Computing statistics (sum, count, mean, std, min, max) over a rolling time window anchored to each observation's timestamp. These are among the most predictive features in behavioral ML tasks — "how much did this user spend in the last 7 days" is more predictive of fraud than "how much did they spend ever". Must be computed point-in-time correctly to avoid label leakage.

- **Feature Crosses**: Creating new features by combining two or more existing features. Examples: price-to-income ratio, click-through rate (clicks/impressions), days-since-last-purchase. Feature crosses encode domain knowledge about relationships between variables that the model cannot easily learn from the individual features alone, especially for linear models that cannot learn interactions without explicit feature crosses.

- **Cyclical Encoding**: Hour-of-day, day-of-week, and month-of-year are cyclical — hour 23 is similar to hour 0, but raw integer encoding puts them far apart. Encoding cyclical features as sin and cos projections (sin(2π * hour/24), cos(2π * hour/24)) preserves the cyclical relationship. Critical for time-series and behavioral features.

- **Target Encoding**: Replacing a categorical value with the mean of the target variable for that category (e.g., replace merchant_id with the average fraud rate for that merchant). Very powerful for high-cardinality categoricals, but must be computed with cross-validation on training folds to prevent target leakage (the category's own label would inflate its encoding).

- **Entity Embeddings**: Representing high-cardinality categorical entities (user IDs, product IDs, merchant IDs) as dense learned vectors rather than one-hot or target encodings. Embeddings can capture complex relationships — users who prefer similar products are nearby in embedding space. Can be pre-trained from collaborative filtering, LLM text descriptions, or learned jointly with the prediction model.

- **SHAP Values (SHapley Additive exPlanations)**: A game-theoretic method for attributing each feature's contribution to a model's prediction. SHAP values are model-agnostic and satisfy desirable fairness properties. They explain individual predictions ("the model predicted fraud because of high_amount = +0.3, new_device = +0.2, low_history = +0.1") and aggregate to feature importance rankings. Essential for model debugging and stakeholder communication.

- **Feature Drift**: The statistical properties of features in production changing from what they were during training (covariate shift). Monitor feature distributions (mean, std, PSI) in production relative to the training set. Feature drift is the leading cause of silent model performance degradation.

## Trade-offs

| Feature Type | Predictive Power | Computation Cost | Interpretability | Risk of Leakage |
|-------------|-----------------|-----------------|-----------------|----------------|
| Raw features | Low | Very Low | High | Low |
| Window aggregations | Very High | Medium | High | High (if not PIT) |
| Feature crosses | High | Low | Medium | Low |
| Entity embeddings | High | High (training) | Low | Low |
| Target encoding | High | Low | Medium | High (without cross-val) |

## When to Use

- **Window aggregations**: Any behavioral prediction task (fraud, churn, recommendation, credit risk) — they are almost always the highest-signal features
- **Cyclical encoding**: Any feature encoding time-of-day, day-of-week, or other periodic variables — always use sin/cos rather than raw integers
- **Target encoding**: High-cardinality categoricals (more than ~50 unique values) where one-hot creates too many dimensions
- **SHAP values**: During model development for feature selection, debugging, and stakeholder explanation; in production for monitoring individual prediction explanations and detecting model drift
- **Entity embeddings**: Recommendation systems, ad targeting, any system where entity IDs are primary keys with millions of unique values