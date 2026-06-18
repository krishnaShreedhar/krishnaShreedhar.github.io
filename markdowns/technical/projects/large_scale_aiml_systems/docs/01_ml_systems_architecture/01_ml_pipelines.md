# ML Pipelines

An ML pipeline is a directed acyclic graph (DAG) of transformations that takes raw data as input and produces a trained, evaluated, and deployed model as output. Well-designed ML pipelines are reproducible, versioned, testable, and observable — applying software engineering discipline to ML workflows.

## End-to-End ML Pipeline

```mermaid
graph TD
    subgraph DataIngestion[Data Ingestion]
        RawData[Raw Data Sources\nDatabases, APIs, Files, Streams]
        Validate[Data Validation\nGreat Expectations\nCheck schema and stats]
        RawData --> Validate
    end

    subgraph FeatureEngineering[Feature Engineering]
        FeatureComp[Feature Computation\nAggregations, transformations\nEmbeddings, encodings]
        FeatureStore[Feature Store\nStore computed features\nfor training and serving]
        Validate --> FeatureComp --> FeatureStore
    end

    subgraph Training[Model Training]
        DataSplit[Train/Val/Test Split\nTime-based or random]
        Train[Model Training\nGPU cluster\nDistributed if needed]
        Evaluate[Model Evaluation\nAccuracy, F1, AUC\nBusiness metrics]
        FeatureStore --> DataSplit --> Train --> Evaluate
    end

    subgraph Deployment[Deployment Decision]
        Compare[Compare vs Champion\nIs challenger better?]
        Register[Register Model\nMLflow model registry\nversion and metadata]
        Deploy[Deploy to Serving]
        Evaluate --> Compare --> Register --> Deploy
    end

    Monitor[Production Monitoring\nData drift, prediction drift\nBusiness metric monitoring]
    Deploy --> Monitor

    style FeatureStore fill:#fef3c7,stroke:#d97706,stroke-width:2px
    style Register fill:#dcfce7,stroke:#16a34a,stroke-width:2px
```

## ML Pipeline Orchestration Tools

```mermaid
graph TD
    subgraph Orchestrators[ML Pipeline Orchestrators]
        Kubeflow[Kubeflow Pipelines\nKubernetes-native\nPython SDK\nML-specific primitives]
        Airflow[Apache Airflow\nGeneral DAG orchestration\nLarge ecosystem\nML plugins available]
        Metaflow[Netflix Metaflow\nData scientist friendly\nAWS-integrated\nVersioned artifacts]
        Prefect[Prefect\nModern Python API\nObservability built-in\nHybrid execution]
        ZenML[ZenML\nML-specific\nPipeline portability\nStack abstraction]
    end

    subgraph Comparison[Tool Selection Criteria]
        UseKubeflow[Kubeflow: K8s-native\nmicroservices team\nfine-grained control]
        UseAirflow[Airflow: existing Airflow\ndata engineering team\nmixed workflows]
        UseMetaflow[Metaflow: data scientists\nAWS shop\nminimal infra overhead]
    end
```

## Training-Serving Skew Prevention

```mermaid
graph LR
    subgraph TrainingSkewProblem[Training-Serving Skew]
        TSkew[Training features: computed from historical DB\nServing features: computed from real-time API\nDifferent logic → model sees different distribution\nPredictions degrade silently]
        style TSkew fill:#fee2e2,stroke:#dc2626
    end

    subgraph FeatureStoreUnification[Feature Store Solution]
        FeatDef[Single Feature Definition\npython def compute_user_spend_7d:\n  ...same SQL logic...\nused for both training and serving]
        OfflineStore[Offline Store: historical\npoint-in-time correct]
        OnlineStore[Online Store: real-time\nsame logic, low latency]
        FeatDef --> OfflineStore & OnlineStore
        style FeatDef fill:#dcfce7,stroke:#16a34a,stroke-width:2px
    end
```

## Key Concepts

- **ML Pipeline as Code**: Define pipelines as code (Python functions, YAML configurations) with version control, code review, and CI/CD. Every pipeline run should be reproducible given the same inputs. Pipelines as code enable collaboration and prevent "notebook chaos" where important ML code lives in undiscoverable Jupyter notebooks.

- **DAG (Directed Acyclic Graph)**: The fundamental structure of ML pipelines. Each node is a computation step; edges represent data dependencies. DAG structure enables parallelism (independent branches run concurrently), caching (reuse completed steps on rerun), and failure recovery (resume from the failed step).

- **Artifact Versioning**: Every artifact produced by a pipeline (datasets, features, models, metrics) is versioned and associated with the pipeline run that produced it. This enables reproducibility (re-run an exact historical pipeline), debugging (trace a bad model to its training data), and rollback.

- **Training-Serving Skew**: The most common and insidious failure mode in ML systems. Training uses historical batch data to compute features; serving computes the same features from real-time data using different code paths. Subtle differences in feature computation logic cause the model to see a different distribution in production than in training. Feature stores with a single feature definition prevent this.

- **Pipeline Triggers**: ML pipelines can be triggered by: schedule (retrain every week), data events (new data volume threshold crossed), model performance degradation (accuracy drops below threshold), or manual approval. Automated triggers require careful guardrails to prevent runaway retraining.

- **Continuous Training (CT)**: Automatically retraining models on fresh data based on triggers. Different from CI/CD for software — CT must include evaluation gates (the new model must be at least as good as the current champion before deployment).

## Trade-offs

| Approach | Engineering Overhead | Flexibility | Reproducibility |
|----------|---------------------|------------|----------------|
| Notebooks only | Very Low | High | Very Low |
| Scripts in git | Low | High | Medium |
| Orchestrated pipelines | Medium | Medium | High |
| ML platform (Kubeflow) | High | Medium | Very High |

## When to Use

- **Scripted pipelines**: Small teams, early stage, single model — low overhead with reasonable reproducibility
- **Kubeflow/Airflow**: Production systems with multiple models, multiple teams, and reliability requirements
- **Feature store**: Any system where features are shared across models or where training-serving skew is a risk (almost always)
- **Continuous training**: When model accuracy is sensitive to data staleness (recommendation, fraud detection, NLP on evolving language)
