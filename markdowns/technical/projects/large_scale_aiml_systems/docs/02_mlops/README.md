# MLOps

MLOps (Machine Learning Operations) applies DevOps principles to the ML lifecycle — automating training, evaluation, deployment, and monitoring of models in production. Unlike traditional software, MLOps must manage both code and data artifacts, and must detect degradation from data distribution shifts rather than just software bugs.

## Overview

```mermaid
mindmap
  root((MLOps))
    Model Registry
      Version control for models
      Stage transitions
      Metadata and lineage
      Artifact storage
      MLflow Registry
    Experiment Tracking
      Hyperparameter logging
      Metric tracking
      Artifact versioning
      Comparison and search
      MLflow Weights and Biases
    Drift Detection
      Data drift - input shift
      Concept drift - label shift
      Prediction drift
      Statistical tests
      Evidently Whylogs
    Retraining Pipelines
      Trigger strategies
      Continuous training
      Champion-challenger
      Evaluation gates
    AB Testing for ML
      Traffic splitting
      Statistical significance
      Business metrics
      Online evaluation
```

## MLOps Maturity Model

```mermaid
graph TD
    subgraph Level0[Level 0 - Manual]
        L0[Manual scripts\nNo pipeline\nModel in notebooks\nDeploy by hand]
        style L0 fill:#fee2e2,stroke:#dc2626
    end

    subgraph Level1[Level 1 - ML Pipelines]
        L1[Automated training pipeline\nExperiment tracking\nModel registry\nManual deployment trigger]
        style L1 fill:#fef3c7,stroke:#d97706
    end

    subgraph Level2[Level 2 - CI/CD for ML]
        L2[Automated retraining on trigger\nAutomated evaluation gate\nAutomated deployment\nProduction monitoring\nDrift detection]
        style L2 fill:#dcfce7,stroke:#16a34a
    end

    Level0 -->|add pipeline + tracking| Level1
    Level1 -->|add automation + monitoring| Level2
```

## Topics in This Section

| File | Topic | Key Concepts |
|------|-------|--------------|
| [01_model_registry.md](01_model_registry.md) | Model Registry | Versioning, stage transitions, lineage |
| [02_experiment_tracking.md](02_experiment_tracking.md) | Experiment Tracking | Metrics, parameters, artifacts |
| [03_drift_detection.md](03_drift_detection.md) | Drift Detection | Data drift, concept drift, PSI |
| [04_retraining_pipelines.md](04_retraining_pipelines.md) | Retraining Pipelines | Triggers, CT, evaluation gates |
| [05_ab_testing.md](05_ab_testing.md) | A/B Testing for ML | Traffic splitting, significance |
