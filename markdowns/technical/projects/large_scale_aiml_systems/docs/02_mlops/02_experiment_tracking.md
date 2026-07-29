---
title: "Experiment Tracking"
subtitle: "Experiment tracking is the practice of logging, organizing, and comparing ML training runs — capturing hyperparameters, metrics, and artifacts to make experimentation reproducible and comparable. Without systematic..."
category: technical
project: large_scale_aiml_systems
project_title: "Large Scale AI/ML Systems"
date: 2025-06-11
reading_time: 3
tags:
  - large-scale-aiml-systems
  - docs
author: "Shreedhar Kodate"
output: "blogs/technical/posts/large_scale_aiml_systems/docs/02_mlops/02_experiment_tracking.html"
---
Experiment tracking is the practice of logging, organizing, and comparing ML training runs — capturing hyperparameters, metrics, and artifacts to make experimentation reproducible and comparable. Without systematic tracking, ML development degenerates into "notebook chaos" where promising experiments are lost, results cannot be reproduced, and teams repeat work.

## Experiment Tracking Architecture

```mermaid
graph TD
    subgraph TrainingCode[Training Code]
        Init[mlflow.start_run\nor wandb.init\nbegin experiment session]
        LogParams[Log Parameters\nlearning_rate 0.001\nbatch_size 32\nmodel_type transformer]
        LogMetrics[Log Metrics per Step\ntrain_loss epoch 1 0.42\nval_loss epoch 1 0.38\nAUC epoch 1 0.84]
        LogArtifacts[Log Artifacts\nconfusion matrix plot\nfeature importance chart\nmodel checkpoint]
        End[End Run\nmark completed or failed]

        Init --> LogParams --> LogMetrics --> LogArtifacts --> End
    end

    subgraph TrackingServer[Tracking Server]
        RunDB[Run Database\nPostgres or SQLite\nparams metrics tags]
        ArtifactStore[Artifact Store\nS3 or GCS\nplots models datasets]
        UI[Experiment UI\ncompare runs\nvisualize metrics\nsearch and filter]

        RunDB <--> UI
        ArtifactStore <--> UI
    end

    TrainingCode --> TrackingServer

    style TrackingServer fill:#dbeafe,stroke:#2563eb,stroke-width:2px
```

## Experiment Comparison Workflow

```mermaid
graph TD
    subgraph HPOSweep[Hyperparameter Sweep]
        Baseline[Baseline Run\nlr=0.01 batch=32\nAUC=0.831]
        Run2[Run 2\nlr=0.001 batch=64\nAUC=0.847]
        Run3[Run 3\nlr=0.0001 batch=128\nAUC=0.839]
        Run4[Run 4\nlr=0.001 batch=32\nAUC=0.851]
        RunN[Run N\n...]
    end

    subgraph Analysis[Experiment Analysis]
        Compare[Compare Runs in UI\nplot val_AUC vs lr\nfind best config]
        Best[Best Run Identified\nRun 4: lr=0.001 batch=32\nAUC=0.851]
        Tag[Tag as Candidate\nbest_of_sweep=true\npromote to registry]
    end

    Baseline & Run2 & Run3 & Run4 & RunN --> Compare --> Best --> Tag

    style Best fill:#dcfce7,stroke:#16a34a,stroke-width:2px
```

## Run Metadata Structure

```mermaid
graph TD
    subgraph RunStructure[Experiment Run Components]
        Run[Experiment Run\nrun_id: unique UUID\nexperiment_id: group of runs\nstatus: RUNNING FINISHED FAILED]

        Params[Parameters\nFixed at start\nlearning_rate: 0.001\nbatch_size: 32\nmodel_arch: resnet50\nNOT updated during training]

        Metrics[Metrics\nUpdated during training\ntrain_loss: logged per step\nval_accuracy: logged per epoch\nsupport min max last mean]

        Tags[Tags\nfree-form key-value\nteam: fraud-ml\nowner: alice\ndata_version: v47]

        Artifacts[Artifacts\nfiles and directories\nmodel weights\nplots and visualizations\ntraining dataset sample]

        Run --> Params & Metrics & Tags & Artifacts
    end
```

## Key Concepts

- **Run**: The atomic unit of experiment tracking — a single training execution with its own parameters, metrics, artifacts, and status. Runs are grouped into experiments (a named collection of related runs). Every run should be tagged with the git commit hash of the training code to ensure reproducibility.

- **Parameters vs Metrics**: Parameters are inputs set before training (hyperparameters, model architecture, dataset version) — logged once at run start and immutable. Metrics are outputs produced during training (loss, accuracy, AUC) — logged at each step or epoch and tracked as time series. This distinction enables metric-based search and comparison across runs.

- **Artifact Logging**: Files produced by the run — model checkpoints, evaluation plots, feature importance charts, sample predictions. Artifacts are stored in blob storage (S3/GCS) and linked to the run. Well-logged artifacts make experiments self-documenting — someone reviewing a run six months later can see exactly what the model learned.

- **Experiment**: A named container grouping related runs — e.g., "fraud-model-v3-hyperparameter-search" or "recommendation-architecture-comparison". Good experiment naming conventions make the tracking system navigable as the number of runs grows to thousands.

- **MLflow**: Open-source experiment tracking with tracking server (SQLite or Postgres backend), artifact store (S3/GCS/HDFS), and web UI. Widely adopted, self-hostable, integrates with model registry. MLflow autolog automatically captures parameters and metrics from supported libraries (scikit-learn, PyTorch, TensorFlow).

- **Weights and Biases (W&B)**: Managed SaaS experiment tracking with richer visualization, team collaboration features, and built-in hyperparameter sweeps (W&B Sweeps). Better UX than MLflow but requires sending data to a third-party service. Preferred by research teams and startups where managed infrastructure is acceptable.

- **Reproducibility**: A run is reproducible if given the same code (git commit), data (dataset version), and parameters (logged hyperparameters), the same model can be retrained. Achieving true reproducibility also requires fixing random seeds and CUDA determinism flags, but these interact with performance — deterministic mode reduces GPU throughput.

## Trade-offs

| Tool | Self-Hosted | Visualization | Team Features | Cost |
|------|------------|--------------|--------------|------|
| MLflow (self-hosted) | Yes | Basic | Limited | Free |
| MLflow (Managed) | No | Basic | Medium | Paid |
| Weights and Biases | No | Excellent | Excellent | Free tier + paid |
| Neptune.ai | No | Good | Good | Paid |
| Custom scripts + S3 | Yes | None | None | Very Low |

## When to Use

- **MLflow**: Teams running their own infrastructure who need a free, self-hostable solution that integrates tightly with the MLflow ecosystem (experiment + registry in one system)
- **W&B**: Research teams and startups where rich visualization and collaboration features are valued over data residency requirements
- **Autolog**: Enable MLflow or W&B autolog for standard frameworks — it captures 80% of what you need with zero additional code
- **Custom tags**: Always tag runs with `git_commit`, `data_version`, and `owner` at minimum — enables filtering and reproducing results months later without documentation