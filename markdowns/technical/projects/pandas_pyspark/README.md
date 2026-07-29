---
title: "Pandas & PySpark — Concepts & Examples"
subtitle: "A self-contained learning project that illustrates core Pandas and PySpark concepts through minimal, working code examples backed by synthetic data. Every constant and hyperparameter is controlled from `config.yaml`."
category: technical
project: pandas_pyspark
project_title: "Pandas & PySpark — Concepts & Examples"
date: 2025-12-22
reading_time: 3
tags:
  - pandas-pyspark
author: "Shreedhar Kodate"
output: "blogs/technical/posts/pandas_pyspark/index.html"
---
A self-contained learning project that illustrates core Pandas and PySpark concepts through minimal, working code examples backed by synthetic data. Every constant and hyperparameter is controlled from `config.yaml`.

## Concepts Covered

### Pandas Core (`src/pandas_core/`)
- **`dataframe_operations.py`** — `.loc`, `.iloc`, `.at`, `.iat` indexing; inner/left/right/outer merges; melt, pivot, stack, unstack, crosstab; MultiIndex creation and `.xs`/`.swaplevel`; copy-vs-view semantics
- **`groupby_patterns.py`** — `agg()`, named aggregations (`NamedAgg`), `transform()` (group-mean normalisation, cumulative sum, rank), `filter()`, `apply()`, time-based `Grouper`
- **`time_series_ops.py`** — Rolling windows (SMA, Bollinger bands), EWM (MACD, signal), expanding windows, shift/lag/lead, `resample()` for OHLCV aggregation

### Pandas Optimization (`src/pandas_optimization/`)
- **`performance_patterns.py`** — Vectorised ops vs `iterrows` vs `itertuples` (timed benchmark); `query()` and `eval()` with numexpr; `pipe()` for composable pipelines; Polars syntax comparison
- **`dtype_optimizer.py`** — Category dtype for low-cardinality strings (memory reporting before/after); `pd.to_numeric(downcast=)` for int/float columns
- **`chunked_processing.py`** — Streaming large CSV with `chunksize`; partial aggregation accumulation across chunks

### PySpark Core (`src/pyspark_core/`)
- **`spark_session.py`** — SparkSession factory from config; AQE, Arrow, broadcast threshold settings
- **`dataframe_ops.py`** — StructType schema definition; `withColumn`, `filter`, `select`, `when/otherwise`; groupBy aggregations; broadcast join vs regular join; SparkSQL with `createOrReplaceTempView`
- **`window_functions.py`** — `rank`, `dense_rank`, `row_number`, `ntile`; `lag`, `lead`; `rowsBetween` rolling mean; cumulative sum within partition

### PySpark Optimization (`src/pyspark_optimization/`)
- **`optimization_patterns.py`** — AQE settings and benefits; `repartition` (shuffle) vs `coalesce` (no shuffle); cache/persist with different `StorageLevel`; broadcast variables for Python dict lookups; `explain(mode='formatted')` physical plans
- **`pandas_udf_examples.py`** — Regular Python UDF (row-by-row) vs Pandas UDF (Arrow-batched, vectorised); `mapInPandas` for distributed inference pattern; side-by-side timing comparison

### ML Pipelines (`src/ml_pipelines/`)
- **`feature_pipeline.py`** — StringIndexer → OneHotEncoder → VectorAssembler → StandardScaler as a Pipeline; fit-once/transform-many; save and load `PipelineModel`
- **`model_pipeline.py`** — GBTClassifier training with Pipeline API; RandomForest + CrossValidator + ParamGridBuilder; AUC-ROC evaluation; model save/load

### Data Quality (`src/data_quality/`)
- **`quality_checks.py`** — Great-Expectations-style checks (null threshold, range, uniqueness, schema, custom predicates); runs on clean and intentionally dirty data; CSV report output

### Notebook (`src/notebooks/`)
- **`pandas_pyspark_demo.ipynb`** — Side-by-side Pandas/PySpark for filter, groupBy, window, join; benchmark charts (iterrows vs vectorised); memory optimisation waterfall; Pandas/PySpark/Polars comparison table

### Documentation (`docs/`)
- **`concepts.md`** — Mermaid diagrams: cluster architecture, Catalyst optimizer pipeline, Pandas memory model; full comparison table
- **`flow_diagrams.md`** — Mermaid diagrams: job execution sequence, transformation vs action, stage/shuffle, ML pipeline, data quality, UDF data flow, AQE flow

## Project Structure

```
pandas_pyspark/
    src/
        config_loader.py            # Shared YAML config loader + logging setup
        pandas_core/
            dataframe_operations.py
            groupby_patterns.py
            time_series_ops.py
        pandas_optimization/
            performance_patterns.py
            dtype_optimizer.py
            chunked_processing.py
        pyspark_core/
            spark_session.py
            dataframe_ops.py
            window_functions.py
        pyspark_optimization/
            optimization_patterns.py
            pandas_udf_examples.py
        ml_pipelines/
            feature_pipeline.py
            model_pipeline.py
        data_quality/
            quality_checks.py
        notebooks/
            pandas_pyspark_demo.ipynb
    docs/
        concepts.md
        flow_diagrams.md
    docker/
        Dockerfile
        docker-compose.yml
        requirements.txt
    config.yaml
    pyproject.toml
    README.md
```

## Usage

### With Docker (recommended)

```bash
# Start JupyterLab
cd docker/
docker compose up pandas-pyspark-lab

# Open browser: http://localhost:8888

# Run all modules in one shot
docker compose --profile runner up pandas-pyspark-runner
```

### Without Docker

```bash
# Install dependencies (uv recommended)
pip install pandas pyspark numpy matplotlib pyarrow pyyaml jupyter polars

# Set PYTHONPATH to src/
export PYTHONPATH=src/

# Run any module directly
python src/pandas_core/dataframe_operations.py
python src/pandas_core/groupby_patterns.py
python src/pandas_core/time_series_ops.py
python src/pandas_optimization/performance_patterns.py
python src/pandas_optimization/dtype_optimizer.py
python src/pandas_optimization/chunked_processing.py
python src/data_quality/quality_checks.py
python src/pyspark_core/dataframe_ops.py
python src/pyspark_core/window_functions.py
python src/pyspark_optimization/optimization_patterns.py
python src/pyspark_optimization/pandas_udf_examples.py
python src/ml_pipelines/feature_pipeline.py
python src/ml_pipelines/model_pipeline.py

# Open notebook
jupyter lab src/notebooks/pandas_pyspark_demo.ipynb
```

## Configuration

All constants live in `config.yaml`. Key sections:

| Section | Key settings |
|---|---|
| `logging` | level, log file path, rotation size |
| `data` | random_seed, num_rows, output_dir |
| `spark` | app_name, master, AQE, Arrow, broadcast threshold |
| `pandas` | chunksize, category_threshold |
| `window` | rolling_window, ewm_span |
| `ml_pipeline` | train_ratio, cv_folds, max_depth_options |
| `data_quality` | null_threshold, age range, revenue bounds |

Logs are written to `logs/pandas_pyspark.log` (rotating, 100 MB per file, 5 backups).
Outputs (CSV, model artefacts, charts) go to `outputs/`.