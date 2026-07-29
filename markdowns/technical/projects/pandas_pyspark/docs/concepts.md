---
title: "Pandas & PySpark — Core Concepts"
subtitle: "graph TD subgraph Driver[\"Driver Process\"] SC[SparkContext / SparkSession] DAG[DAG Scheduler] TS[Task Scheduler] end"
category: technical
project: pandas_pyspark
project_title: "Pandas & PySpark — Concepts & Examples"
date: 2025-03-05
reading_time: 4
tags:
  - pandas-pyspark
  - docs
author: "Shreedhar Kodate"
output: "blogs/technical/posts/pandas_pyspark/docs/concepts.html"
---
## 1. PySpark Cluster Architecture

```mermaid
graph TD
    subgraph Driver["Driver Process"]
        SC[SparkContext / SparkSession]
        DAG[DAG Scheduler]
        TS[Task Scheduler]
    end

    subgraph ClusterManager["Cluster Manager (YARN / K8s / Standalone)"]
        CM[Resource Negotiator]
    end

    subgraph Executor1["Executor 1 (Worker Node)"]
        T1A[Task]
        T1B[Task]
        CACHE1[Block Manager / Cache]
    end

    subgraph Executor2["Executor 2 (Worker Node)"]
        T2A[Task]
        T2B[Task]
        CACHE2[Block Manager / Cache]
    end

    subgraph Executor3["Executor 3 (Worker Node)"]
        T3A[Task]
        T3B[Task]
        CACHE3[Block Manager / Cache]
    end

    SC -->|submits job| DAG
    DAG -->|builds stages| TS
    TS -->|requests resources| CM
    CM -->|allocates containers| Executor1
    CM -->|allocates containers| Executor2
    CM -->|allocates containers| Executor3
    TS -->|sends tasks| T1A
    TS -->|sends tasks| T1B
    TS -->|sends tasks| T2A
    TS -->|sends tasks| T2B
    TS -->|sends tasks| T3A
    TS -->|sends tasks| T3B
    T1A -->|shuffle write| CACHE2
    T2A -->|shuffle write| CACHE3
    T1A & T1B & T2A & T2B & T3A & T3B -->|results| SC

    style Driver fill:#2d3748,color:#fff
    style ClusterManager fill:#2b6cb0,color:#fff
    style Executor1 fill:#276749,color:#fff
    style Executor2 fill:#276749,color:#fff
    style Executor3 fill:#276749,color:#fff
```

**Key roles:**
- **Driver**: the process running your Python script. Owns the SparkContext, constructs the DAG, and coordinates execution.
- **DAG Scheduler**: breaks the logical plan into *stages* separated by shuffle boundaries.
- **Task Scheduler**: allocates individual *tasks* (one per partition) to executor slots.
- **Cluster Manager**: negotiates physical resources (YARN / Kubernetes / Spark Standalone).
- **Executor**: JVM process that runs tasks and stores cached RDD/DataFrame partitions in its Block Manager.

---

## 2. Catalyst Optimizer Pipeline

```mermaid
flowchart LR
    subgraph User["User Code"]
        DF["df.filter(...).groupBy(...).agg(...)"]
    end

    subgraph Catalyst["Catalyst Optimizer"]
        ULP["Unresolved\nLogical Plan"]
        ALP["Analyzed\nLogical Plan\n(catalog lookup)"]
        OLP["Optimized\nLogical Plan\n(predicate pushdown,\ncolumn pruning,\nconstant folding)"]
        PP["Physical Plans\n(multiple candidates)"]
        BPP["Best Physical Plan\n(cost model selection)"]
    end

    subgraph Tungsten["Tungsten Engine"]
        CG["Whole-stage\nCode Generation"]
        EX["Execution\n(RDD DAG)"]
    end

    DF --> ULP --> ALP --> OLP --> PP --> BPP --> CG --> EX

    style User fill:#2d3748,color:#fff
    style Catalyst fill:#2b6cb0,color:#fff
    style Tungsten fill:#276749,color:#fff
```

**Optimisation rules applied:**
| Rule | Example |
|---|---|
| Predicate Pushdown | Move `.filter()` as close to data source as possible |
| Column Pruning | Only read columns referenced downstream |
| Constant Folding | Evaluate `1 + 1` at plan time |
| Join Reordering | Place smaller table on the build side |
| Broadcast Join | Auto-broadcast tables smaller than `autoBroadcastJoinThreshold` |

---

## 3. Pandas Memory Model

```mermaid
graph TD
    subgraph Process["Python Process (single machine)"]
        DF_OBJ["DataFrame Object\n(Python heap)"]

        subgraph Blocks["Block Manager (internal)"]
            direction LR
            B1["float64 block\n[revenue, discount_pct]\n(contiguous NumPy array)"]
            B2["int64 block\n[units, salesperson_id]\n(contiguous NumPy array)"]
            B3["object block\n[region, product]\n(Python object pointers)"]
            B4["datetime64 block\n[timestamp]"]
        end

        IDX["Index\n(RangeIndex / DatetimeIndex)"]
    end

    DF_OBJ --> B1 & B2 & B3 & B4 & IDX

    style Process fill:#2d3748,color:#fff
    style Blocks fill:#2b6cb0,color:#fff
```

**Memory optimisation strategies:**
1. **Category dtype**: replaces repeated string objects with integer codes + a small lookup table. Saves up to 90% for low-cardinality columns.
2. **Numeric downcast**: `int64` (8 bytes) -> `int8` (1 byte) when range allows. `float64` -> `float32`.
3. **Chunked reading**: process large files in `chunksize` slices to avoid loading everything into RAM.
4. **Copy-on-write (pandas ≥ 2.0)**: slices share memory until mutated, reducing unnecessary copies.

---

## 4. Pandas vs PySpark vs Polars — Comparison Table

| Concept | Pandas | PySpark | Polars |
|---|---|---|---|
| **Execution model** | Eager (immediate) | Lazy (DAG, action triggers) | Eager default / `.lazy()` available |
| **Scale** | Single machine | Distributed cluster | Single machine (multi-threaded) |
| **Memory** | NumPy blocks (mutable) | Immutable partitioned RDDs | Apache Arrow columnar (immutable) |
| **Backend language** | Python / NumPy (C) | Python API → JVM (Scala/Java) | Python API → Rust |
| **GroupBy** | `df.groupby().agg()` | `df.groupBy().agg()` | `df.group_by().agg()` |
| **Window functions** | `df.rolling()` / `expanding()` | `Window.partitionBy()` | `pl.col(...).rolling_mean()` |
| **Join** | `df.merge(other)` | `df.join(other)` | `df.join(other)` |
| **UDFs** | Native Python | Python UDF (slow) / Pandas UDF (fast) | Native Rust expressions |
| **SQL** | `pandasql` / DuckDB | `spark.sql(...)` | `pl.SQLContext` |
| **Schema** | Inferred, loose | `StructType`, strict | Inferred / declared, strict |
| **Streaming** | No | Structured Streaming | No |
| **Serialisation** | Pickle / CSV / Parquet | Kryo / Parquet / ORC | IPC / Parquet / CSV |
| **Best for** | < 1 GB, exploration | > 1 GB, distributed ETL / ML | < 50 GB, fast local transforms |
| **Installation** | `pip install pandas` | `pip install pyspark` | `pip install polars` |

---

## 5. Key API Cross-Reference

```mermaid
mindmap
  root((Data Operations))
    Filtering
      Pandas: df[mask]
      Pandas: df.query()
      PySpark: df.filter(F.col() > x)
      Polars: df.filter(pl.col() > x)
    GroupBy
      Pandas: groupby().agg()
      Pandas: groupby().transform()
      PySpark: groupBy().agg()
      Polars: group_by().agg()
    Windows
      Pandas: rolling / ewm / expanding
      PySpark: Window.partitionBy().orderBy()
      Polars: rolling_mean / over()
    Joins
      Pandas: merge(how=inner/left/right/outer)
      PySpark: join(how=inner/left/right/outer)
      PySpark: broadcast(df)
      Polars: join(how=...)
    Reshaping
      Pandas: melt / pivot / stack / unstack
      PySpark: stack / unpivot
      Polars: melt / pivot
    IO
      CSV: read_csv / spark.read.csv / pl.read_csv
      Parquet: read_parquet / spark.read.parquet / pl.read_parquet
      JSON: read_json / spark.read.json / pl.read_json
```

---

## 6. When to Use Which Framework

```mermaid
flowchart TD
    START(["Data processing task"]) --> Q1{"Data size?"}
    Q1 -->|"< 1 GB"| Q2{"Need speed\nover familiarity?"}
    Q1 -->|"1 GB – 100 GB"| Q3{"On a cluster?"}
    Q1 -->|"> 100 GB"| PYSPARK(["PySpark\n(distributed)"])

    Q2 -->|Yes| POLARS(["Polars\n(Rust, fast, local)"])
    Q2 -->|No| PANDAS(["Pandas\n(mature ecosystem)"])

    Q3 -->|Yes| PYSPARK
    Q3 -->|No| Q4{"CPU-bound\ntransformations?"}
    Q4 -->|Yes| POLARS
    Q4 -->|No| PANDAS

    style PANDAS fill:#4C72B0,color:#fff
    style PYSPARK fill:#DD8452,color:#fff
    style POLARS fill:#55A868,color:#fff
```