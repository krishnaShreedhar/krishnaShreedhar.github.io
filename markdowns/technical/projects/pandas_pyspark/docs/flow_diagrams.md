# Flow Diagrams — Pandas & PySpark

## 1. Spark Job Execution Flow

From user code to task execution — showing how transformations and actions produce stages and tasks.

```mermaid
sequenceDiagram
    participant U  as User Code
    participant D  as Driver (DAG Scheduler)
    participant TS as Task Scheduler
    participant E  as Executor(s)
    participant S  as Storage / Shuffle

    Note over U,D: Transformations are lazy — no work yet
    U->>D: df.filter(...)        [Transformation]
    U->>D: .groupBy(...)         [Transformation]
    U->>D: .agg(...)             [Transformation]
    U->>D: .count()              [ACTION — triggers execution]

    D->>D: Build DAG of RDD stages
    D->>D: Identify shuffle boundaries → Stage 1, Stage 2

    Note over D,TS: Stage 1: scan + filter (no shuffle)
    D->>TS: Submit Stage 1 tasks (one per input partition)
    TS->>E: Launch Task 1.1 (partition 0)
    TS->>E: Launch Task 1.2 (partition 1)
    TS->>E: Launch Task 1.N (partition N)
    E->>S: Write shuffle data (map output)

    Note over D,TS: Stage 2: shuffle read + groupBy aggregate
    D->>TS: Submit Stage 2 tasks
    S->>E: Shuffle fetch (reduce input)
    E->>E: Local aggregation
    E->>D: Return partial results

    D->>U: Return final result (count = integer)
```

---

## 2. Transformation vs Action Decision Tree

```mermaid
flowchart TD
    OP["DataFrame / RDD operation"] --> Q{"Does it\nreturn a new\nDF / RDD?"}
    Q -->|Yes| TRANS["TRANSFORMATION\n(lazy, added to DAG)\n\nExamples:\n.filter() .select()\n.withColumn() .groupBy()\n.join() .repartition()"]
    Q -->|No| ACTION["ACTION\n(triggers execution)\n\nExamples:\n.count() .show()\n.collect() .write()\n.take() .first()"]
    TRANS --> DAG["DAG grows\n(no computation yet)"]
    ACTION --> EXEC["Catalyst optimises DAG\nStages split at shuffles\nTasks dispatched to executors\nResult returned / written"]

    style TRANS fill:#2b6cb0,color:#fff
    style ACTION fill:#c53030,color:#fff
    style EXEC fill:#276749,color:#fff
```

---

## 3. Spark Stage and Shuffle Flow

```mermaid
graph LR
    subgraph Stage1["Stage 1 — Scan + Filter (no shuffle)"]
        P1["Partition 0\n(scan + filter)"]
        P2["Partition 1\n(scan + filter)"]
        P3["Partition N\n(scan + filter)"]
    end

    subgraph Shuffle["Shuffle (network transfer)"]
        SB["Shuffle Buffers\n(disk / memory)"]
    end

    subgraph Stage2["Stage 2 — GroupBy Aggregate"]
        R1["Reducer 0\n(group key hash=0)"]
        R2["Reducer 1\n(group key hash=1)"]
        R3["Reducer M\n(group key hash=M)"]
    end

    P1 & P2 & P3 -->|map output| SB
    SB -->|fetch by hash partition| R1 & R2 & R3
    R1 & R2 & R3 -->|partial aggregates| FINAL["Final Result\n(Driver)"]

    style Stage1 fill:#2b6cb0,color:#fff
    style Shuffle fill:#744210,color:#fff
    style Stage2 fill:#276749,color:#fff
```

---

## 4. ML Pipeline Flow

```mermaid
flowchart LR
    subgraph Input["Raw Data"]
        RAW["DataFrame\nregion, product, age,\nrevenue, units, label"]
    end

    subgraph FeatureEng["Feature Engineering Pipeline (fit on train only)"]
        SI["StringIndexer\nregion -> region_index\nproduct -> product_index"]
        OHE["OneHotEncoder\nregion_index -> region_ohe\nproduct_index -> product_ohe"]
        VA["VectorAssembler\n[ohe cols + numeric cols]\n-> raw_features"]
        SS["StandardScaler\nraw_features -> features\n(mean=0, std=1)"]
    end

    subgraph Training["Model Training"]
        SPLIT["Train / Test Split\n(80% / 20%)"]
        GBT["GBTClassifier\nor RandomForest\n(Pipeline API)"]
        CV["CrossValidator\n+ ParamGridBuilder\n(k-fold)"]
    end

    subgraph Eval["Evaluation & Persistence"]
        PRED["Predictions\n(probability, prediction)"]
        EVAL["BinaryClassificationEvaluator\nAUC-ROC"]
        SAVE["model.write().save(path)"]
        LOAD["PipelineModel.load(path)"]
    end

    RAW --> SPLIT
    SPLIT -->|train set| SI --> OHE --> VA --> SS --> GBT
    GBT --> CV --> EVAL
    SPLIT -->|test set| PRED
    CV -->|best model| PRED --> EVAL --> SAVE --> LOAD

    style Input fill:#2d3748,color:#fff
    style FeatureEng fill:#2b6cb0,color:#fff
    style Training fill:#276749,color:#fff
    style Eval fill:#744210,color:#fff
```

---

## 5. Data Quality Check Flow

```mermaid
flowchart TD
    DATA["Input DataFrame"] --> SUITE["DataQualitySuite\n(register checks)"]

    SUITE --> NC["NullCheck\n(null fraction per column)"]
    SUITE --> RC["RangeCheck\n(min/max bounds)"]
    SUITE --> UC["UniquenessCheck\n(duplicate detection)"]
    SUITE --> SC["SchemaCheck\n(required columns + dtypes)"]
    SUITE --> PC["PredicateCheck\n(custom business rule)"]

    NC & RC & UC & SC & PC --> RES["CheckResult list\n(passed, actual, expected, message)"]

    RES --> PASS["PASS checks\n(logged at INFO)"]
    RES --> FAIL["FAIL checks\n(logged at WARNING)"]

    PASS & FAIL --> REPORT["QualityReportGenerator\n-> CSV report\n(suite_quality_report.csv)"]
    REPORT --> DECIDE{{"All checks pass?"}}
    DECIDE -->|Yes| PROCEED["Proceed to downstream\nprocessing / ML"]
    DECIDE -->|No| BLOCK["Block pipeline\nAlert / Quarantine data"]

    style PASS fill:#276749,color:#fff
    style FAIL fill:#c53030,color:#fff
    style BLOCK fill:#744210,color:#fff
    style PROCEED fill:#2b6cb0,color:#fff
```

---

## 6. Pandas UDF vs Python UDF Data Flow

```mermaid
flowchart TD
    subgraph Python_UDF["Regular Python UDF (slow path)"]
        P1["JVM partition data\n(Java objects)"]
        P2["Serialize row-by-row\n(pickle)"]
        P3["Python process\nprocesses one row\nat a time"]
        P4["Deserialize results\nrow-by-row"]
        P5["Return to JVM"]
        P1 --> P2 --> P3 --> P4 --> P5
    end

    subgraph Pandas_UDF["Pandas UDF — Vectorized (fast path)"]
        Q1["JVM partition data\n(Java objects)"]
        Q2["Serialize entire batch\nvia Apache Arrow\n(zero-copy columnar)"]
        Q3["Python process\nreceives Pandas Series\n(entire batch at once)"]
        Q4["NumPy / Pandas\nvectorized operation"]
        Q5["Arrow serialize\nresult Series"]
        Q6["Return to JVM"]
        Q1 --> Q2 --> Q3 --> Q4 --> Q5 --> Q6
    end

    COMPARE{{"Speedup comparison"}}
    P5 --> COMPARE
    Q6 --> COMPARE
    COMPARE --> NOTE["Pandas UDF is typically\n10-100x faster for\nlarge partitions\ndue to Arrow batch transfer"]

    style Python_UDF fill:#c53030,color:#fff
    style Pandas_UDF fill:#276749,color:#fff
    style NOTE fill:#2b6cb0,color:#fff
```

---

## 7. Adaptive Query Execution (AQE) Flow

```mermaid
flowchart TD
    Q["User Query\n(SQL / DataFrame API)"] --> PLAN["Initial Physical Plan\n(shuffle.partitions = 200)"]
    PLAN --> STAGE1["Stage 1 Execution\n(collect partition statistics)"]
    STAGE1 --> STATS["Runtime Statistics\n(actual partition sizes,\nrow counts, skew info)"]

    STATS --> AQE{"AQE\nOptimiser"}
    AQE --> COAL["Coalesce small partitions\n(reduce from 200 -> few)\nAvoid tiny task overhead"]
    AQE --> BJ["Convert SortMerge Join\n-> Broadcast Join\n(one side turned out small)"]
    AQE --> SKEW["Split skewed partitions\n(split large -> multiple small)\nEliminate stragglers"]

    COAL & BJ & SKEW --> STAGE2["Stage 2 Execution\n(re-optimised plan)"]
    STAGE2 --> RESULT["Final Result"]

    style AQE fill:#2b6cb0,color:#fff
    style RESULT fill:#276749,color:#fff
```
