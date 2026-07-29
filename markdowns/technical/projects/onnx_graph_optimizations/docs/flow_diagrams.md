---
title: "ONNX Optimization Pipeline - Flow Diagrams"
subtitle: "Detailed workflow diagrams for every stage of the ONNX optimization lifecycle. Each diagram corresponds directly to code in the `src/` modules."
category: technical
project: onnx_graph_optimizations
project_title: "ONNX Graph Optimizations"
date: 2025-10-03
reading_time: 5
tags:
  - onnx-graph-optimizations
  - docs
author: "Shreedhar Kodate"
output: "blogs/technical/posts/onnx_graph_optimizations/docs/flow_diagrams.html"
---
Detailed workflow diagrams for every stage of the ONNX optimization lifecycle.
Each diagram corresponds directly to code in the `src/` modules.

---

## 1. Model Export Pipeline

This flow covers `src/model_exporter/pytorch_exporter.py` and
`src/model_exporter/sklearn_exporter.py`.

```mermaid
flowchart TD
    START([Start: define model architecture])

    subgraph PyTorch["PyTorch → ONNX (pytorch_exporter.py)"]
        direction TB
        PT1["Define SimpleCNN\nConv→BN→ReLU→Conv→BN→ReLU\n→AvgPool→Flatten→Linear×2"]
        PT2["model.eval()\n(disables dropout/BN training mode)"]
        PT3["Create dummy input\ntorch.randn(B, C, H, W)"]
        PT4["torch.onnx.export()\nopset=17, dynamic_axes={'input': {0: 'batch_size'}}\ndo_constant_folding=True"]
        PT5["onnx.checker.check_model()"]
        PT6["onnx.shape_inference.infer_shapes()"]
        PT7["Log: opset, node_count,\ninput/output shapes"]
    end

    subgraph Sklearn["sklearn → ONNX (sklearn_exporter.py)"]
        direction TB
        SK1["make_classification()\nsynthetic dataset"]
        SK2["Pipeline(StandardScaler\n+ LogisticRegression)"]
        SK3["pipeline.fit(X, y)"]
        SK4["skl2onnx.convert_sklearn()\nFloatTensorType, opset=17"]
        SK5["onnx.checker.check_model()"]
        SK6["Log: node types,\noperator count"]
    end

    ONNX_FILE[".onnx file\n(outputs/models/)"]

    START --> PT1 --> PT2 --> PT3 --> PT4 --> PT5 --> PT6 --> PT7 --> ONNX_FILE
    START --> SK1 --> SK2 --> SK3 --> SK4 --> SK5 --> SK6 --> ONNX_FILE
```

### Export Key Decision Points

```mermaid
flowchart TD
    Q1{"Dynamic batch\nsize needed?"}
    Q1 -- Yes --> DYN["dynamic_axes={'input': {0: 'batch_size'}}"]
    Q1 -- No  --> STAT["No dynamic_axes\n(fixed shape, smaller model)"]

    Q2{"do_constant_folding?"}
    Q2 -- Yes --> CF["Pre-folds constants\nat export time"]
    Q2 -- No  --> NCF["Larger graph\nbut more transparent"]

    Q3{"opset version?"}
    Q3 -- "17 (recommended)" --> OPS17["Broad ORT support\nLayerNorm as first-class op"]
    Q3 -- "<11"              --> OPS_OLD["Missing Resize,\nRange ops"]
    Q3 -- ">20"             --> OPS_NEW["Newest features,\ncheck ORT version support"]
```

---

## 2. Graph Optimization Pipeline

This flow covers `src/graph_optimizer/optimization_pipeline.py`,
`constant_folder.py`, and `fusion_analyzer.py`.

```mermaid
flowchart TD
    INPUT_MODEL["Input: raw .onnx file\n(from export step)"]

    subgraph Scan["Pre-optimization Analysis"]
        direction TB
        FU["FusionAnalyzer.find_patterns()\nDetect Conv+BN+Relu,\nMatMul+Add, etc."]
        CF["ConstantFolder.analyse()\nCount foldable nodes"]
        GI["GraphInspector.inspect()\nNode count, op types, shapes"]
    end

    subgraph Levels["Apply ORT Levels (OptimizationPipeline)"]
        direction TB
        L0["SessionOptions.graph_optimization_level\n= ORT_DISABLE_ALL\n→ save optimized_ORT_DISABLE_ALL.onnx"]
        L1["= ORT_ENABLE_BASIC\n→ constant folding, identity elim\n→ save optimized_ORT_ENABLE_BASIC.onnx"]
        L2["= ORT_ENABLE_EXTENDED\n→ + Conv+BN folding, op fusion\n→ save optimized_ORT_ENABLE_EXTENDED.onnx"]
        L3["= ORT_ENABLE_ALL\n→ + layout optimization\n→ save optimized_ORT_ENABLE_ALL.onnx"]
    end

    subgraph PostAnalysis["Post-optimization Analysis"]
        direction TB
        NC["NodeCounter.compare_counts()\nNode reduction per level"]
        FA["FusionAnalyzer.compare_fusion()\nBN nodes before/after"]
        CF2["ConstantFolder.compare()\nFolded node delta"]
    end

    REPORT["Structured Log Report\n+ comparison charts"]

    INPUT_MODEL --> Scan
    Scan --> Levels
    L0 --> L1 --> L2 --> L3
    Levels --> PostAnalysis
    PostAnalysis --> REPORT
```

### BatchNorm Folding - Node-level Detail

```mermaid
flowchart LR
    subgraph RAW["Before (ORT_ENABLE_BASIC)"]
        direction TB
        CW["Conv weights\nW: [32,1,3,3]"]
        CB["Conv bias\nb: [32]"]
        C["Conv node"]
        BNW["BN.weight γ\nBN.bias β\nBN.mean μ\nBN.var σ²"]
        BN["BatchNorm node\n(ε=1e-5)"]
        RL["Relu node"]
        CW & CB --> C --> BN
        BNW --> BN --> RL
    end

    subgraph FUSED["After (ORT_ENABLE_EXTENDED)"]
        direction TB
        FW["Fused Conv weights\nW' = W · (γ/√(σ²+ε))"]
        FB["Fused Conv bias\nb' = b·(γ/√(σ²+ε)) + β − μ·(γ/√(σ²+ε))"]
        FC["Conv node\n(BN absorbed)"]
        FR["Relu node"]
        FW & FB --> FC --> FR
    end

    RAW -- "ORT_ENABLE_EXTENDED" --> FUSED
```

---

## 3. Inference Request Flow with EP Selection

This flow covers `src/inference_engine/ort_inference.py` and
`src/inference_engine/execution_providers.py`.

```mermaid
sequenceDiagram
    participant App as Application
    participant Sel as ExecutionProviderSelector
    participant ORT as ort.InferenceSession
    participant EP1 as CUDAExecutionProvider
    participant EP2 as CPUExecutionProvider

    App->>Sel: build_provider_list(['CUDA', 'CPU'])
    Sel->>Sel: ort.get_available_providers()
    alt CUDA available
        Sel-->>App: ['CUDAExecutionProvider', 'CPUExecutionProvider']
    else CUDA not available
        Sel-->>App: ['CPUExecutionProvider']
    end

    App->>ORT: InferenceSession(model, providers=[...])
    ORT->>ORT: Load and optimize graph
    ORT->>ORT: Partition nodes to EPs

    loop For each node in graph
        ORT->>EP1: Can you run this op?
        alt EP1 supports op
            EP1-->>ORT: Yes
            ORT->>EP1: Execute node
        else EP1 doesn't support op
            ORT->>EP2: Execute node (fallback)
        end
    end

    App->>ORT: session.run(None, {"input": data})
    ORT->>EP1: Execute Conv, Gemm (GPU ops)
    EP1-->>ORT: intermediate tensor
    ORT->>EP2: Execute unsupported ops (CPU fallback)
    EP2-->>ORT: output tensor
    ORT-->>App: [output_array]
```

### Session Options Configuration Flow

```mermaid
flowchart TD
    OPT["ort.SessionOptions()"]
    OPT --> GL["graph_optimization_level\n= ORT_ENABLE_ALL"]
    OPT --> OF["optimized_model_filepath\n= 'outputs/models/opt.onnx'"]
    OPT --> PROF["enable_profiling = True\n(optional, see config.inference.enable_profiling)"]
    OPT --> THR["inter_op_num_threads = 0\n(auto-detect)\nintra_op_num_threads = 0"]

    GL & OF & PROF & THR --> SESSION["ort.InferenceSession(model, opts, providers)"]
    SESSION --> RUN["session.run(output_names, input_dict)"]
    RUN --> OUT["list of np.ndarray outputs"]
```

---

## 4. Graph Analysis Workflow

This flow covers `src/graph_analysis/graph_inspector.py`,
`shape_analyzer.py`, and `node_counter.py`.

```mermaid
flowchart TD
    ONNX_FILE[".onnx file"]

    subgraph Load["Load & Validate"]
        direction TB
        LD["onnx.load(path)"]
        SI["onnx.shape_inference.infer_shapes()\n→ fills value_info with tensor shapes"]
        CHK["onnx.checker.check_model()\n→ raises on structural errors"]
    end

    subgraph Structure["Structural Inspection (GraphInspector)"]
        direction TB
        META["Log metadata:\n  ir_version, opset, graph name"]
        IOS["Log inputs/outputs:\n  name, shape, dtype"]
        INITS["Log initializers:\n  name, shape, numel, dtype\n  total_parameters count"]
        NODES["Inspect nodes:\n  op_type, name, inputs, outputs\n  attributes (kernel_size, strides…)"]
        OPP["Detect optimization opportunities:\n  unfused Conv+BN+Relu patterns\n  Dropout at inference"]
    end

    subgraph Shapes["Shape Propagation (ShapeAnalyzer)"]
        direction TB
        MAP["Build shape_map:\n  tensor_name → [dims]"]
        PROP["Print per-node:\n  [idx] op_type  in=[x:shape]  →  out=[y:shape]"]
        DYN["Find dynamic dimensions:\n  symbolic dims (batch_size, seq_len)"]
    end

    subgraph Counting["Operator Counting (NodeCounter)"]
        direction TB
        CNT["Count by op_type:\n  Conv×2, BN×2, Relu×3 …"]
        SORT["Sort descending by count"]
        CHART1["plot_distribution()\n  → horizontal bar chart\n  → saved to outputs/models/"]
        CHART2["compare_counts(models_dict)\n  → grouped bar across opt levels"]
    end

    ONNX_FILE --> Load
    Load --> Structure
    Load --> Shapes
    Load --> Counting
```

### Shape Inference Detail

```mermaid
flowchart LR
    subgraph Before["Before infer_shapes"]
        GI["graph.input → shape known"]
        VI_EMPTY["graph.value_info → EMPTY"]
        GO["graph.output → shape unknown"]
    end

    subgraph After["After infer_shapes"]
        GI2["graph.input → shape known"]
        VI_FULL["graph.value_info\n→ ALL intermediate tensor shapes\npopulated by symbolic propagation"]
        GO2["graph.output → shape resolved"]
    end

    Before -- "onnx.shape_inference.infer_shapes()" --> After
```

---

## 5. End-to-End Benchmarking Flow

This flow covers `src/inference_engine/benchmark.py`.

```mermaid
flowchart TD
    START([Start benchmark])

    subgraph Setup["Setup (per model × batch_size)"]
        direction TB
        SESS["Create ort.InferenceSession\nORT_ENABLE_ALL"]
        DUMMY["Build dummy input\nnp.random.randn(batch_size, C, H, W).astype(float32)"]
    end

    subgraph WarmUp["Warm-up Phase"]
        direction TB
        WU_LOOP["for _ in range(warmup_iterations=10):\n    session.run(None, inputs)"]
        WU_NOTE["Purpose: prime CPU caches,\nJIT-compile ORT kernels"]
    end

    subgraph Timed["Timed Loop"]
        direction TB
        TLOOP["for _ in range(benchmark_iterations=100):"]
        T0["t0 = time.perf_counter()"]
        RUN["session.run(None, inputs)"]
        RECORD["latencies_ms.append((t1-t0)*1000)"]
    end

    subgraph Stats["Statistics"]
        direction TB
        MEAN["mean_latency_ms"]
        STD["std_latency_ms"]
        P50["p50_latency_ms  (median)"]
        P95["p95_latency_ms  (tail latency)"]
        P99["p99_latency_ms  (worst-case)"]
        THRU["throughput = batch_size / (mean_ms / 1000)"]
    end

    REPORT["BenchmarkResult dataclass\n→ logged to file + console"]

    START --> Setup --> WarmUp --> Timed --> Stats --> REPORT
```

### Benchmark Interpretation Guide

| Metric | When to care | What it tells you |
|--------|-------------|-------------------|
| `mean_latency_ms` | Always | Average cost per inference call |
| `p50_latency_ms` | Steady-state systems | Typical latency (50% of calls faster) |
| `p95_latency_ms` | SLA-bound services | Latency seen by 95% of requests |
| `p99_latency_ms` | Latency-sensitive APIs | Near-worst-case; affects tail users |
| `throughput_samples_per_s` | Batch workloads | Maximum data processing rate |

**Throughput vs Latency trade-off**: Larger batch sizes increase throughput
but also increase p99 latency.  Choose batch size based on your SLA.

---

## 6. Complete System Architecture

```mermaid
graph TD
    CFG["config.yaml\n(single source of truth)"]

    CFG --> EXP
    CFG --> OPT
    CFG --> INF
    CFG --> ANA

    subgraph EXP["src/model_exporter/"]
        PE["pytorch_exporter.py\nSimpleCNN → ONNX"]
        SE["sklearn_exporter.py\nPipeline → ONNX"]
    end

    subgraph ANA["src/graph_analysis/"]
        GI2["graph_inspector.py\nnodes, edges, metadata"]
        SA["shape_analyzer.py\ntensor shapes"]
        NC2["node_counter.py\nop distribution, charts"]
    end

    subgraph OPT["src/graph_optimizer/"]
        OP["optimization_pipeline.py\n4 ORT levels"]
        CF2["constant_folder.py\nfold analysis"]
        FA["fusion_analyzer.py\npattern detection"]
    end

    subgraph INF["src/inference_engine/"]
        OI["ort_inference.py\nInferenceSession"]
        BM["benchmark.py\nlatency, throughput"]
        EP["execution_providers.py\nEP selection"]
    end

    ONNX_MODELS["outputs/models/\n  cnn_model.onnx\n  sklearn_pipeline.onnx\n  optimized_*.onnx"]
    CHARTS["outputs/models/\n  *.png charts"]
    LOGS["logs/\n  onnx_optimizations.log"]

    EXP --> ONNX_MODELS
    ONNX_MODELS --> ANA
    ONNX_MODELS --> OPT
    OPT --> ONNX_MODELS
    ONNX_MODELS --> INF
    ANA --> CHARTS
    INF --> LOGS
    OPT --> LOGS
    ANA --> LOGS
```