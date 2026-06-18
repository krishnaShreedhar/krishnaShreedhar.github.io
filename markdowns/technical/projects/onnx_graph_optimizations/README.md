# ONNX Graph Optimizations

A self-contained project illustrating ONNX features and graph optimization techniques
with minimal, runnable code examples.  Every concept is paired with working code,
mermaid diagrams, and extensive logging so a learner can trace exactly what happens at
each step.

---

## What This Project Covers

| Topic | Module | Description |
|-------|--------|-------------|
| PyTorch → ONNX export | `src/model_exporter/pytorch_exporter.py` | CNN export with shape validation |
| sklearn → ONNX export | `src/model_exporter/sklearn_exporter.py` | Pipeline export via skl2onnx |
| ORT optimization levels | `src/graph_optimizer/optimization_pipeline.py` | DISABLE_ALL → BASIC → EXTENDED → ALL |
| Constant folding | `src/graph_optimizer/constant_folder.py` | Detect and quantify foldable nodes |
| Fusion detection | `src/graph_optimizer/fusion_analyzer.py` | Conv+BN+Relu, MatMul+Add, Gelu, etc. |
| ORT inference | `src/inference_engine/ort_inference.py` | Session creation, single + batch inference |
| Benchmarking | `src/inference_engine/benchmark.py` | Latency, throughput, p50/p95/p99 |
| Execution Providers | `src/inference_engine/execution_providers.py` | CPU / CUDA / TensorRT selection |
| Graph inspection | `src/graph_analysis/graph_inspector.py` | Nodes, edges, initializers, metadata |
| Shape propagation | `src/graph_analysis/shape_analyzer.py` | Tensor shapes through the graph |
| Operator counting | `src/graph_analysis/node_counter.py` | Distribution charts, multi-level compare |
| Full pipeline demo | `src/notebooks/onnx_optimization_demo.ipynb` | End-to-end notebook |

---

## Project Structure

```
onnx_graph_optimizations/
    src/
        notebooks/              # Jupyter notebook: full pipeline demo
        model_exporter/         # pytorch_exporter.py, sklearn_exporter.py
        graph_optimizer/        # optimization_pipeline.py, constant_folder.py, fusion_analyzer.py
        inference_engine/       # ort_inference.py, benchmark.py, execution_providers.py
        graph_analysis/         # graph_inspector.py, shape_analyzer.py, node_counter.py
    docs/
        concepts.md             # ONNX concepts with mermaid diagrams
        flow_diagrams.md        # Pipeline flow diagrams
    docker/
        Dockerfile
        docker-compose.yml
        requirements.txt
    logs/                       # Runtime log files (auto-created)
    outputs/
        models/                 # Exported and optimized .onnx files + charts
    README.md
    pyproject.toml
    config.yaml                 # Single source of truth for all constants
```

---

## ONNX Core Concepts

### What is ONNX?

ONNX (Open Neural Network Exchange) is an open format for representing machine learning
models.  It defines:

- A **computation graph** (nodes connected by typed edges).
- A set of **operator specifications** (Conv, BatchNorm, Relu, Gemm, …).
- A **serialisation format** (protobuf `.onnx` file).

Any framework can export to ONNX; any runtime can load it.

### ONNX Graph Structure

```
ModelProto
  └── GraphProto
        ├── node[]          operator nodes (Conv, BN, Relu, …)
        ├── input[]         external inputs (e.g., image tensor)
        ├── output[]        graph outputs
        ├── initializer[]   constant tensors (weights, biases)
        └── value_info[]    intermediate tensor shapes (after shape inference)
```

### ORT Optimization Levels

| Level | What it does |
|-------|-------------|
| `ORT_DISABLE_ALL` | No optimizations. Useful for debugging raw graph. |
| `ORT_ENABLE_BASIC` | Constant folding, identity elimination, slice removal. |
| `ORT_ENABLE_EXTENDED` | Operator fusion: Conv+BN folding, Conv+BN+Relu → single kernel, Gelu, LayerNorm. |
| `ORT_ENABLE_ALL` | Layout optimization (NCHW→NHWC etc.) + all above. |

### BatchNorm Folding (most impactful fusion)

At inference time, BatchNorm is a **linear transform** on the preceding Conv output.
ORT folds BN parameters (γ, β, μ, σ²) into Conv weights and bias:

```
W_fused = W_conv · (γ / √(σ² + ε))
b_fused = b_conv · (γ / √(σ² + ε)) + β - μ · (γ / √(σ² + ε))
```

Result: **BatchNorm node disappears** from the graph.  For a network with N Conv+BN pairs,
you eliminate N nodes with zero accuracy loss.

---

## Quick Start

### 1. Install dependencies

```bash
cd projects/onnx_graph_optimizations
pip install -e ".[notebook]"
```

For GPU support (CUDA):
```bash
pip install onnxruntime-gpu   # replaces onnxruntime
```

### 2. Export the CNN model

```bash
python src/model_exporter/pytorch_exporter.py config.yaml
# → outputs/models/cnn_model.onnx
```

### 3. Export the sklearn pipeline

```bash
python src/model_exporter/sklearn_exporter.py config.yaml
# → outputs/models/sklearn_pipeline.onnx
```

### 4. Inspect the graph

```bash
python src/graph_analysis/graph_inspector.py config.yaml outputs/models/cnn_model.onnx
python src/graph_analysis/shape_analyzer.py config.yaml outputs/models/cnn_model.onnx
python src/graph_analysis/node_counter.py config.yaml outputs/models/cnn_model.onnx
```

### 5. Apply optimizations

```bash
python src/graph_optimizer/optimization_pipeline.py config.yaml outputs/models/cnn_model.onnx
# → outputs/models/optimized_ORT_DISABLE_ALL.onnx
# → outputs/models/optimized_ORT_ENABLE_BASIC.onnx
# → outputs/models/optimized_ORT_ENABLE_EXTENDED.onnx
# → outputs/models/optimized_ORT_ENABLE_ALL.onnx
```

### 6. Analyse fusion patterns

```bash
python src/graph_optimizer/fusion_analyzer.py config.yaml outputs/models/cnn_model.onnx
python src/graph_optimizer/constant_folder.py config.yaml outputs/models/cnn_model.onnx
```

### 7. Benchmark inference

```bash
python src/inference_engine/benchmark.py config.yaml \
    outputs/models/cnn_model.onnx \
    outputs/models/optimized_ORT_ENABLE_ALL.onnx
```

### 8. Check available EPs

```bash
python src/inference_engine/execution_providers.py config.yaml outputs/models/cnn_model.onnx
```

### 9. Run the full notebook

```bash
cd projects/onnx_graph_optimizations
jupyter notebook src/notebooks/onnx_optimization_demo.ipynb
```

### 10. Docker

```bash
cd projects/onnx_graph_optimizations/docker
docker-compose up --build
# Open http://localhost:8888
```

---

## Configuration

All constants live in `config.yaml`.  No command-line arguments needed — scripts read the config file as their first positional argument (defaults to `config.yaml`).

Key sections:

```yaml
pytorch_model:
  in_channels: 1        # grayscale (MNIST-style)
  num_classes: 10
  input_height: 28
  input_width: 28

optimization:
  levels: [ORT_DISABLE_ALL, ORT_ENABLE_BASIC, ORT_ENABLE_EXTENDED, ORT_ENABLE_ALL]
  save_optimized: true

inference:
  benchmark_iterations: 100
  warmup_iterations: 10
  batch_sizes: [1, 4, 8, 16, 32]
```

---

## Outputs

After running the full pipeline you will find:

```
outputs/models/
    cnn_model.onnx
    sklearn_pipeline.onnx
    optimized_ORT_DISABLE_ALL.onnx
    optimized_ORT_ENABLE_BASIC.onnx
    optimized_ORT_ENABLE_EXTENDED.onnx
    optimized_ORT_ENABLE_ALL.onnx
    baseline_operator_distribution.png
    optimization_node_reduction.png
    inference_performance.png
    baseline_vs_optimized.png
    multi_level_node_counts.png

logs/
    onnx_optimizations.log     # structured per-module log with timestamps
```

---

## Documentation

- `docs/concepts.md` — ONNX ecosystem, graph structure, optimization levels, EP selection,
  all with Mermaid diagrams and reference tables.
- `docs/flow_diagrams.md` — Step-by-step flow diagrams for export, optimization, inference,
  analysis, and benchmarking pipelines.

---

## Key Libraries

| Library | Version | Role |
|---------|---------|------|
| `torch` | ≥2.3 | Model definition and ONNX export |
| `onnx` | ≥1.16 | Graph loading, checking, shape inference |
| `onnxruntime` | ≥1.18 | Optimized inference + graph optimization |
| `scikit-learn` | ≥1.5 | sklearn pipeline for export demo |
| `skl2onnx` | ≥1.17 | sklearn → ONNX conversion |
| `numpy` | ≥1.26 | Input/output array handling |
| `matplotlib` | ≥3.9 | Operator distribution charts |
| `pyyaml` | ≥6.0 | YAML config parsing |
