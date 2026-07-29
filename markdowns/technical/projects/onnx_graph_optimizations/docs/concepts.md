---
title: "ONNX Concepts and Graph Optimization"
subtitle: "This document explains core ONNX concepts with diagrams, tables, and worked examples. Every diagram is rendered as a Mermaid block so it displays natively in GitHub, GitLab, and most modern Markdown viewers."
category: technical
project: onnx_graph_optimizations
project_title: "ONNX Graph Optimizations"
date: 2025-12-29
reading_time: 6
tags:
  - onnx-graph-optimizations
  - docs
author: "Shreedhar Kodate"
output: "blogs/technical/posts/onnx_graph_optimizations/docs/concepts.html"
---
This document explains core ONNX concepts with diagrams, tables, and worked examples.
Every diagram is rendered as a Mermaid block so it displays natively in GitHub, GitLab,
and most modern Markdown viewers.

---

## 1. The ONNX Ecosystem

ONNX (Open Neural Network Exchange) is an open standard that decouples model training
from deployment.  A model trained in any supported framework can be exported to a single
`.onnx` file and executed on any compliant runtime across diverse hardware.

```mermaid
flowchart LR
    subgraph Frameworks["Training Frameworks"]
        direction TB
        PT["PyTorch"]
        TF["TensorFlow / Keras"]
        SK["scikit-learn\n(via skl2onnx)"]
        PX["PaddlePaddle"]
        MX["MXNet"]
    end

    subgraph ONNX["ONNX Standard"]
        direction TB
        SPEC["ONNX Opset Spec\n(standardised ops)"]
        GRAPH["onnx.ModelProto\n(.onnx file)"]
        CHECK["onnx.checker\nonnx.shape_inference"]
    end

    subgraph Runtimes["Inference Runtimes"]
        direction TB
        ORT["ONNX Runtime (ORT)"]
        TRT["TensorRT\n(NVIDIA)"]
        OV["OpenVINO\n(Intel)"]
        COREML["Core ML\n(Apple)"]
        TVM["Apache TVM"]
    end

    subgraph Hardware["Target Hardware"]
        direction TB
        CPU["CPU\n(x86 / ARM)"]
        NVGPU["NVIDIA GPU\n(CUDA)"]
        INTELHW["Intel CPU/iGPU\n(VNNI, AMX)"]
        APPLE["Apple Silicon\n(Neural Engine)"]
        EDGE["Edge / MCU\n(ONNX Runtime Mobile)"]
    end

    PT --> ONNX
    TF --> ONNX
    SK --> ONNX
    PX --> ONNX
    MX --> ONNX

    ONNX --> ORT
    ONNX --> TRT
    ONNX --> OV
    ONNX --> COREML
    ONNX --> TVM

    ORT --> CPU
    ORT --> NVGPU
    TRT --> NVGPU
    OV --> INTELHW
    COREML --> APPLE
    ORT --> EDGE
```

---

## 2. ONNX Graph Structure

An ONNX model is a **directed acyclic graph (DAG)**.  Every element has a precise role:

| Element | Description |
|---------|-------------|
| **Node** | An operator (Conv, Relu, MatMul …). Has typed inputs/outputs. |
| **Edge** | A named tensor flowing between nodes. |
| **Initializer** | A constant tensor (weight, bias). Stored in the model file. |
| **Graph Input** | External data fed at runtime (e.g., image batch). |
| **Graph Output** | Results returned to the caller. |
| **value_info** | Shape/type annotations on intermediate tensors (added by shape inference). |

```mermaid
graph TD
    subgraph Inputs["Graph Inputs"]
        IN["input\n[batch, 1, 28, 28]"]
    end

    subgraph Initializers["Initializers (Weights)"]
        W1["conv1.weight\n[32, 1, 3, 3]"]
        B1["conv1.bias\n[32]"]
        BNW["bn1.weight / bias / mean / var"]
        W2["conv2.weight\n[64, 32, 3, 3]"]
        FC1W["fc1.weight  [128, 1024]"]
        FC2W["fc2.weight  [10, 128]"]
    end

    subgraph Nodes["Operator Nodes"]
        C1["Conv\n(stride=1, pad=1)"]
        BN1["BatchNormalization\n(ε=1e-5)"]
        R1["Relu"]
        C2["Conv\n(stride=1, pad=1)"]
        BN2["BatchNormalization"]
        R2["Relu"]
        POOL["GlobalAveragePool"]
        FL["Flatten"]
        FC1["Gemm"]
        R3["Relu"]
        FC2["Gemm"]
    end

    subgraph Outputs["Graph Outputs"]
        OUT["output\n[batch, 10]"]
    end

    IN --> C1
    W1 --> C1
    B1 --> C1
    C1 --> BN1
    BNW --> BN1
    BN1 --> R1
    R1 --> C2
    W2 --> C2
    C2 --> BN2
    BN2 --> R2
    R2 --> POOL
    POOL --> FL
    FL --> FC1
    FC1W --> FC1
    FC1 --> R3
    R3 --> FC2
    FC2W --> FC2
    FC2 --> OUT
```

---

## 3. ORT Optimization Pipeline

ONNX Runtime applies graph-level optimizations before running any inference.
The four levels are cumulative — each higher level includes all optimizations from lower levels.

```mermaid
flowchart TD
    RAW["Raw ONNX Graph\n(exported from framework)"]

    L0["ORT_DISABLE_ALL\nNo optimizations\nFast session init\nDebug-friendly"]
    L1["ORT_ENABLE_BASIC\n+ Constant folding\n+ Identity elimination\n+ Slice elimination\n+ Redundant node removal"]
    L2["ORT_ENABLE_EXTENDED\n+ Conv+BN folding (→ fewer nodes)\n+ Conv+BN+Relu fusion\n+ Gelu fusion\n+ LayerNorm fusion\n+ Attention fusion (transformers)"]
    L3["ORT_ENABLE_ALL\n+ Layout optimization\n(NCHW → NHWC or NCHWc\nfor accelerator efficiency)\n+ Provider-specific kernels"]

    OPT["Optimized ONNX Graph\n(saved via optimized_model_filepath)"]

    RAW --> L0
    L0 --> L1
    L1 --> L2
    L2 --> L3
    L3 --> OPT
```

### Optimization Level Reference Table

| Level | Constant Folding | Redundant Elimination | Op Fusion | Layout Opt | Use Case |
|-------|:---:|:---:|:---:|:---:|----------|
| `ORT_DISABLE_ALL` | - | - | - | - | Debugging, profiling raw graph |
| `ORT_ENABLE_BASIC` | Yes | Yes | - | - | Minimal safe optimisation |
| `ORT_ENABLE_EXTENDED` | Yes | Yes | Yes | - | Production CPU inference |
| `ORT_ENABLE_ALL` | Yes | Yes | Yes | Yes | Maximum throughput (EP-specific) |

---

## 4. Operator Fusion Patterns

Fusion merges multiple small kernels into one, eliminating intermediate tensor writes and
reducing kernel-launch overhead.

```mermaid
flowchart LR
    subgraph Before["Before Fusion (3 nodes, 2 intermediate tensors)"]
        direction TB
        C["Conv\nnode"]
        BN["BatchNorm\nnode"]
        RL["Relu\nnode"]
        T1([" tensor_a "])
        T2([" tensor_b "])
        C --> T1 --> BN --> T2 --> RL
    end

    subgraph After["After Fusion (1 node, 0 intermediate tensors)"]
        direction TB
        FUSED["ConvBatchNormRelu\n(single fused kernel)\nBN weights absorbed\ninto Conv parameters"]
    end

    Before -- "ORT_ENABLE_EXTENDED" --> After
```

### BatchNorm Folding - What Happens Mathematically

BatchNorm at inference (running statistics, not per-batch) computes:

```
y = (x - mean) / sqrt(var + ε) * γ + β
```

This is a linear transform that can be folded into the preceding Conv:

```
W_fused = W_conv * (γ / sqrt(var + ε))
b_fused = b_conv * (γ / sqrt(var + ε)) + β - mean * (γ / sqrt(var + ε))
```

After folding: `y = W_fused * x + b_fused` — one fewer node, no intermediate tensor.

### Common Fusion Patterns

```mermaid
graph LR
    subgraph CNN["CNN Fusions (ORT_ENABLE_EXTENDED)"]
        P1["Conv + BatchNorm + Relu → ConvBatchNormRelu"]
        P2["Conv + BatchNorm → ConvBatchNorm"]
        P3["MatMul + Add → Gemm (or FusedMatMul)"]
    end

    subgraph Transformer["Transformer Fusions (onnxruntime.transformers)"]
        T1["QKV MatMuls + Softmax + projection → MultiHeadAttention"]
        T2["Div + Erf + Add + Mul + Mul → FastGelu"]
        T3["ReduceMean + Sub + Pow + … → LayerNormalization"]
        T4["Embedding + Position + LayerNorm → EmbedLayerNormalization"]
    end
```

---

## 5. Execution Provider Selection Flow

ORT selects operators for each EP in priority order.  Unsupported ops fall through
to the next provider in the list (ultimately always CPU).

```mermaid
flowchart TD
    START([Inference Request])

    Q1{"CUDA EP\navailable?"}
    Q2{"Op supported\nby CUDA EP?"}
    Q3{"TensorRT EP\navailable?"}
    Q4{"Op supported\nby TRT?"}
    CPU["CPUExecutionProvider\n(always available)"]
    CUDA["CUDAExecutionProvider\n(NVIDIA GPU)"]
    TRT["TensorrtExecutionProvider\n(NVIDIA TensorRT)"]

    RESULT([Output Tensor])

    START --> Q3
    Q3 -- Yes --> Q4
    Q3 -- No  --> Q1
    Q4 -- Yes --> TRT --> RESULT
    Q4 -- No  --> Q1
    Q1 -- Yes --> Q2
    Q1 -- No  --> CPU --> RESULT
    Q2 -- Yes --> CUDA --> RESULT
    Q2 -- No  --> CPU --> RESULT
```

### Execution Provider Reference Table

| EP | Package | Hardware | Notes |
|----|---------|----------|-------|
| `CPUExecutionProvider` | `onnxruntime` | Any CPU | Always available. Uses MLAS. |
| `CUDAExecutionProvider` | `onnxruntime-gpu` | NVIDIA GPU | Requires CUDA + cuDNN. |
| `TensorrtExecutionProvider` | `onnxruntime-gpu` | NVIDIA GPU | Best throughput; requires TensorRT. |
| `ROCmExecutionProvider` | `onnxruntime-rocm` | AMD GPU | Requires ROCm stack. |
| `CoreMLExecutionProvider` | `onnxruntime` | Apple Silicon | macOS / iOS. Neural Engine delegate. |
| `OpenVINOExecutionProvider` | `onnxruntime-openvino` | Intel CPU/GPU/VPU | Requires OpenVINO toolkit. |
| `DirectMLExecutionProvider` | `onnxruntime-directml` | Windows GPU | DirectX 12. Windows only. |
| `QNNExecutionProvider` | `onnxruntime-qnn` | Qualcomm NPU | Snapdragon NPU / HTP. |

---

## 6. ONNX Opset Versioning

An opset version specifies which set of operator definitions the model uses.
Higher opsets add new operators and may change semantics of existing ones.

```mermaid
timeline
    title ONNX Opset History (selected)
    Opset 9  : BatchNorm inference mode stabilised
    Opset 11 : Resize / Range / Cumsum added
    Opset 13 : Quantisation operators (QLinearConv, etc.)
    Opset 14 : Trilu, Identity upgrade
    Opset 15 : Optional / OptionalGetElement
    Opset 16 : ScatterElements, Greedy decoding
    Opset 17 : STFT, LayerNorm as first-class op
    Opset 18 : Pad / Resize / LpPool updates
    Opset 19 : AveragePool / Dequantize updates
    Opset 20 : IsInf / IsNaN upgrades, DFT
    Opset 21 : Flatten / Identity upgrade (current stable)
```

**Rule of thumb**: use `opset_version=17` for broad ORT compatibility.
PyTorch `torch.onnx.export` supports up to the current opset automatically.

---

## 7. Key ONNX Python APIs

```python
import onnx
import onnx.shape_inference

# Load a model
model = onnx.load("model.onnx")

# Structural validation (raises on error)
onnx.checker.check_model(model)

# Shape inference — fills value_info for all intermediate tensors
model = onnx.shape_inference.infer_shapes(model)

# Inspect the graph
graph = model.graph
for node in graph.node:
    print(node.op_type, node.input, node.output)

# Inspect initializers (weights)
for init in graph.initializer:
    print(init.name, list(init.dims))

# Access input/output shapes
for vi in graph.input:
    for dim in vi.type.tensor_type.shape.dim:
        print(dim.dim_value or dim.dim_param)
```

```python
import onnxruntime as ort

# Create a session with optimization
opts = ort.SessionOptions()
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
opts.optimized_model_filepath = "optimized.onnx"

session = ort.InferenceSession("model.onnx", sess_options=opts,
                                providers=["CPUExecutionProvider"])
# Run inference
outputs = session.run(None, {"input": input_array})
```