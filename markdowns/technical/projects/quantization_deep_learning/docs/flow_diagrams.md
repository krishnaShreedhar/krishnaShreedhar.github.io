# Quantization Workflow Diagrams

## 1. Standard PTQ Workflow (8 Steps)

The Post-Training Quantization pipeline transforms a trained FP32 model into an INT8 model without retraining. The key steps are: data collection, statistics gathering, scale computation, and model conversion.

```mermaid
flowchart TD
    S1["Step 1: Train FP32 Model\nUse standard training pipeline\nAchieve target accuracy in FP32"]
    
    S2["Step 2: Select Quantization Backend\ntorch.backends.quantized.engine = 'qnnpack'\n(ARM/mobile) or 'fbgemm' (x86/server)"]
    
    S3["Step 3: Fuse Operators\ntorch.quantization.fuse_modules()\nFuse Conv + BatchNorm + ReLU\n→ Reduces quantization rounding steps"]
    
    S4["Step 4: Assign QConfig\nmodel.qconfig = get_default_qconfig(backend)\nQConfig specifies:\n  - activation observer (MinMax, Histogram, etc.)\n  - weight observer (per-channel MinMax)"]
    
    S5["Step 5: Prepare Model\ntorch.quantization.prepare(model)\n→ Inserts observer modules before/after\n  every quantizable operation"]
    
    S6["Step 6: Run Calibration Data\nRun 100–1000 representative samples\nthrough the prepared model\n→ Observers collect statistics"]
    
    S7["Step 7: Convert to INT8\ntorch.quantization.convert(model)\n→ Replaces float ops with int ops\n→ Freezes scale/zero_point from observers\n→ Removes observer modules"]
    
    S8["Step 8: Validate & Deploy\nMeasure accuracy on test set\nBenchmark latency & throughput\nCompare FP32 vs INT8 metrics"]
    
    S1 --> S2
    S2 --> S3
    S3 --> S4
    S4 --> S5
    S5 --> S6
    S6 --> S7
    S7 --> S8

    style S1 fill:#4488ff,color:#fff
    style S5 fill:#ff9900,color:#000
    style S7 fill:#44aa44,color:#fff
    style S8 fill:#884488,color:#fff
```

---

## 2. Standard QAT Workflow (6 Steps)

Quantization-Aware Training improves upon PTQ by adapting model weights to quantization error during training. The Straight-Through Estimator (STE) enables gradient flow through the non-differentiable rounding operation.

```mermaid
flowchart TD
    Q1["Step 1: Start from Pretrained FP32 Model\nUse converged FP32 checkpoint\nHigher initial accuracy → better final QAT result"]
    
    Q2["Step 2: Prepare for QAT\ntorch.quantization.prepare_qat(model)\nFuses Conv-BN-ReLU patterns\nInserts FakeQuantize nodes:\n  - Acts on activations (after each layer)\n  - Acts on weights (before each Conv/Linear)"]
    
    Q3["Step 3: Fine-tune with Simulated Quantization\nForward pass uses fake-quantized values:\n  x_q = round(x/scale) * scale  (FP32 domain)\nLower LR than original training (1e-5 to 1e-4)\nTypically 10–25% of original training epochs"]
    
    Q4["Step 4: STE Enables Weight Updates\nBackward pass gradient through FakeQuantize:\n  ∂L/∂x ≈ ∂L/∂x_q  (identity STE)\nWeights adapt to minimize loss\nwith quantization noise baked in"]
    
    Q5["Step 5: Convert to True INT8\nmodel.eval()\ntorch.quantization.convert(model)\nRemoves FakeQuantize nodes\nConverts float ops to integer ops\nFreezes learned scale/zero_point values"]
    
    Q6["Step 6: Validate & Deploy\nExpected: 0.1–0.5% accuracy drop vs FP32\nVs PTQ: 0.5–2% accuracy drop\nSame inference speed as PTQ INT8"]
    
    Q1 --> Q2
    Q2 --> Q3
    Q3 --> Q4
    Q4 --> Q3
    Q4 --> Q5
    Q5 --> Q6

    style Q1 fill:#4488ff,color:#fff
    style Q2 fill:#ff9900,color:#000
    style Q4 fill:#ffdd44,color:#000
    style Q5 fill:#44aa44,color:#fff
    style Q6 fill:#884488,color:#fff
```

---

## 3. Calibration Range Selection Flow

Calibration determines the optimal clipping range [min, max] for quantizing activations. The choice of calibration method significantly impacts quantization accuracy.

```mermaid
flowchart TD
    IN["Input: Activation Tensor\nfrom calibration data (N samples)"]
    
    COLLECT["Collect Statistics\nPass calibration batches through model\nObservers record activation values"]
    
    HIST["Build Histogram\nBin activations into kl_bins buckets\n(captures full distribution shape)"]
    
    subgraph METHODS ["Calibration Method Choices"]
        direction LR
        MM["Min-Max\nrange = [min, max]\nFast, exact, outlier-sensitive"]
        PCT["Percentile\nrange = [p-th, (100-p)-th]\nClip outliers at chosen %"]
        MSE_C["MSE Search\nFor α in [0.8, 1.0]:\n  range = [−α×max, +α×max]\n  Pick α minimizing MSE"]
        KL_C["KL-Divergence\nFor threshold T in [max/2, max]:\n  Clip to T, quantize to 256 bins\n  Minimize KL(FP32 || INT8)"]
    end
    
    SCALE["Compute scale & zero_point\nsymmetric: scale = max_abs / 127\nasymmetric: scale = range / 255"]
    
    VALIDATE["Validate Range\nCompute SQNR for chosen range\nIf SQNR < 30 dB: try different method"]
    
    ASSIGN["Assign to Model\nFreeze scale/zero_point per layer\nReady for torch.quantization.convert()"]
    
    IN --> COLLECT
    COLLECT --> HIST
    HIST --> METHODS
    METHODS --> SCALE
    SCALE --> VALIDATE
    VALIDATE --> ASSIGN

    style METHODS fill:#f8f8f8
    style MM fill:#ddeeFF
    style PCT fill:#ddffd4
    style MSE_C fill:#fff4dd
    style KL_C fill:#ffd4d4
```

---

## 4. Sensitivity Analysis → Mixed-Precision Assignment Flow

This flow identifies the optimal bit-width per layer by measuring how much each layer contributes to accuracy degradation when quantized.

```mermaid
flowchart TD
    START["FP32 Baseline\nA_fp32 = evaluate(fp32_model, test_data)"]

    LAYERS["Enumerate Quantizable Layers\nL = {conv1, conv2, fc1, fc2, ...}"]
    
    FOR_EACH["For each layer L_i in L"]
    
    CLONE["Clone FP32 model\nmodel_i = deepcopy(fp32_model)"]
    
    QUANT_LAYER["Quantize ONLY layer L_i\nmodel_i.L_i.qconfig = default_qconfig\nAll other layers: qconfig = None\ntorch.quantization.prepare() → calibrate → convert()"]
    
    EVAL["Evaluate model_i\nA_i = evaluate(model_i, test_data)"]
    
    SENSITIVITY["Record sensitivity\nsens(L_i) = A_fp32 - A_i"]
    
    ALL_DONE{"All layers\nanalyzed?"}
    
    RANK["Rank layers by sensitivity\nHighest drop = Most sensitive"]
    
    THRESHOLD["Apply threshold from config:\nsensitive_layer_threshold = 0.01\n(1% accuracy drop)"]
    
    ASSIGN_FP16["sens(L_i) > threshold\n→ Assign FP16\n(2 bytes/weight)"]
    
    ASSIGN_INT8["sens(L_i) ≤ threshold\n→ Assign INT8\n(1 byte/weight)"]
    
    MIXED["Mixed Precision Model\n- Sensitive layers: FP16 compute\n- Insensitive layers: INT8 compute\n- Estimate size: Σ bytes per layer"]
    
    COMPARE["Compare vs Full INT8:\n- Size reduction achieved\n- Accuracy recovered\n- Latency profile"]

    START --> LAYERS
    LAYERS --> FOR_EACH
    FOR_EACH --> CLONE
    CLONE --> QUANT_LAYER
    QUANT_LAYER --> EVAL
    EVAL --> SENSITIVITY
    SENSITIVITY --> ALL_DONE
    ALL_DONE -->|No| FOR_EACH
    ALL_DONE -->|Yes| RANK
    RANK --> THRESHOLD
    THRESHOLD --> ASSIGN_FP16
    THRESHOLD --> ASSIGN_INT8
    ASSIGN_FP16 --> MIXED
    ASSIGN_INT8 --> MIXED
    MIXED --> COMPARE

    style START fill:#4488ff,color:#fff
    style ASSIGN_FP16 fill:#ffaa44,color:#000
    style ASSIGN_INT8 fill:#ff9900,color:#000
    style MIXED fill:#44aa44,color:#fff
    style COMPARE fill:#884488,color:#fff
```

---

## 5. BatchNorm Folding Workflow

BatchNorm folding eliminates the BatchNorm layer by absorbing its parameters into the preceding Conv layer. This is done before quantization to reduce the number of quantized operations.

```mermaid
flowchart LR
    subgraph BEFORE ["Before Folding"]
        direction TB
        C1["Conv2d\ny = W*x + b"]
        BN1["BatchNorm2d\ny = γ*(x-μ)/σ + β"]
        C1 --> BN1
    end
    
    FOLD["Fold BN into Conv\nW_new = W * γ/σ\nb_new = (b-μ)*γ/σ + β"]
    
    subgraph AFTER ["After Folding"]
        direction TB
        C2["Conv2d (with bias)\ny = W_new*x + b_new"]
        NOTE["BN layer eliminated!\nSame mathematical output\nFewer quantized operations"]
        C2 --> NOTE
    end
    
    BEFORE --> FOLD --> AFTER

    style BEFORE fill:#ffe8e8
    style AFTER fill:#e8ffe8
```

**Mathematical derivation:**
```
BN forward:   y_bn = γ * (conv(x) - μ) / σ + β
Combined:     y = (W * γ/σ) * x + ((b - μ) * γ/σ + β)
```

Benefits:
- Removes one layer of computation (faster inference)
- Reduces number of quantization nodes
- Eliminates BN's division and addition operations from the critical path
