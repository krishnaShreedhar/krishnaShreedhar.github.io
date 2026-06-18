# Deep Learning Quantization Concepts

## 1. Quantization Fundamentals

Quantization maps a continuous floating-point tensor to a discrete integer representation. The core mathematical transformation is:

```
x_int = clamp(round(x_float / scale) + zero_point, quant_min, quant_max)
x_dequant = (x_int - zero_point) * scale
```

**Parameters:**
- `scale` — step size between adjacent quantization levels
- `zero_point` — integer value that represents 0.0 in float space
- `quant_min` / `quant_max` — integer range bounds (e.g., -128 to 127 for INT8)

```mermaid
flowchart LR
    FP32["FP32 Tensor\nx ∈ ℝ"]
    SCALE["Compute scale & zero_point\nscale = range / (2^bits - 1)\nzero_point = round(-min/scale)"]
    QUANT["Quantize\nx_int = round(x/scale) + zp\nclamp to [qmin, qmax]"]
    INT8["INT8 Tensor\nx_int ∈ {-128..127}"]
    DEQUANT["Dequantize\nx_fp = (x_int - zp) × scale"]
    OUT["Approx FP32\nx̂ ∈ ℝ (discretized)"]

    FP32 --> SCALE
    SCALE --> QUANT
    FP32 --> QUANT
    QUANT --> INT8
    INT8 --> DEQUANT
    DEQUANT --> OUT

    style INT8 fill:#ff9900,color:#000
    style FP32 fill:#4488ff,color:#fff
    style OUT fill:#44aa44,color:#fff
```

**Quantization Error:** `ε = x_float - x_dequant`

The maximum error per value is `scale/2` (half a quantization step). Minimizing `scale` (choosing a tighter range) reduces error but increases clipping risk.

---

## 2. PTQ vs QAT Workflow Comparison

```mermaid
flowchart TD
    subgraph PTQ ["Post-Training Quantization (PTQ)"]
        direction TB
        P1["Train FP32 Model\n(standard training)"]
        P2["Collect calibration data\n(100–1000 representative samples)"]
        P3["Run calibration data through model\n(observers record min/max/histogram)"]
        P4["Compute scale & zero_point\nper layer (or per channel)"]
        P5["Convert to INT8\ntorch.quantization.convert()"]
        P6["Deploy INT8 model\n(no further training needed)"]

        P1 --> P2
        P2 --> P3
        P3 --> P4
        P4 --> P5
        P5 --> P6
    end

    subgraph QAT ["Quantization-Aware Training (QAT)"]
        direction TB
        Q1["Train FP32 Model\n(or start from pretrained)"]
        Q2["Insert FakeQuantize nodes\ntorch.quantization.prepare_qat()"]
        Q3["Fine-tune with QAT\n(forward: simulated INT8)\n(backward: STE gradients)"]
        Q4["Model adapts weights\nto quantization noise"]
        Q5["Remove FakeQuantize nodes\ntorch.quantization.convert()"]
        Q6["Deploy INT8 model\n(higher accuracy than PTQ)"]

        Q1 --> Q2
        Q2 --> Q3
        Q3 --> Q4
        Q4 --> Q5
        Q5 --> Q6
    end

    style PTQ fill:#e8f4f8
    style QAT fill:#f8f4e8
```

| Property | PTQ | QAT |
|----------|-----|-----|
| Training required | None (only calibration) | Yes (fine-tuning) |
| Calibration data | 100–1000 samples | Full training dataset |
| Accuracy drop | 0.5–2% typical | 0.1–0.5% typical |
| Time to quantize | Minutes | Hours |
| Best for | Quick deployment | Production accuracy |

---

## 3. Calibration Techniques Decision Tree

```mermaid
flowchart TD
    START["Need to calibrate quantization range?"]
    Q_DATA{"Calibration\ndata available?"}
    Q_OUTLIERS{"Are there\noutliers in\nactivations?"}
    Q_SPEED{"Speed vs\naccuracy\ntrade-off?"}

    MM["Min-Max Calibrator\nrange = [min(x), max(x)]\n✓ Simple, exact\n✗ Outlier-sensitive"]
    PCT["Percentile Calibrator\nrange = [p%, (100-p)%]\n✓ Robust to outliers\n✗ Requires tuning p"]
    MSE["MSE Calibrator\nOptimize: argmin_α MSE(x, Q(αx))\n✓ Minimizes quantization error\n✗ Grid search overhead"]
    KL["KL-Divergence Calibrator\nMinimize KL(P_fp32 || P_int8)\n✓ Best for activations\n✗ Most computationally expensive"]
    DYN["Dynamic Quantization\nNo calibration needed!\nActivations quantized at runtime"]

    START --> Q_DATA
    Q_DATA -->|No| DYN
    Q_DATA -->|Yes| Q_OUTLIERS
    Q_OUTLIERS -->|No| MM
    Q_OUTLIERS -->|Yes| Q_SPEED
    Q_SPEED -->|Speed| PCT
    Q_SPEED -->|Accuracy| MSE
    Q_SPEED -->|Best accuracy| KL

    style MM fill:#ddeeFF
    style PCT fill:#ddffd4
    style MSE fill:#fff4dd
    style KL fill:#ffd4d4
    style DYN fill:#f4ddff
```

---

## 4. Mixed-Precision Sensitivity Analysis Flow

```mermaid
flowchart TD
    FP32["FP32 Baseline Model\nAccuracy = A_baseline"]
    
    LOOP["For each quantizable layer L_i:"]
    
    SINGLE["Quantize ONLY layer L_i to INT8\n(all other layers stay FP32)"]
    
    MEASURE["Measure accuracy A_i\nwith layer L_i quantized"]
    
    DROP["sensitivity(L_i) = A_baseline - A_i\n(accuracy drop from one-layer quantization)"]
    
    RANK["Rank all layers by sensitivity\n(high drop = sensitive layer)"]
    
    THRESH{"sensitivity(L_i)\n> threshold?"}
    
    FP16["Assign FP16 to L_i\n(keep high precision)"]
    INT8["Assign INT8 to L_i\n(quantize for speed/size)"]
    
    MIXEDPREC["Mixed Precision Model\nSensitive layers → FP16\nInsensitive layers → INT8"]
    
    FP32 --> LOOP
    LOOP --> SINGLE
    SINGLE --> MEASURE
    MEASURE --> DROP
    DROP --> LOOP
    LOOP --> RANK
    RANK --> THRESH
    THRESH -->|Yes| FP16
    THRESH -->|No| INT8
    FP16 --> MIXEDPREC
    INT8 --> MIXEDPREC

    style FP32 fill:#4488ff,color:#fff
    style FP16 fill:#ffaa44,color:#000
    style INT8 fill:#ff9900,color:#000
    style MIXEDPREC fill:#44aa44,color:#fff
```

---

## 5. QAT Training Loop with Fake Quantization

```mermaid
sequenceDiagram
    participant Data as Training Data
    participant FQ as FakeQuantize Node
    participant Model as Model Layers
    participant Loss as Loss Function
    participant Optimizer as Optimizer (Adam/SGD)

    loop Each Training Batch
        Data ->> FQ: x_float (FP32 input)
        Note over FQ: Forward: quantize then dequantize<br/>x_q = round(x/scale)*scale<br/>(stays in FP32, on INT8 grid)
        FQ ->> Model: x_q (simulated INT8 values)
        Model ->> Loss: predictions (FP32)
        Loss ->> Optimizer: loss value
        Note over Optimizer: Backward: STE approximation<br/>∂L/∂x ≈ ∂L/∂x_q (identity)<br/>Gradient passes through FakeQuantize
        Optimizer ->> Model: update FP32 weights
        Note over Model: Weights adapt to minimize<br/>loss under INT8 constraints
    end

    Model ->> Model: convert() → INT8<br/>Remove FakeQuantize nodes<br/>Deploy with real INT8 arithmetic
```

---

## 6. Quantization Format Reference Table

| Format | Bits | Range | Use Case | Precision Loss |
|--------|------|-------|----------|----------------|
| FP32 | 32 | ±3.4×10^38 | Training, baseline | None (reference) |
| FP16 | 16 | ±65504 | Mixed-precision training | Low |
| BF16 | 16 | ±3.4×10^38 | Large model training (TPUs) | Low (wider range than FP16) |
| FP8 (E4M3) | 8 | ±448 | Training on H100 GPUs | Medium |
| FP8 (E5M2) | 8 | ±57344 | Gradient quantization | Medium |
| INT8 | 8 | -128 to 127 | Inference (PTQ/QAT) | Medium |
| INT4 | 4 | -8 to 7 | Aggressive compression, LLMs | High |
| NF4 | 4 | Normal float levels | QLoRA fine-tuning | High (structured) |
| UINT8 | 8 | 0 to 255 | Post-ReLU activations | Medium |

**Notes:**
- BF16 has the same exponent range as FP32 (good for training stability) but only 7 mantissa bits.
- NF4 (Normal Float 4) uses quantization levels optimally spaced for normally distributed weights (used in bitsandbytes / QLoRA).
- FP8 is hardware-native on NVIDIA H100 and newer.

---

## 7. Quantization Granularity Levels

| Granularity | Scope | Scale Tensors | Pros | Cons |
|-------------|-------|---------------|------|------|
| Per-tensor | Entire tensor | 1 scalar | Fastest, smallest overhead | Worst quality (one scale fits all) |
| Per-channel | Per output channel of Conv/Linear | C scalars (C = out_channels) | Good quality for weights | C times more scale storage |
| Per-group | Groups of N elements | tensor / group_size scales | Flexible, used in GPTQ/AWQ | More complex dequant |
| Per-token | Per sequence position (LLMs) | seq_len scalars | Handles token distribution variance | Only for transformers |
| Per-row | Each row of a matrix | rows scalars | Good for Linear in LLMs | More overhead than per-tensor |

**Recommendation:**
- **Weights**: Per-channel quantization (dramatically better than per-tensor for Conv layers)
- **Activations**: Per-tensor quantization (per-channel activations are expensive at runtime)
- **LLM weights**: Per-group quantization (group size 64 or 128 is common in GPTQ/AWQ)

---

## 8. SQNR Formula and Interpretation

Signal-to-Quantization-Noise Ratio measures how much of the original signal is preserved:

```
SQNR = 10 × log₁₀( E[x²] / E[(x - x̂)²] )
```

Where:
- `x` = original FP32 tensor
- `x̂` = dequantized tensor (FP32 values on INT8 grid)
- `E[x²]` = signal power
- `E[(x - x̂)²]` = noise power (quantization error power)

**Reference values:**
| SQNR | Interpretation |
|------|---------------|
| > 50 dB | Excellent — virtually no perceptible quality loss |
| 40–50 dB | Good — typical INT8 quantization quality |
| 30–40 dB | Acceptable — some accuracy degradation |
| < 30 dB | Poor — significant accuracy drop expected |
| −∞ dB | Complete signal destruction |

**Theoretical INT8 SQNR** (symmetric, uniform, Gaussian input): ~49.9 dB
**Theoretical INT4 SQNR**: ~25.8 dB
