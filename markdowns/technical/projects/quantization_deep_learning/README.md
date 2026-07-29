---
title: "Deep Learning Quantization"
subtitle: "A comprehensive project illustrating deep learning quantization concepts through minimal working code examples, visualizations, mermaid diagrams, and extensive documentation."
category: technical
project: quantization_deep_learning
project_title: "Deep Learning Quantization"
date: 2025-12-22
reading_time: 4
tags:
  - quantization-deep-learning
author: "Shreedhar Kodate"
output: "blogs/technical/posts/quantization_deep_learning/index.html"
---
A comprehensive project illustrating deep learning quantization concepts through minimal working code examples, visualizations, mermaid diagrams, and extensive documentation.

## Overview

Quantization reduces model precision from 32-bit floating point (FP32) to lower-bit representations (INT8, INT4, FP16, etc.), achieving:

- **2-4x model size reduction**
- **2-4x inference speedup** on supported hardware
- **Minimal accuracy degradation** when done correctly

This project covers four core quantization techniques:

| Technique | Description | When to Use |
|-----------|-------------|-------------|
| Post-Training Quantization (PTQ) | Quantize after training, no retraining | Fastest, baseline accuracy drop |
| Quantization-Aware Training (QAT) | Simulate quantization during training | Best accuracy, requires training |
| Calibration | Determine optimal quantization ranges | Part of PTQ pipeline |
| Mixed Precision | Different precision per layer | Sensitive layers stay at FP16 |

## Project Structure

```
quantization_deep_learning/
    src/
        ptq/                    # Post-Training Quantization implementations
            static_quantization.py     # Static PTQ with calibration
            dynamic_quantization.py    # Dynamic PTQ for LSTM/Linear
            weight_analysis.py         # Weight distribution visualization
        qat/                    # Quantization-Aware Training
            qat_trainer.py             # QAT training loop
            fake_quantization.py       # Fake quantization node demo
            ste_demo.py                # Straight-Through Estimator demo
        calibration/            # Calibration techniques
            calibrators.py             # Min-Max, Percentile, MSE, KL-Div calibrators
            range_analyzer.py          # Calibration range analysis and visualization
        evaluation/             # Benchmarking and analysis
            benchmark.py               # Latency, throughput, size benchmarks
            error_metrics.py           # SQNR, cosine similarity, MSE metrics
            sensitivity_analysis.py    # Per-layer sensitivity and mixed-precision
        notebooks/
            quantization_demo.ipynb    # End-to-end demonstration notebook
    docs/
        concepts.md             # Quantization theory with mermaid diagrams
        flow_diagrams.md        # Quantization workflow diagrams
    docker/
        Dockerfile
        docker-compose.yml
        requirements.txt
    config.yaml                 # All hyperparameters and constants
    pyproject.toml
```

## Quantization Concepts

### 1. Quantization Fundamentals

Quantization maps a continuous range of floating-point values to a discrete set of integers:

```
x_int = round(x_float / scale) + zero_point
x_dequant = (x_int - zero_point) * scale
```

Where:
- `scale`: step size between quantized levels
- `zero_point`: integer value corresponding to 0.0 in float
- **Symmetric**: zero_point = 0, scale = max(|x|) / 127
- **Asymmetric**: zero_point != 0, covers [min, max] range

### 2. Post-Training Quantization (PTQ)

PTQ quantizes a trained FP32 model without additional training:

**Static PTQ**:
1. Collect calibration data statistics
2. Compute scale/zero_point per layer
3. Convert model to INT8 operators

**Dynamic PTQ**:
- Weights quantized statically at conversion time
- Activations quantized dynamically at runtime
- Best for LSTM, Linear layers with variable input ranges

### 3. Quantization-Aware Training (QAT)

QAT simulates quantization during training using fake quantization nodes:

1. Insert `FakeQuantize` nodes in the computation graph
2. Forward pass uses quantized values (quantize → dequantize)
3. Backward pass uses Straight-Through Estimator (STE)
4. Model learns to minimize quantization error
5. Convert to INT8 after training

### 4. Calibration Techniques

| Method | Description | Pros | Cons |
|--------|-------------|------|------|
| Min-Max | Range = [min(x), max(x)] | Simple, exact range | Sensitive to outliers |
| Percentile | Range = [p%, (1-p)%] | Robust to outliers | Requires tuning |
| MSE | Minimize MSE between FP32/INT8 | Optimal for Gaussian | Computationally expensive |
| KL-Divergence | Minimize KL(FP32 \|\| INT8) | Best for activations | Most complex |

### 5. Sensitivity Analysis and Mixed Precision

Not all layers are equally sensitive to quantization. The workflow:

1. Baseline: measure FP32 accuracy
2. Per-layer: quantize one layer at a time, measure accuracy drop
3. Rank: sort layers by sensitivity (accuracy drop)
4. Assign: sensitive layers (drop > threshold) → FP16, rest → INT8

## Usage

All configuration is in `config.yaml`. No CLI arguments needed.

### Running PTQ

```python
# src/ptq/static_quantization.py
python src/ptq/static_quantization.py
```

This will:
1. Train a small CNN on synthetic data
2. Apply static PTQ with calibration
3. Compare FP32 vs INT8 model size and inference time
4. Save outputs to `outputs/models/`

### Running QAT

```python
python src/qat/qat_trainer.py
```

### Running Calibration Analysis

```python
python src/calibration/calibrators.py
python src/calibration/range_analyzer.py
```

### Running Evaluation

```python
python src/evaluation/benchmark.py
python src/evaluation/sensitivity_analysis.py
```

### Notebook Demo

```bash
jupyter notebook src/notebooks/quantization_demo.ipynb
```

## Docker Usage

```bash
cd docker
docker compose up --build
```

This starts a Jupyter Lab server at `http://localhost:8888`.

## Key Results (Expected)

| Metric | FP32 | PTQ INT8 | QAT INT8 |
|--------|------|----------|----------|
| Model Size | 1.0x | ~0.25x | ~0.25x |
| Inference Latency | 1.0x | ~0.5-0.7x | ~0.5-0.7x |
| Accuracy | baseline | -0.5 to -2% | -0.1 to -0.5% |
| SQNR | inf | ~35-45 dB | ~45-55 dB |

## References

- PyTorch Quantization Documentation: https://pytorch.org/docs/stable/quantization.html
- Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference (Jacob et al., 2018)
- Data-Free Quantization Through Weight Equalization and Bias Correction (Nagel et al., 2019)
- ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers (Yao et al., 2022)