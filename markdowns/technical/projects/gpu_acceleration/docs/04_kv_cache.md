# Module 4 — KV-Cache from Scratch

## Overview

This module builds a KV-cache from first principles and measures the
throughput improvement it delivers during autoregressive inference.

| File | Teaches |
|------|---------|
| `attention.py` | Naive MHA (full K/V recompute) vs cached MHA |
| `kv_cache_manager.py` | Pre-allocated GPU buffer for K/V tensors |
| `inference_engine.py` | Prefill + decode phases, throughput benchmark |

---

## The Problem: Autoregressive Decode is Expensive

To generate token `t`, the model needs to attend to **all previous tokens** `0…t-1`.
Without caching:

```
Step 1: attend over [t₀]              — 1 token
Step 2: attend over [t₀, t₁]         — 2 tokens
Step 3: attend over [t₀, t₁, t₂]     — 3 tokens
...
Step T: attend over [t₀, …, t_{T-1}] — T tokens
```

Total K/V computation: **O(T²)** — quadratic in sequence length.

The KV-cache reuses previously computed K and V tensors, reducing per-step
cost to **O(T)** amortized.

---

## KV-Cache Mechanics

### Without Cache

```mermaid
sequenceDiagram
    participant Model
    participant GPU

    note over Model,GPU: Decode step t=5 (history=[t0..t4])

    Model->>GPU: Q = x_t5 · Wq
    Model->>GPU: K = [t0..t5] · Wk  ← recomputed every step!
    Model->>GPU: V = [t0..t5] · Wv  ← recomputed every step!
    GPU->>GPU: Attention(Q, K, V)
    GPU-->>Model: output token t5
```

### With Cache

```mermaid
sequenceDiagram
    participant Model
    participant KVCache as "KV Cache (GPU buffer)"
    participant GPU

    note over Model,GPU: Decode step t=5

    Model->>GPU: Q = x_t5 · Wq
    Model->>GPU: K_new = x_t5 · Wk  ← only new token!
    Model->>GPU: V_new = x_t5 · Wv  ← only new token!
    Model->>KVCache: append K_new at position 5
    Model->>KVCache: append V_new at position 5
    KVCache-->>GPU: K[0..5], V[0..5]  ← full history from cache
    GPU->>GPU: Attention(Q, K[0..5], V[0..5])
    GPU-->>Model: output token t5
```

**Savings**: For T=512, cached avoids **512× recomputation** of K/V at the final step.

---

## KV-Cache Data Structure

```mermaid
flowchart TD
    subgraph "KVCacheManager (GPU)"
        K["k_cache\n[num_layers, B, H, max_seq_len, head_dim]"]
        V["v_cache\n[num_layers, B, H, max_seq_len, head_dim]"]
    end

    L0["Layer 0"] -->|"update(0, k_new, v_new, pos)"| K
    L0 --> V
    L1["Layer 1"] -->|"update(1, k_new, v_new, pos)"| K
    L1 --> V

    K -->|"read k_cache[0, :, :, :pos]"| ATT0["Attention Layer 0"]
    V --> ATT0
```

### Memory Cost

For a 6-layer model, 8 heads, 64 head_dim, batch=4, max_seq=1024 (BF16):

```
2 (K+V) × 6 × 4 × 8 × 1024 × 64 × 2 bytes = 48 MB
```

On H200 (141 GB), this is negligible. Scaling to 70B LLMs with 32k context
requires careful management (paged attention, etc.), but the principle is identical.

---

## Two-Phase Generation

```mermaid
flowchart LR
    subgraph Prefill
        P["Prompt tokens\n[t0..t_{n-1}]"] -->|"forward (full seq)"| KVC["Populate KV cache\nfor all prompt tokens"]
    end

    subgraph Decode
        KVC --> D1["Step 1: new token t_n"]
        D1 --> KVC2["Append to cache"]
        KVC2 --> D2["Step 2: new token t_{n+1}"]
        D2 --> D3["..."]
        D3 --> DN["Step T: token t_{n+T-1}"]
    end
```

- **Prefill** processes the entire prompt in one batched forward pass — high GPU utilization.
- **Decode** processes one token at a time — low arithmetic intensity (memory-bound).

This is why decode throughput is limited by HBM bandwidth, not FLOPS.
H200's 3.35 TB/s HBM makes it ~4× faster at decode than A100.

---

## Benchmark Results (expected on 2× H200)

| Strategy | Prompt=64, Generate=128 | Speedup |
|----------|-------------------------|---------|
| Naive (recompute) | ~800 tok/s | 1× |
| KV-cached | ~15,000 tok/s | ~19× |

Actual numbers depend on model size, batch size, and dtype.

---

## Running

```bash
python -m src.kv_cache.inference_engine
```

### Key config knobs (`configs/kv_cache.yaml`)

| Key | Effect |
|-----|--------|
| `kv_cache.max_batch_size` | Pre-allocation batch dimension |
| `kv_cache.max_seq_len` | Maximum sequence length the cache supports |
| `kv_cache.dtype` | `bfloat16` halves cache memory vs `float32` |
| `inference.prompt_tokens` | Prefill length |
| `inference.max_new_tokens` | Decode length |
| `inference.benchmark_iters` | Runs to average for stable throughput numbers |
| `logging.level` | `DEBUG` shows per-step timing |
