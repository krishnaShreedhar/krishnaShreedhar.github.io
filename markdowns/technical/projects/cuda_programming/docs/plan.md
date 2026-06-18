# CUDA & cuDNN Tutorial Implementation Plan

## Overview

25 standalone tutorials covering the full GPU programming stack, from fundamental
execution model concepts through advanced library usage and profiling.
Each tutorial is a self-contained CUDA C++ program built from YAML config,
with extensive logging and ablation experiments.

## Concept Groupings

### Group 1: Execution Model (Tutorials 01–02)

**Motivation**: Before optimizing anything, understand how work is organized on the GPU.

| # | Concept | Key Insight |
|---|---------|-------------|
| 01 | Threads, Blocks & Grids | Hierarchy maps to hardware (threads → SM lanes, blocks → SM) |
| 02 | Warps & SIMT | 32 threads execute lockstep; divergence serializes both paths |

**Learning path**: Start here. These fundamentals affect every optimization decision.

**Code structure**: Vector addition (01) and branch-cost measurement (02).

---

### Group 2: Memory Hierarchy (Tutorials 03–05)

**Motivation**: GPU performance is nearly always memory-bound. The memory hierarchy
determines achievable bandwidth and latency.

| # | Concept | Key Insight |
|---|---------|-------------|
| 03 | Memory Types | Register > Shared > L2 > Global; know when to use each |
| 04 | Memory Coalescing | 32-thread warp should issue one 128-byte transaction |
| 05 | Bank Conflicts | 32 shared-memory banks; same bank from two threads = serialized |

**Learning path**: Understand access patterns before writing any kernel.
Measure bandwidth for each memory type. Apply coalescing and bank-conflict
analysis to your kernels.

---

### Group 3: Concurrency & Synchronization (Tutorials 06–09)

**Motivation**: GPUs hide latency through massive parallelism, but coordination
is required for correctness and performance measurement.

| # | Concept | Key Insight |
|---|---------|-------------|
| 06 | Thread Synchronization | `__syncthreads()` is a block barrier; warps within a block can diverge |
| 07 | Occupancy | More resident warps = more latency hiding; not always the bottleneck |
| 08 | CUDA Streams | Overlap H2D copy, kernel, D2H copy across separate copy engines |
| 09 | Events & Timing | CPU chrono includes launch overhead; GPU events give kernel-only time |

---

### Group 4: Memory Allocation Strategies (Tutorials 10–11)

**Motivation**: How memory is allocated affects H2D/D2H bandwidth and page-fault
behavior.

| # | Concept | Key Insight |
|---|---------|-------------|
| 10 | Unified Memory | Demand paging → page faults; prefetch + advise eliminates most |
| 11 | Pinned Memory | Skips bounce buffer; doubles H2D/D2H bandwidth on average |

---

### Group 5: Parallel Patterns (Tutorials 12–14)

**Motivation**: Core algorithmic building blocks for GPU programming.

| # | Concept | Key Insight |
|---|---------|-------------|
| 12 | Atomic Operations | Global atomics serialize; private smem histogram then merge |
| 13 | Parallel Reduction | Naive → optimized (no divergence) → warp shuffle |
| 14 | Warp Shuffle | Register-level exchange; no shared mem, no bank conflicts |

---

### Group 6: GEMM Optimization (Tutorials 15–17)

**Motivation**: GEMM is the core operation of deep learning. Understanding the
optimization stack from naive → tiled → Tensor Core → CUTLASS → cuBLAS
builds intuition for all matrix ops.

| # | Concept | Key Insight |
|---|---------|-------------|
| 15 | Tiled MatMul | Shared memory tiles reduce global memory traffic by TILE× |
| 16 | Tensor Cores WMMA | FP16 MMA in warp instruction; 5–10× FP32 CUDA cores |
| 17 | CUTLASS | Template-level control over tile sizes, epilogue, pipelining |

---

### Group 7: Advanced & Profiling (Tutorials 18–19, 25)

**Motivation**: Production-quality GPU code requires low-level control and
systematic profiling.

| # | Concept | Key Insight |
|---|---------|-------------|
| 18 | PTX & SASS | Inspect compiler output; `#pragma unroll` + inline PTX for ILP |
| 19 | Nsight Profiling | NVTX annotations; key Nsight Compute metrics |
| 25 | CUDA Graphs | Amortize launch overhead; single graph launch replaces N API calls |

---

### Group 8: CUDA Libraries (Tutorials 20–24)

**Motivation**: Production code uses cuBLAS and cuDNN rather than hand-written kernels.
Understanding the descriptor/algorithm/workspace model enables correct usage
and systematic performance tuning.

| # | Concept | Key Insight |
|---|---------|-------------|
| 20 | cuBLAS | `cublasSgemm`, TF32 math mode, column-major convention |
| 21 | cuDNN Descriptors | Handle, tensor, filter, conv descriptors; layout matters |
| 22 | cuDNN Algorithm Selection | `cudnnFindConvolutionForwardAlgorithm` benchmarks all algos |
| 23 | cuDNN Workspace | More workspace = faster algorithm; explicit budget control |
| 24 | cuDNN Fused Ops | Conv+Bias+ReLU in one kernel; halves bandwidth for this pattern |

---

## Build Order

All tutorials are independent. Build with:
```bash
cmake -S . -B build -DCUTLASS_DIR=/workspace/cutlass
cmake --build build --target all_tutorials -j$(nproc)
```

## Configuration System

All parameters flow from YAML — no CLI arguments.
```
configs/global.yaml                      # shared defaults
configs/tutorials/NN_name.yaml           # per-tutorial overrides
```

Config loading uses `ConfigLoader::from_file()` with `merge_defaults()`.

## Logging

All output goes to `logs/<tutorial_name>.log` and stdout simultaneously.
Log level is controlled per-tutorial via the YAML `logging.level` key.
Format: `[YYYY-MM-DD HH:MM:SS.mmm] [LEVEL  ] [tutorial_name] message`

## Expected Hardware

- GPU: NVIDIA H200 (sm_90) — 1 or 2 GPUs
- Memory: 80–141 GB HBM3 per GPU
- Peak FP16 Tensor Core: ~1979 TFLOPS (dense)
- Peak HBM bandwidth: ~3350 GB/s
