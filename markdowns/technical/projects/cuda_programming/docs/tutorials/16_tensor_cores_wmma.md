---
title: "Tutorial 16: Tensor Cores & WMMA"
subtitle: "See `src/tutorials/16_tensor_cores_wmma/main.cu` for detailed inline comments explaining each CUDA concept, all API calls, and the ablation experiments."
category: technical
project: cuda_programming
project_title: "CUDA & cuDNN Programming Tutorials"
date: 2025-12-17
reading_time: 1
tags:
  - cuda-programming
  - docs
author: "Shreedhar Kodate"
output: "blogs/technical/posts/cuda_programming/docs/tutorials/16_tensor_cores_wmma.html"
---
## Concept

See `src/tutorials/16_tensor_cores_wmma/main.cu` for detailed inline comments explaining
each CUDA concept, all API calls, and the ablation experiments.

## Key APIs

Refer to the source file for full API usage and inline explanations.

## Execution Flow

```mermaid
flowchart TD
    A[main: load config + init logger] --> B[cudaSetDevice + print_device_info]
    B --> C[Allocate host and device memory]
    C --> D[Run kernel variants / ablations]
    D --> E[CUDA event timing per variant]
    E --> F[Log results: time, bandwidth, correctness]
    F --> G[Cleanup: cudaFree + handle destroy]
    G --> H[Exit 0]
```

## Ablation Results (placeholder — fill after running on H200)

| Variant | Mean Time (ms) | Bandwidth / GFLOPS | Notes |
|---------|---------------|-------------------|-------|
| TBD     | TBD           | TBD               | TBD   |

## What to Observe in Nsight Compute

Run:
```bash
ncu --set full ./build/bin/tutorial_16_tensor_cores_wmma
```

Key metrics:
- `sm__throughput.avg.pct_of_peak_sustained_active` — SM utilization
- `dram__bytes.sum` — HBM traffic
- `smsp__warp_issue_stalled_long_scoreboard_per_warp` — memory stalls
- `l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum` — L1/global load bytes