---
title: "CUDA & cuDNN Programming Tutorials"
subtitle: "25 standalone CUDA C++ tutorials covering GPU programming from fundamentals to advanced library usage. Each tutorial targets NVIDIA H200 (sm_90) and implements one concept with ablation experiments, extensive..."
category: technical
project: cuda_programming
project_title: "CUDA & cuDNN Programming Tutorials"
date: 2025-05-29
reading_time: 3
tags:
  - cuda-programming
author: "Shreedhar Kodate"
output: "blogs/technical/posts/cuda_programming/index.html"
---
25 standalone CUDA C++ tutorials covering GPU programming from fundamentals to
advanced library usage. Each tutorial targets NVIDIA H200 (sm_90) and
implements one concept with ablation experiments, extensive logging, and YAML
configuration.

## Project Structure

```
cuda_programming/
├── src/
│   ├── common/
│   │   ├── logger.hpp           # Thread-safe logger (reads level from YAML)
│   │   ├── config_loader.hpp    # yaml-cpp wrapper: get<T>(section, key)
│   │   └── cuda_utils.hpp       # CUDA_CHECK, CUDNN_CHECK, DeviceInfo
│   ├── tutorials/
│   │   ├── 01_threads_blocks_grids/main.cu
│   │   ├── 02_warps_simt/main.cu
│   │   └── ... (25 tutorials total)
│   └── notebooks/
├── configs/
│   ├── global.yaml
│   └── tutorials/
│       └── NN_name.yaml         # one per tutorial
├── docs/
│   ├── plan.md                  # concept groupings and learning path
│   └── tutorials/               # per-tutorial explainers
├── docker/
│   ├── Dockerfile               # nvidia/cuda:13.0.1-devel-ubuntu22.04
│   ├── docker-compose.yml
│   └── .env
├── logs/                        # runtime logs (created at first run)
├── CMakeLists.txt
└── pyproject.toml
```

## Prerequisites

- NVIDIA H200 GPU (sm_90), driver >= 550
- CUDA 13.0.1
- cuDNN 9.x
- cmake >= 3.24
- yaml-cpp development headers
- CUTLASS (for tutorial 17, auto-cloned in Docker)

## Build

### With Docker (recommended)

```bash
cd docker/
docker-compose build
docker-compose run cuda_tutorials bash
```

### Local build

```bash
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCUTLASS_DIR=/path/to/cutlass
cmake --build build --target all_tutorials -j$(nproc)
```

Binaries are placed in `build/bin/`.

## Run a Tutorial

All config via YAML — no CLI arguments:

```bash
# Adjust configs/tutorials/01_threads_blocks_grids.yaml, then:
./build/bin/tutorial_01_threads_blocks_grids

# Logs appear in logs/01_threads_blocks_grids.log and stdout
```

## Tutorials

| # | Topic | Key Concept |
|---|-------|-------------|
| 01 | Threads, Blocks & Grids | Execution hierarchy, vector addition sweep |
| 02 | Warps & SIMT | Warp divergence cost measurement |
| 03 | CUDA Memory Types | Register / shared / constant / global bandwidth |
| 04 | Memory Coalescing | Coalesced vs strided access bandwidth |
| 05 | Shared Memory & Bank Conflicts | Naive vs padded transpose |
| 06 | Thread Synchronization | Prefix sum with/without `__syncthreads()` |
| 07 | Occupancy | `cudaOccupancyMaxActiveBlocksPerMultiprocessor` |
| 08 | CUDA Streams | H2D + compute + D2H overlap |
| 09 | CUDA Events & Timing | GPU events vs CPU chrono |
| 10 | Unified Memory | Demand paging, prefetch, ReadMostly advise |
| 11 | Pinned Memory | Pageable vs pinned H2D/D2H bandwidth |
| 12 | Atomic Operations | Global vs shared-memory histogram |
| 13 | Parallel Reduction | Naive, optimized, warp shuffle variants |
| 14 | Warp Shuffle | `__shfl_down_sync`, broadcast, XOR butterfly |
| 15 | Tiled MatMul | Shared memory tiling vs naive GEMM |
| 16 | Tensor Cores / WMMA | FP16 wmma vs FP32 CUDA cores |
| 17 | CUTLASS | `cutlass::gemm::device::Gemm` template |
| 18 | PTX & SASS | Inline PTX, `#pragma unroll`, register counts |
| 19 | Nsight Profiling | NVTX ranges, Nsight Compute instructions |
| 20 | cuBLAS | `cublasSgemm`, TF32 Tensor Core math mode |
| 21 | cuDNN Descriptors | Handle, tensor, filter, conv descriptors |
| 22 | cuDNN Algorithm Selection | `cudnnFindConvolutionForwardAlgorithmEx` |
| 23 | cuDNN Workspace | Workspace cap vs algorithm fallback |
| 24 | cuDNN Fused Ops | Conv+Bias+ReLU unfused vs fused |
| 25 | CUDA Graphs | Graph capture, instantiate, replay |

## Configuration System

All parameters are read from YAML — no command-line arguments.

```yaml
# configs/global.yaml
logging:
  level: INFO
  file: global.log
device:
  id: 0

# configs/tutorials/01_threads_blocks_grids.yaml
logging:
  level: DEBUG
  file: 01_threads_blocks_grids.log
tutorial:
  N: 1048576
  block_dims: [32, 64, 128, 256, 512]
  num_iterations: 10
```

## Profiling

```bash
# Nsight Systems (timeline view)
nsys profile --trace=cuda,nvtx ./build/bin/tutorial_19_nsight_profiling

# Nsight Compute (kernel metrics)
ncu --set full --target-processes all ./build/bin/tutorial_15_tiled_matmul
```

## Hardware Target

- GPU: NVIDIA H200 SXM (sm_90), 141 GB HBM3
- Peak HBM bandwidth: ~3350 GB/s
- Peak FP16 Tensor Core: ~1979 TFLOPS (dense)
- Compile flag: `-arch=sm_90`