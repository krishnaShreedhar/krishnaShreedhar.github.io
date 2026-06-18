// =============================================================================
// Tutorial 13: Parallel Reduction
//
// Concept:
//   Sum reduction of N elements requires log2(N) steps when parallelized.
//   The three variants illustrate progressively optimized implementations.
//
//   Variant A — Naive (stride 1, warp divergence):
//     step=1:  t0 +=t1,  t2+=t3,  t4+=t5,  ...  (half idle each step)
//     Problem: in each step, half the warps have divergent paths.
//
//   Variant B — Optimized (interleaved, no warp divergence):
//     Active threads are always the first half → no divergence within warps.
//
//   Variant C — Warp shuffle final stage:
//     Use __shfl_down_sync for last 32 elements — no shared memory needed.
//     This eliminates 5 rounds of shared-mem writes/reads.
// =============================================================================

#include <cmath>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "../../common/config_loader.hpp"
#include "../../common/cuda_utils.hpp"
#include "../../common/logger.hpp"

// ---------------------------------------------------------------------------
// Kernel A: Naive stride-1 reduction (divergent warps)
// ---------------------------------------------------------------------------
__global__ void reduce_naive(const float* __restrict__ in,
                              float*       __restrict__ out,
                              int n) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    smem[tid] = (idx < n) ? in[idx] : 0.f;
    __syncthreads();

    // Stride starts at 1, doubles each step.
    // In first step: t0 reads t0+t1, t2 reads t2+t3, t1/t3 idle → divergence
    for (int stride = 1; stride < blockDim.x; stride <<= 1) {
        if (tid % (2 * stride) == 0) {         // half the warps take this branch
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = smem[0];
}

// ---------------------------------------------------------------------------
// Kernel B: Optimized interleaved reduction (no divergence)
// ---------------------------------------------------------------------------
__global__ void reduce_optimized(const float* __restrict__ in,
                                  float*       __restrict__ out,
                                  int n) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    smem[tid] = (idx < n) ? in[idx] : 0.f;
    __syncthreads();

    // Stride starts at half block size, halves each step.
    // Active threads are always the first (stride) threads → no divergence
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = smem[0];
}

// ---------------------------------------------------------------------------
// Warp-level reduction using shuffle (used in Kernel C final stage)
// ---------------------------------------------------------------------------
__device__ inline float warp_reduce_sum(float val) {
    // __shfl_down_sync: each thread receives value from thread (lane + offset)
    // mask 0xffffffff = all 32 lanes participate
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    return val;
}

// ---------------------------------------------------------------------------
// Kernel C: Warp shuffle final stage
// ---------------------------------------------------------------------------
__global__ void reduce_warp_shuffle(const float* __restrict__ in,
                                     float*       __restrict__ out,
                                     int n) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    smem[tid] = (idx < n) ? in[idx] : 0.f;
    __syncthreads();

    // Main reduction in shared memory (down to 32 elements)
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }

    // Final 32 elements: use warp shuffle (no shared mem, no __syncthreads)
    float val = (tid < 32) ? smem[tid] : 0.f;
    val = warp_reduce_sum(val);

    if (tid == 0) out[blockIdx.x] = val;
}

// ---------------------------------------------------------------------------
// Run a two-pass reduction (kernel + CPU final sum) and return result
// ---------------------------------------------------------------------------
static float two_pass_reduce(auto kernel_fn,
                              float* d_in,
                              float* d_partial,
                              int    n,
                              int    block_dim,
                              int    grid_dim,
                              size_t smem) {
    kernel_fn(d_in, d_partial, n, grid_dim, block_dim, smem);
    CUDA_KERNEL_CHECK();

    std::vector<float> h_partial(grid_dim);
    CUDA_CHECK(cudaMemcpy(h_partial.data(), d_partial,
                          grid_dim * sizeof(float), cudaMemcpyDeviceToHost));
    return std::accumulate(h_partial.begin(), h_partial.end(), 0.f);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    const std::string project_root  = PROJECT_ROOT;
    const std::string config_path   = project_root +
        "/configs/tutorials/13_parallel_reduction.yaml";
    const std::string global_config = project_root + "/configs/global.yaml";

    auto cfg = cuda_tutorials::ConfigLoader::from_file(config_path);
    cfg.merge_defaults(global_config);

    auto logger = cuda_tutorials::Logger::create(config_path, project_root,
                                                  "13_parallel_reduction");
    logger.log_info("=== Tutorial 13: Parallel Reduction ===");

    const int device_id = cfg.get<int>("device", "id");
    const int N         = cfg.get<int>("tutorial", "N");
    const int block_dim = cfg.get<int>("tutorial", "block_size");
    const int num_iter  = cfg.get<int>("tutorial", "num_iterations");

    CUDA_CHECK(cudaSetDevice(device_id));
    auto dev_info = cuda_tutorials::get_device_info(device_id);
    cuda_tutorials::print_device_info(logger, dev_info);

    std::ostringstream oss;
    oss << "Config: N=" << N << "  block_size=" << block_dim << "  num_iter=" << num_iter;
    logger.log_info(oss.str());

    std::vector<float> h_in(N);
    for (int i = 0; i < N; ++i) h_in[i] = 1.0f;
    float cpu_sum = static_cast<float>(N);  // all ones

    float *d_in{}, *d_partial{};
    const int grid_dim = (N + block_dim - 1) / block_dim;
    CUDA_CHECK(cudaMalloc(&d_in,      N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_partial, grid_dim * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    const size_t smem = block_dim * sizeof(float);
    cuda_tutorials::CudaEvent ev_start, ev_stop;

    struct Variant { const char* name; float result; float ms; };
    std::vector<Variant> variants;

    auto bench_variant = [&](auto launch_fn, const char* name) {
        // Warm-up + correctness
        launch_fn(d_in, d_partial, N, grid_dim, block_dim, smem);
        CUDA_KERNEL_CHECK();
        std::vector<float> hp(grid_dim);
        CUDA_CHECK(cudaMemcpy(hp.data(), d_partial, grid_dim * sizeof(float), cudaMemcpyDeviceToHost));
        float result = std::accumulate(hp.begin(), hp.end(), 0.f);

        // Timing
        ev_start.record();
        for (int i = 0; i < num_iter; ++i)
            launch_fn(d_in, d_partial, N, grid_dim, block_dim, smem);
        ev_stop.record();
        CUDA_CHECK(cudaEventSynchronize(ev_stop.event));
        float t = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&t, ev_start.event, ev_stop.event));
        float ms = t / num_iter;

        double bw = static_cast<double>(N) * sizeof(float) / (ms * 1e-3) / 1e9;
        bool ok = std::abs(result - cpu_sum) / cpu_sum < 1e-4f;

        oss.str("");
        oss << name << ": " << std::fixed << std::setprecision(3) << ms
            << " ms  " << std::setprecision(2) << bw << " GB/s"
            << "  result=" << result << (ok ? "  CORRECT" : "  WRONG");
        logger.log_info(oss.str());
        variants.push_back({name, result, ms});
    };

    bench_variant([](float* a, float* b, int n, int g, int bd, size_t s){
        reduce_naive<<<g, bd, s>>>(a, b, n);
    }, "Naive (divergent)       ");

    bench_variant([](float* a, float* b, int n, int g, int bd, size_t s){
        reduce_optimized<<<g, bd, s>>>(a, b, n);
    }, "Optimized (no diverge)  ");

    bench_variant([](float* a, float* b, int n, int g, int bd, size_t s){
        reduce_warp_shuffle<<<g, bd, s>>>(a, b, n);
    }, "Warp shuffle final      ");

    if (variants.size() == 3) {
        oss.str("");
        oss << "Optimized vs Naive     : "
            << std::fixed << std::setprecision(2)
            << (variants[0].ms / variants[1].ms) << "x speedup";
        logger.log_info(oss.str());
        oss.str("");
        oss << "Warp shuffle vs Naive  : "
            << (variants[0].ms / variants[2].ms) << "x speedup";
        logger.log_info(oss.str());
    }

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_partial));

    logger.log_info("Tutorial 13 complete.");
    return 0;
}
