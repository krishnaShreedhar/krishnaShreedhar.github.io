// =============================================================================
// Tutorial 03: CUDA Memory Types
//
// Concept:
//   CUDA exposes several memory spaces with different bandwidth, latency,
//   capacity, and scope characteristics:
//
//   Memory Type     | Speed    | Scope       | Lifetime   | Capacity
//   --------------- | -------- | ----------- | ---------- | --------
//   Registers       | fastest  | per thread  | kernel     | ~255 regs/thread
//   Shared memory   | ~L1      | per block   | kernel     | 48–228 KB/SM
//   L2 cache        | ~TB/s    | device-wide | automatic  | 40–50 MB (H200)
//   Global memory   | HBM3     | device-wide | allocation | 80–141 GB (H200)
//   Constant memory | cached   | device-wide | allocation | 64 KB
//   Local memory    | global   | per thread  | kernel     | spills only
//
// Experiment:
//   We benchmark effective bandwidth for:
//     (a) Global memory — large array copy
//     (b) Shared memory — load into smem, do work, store back
//     (c) Constant memory — broadcast read from __constant__ array
//     (d) Register — trivial intra-register accumulation (throughput)
//
//   Access patterns: sequential / strided / random
// =============================================================================

#include <algorithm>
#include <cmath>
#include <cstdlib>
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
// Constant memory (max 64 KB = 16 K floats)
// ---------------------------------------------------------------------------
static constexpr int kConstSize = 1024;
__constant__ float d_const[kConstSize];

// ---------------------------------------------------------------------------
// Kernel: global memory sequential copy (bandwidth benchmark)
// ---------------------------------------------------------------------------
__global__ void global_copy_sequential(const float* __restrict__ src,
                                       float*       __restrict__ dst,
                                       int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = src[idx];
}

// ---------------------------------------------------------------------------
// Kernel: global memory strided access (non-coalesced)
// ---------------------------------------------------------------------------
__global__ void global_copy_strided(const float* __restrict__ src,
                                    float*       __restrict__ dst,
                                    int n,
                                    int stride) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * stride;
    if (idx < n) dst[idx] = src[idx];
}

// ---------------------------------------------------------------------------
// Kernel: global memory random access via index permutation
// ---------------------------------------------------------------------------
__global__ void global_copy_random(const float*  __restrict__ src,
                                   float*        __restrict__ dst,
                                   const int*    __restrict__ perm,
                                   int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = src[perm[idx]];
}

// ---------------------------------------------------------------------------
// Kernel: shared memory — load tile, sum, store
// ---------------------------------------------------------------------------
__global__ void shared_mem_bandwidth(const float* __restrict__ src,
                                     float*       __restrict__ dst,
                                     int n) {
    extern __shared__ float smem[];  // size = blockDim.x * sizeof(float)

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Load from global into shared
    if (idx < n) smem[tid] = src[idx];
    __syncthreads();

    // Trivial transform in shared memory
    smem[tid] = smem[tid] * 1.0f;
    __syncthreads();

    // Store from shared back to global
    if (idx < n) dst[idx] = smem[tid];
}

// ---------------------------------------------------------------------------
// Kernel: constant memory broadcast read
// ---------------------------------------------------------------------------
__global__ void constant_mem_read(float*       __restrict__ dst,
                                  int n,
                                  int const_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // All threads in a warp read the SAME constant address → broadcast
        dst[idx] = d_const[idx % const_size];
    }
}

// ---------------------------------------------------------------------------
// Kernel: register arithmetic (compute-bound, zero memory traffic after load)
// ---------------------------------------------------------------------------
__global__ void register_compute(const float* __restrict__ src,
                                 float*       __restrict__ dst,
                                 int n,
                                 int iters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Load into register once
    float val = src[idx];

    // Accumulate in registers — stays on-chip, no memory traffic per iter
    for (int i = 0; i < iters; ++i) {
        val = val * 1.000001f + 0.000001f;
    }

    dst[idx] = val;
}

// ---------------------------------------------------------------------------
// Bandwidth helper
// ---------------------------------------------------------------------------
static double measure_bandwidth(double bytes, float time_ms) {
    return bytes / (time_ms * 1e-3) / 1e9;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    const std::string project_root  = PROJECT_ROOT;
    const std::string config_path   = project_root + "/configs/tutorials/03_cuda_memory_types.yaml";
    const std::string global_config = project_root + "/configs/global.yaml";

    auto cfg = cuda_tutorials::ConfigLoader::from_file(config_path);
    cfg.merge_defaults(global_config);

    auto logger = cuda_tutorials::Logger::create(config_path, project_root,
                                                  "03_cuda_memory_types");
    logger.log_info("=== Tutorial 03: CUDA Memory Types ===");

    const int device_id    = cfg.get<int>("device", "id");
    const int N            = cfg.get<int>("tutorial", "N");
    const int num_iter     = cfg.get<int>("tutorial", "num_iterations");
    const int stride       = cfg.get<int>("tutorial", "stride");
    auto access_patterns   = cfg.get_vector<std::string>("tutorial", "access_patterns");

    CUDA_CHECK(cudaSetDevice(device_id));
    auto dev_info = cuda_tutorials::get_device_info(device_id);
    cuda_tutorials::print_device_info(logger, dev_info);

    std::ostringstream oss;
    oss << "Config: N=" << N << "  num_iter=" << num_iter << "  stride=" << stride;
    logger.log_info(oss.str());

    // Allocate host
    std::vector<float> h_src(N), h_const_data(kConstSize);
    std::vector<int>   h_perm(N);
    for (int i = 0; i < N; ++i) {
        h_src[i]  = static_cast<float>(i) * 0.001f;
        h_perm[i] = i;
    }
    // Fisher-Yates shuffle for random permutation
    for (int i = N - 1; i > 0; --i) {
        int j = rand() % (i + 1);
        std::swap(h_perm[i], h_perm[j]);
    }
    for (int i = 0; i < kConstSize; ++i) h_const_data[i] = static_cast<float>(i);

    // Device buffers
    float *d_src{}, *d_dst{};
    int   *d_perm{};
    CUDA_CHECK(cudaMalloc(&d_src,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dst,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_perm, N * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_src,  h_src.data(),  N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_perm, h_perm.data(), N * sizeof(int),   cudaMemcpyHostToDevice));

    // Upload constant memory
    CUDA_CHECK(cudaMemcpyToSymbol(d_const, h_const_data.data(),
                                  kConstSize * sizeof(float)));

    const int block_dim = 256;
    const int grid_dim  = (N + block_dim - 1) / block_dim;
    cuda_tutorials::CudaEvent ev_start, ev_stop;

    auto time_kernel = [&](auto launch_fn) -> float {
        launch_fn(); // warm-up
        CUDA_KERNEL_CHECK();
        ev_start.record();
        for (int i = 0; i < num_iter; ++i) launch_fn();
        ev_stop.record();
        CUDA_CHECK(cudaEventSynchronize(ev_stop.event));
        float total_ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&total_ms, ev_start.event, ev_stop.event));
        return total_ms / num_iter;
    };

    logger.log_info("--- Memory type benchmarks ---");

    // (a) Global memory — sequential
    {
        float ms = time_kernel([&]{
            global_copy_sequential<<<grid_dim, block_dim>>>(d_src, d_dst, N);
        });
        double bw = measure_bandwidth(2.0 * N * sizeof(float), ms);
        oss.str("");
        oss << "Global sequential : " << std::fixed << std::setprecision(3) << ms
            << " ms  " << std::setprecision(2) << bw << " GB/s";
        logger.log_info(oss.str());
    }

    // (b) Global memory — strided
    {
        int eff_n = N / stride;
        int g = (eff_n + block_dim - 1) / block_dim;
        float ms = time_kernel([&]{
            global_copy_strided<<<g, block_dim>>>(d_src, d_dst, N, stride);
        });
        // Only eff_n elements moved but memory transactions are still issued per-thread
        double bw = measure_bandwidth(2.0 * eff_n * sizeof(float), ms);
        oss.str("");
        oss << "Global strided(x" << stride << ")  : " << std::fixed
            << std::setprecision(3) << ms << " ms  "
            << std::setprecision(2) << bw << " GB/s (effective)";
        logger.log_info(oss.str());
    }

    // (c) Global memory — random
    {
        float ms = time_kernel([&]{
            global_copy_random<<<grid_dim, block_dim>>>(d_src, d_dst, d_perm, N);
        });
        double bw = measure_bandwidth(2.0 * N * sizeof(float), ms);
        oss.str("");
        oss << "Global random     : " << std::fixed << std::setprecision(3) << ms
            << " ms  " << std::setprecision(2) << bw << " GB/s";
        logger.log_info(oss.str());
    }

    // (d) Shared memory — tile load/store
    {
        size_t smem = block_dim * sizeof(float);
        float ms = time_kernel([&]{
            shared_mem_bandwidth<<<grid_dim, block_dim, smem>>>(d_src, d_dst, N);
        });
        // Traffic: 1 global read + 1 global write (smem is on-chip)
        double bw = measure_bandwidth(2.0 * N * sizeof(float), ms);
        oss.str("");
        oss << "Shared memory     : " << std::fixed << std::setprecision(3) << ms
            << " ms  " << std::setprecision(2) << bw << " GB/s (global I/O)";
        logger.log_info(oss.str());
    }

    // (e) Constant memory — broadcast read
    {
        float ms = time_kernel([&]{
            constant_mem_read<<<grid_dim, block_dim>>>(d_dst, N, kConstSize);
        });
        double bw = measure_bandwidth(1.0 * N * sizeof(float), ms);
        oss.str("");
        oss << "Constant memory   : " << std::fixed << std::setprecision(3) << ms
            << " ms  " << std::setprecision(2) << bw << " GB/s (write only meas'd)";
        logger.log_info(oss.str());
    }

    // (f) Register arithmetic — measure compute throughput
    {
        const int reg_iters = 100;
        float ms = time_kernel([&]{
            register_compute<<<grid_dim, block_dim>>>(d_src, d_dst, N, reg_iters);
        });
        double gflops = 2.0 * N * reg_iters / (ms * 1e-3) / 1e9;
        oss.str("");
        oss << "Register compute  : " << std::fixed << std::setprecision(3) << ms
            << " ms  " << std::setprecision(2) << gflops << " GFLOPS (2 ops/iter)";
        logger.log_info(oss.str());
    }

    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
    CUDA_CHECK(cudaFree(d_perm));

    logger.log_info("Tutorial 03 complete.");
    return 0;
}
