// =============================================================================
// Tutorial 14: Warp Shuffle Intrinsics
//
// Concept:
//   Warp shuffle instructions allow threads within a warp to directly exchange
//   register values without going through shared memory.
//
//   Intrinsics (sm_30+):
//     __shfl_sync(mask, val, srcLane)    — broadcast from lane srcLane
//     __shfl_down_sync(mask, val, delta) — shift "down": val from lane+delta
//     __shfl_up_sync(mask, val, delta)   — shift "up":   val from lane-delta
//     __shfl_xor_sync(mask, val, laneMask) — butterfly exchange
//
//   Benefits over shared memory:
//     - No bank conflicts (register file access)
//     - No __syncthreads() needed within a warp (implicit sync)
//     - Lower latency than L1 shared memory reads
//
// Experiment:
//   (a) Warp-level reduction: sum/max/min using __shfl_down_sync
//   (b) Broadcast: thread 0 sends a value to all lanes via __shfl_sync
//   (c) XOR butterfly: transpose pairs within a warp
//   Compare with shared-memory equivalent for performance
// =============================================================================

#include <algorithm>
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

static constexpr unsigned FULL_MASK = 0xffffffff;

// ---------------------------------------------------------------------------
// Warp-level reduce: sum
// ---------------------------------------------------------------------------
__device__ inline float warp_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(FULL_MASK, val, offset);
    return val;  // valid in lane 0
}
// Warp-level reduce: max
__device__ inline float warp_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(FULL_MASK, val, offset));
    return val;
}
// Warp-level reduce: min
__device__ inline float warp_min(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fminf(val, __shfl_down_sync(FULL_MASK, val, offset));
    return val;
}

// ---------------------------------------------------------------------------
// Kernel: block-level reduction using warp shuffle (op=0 sum, 1 max, 2 min)
// ---------------------------------------------------------------------------
__global__ void block_reduce_shuffle(const float* __restrict__ in,
                                      float*       __restrict__ out,
                                      int n, int op) {
    extern __shared__ float warp_results[];  // one float per warp

    int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % 32;
    int wid  = threadIdx.x / 32;

    float val;
    if (tid < n) val = in[tid];
    else         val = (op == 0) ? 0.f : (op == 1 ? -1e30f : 1e30f);

    // Warp-level reduction
    if (op == 0)      val = warp_sum(val);
    else if (op == 1) val = warp_max(val);
    else              val = warp_min(val);

    // First lane of each warp writes to shared memory
    if (lane == 0) warp_results[wid] = val;
    __syncthreads();

    // Final reduction across warps (done by first warp only)
    int num_warps = blockDim.x / 32;
    if (threadIdx.x < num_warps) {
        val = warp_results[threadIdx.x];
        if (op == 0)      val = warp_sum(val);
        else if (op == 1) val = warp_max(val);
        else              val = warp_min(val);
    }

    if (threadIdx.x == 0) out[blockIdx.x] = val;
}

// ---------------------------------------------------------------------------
// Kernel: shared-memory equivalent (for comparison)
// ---------------------------------------------------------------------------
__global__ void block_reduce_smem(const float* __restrict__ in,
                                   float*       __restrict__ out,
                                   int n) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    smem[tid] = (idx < n) ? in[idx] : 0.f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = smem[0];
}

// ---------------------------------------------------------------------------
// Kernel: broadcast demo — lane 0 sends its value to all lanes in each warp
// ---------------------------------------------------------------------------
__global__ void broadcast_demo(float* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Each thread starts with its own lane ID as a value
    float val = static_cast<float>(threadIdx.x % 32);  // lane 0 has 0.0

    // Broadcast from lane 0: all threads in warp now have val=0.0
    val = __shfl_sync(FULL_MASK, val, 0);

    out[idx] = val;  // should all be 0.0 within each warp
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    const std::string project_root  = PROJECT_ROOT;
    const std::string config_path   = project_root + "/configs/tutorials/14_warp_shuffle.yaml";
    const std::string global_config = project_root + "/configs/global.yaml";

    auto cfg = cuda_tutorials::ConfigLoader::from_file(config_path);
    cfg.merge_defaults(global_config);

    auto logger = cuda_tutorials::Logger::create(config_path, project_root,
                                                  "14_warp_shuffle");
    logger.log_info("=== Tutorial 14: Warp Shuffle Intrinsics ===");

    const int device_id = cfg.get<int>("device", "id");
    const int N         = cfg.get<int>("tutorial", "N");
    const int num_iter  = cfg.get<int>("tutorial", "num_iterations");
    auto ops            = cfg.get_vector<std::string>("tutorial", "warp_reduce_ops");

    CUDA_CHECK(cudaSetDevice(device_id));
    auto dev_info = cuda_tutorials::get_device_info(device_id);
    cuda_tutorials::print_device_info(logger, dev_info);

    std::ostringstream oss;
    oss << "Config: N=" << N << "  num_iter=" << num_iter;
    logger.log_info(oss.str());

    std::vector<float> h_in(N);
    for (int i = 0; i < N; ++i) h_in[i] = static_cast<float>(i % 1000) * 0.001f + 0.5f;

    const int block_dim = 256;
    const int grid_dim  = (N + block_dim - 1) / block_dim;

    float *d_in{}, *d_out{};
    CUDA_CHECK(cudaMalloc(&d_in,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, grid_dim * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    cuda_tutorials::CudaEvent ev_start, ev_stop;
    size_t smem_shuffle = (block_dim / 32) * sizeof(float);  // one float per warp
    size_t smem_shared  = block_dim * sizeof(float);

    auto bench = [&](auto fn, size_t smem, const char* label) -> float {
        fn(); CUDA_KERNEL_CHECK();
        ev_start.record();
        for (int i = 0; i < num_iter; ++i) fn();
        ev_stop.record();
        CUDA_CHECK(cudaEventSynchronize(ev_stop.event));
        float t = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&t, ev_start.event, ev_stop.event));
        float ms = t / num_iter;
        double bw = static_cast<double>(N) * sizeof(float) / (ms * 1e-3) / 1e9;
        oss.str("");
        oss << label << ": " << std::fixed << std::setprecision(3) << ms
            << " ms  " << std::setprecision(2) << bw << " GB/s";
        logger.log_info(oss.str());
        return ms;
    };

    // Run for each op from config
    for (auto& op_str : ops) {
        int op = (op_str == "sum") ? 0 : (op_str == "max") ? 1 : 2;
        std::string label_shfl = "Shuffle " + op_str + "  ";
        std::string label_smem = "Smem    " + op_str + "  ";

        float t_shfl = bench([&]{
            block_reduce_shuffle<<<grid_dim, block_dim, smem_shuffle>>>(
                d_in, d_out, N, op);
        }, smem_shuffle, label_shfl.c_str());

        float t_smem = bench([&]{
            block_reduce_smem<<<grid_dim, block_dim, smem_shared>>>(
                d_in, d_out, N);
        }, smem_shared, label_smem.c_str());

        oss.str("");
        oss << "Shuffle vs smem speedup (" << op_str << "): "
            << std::fixed << std::setprecision(2) << (t_smem / t_shfl) << "x";
        logger.log_info(oss.str());
    }

    // Broadcast demo
    {
        logger.log_info("--- Broadcast demo ---");
        block_reduce_shuffle<<<grid_dim, block_dim, smem_shuffle>>>(d_in, d_out, N, 0);
        broadcast_demo<<<grid_dim, block_dim>>>(d_out, N);
        CUDA_KERNEL_CHECK();

        std::vector<float> h_out(N);
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

        // Every element should be 0.0 (lane 0's value)
        float max_err = 0.f;
        for (int i = 0; i < N; ++i) max_err = std::max(max_err, std::abs(h_out[i]));
        oss.str("");
        oss << "Broadcast from lane 0: all values should be 0.0, max_abs=" << max_err
            << (max_err < 1e-6f ? "  CORRECT" : "  WRONG");
        logger.log_info(oss.str());
    }

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    logger.log_info("Tutorial 14 complete.");
    return 0;
}
