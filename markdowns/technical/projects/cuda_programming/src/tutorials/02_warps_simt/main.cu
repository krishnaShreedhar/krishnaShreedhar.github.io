// =============================================================================
// Tutorial 02: Warps & SIMT (Single Instruction, Multiple Threads)
//
// Concept:
//   The GPU schedules threads in groups of 32 called WARPS. All threads in a
//   warp execute the same instruction each cycle (SIMT model).
//
//   WARP DIVERGENCE occurs when threads within the same warp take different
//   branches. Both paths execute serially, doubling (or worse) the latency.
//
//   Example:
//     if (threadIdx.x % 2 == 0)   // alternates per thread => divergence
//     if (threadIdx.x / 32 % 2)   // alternates per warp   => no divergence
//
// Experiment:
//   Two kernels perform the same arithmetic but with different branch patterns:
//   (a) uniform     — all threads in a warp take the same branch
//   (b) divergent   — threads within a warp take alternating branches
//
//   The divergence_stride YAML param controls granularity:
//     stride=32 → branching at warp granularity (no divergence)
//     stride=1  → alternating per thread (maximum divergence)
// =============================================================================

#include <cmath>
#include <filesystem>
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
// Kernel A: Uniform branching (no divergence within warp)
//   Branch is determined by (global_id / warp_size) — warp-level decision.
//   All 32 threads in a warp land on the SAME branch.
// ---------------------------------------------------------------------------
__global__ void kernel_uniform_branch(const float* __restrict__ in,
                                      float*       __restrict__ out,
                                      int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Branch at warp granularity: warpId is even → path A, else path B.
    // Every thread in the warp executes the same path — zero divergence.
    int warp_id = idx / 32;
    float val = in[idx];
    if (warp_id % 2 == 0) {
        // Path A: simple arithmetic chain
        val = val * 1.1f + 0.5f;
        val = sqrtf(val);
    } else {
        // Path B: different arithmetic chain
        val = val * 0.9f - 0.5f;
        val = val * val;
    }
    out[idx] = val;
}

// ---------------------------------------------------------------------------
// Kernel B: Divergent branching
//   Branch is determined by (global_id % stride) — thread-level decision
//   when stride < warp_size. With stride=1, odd and even threads diverge.
// ---------------------------------------------------------------------------
__global__ void kernel_divergent_branch(const float* __restrict__ in,
                                        float*       __restrict__ out,
                                        int n,
                                        int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Branch at thread (or sub-warp) granularity.
    // With stride=1: every other thread diverges → worst case.
    // With stride=32: equivalent to warp-level (same as uniform kernel).
    float val = in[idx];
    if ((idx / stride) % 2 == 0) {
        val = val * 1.1f + 0.5f;
        val = sqrtf(val);
    } else {
        val = val * 0.9f - 0.5f;
        val = val * val;
    }
    out[idx] = val;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    const std::string project_root  = PROJECT_ROOT;
    const std::string config_path   = project_root + "/configs/tutorials/02_warps_simt.yaml";
    const std::string global_config = project_root + "/configs/global.yaml";

    auto cfg = cuda_tutorials::ConfigLoader::from_file(config_path);
    cfg.merge_defaults(global_config);

    auto logger = cuda_tutorials::Logger::create(config_path, project_root,
                                                  "02_warps_simt");
    logger.log_info("=== Tutorial 02: Warps & SIMT ===");

    const int device_id        = cfg.get<int>("device", "id");
    const int N                = cfg.get<int>("tutorial", "N");
    const int divergence_stride = cfg.get<int>("tutorial", "divergence_stride");
    const int num_iterations   = cfg.get<int>("tutorial", "num_iterations");

    CUDA_CHECK(cudaSetDevice(device_id));
    auto dev_info = cuda_tutorials::get_device_info(device_id);
    cuda_tutorials::print_device_info(logger, dev_info);

    std::ostringstream oss;
    oss << "Config: N=" << N
        << "  divergence_stride=" << divergence_stride
        << "  num_iterations=" << num_iterations;
    logger.log_info(oss.str());

    // Allocate and initialize
    std::vector<float> h_in(N), h_out_uniform(N), h_out_divergent(N);
    for (int i = 0; i < N; ++i) h_in[i] = 1.0f + static_cast<float>(i % 1000) * 0.001f;

    float *d_in{}, *d_out{};
    CUDA_CHECK(cudaMalloc(&d_in,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    const int block_dim = 256;
    const int grid_dim  = (N + block_dim - 1) / block_dim;

    oss.str("");
    oss << "Launch config: block_dim=" << block_dim
        << "  grid_dim=" << grid_dim;
    logger.log_debug(oss.str());

    // ---------------------------------------------------------------------------
    // Benchmark: Uniform (no divergence)
    // ---------------------------------------------------------------------------
    // Warm-up
    kernel_uniform_branch<<<grid_dim, block_dim>>>(d_in, d_out, N);
    CUDA_KERNEL_CHECK();

    cuda_tutorials::CudaEvent ev_start, ev_stop;
    std::vector<float> times_uniform(num_iterations);
    for (int i = 0; i < num_iterations; ++i) {
        ev_start.record();
        kernel_uniform_branch<<<grid_dim, block_dim>>>(d_in, d_out, N);
        ev_stop.record();
        times_uniform[i] = ev_stop.elapsed_ms(ev_start);
    }
    double mean_uniform = std::accumulate(times_uniform.begin(),
                                          times_uniform.end(), 0.0) / num_iterations;

    // ---------------------------------------------------------------------------
    // Benchmark: Divergent branching
    // ---------------------------------------------------------------------------
    kernel_divergent_branch<<<grid_dim, block_dim>>>(d_in, d_out, N, divergence_stride);
    CUDA_KERNEL_CHECK();

    std::vector<float> times_divergent(num_iterations);
    for (int i = 0; i < num_iterations; ++i) {
        ev_start.record();
        kernel_divergent_branch<<<grid_dim, block_dim>>>(d_in, d_out, N, divergence_stride);
        ev_stop.record();
        times_divergent[i] = ev_stop.elapsed_ms(ev_start);
    }
    double mean_divergent = std::accumulate(times_divergent.begin(),
                                            times_divergent.end(), 0.0) / num_iterations;

    double slowdown = mean_divergent / mean_uniform;

    // ---------------------------------------------------------------------------
    // Results
    // ---------------------------------------------------------------------------
    oss.str("");
    oss << "Uniform branch   : " << std::fixed << std::setprecision(3)
        << mean_uniform << " ms";
    logger.log_info(oss.str());

    oss.str("");
    oss << "Divergent branch : " << std::fixed << std::setprecision(3)
        << mean_divergent << " ms  (stride=" << divergence_stride << ")";
    logger.log_info(oss.str());

    oss.str("");
    oss << "Divergence slowdown: " << std::setprecision(2) << slowdown << "x";
    logger.log_info(oss.str());

    // Explanation: when stride < warp_size (32), threads inside a warp diverge.
    // The GPU must execute both code paths with the inactive half masked off,
    // roughly doubling instruction count per warp.
    if (divergence_stride < 32) {
        logger.log_info(
            "Explanation: stride < 32 → intra-warp divergence detected. "
            "GPU serializes both branches; effective SIMD efficiency drops.");
    } else {
        logger.log_info(
            "Explanation: stride >= 32 → each warp takes one branch uniformly. "
            "No divergence penalty expected.");
    }

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    logger.log_info("Tutorial 02 complete.");
    return 0;
}
