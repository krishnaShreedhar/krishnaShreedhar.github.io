// =============================================================================
// Tutorial 01: Threads, Blocks & Grids
//
// Concept:
//   CUDA's execution hierarchy has three levels:
//     Thread  — a single GPU lane, identified by threadIdx
//     Block   — a group of threads that share shared memory and can synchronize
//     Grid    — the collection of all blocks for one kernel launch
//
//   Thread index in 1D: global_id = blockIdx.x * blockDim.x + threadIdx.x
//
// Experiment:
//   Vector addition C[i] = A[i] + B[i].
//   We sweep block_dim over [32, 64, 128, 256, 512] and measure throughput.
//   All config is read from YAML — no CLI arguments.
//
// Key observations:
//   - Larger blocks amortize launch overhead across more threads
//   - Peak bandwidth is reached at block_dim ≥ 128 for this workload
//   - Grid dimension = ceil(N / block_dim) — fractional blocks handled with
//     a bounds-check inside the kernel
// =============================================================================

#include <cmath>
#include <filesystem>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "../../common/config_loader.hpp"
#include "../../common/cuda_utils.hpp"
#include "../../common/logger.hpp"

// ---------------------------------------------------------------------------
// Kernel: vector addition with bounds-check
// Each thread processes one element.
// ---------------------------------------------------------------------------
__global__ void vector_add_kernel(const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  float*       __restrict__ c,
                                  int n) {
    // Compute the global thread index (1D grid, 1D block)
    int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
              static_cast<int>(threadIdx.x);

    // Guard: last block may be partially filled when N is not divisible by blockDim
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// ---------------------------------------------------------------------------
// Measure bandwidth for a given block_dim over num_iterations
// Returns mean time in ms and GB/s
// ---------------------------------------------------------------------------
static void run_ablation(cuda_tutorials::Logger&        logger,
                         const float*                    d_a,
                         const float*                    d_b,
                         float*                          d_c,
                         int                             n,
                         int                             block_dim,
                         int                             num_iterations,
                         double&                         out_mean_ms,
                         double&                         out_bandwidth_gbs) {
    // Grid dimension: how many blocks are needed to cover N elements
    int grid_dim = (n + block_dim - 1) / block_dim;

    std::ostringstream oss;
    oss << "block_dim=" << block_dim
        << "  grid_dim=" << grid_dim
        << "  total_threads=" << static_cast<long long>(grid_dim) * block_dim;
    logger.log_debug(oss.str());

    // Warm-up launch to prime caches / JIT
    vector_add_kernel<<<grid_dim, block_dim>>>(d_a, d_b, d_c, n);
    CUDA_KERNEL_CHECK();

    // Timed iterations using CUDA events
    cuda_tutorials::CudaEvent ev_start, ev_stop;
    std::vector<float> times_ms(num_iterations);

    for (int iter = 0; iter < num_iterations; ++iter) {
        ev_start.record();
        vector_add_kernel<<<grid_dim, block_dim>>>(d_a, d_b, d_c, n);
        ev_stop.record();
        times_ms[iter] = ev_stop.elapsed_ms(ev_start);
    }

    // Compute mean
    double sum = std::accumulate(times_ms.begin(), times_ms.end(), 0.0);
    out_mean_ms = sum / num_iterations;

    // Memory traffic: 2 reads + 1 write, each float (4 bytes)
    double bytes       = static_cast<double>(n) * sizeof(float) * 3.0;
    out_bandwidth_gbs  = bytes / (out_mean_ms * 1e-3) / 1e9;

    oss.str("");
    oss << "block_dim=" << std::setw(4) << block_dim
        << "  mean_time=" << std::fixed << std::setprecision(3) << out_mean_ms << " ms"
        << "  bandwidth=" << std::setprecision(2) << out_bandwidth_gbs << " GB/s";
    logger.log_info(oss.str());
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    // Paths relative to the project root — resolved at build time via CMake
    // PROJECT_ROOT is injected by CMake as a compile definition.
    const std::string project_root   = PROJECT_ROOT;
    const std::string config_path    = project_root +
        "/configs/tutorials/01_threads_blocks_grids.yaml";
    const std::string global_config  = project_root + "/configs/global.yaml";

    // Load config (tutorial-specific, with global defaults merged in)
    auto cfg = cuda_tutorials::ConfigLoader::from_file(config_path);
    cfg.merge_defaults(global_config);

    // Initialize logger
    auto logger = cuda_tutorials::Logger::create(config_path, project_root,
                                                  "01_threads_blocks_grids");
    logger.log_info("=== Tutorial 01: Threads, Blocks & Grids ===");

    // Read parameters from YAML
    const int device_id     = cfg.get<int>("device", "id");
    const int N             = cfg.get<int>("tutorial", "N");
    const int num_iter      = cfg.get<int>("tutorial", "num_iterations");
    auto block_dims         = cfg.get_vector<int>("tutorial", "block_dims");

    // Select GPU
    CUDA_CHECK(cudaSetDevice(device_id));
    auto dev_info = cuda_tutorials::get_device_info(device_id);
    cuda_tutorials::print_device_info(logger, dev_info);

    std::ostringstream oss;
    oss << "Config: N=" << N << "  num_iterations=" << num_iter
        << "  block_dims=[";
    for (size_t i = 0; i < block_dims.size(); ++i) {
        oss << block_dims[i] << (i + 1 < block_dims.size() ? ", " : "]");
    }
    logger.log_info(oss.str());

    // Allocate host data and initialize
    std::vector<float> h_a(N), h_b(N), h_c(N);
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i) * 0.001f;
        h_b[i] = static_cast<float>(N - i) * 0.001f;
    }

    // Allocate device memory
    float *d_a{}, *d_b{}, *d_c{};
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, N * sizeof(float)));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // ---------------------------------------------------------------------------
    // Ablation: sweep block_dim
    // ---------------------------------------------------------------------------
    logger.log_info("--- Block-dim ablation sweep ---");
    struct Result { int block_dim; double mean_ms; double bw_gbs; };
    std::vector<Result> results;

    for (int bd : block_dims) {
        // block_dim must be <= max threads per block
        if (bd > dev_info.max_threads_per_block) {
            oss.str("");
            oss << "Skipping block_dim=" << bd
                << " (exceeds max_threads_per_block="
                << dev_info.max_threads_per_block << ")";
            logger.log_warn(oss.str());
            continue;
        }
        double mean_ms{}, bw_gbs{};
        run_ablation(logger, d_a, d_b, d_c, N, bd, num_iter, mean_ms, bw_gbs);
        results.push_back({bd, mean_ms, bw_gbs});
    }

    // ---------------------------------------------------------------------------
    // Correctness check with the best block_dim
    // ---------------------------------------------------------------------------
    if (!results.empty()) {
        // Use first result's block_dim for final correctness pass
        int best_bd = results[0].block_dim;
        int grid    = (N + best_bd - 1) / best_bd;
        vector_add_kernel<<<grid, best_bd>>>(d_a, d_b, d_c, N);
        CUDA_KERNEL_CHECK();

        CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, N * sizeof(float), cudaMemcpyDeviceToHost));

        double max_err = 0.0;
        for (int i = 0; i < N; ++i) {
            double expected = static_cast<double>(h_a[i]) + static_cast<double>(h_b[i]);
            max_err = std::max(max_err, std::abs(static_cast<double>(h_c[i]) - expected));
        }
        oss.str("");
        oss << "Correctness check: max_abs_error=" << max_err
            << (max_err < 1e-5 ? "  PASS" : "  FAIL");
        logger.log_info(oss.str());
    }

    // ---------------------------------------------------------------------------
    // Summary table
    // ---------------------------------------------------------------------------
    logger.log_info("--- Summary ---");
    for (auto& r : results) {
        oss.str("");
        oss << "block_dim=" << std::setw(4) << r.block_dim
            << "  mean=" << std::fixed << std::setprecision(3) << r.mean_ms << " ms"
            << "  BW=" << std::setprecision(2) << r.bw_gbs << " GB/s";
        logger.log_info(oss.str());
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    logger.log_info("Tutorial 01 complete.");
    return 0;
}
