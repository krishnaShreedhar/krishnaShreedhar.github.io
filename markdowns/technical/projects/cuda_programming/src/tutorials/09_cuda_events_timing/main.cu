// =============================================================================
// Tutorial 09: CUDA Events & Timing
//
// Concept:
//   Measuring GPU kernel time correctly requires GPU-side timestamps, not
//   CPU-side clocks. CPU clocks measure the time from kernel *launch* to
//   *device synchronization*, which includes:
//     - CPU→GPU launch overhead (~5–20 µs)
//     - Time waiting in the GPU command queue
//     - Actual kernel execution time
//   GPU events inserted into the command stream record timestamps on the
//   GPU timeline, giving accurate kernel execution time.
//
//   cudaEventRecord(event, stream) — inserts a timestamp marker
//   cudaEventElapsedTime(&ms, start, stop) — returns GPU-side elapsed time
//
// Experiment:
//   Run the same kernel num_iterations times, collect timing with:
//     (a) CUDA events — accurate GPU time
//     (b) CPU std::chrono — includes launch overhead
//   Discard warmup_iterations then compute mean ± std.
// =============================================================================

#include <chrono>
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
// Kernel: memory bandwidth workload (vector scale)
// ---------------------------------------------------------------------------
__global__ void scale_kernel(const float* __restrict__ in,
                              float*       __restrict__ out,
                              int n,
                              float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = in[idx] * scale;
}

// ---------------------------------------------------------------------------
// Statistics helpers
// ---------------------------------------------------------------------------
static double compute_mean(const std::vector<double>& v) {
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}
static double compute_std(const std::vector<double>& v, double mean) {
    double sq = 0.0;
    for (double x : v) sq += (x - mean) * (x - mean);
    return std::sqrt(sq / v.size());
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    const std::string project_root  = PROJECT_ROOT;
    const std::string config_path   = project_root +
        "/configs/tutorials/09_cuda_events_timing.yaml";
    const std::string global_config = project_root + "/configs/global.yaml";

    auto cfg = cuda_tutorials::ConfigLoader::from_file(config_path);
    cfg.merge_defaults(global_config);

    auto logger = cuda_tutorials::Logger::create(config_path, project_root,
                                                  "09_cuda_events_timing");
    logger.log_info("=== Tutorial 09: CUDA Events & Timing ===");

    const int device_id     = cfg.get<int>("device", "id");
    const int N             = cfg.get<int>("tutorial", "N");
    const int num_iter      = cfg.get<int>("tutorial", "num_iterations");
    const int warmup_iter   = cfg.get<int>("tutorial", "warmup_iterations");

    CUDA_CHECK(cudaSetDevice(device_id));
    auto dev_info = cuda_tutorials::get_device_info(device_id);
    cuda_tutorials::print_device_info(logger, dev_info);

    std::ostringstream oss;
    oss << "Config: N=" << N
        << "  num_iter=" << num_iter
        << "  warmup=" << warmup_iter;
    logger.log_info(oss.str());

    float *d_in{}, *d_out{};
    CUDA_CHECK(cudaMalloc(&d_in,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_in, 0x3f, N * sizeof(float)));

    const int block_dim = 256;
    const int grid_dim  = (N + block_dim - 1) / block_dim;

    // CUDA events for GPU timing
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    std::vector<double> gpu_times_ms, cpu_times_ms;

    // Warm-up (not recorded)
    logger.log_info("Running " + std::to_string(warmup_iter) + " warm-up iterations...");
    for (int i = 0; i < warmup_iter; ++i) {
        scale_kernel<<<grid_dim, block_dim>>>(d_in, d_out, N, 1.001f);
        CUDA_KERNEL_CHECK();
    }

    // Timed iterations
    logger.log_info("Running " + std::to_string(num_iter) + " timed iterations...");
    for (int i = 0; i < num_iter; ++i) {
        // --- GPU event timing ---
        CUDA_CHECK(cudaEventRecord(ev_start));
        scale_kernel<<<grid_dim, block_dim>>>(d_in, d_out, N, 1.001f);
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));

        float gpu_ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, ev_start, ev_stop));
        gpu_times_ms.push_back(static_cast<double>(gpu_ms));

        // --- CPU timing (measures launch + wait overhead) ---
        auto cpu_t0 = std::chrono::high_resolution_clock::now();
        scale_kernel<<<grid_dim, block_dim>>>(d_in, d_out, N, 1.001f);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto cpu_t1 = std::chrono::high_resolution_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(cpu_t1 - cpu_t0).count();
        cpu_times_ms.push_back(cpu_ms);

        if (i % 10 == 0) {
            oss.str("");
            oss << "iter " << std::setw(3) << i
                << "  GPU=" << std::fixed << std::setprecision(3) << gpu_ms
                << " ms  CPU=" << cpu_ms << " ms";
            logger.log_debug(oss.str());
        }
    }

    // Statistics
    double gpu_mean = compute_mean(gpu_times_ms);
    double gpu_std  = compute_std(gpu_times_ms, gpu_mean);
    double cpu_mean = compute_mean(cpu_times_ms);
    double cpu_std  = compute_std(cpu_times_ms, cpu_mean);

    double launch_overhead = cpu_mean - gpu_mean;
    double bw_gbs = 2.0 * N * sizeof(float) / (gpu_mean * 1e-3) / 1e9;

    logger.log_info("--- Results ---");
    oss.str("");
    oss << "GPU event timing  : mean=" << std::fixed << std::setprecision(3)
        << gpu_mean << " ms  std=" << gpu_std << " ms";
    logger.log_info(oss.str());

    oss.str("");
    oss << "CPU chrono timing : mean=" << std::fixed << std::setprecision(3)
        << cpu_mean << " ms  std=" << cpu_std << " ms";
    logger.log_info(oss.str());

    oss.str("");
    oss << "Launch+sync overhead (CPU - GPU): ~" << std::fixed
        << std::setprecision(3) << launch_overhead << " ms";
    logger.log_info(oss.str());

    oss.str("");
    oss << "Effective bandwidth (GPU time): "
        << std::fixed << std::setprecision(2) << bw_gbs << " GB/s";
    logger.log_info(oss.str());

    logger.log_info(
        "Explanation: CPU timing includes GPU command queue latency and "
        "cudaDeviceSynchronize() spin-wait. For accurate kernel profiling, "
        "always use cudaEventRecord/cudaEventElapsedTime. "
        "For multi-kernel pipelines, record events at the exact points of interest.");

    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    logger.log_info("Tutorial 09 complete.");
    return 0;
}
