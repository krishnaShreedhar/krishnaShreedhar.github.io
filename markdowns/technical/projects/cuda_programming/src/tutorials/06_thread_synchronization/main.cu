// =============================================================================
// Tutorial 06: Thread Synchronization
//
// Concept:
//   __syncthreads() is a block-level barrier: all threads in the block must
//   reach this point before any thread can continue. It ensures shared memory
//   writes by earlier phases are visible to all threads in later phases.
//
//   WITHOUT __syncthreads(), threads may read partial/stale data from shared
//   memory that was written by another thread still in-flight → data races →
//   wrong results.
//
// Experiment: Prefix Sum (Inclusive Scan) using Hillis-Steele algorithm
//   Phase log2(block_size) iterations:
//     for stride in [1, 2, 4, ..., block_size/2]:
//       smem[t] += smem[t - stride]  (if t >= stride)
//       __syncthreads()              ← without this, UB
//
//   Ablation: run with and without sync barriers, check result vs CPU.
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
// Kernel: correct inclusive prefix sum with __syncthreads()
// Processes one block-sized chunk at a time.
// Input: in[0..n-1], output: out[i] = sum(in[0..i])  (within each block)
// ---------------------------------------------------------------------------
__global__ void prefix_sum_synced(const float* __restrict__ in,
                                   float*       __restrict__ out,
                                   int n) {
    extern __shared__ float smem[];
    int tid  = threadIdx.x;
    int idx  = blockIdx.x * blockDim.x + tid;

    smem[tid] = (idx < n) ? in[idx] : 0.0f;
    __syncthreads();  // Ensure all threads have loaded before scanning

    // Hillis-Steele (up-sweep)
    for (int stride = 1; stride < blockDim.x; stride <<= 1) {
        float val = (tid >= stride) ? smem[tid - stride] : 0.0f;
        __syncthreads();    // Wait for all reads from previous phase
        smem[tid] += val;
        __syncthreads();    // Wait for all writes before next read phase
    }

    if (idx < n) out[idx] = smem[tid];
}

// ---------------------------------------------------------------------------
// Kernel: broken prefix sum WITHOUT __syncthreads()
//   Demonstrates undefined behavior — results are incorrect and non-deterministic.
//   We still launch it so the programmer can observe the wrong output.
// ---------------------------------------------------------------------------
__global__ void prefix_sum_no_sync(const float* __restrict__ in,
                                    float*       __restrict__ out,
                                    int n) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    smem[tid] = (idx < n) ? in[idx] : 0.0f;
    // Missing __syncthreads() here — threads may not see all initial values

    for (int stride = 1; stride < blockDim.x; stride <<= 1) {
        float val = (tid >= stride) ? smem[tid - stride] : 0.0f;
        // Missing __syncthreads() — races between reading old and new values
        smem[tid] += val;
        // Missing __syncthreads() — writes may not be visible to other threads
    }

    if (idx < n) out[idx] = smem[tid];
}

// ---------------------------------------------------------------------------
// CPU reference: block-independent prefix sum
// ---------------------------------------------------------------------------
static void cpu_prefix_sum(const std::vector<float>& in,
                            std::vector<float>&       out,
                            int block_size) {
    int n = static_cast<int>(in.size());
    out.resize(n);
    for (int b = 0; b * block_size < n; ++b) {
        int start = b * block_size;
        int end   = std::min(start + block_size, n);
        float acc = 0.f;
        for (int i = start; i < end; ++i) {
            acc += in[i];
            out[i] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    const std::string project_root  = PROJECT_ROOT;
    const std::string config_path   = project_root +
        "/configs/tutorials/06_thread_synchronization.yaml";
    const std::string global_config = project_root + "/configs/global.yaml";

    auto cfg = cuda_tutorials::ConfigLoader::from_file(config_path);
    cfg.merge_defaults(global_config);

    auto logger = cuda_tutorials::Logger::create(config_path, project_root,
                                                  "06_thread_synchronization");
    logger.log_info("=== Tutorial 06: Thread Synchronization ===");

    const int device_id      = cfg.get<int>("device", "id");
    const int N              = cfg.get<int>("tutorial", "N");
    const int block_size     = cfg.get<int>("tutorial", "block_size");
    const int num_iter       = cfg.get<int>("tutorial", "num_iterations");
    const bool run_no_sync   = cfg.get<bool>("tutorial", "run_without_sync");

    CUDA_CHECK(cudaSetDevice(device_id));
    auto dev_info = cuda_tutorials::get_device_info(device_id);
    cuda_tutorials::print_device_info(logger, dev_info);

    std::ostringstream oss;
    oss << "Config: N=" << N << "  block_size=" << block_size
        << "  num_iter=" << num_iter << "  run_no_sync=" << run_no_sync;
    logger.log_info(oss.str());

    std::vector<float> h_in(N), h_out_gpu(N), h_out_cpu;
    for (int i = 0; i < N; ++i) h_in[i] = 1.0f;  // sum of i+1 elements expected

    cpu_prefix_sum(h_in, h_out_cpu, block_size);

    float *d_in{}, *d_out{};
    CUDA_CHECK(cudaMalloc(&d_in,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    int grid = (N + block_size - 1) / block_size;
    size_t smem = block_size * sizeof(float);

    cuda_tutorials::CudaEvent ev_start, ev_stop;

    // ---------------------------------------------------------------------------
    // Synced version
    // ---------------------------------------------------------------------------
    prefix_sum_synced<<<grid, block_size, smem>>>(d_in, d_out, N);
    CUDA_KERNEL_CHECK();

    ev_start.record();
    for (int i = 0; i < num_iter; ++i)
        prefix_sum_synced<<<grid, block_size, smem>>>(d_in, d_out, N);
    ev_stop.record();
    CUDA_CHECK(cudaEventSynchronize(ev_stop.event));
    float t_sync_total = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&t_sync_total, ev_start.event, ev_stop.event));
    float t_sync = t_sync_total / num_iter;

    CUDA_CHECK(cudaMemcpy(h_out_gpu.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

    double max_err_sync = 0.0;
    for (int i = 0; i < N; ++i)
        max_err_sync = std::max(max_err_sync,
                                std::abs(static_cast<double>(h_out_gpu[i] - h_out_cpu[i])));
    oss.str("");
    oss << "Synced prefix sum: " << std::fixed << std::setprecision(3) << t_sync
        << " ms  max_error=" << max_err_sync
        << (max_err_sync < 1e-3 ? "  CORRECT" : "  WRONG");
    logger.log_info(oss.str());

    // ---------------------------------------------------------------------------
    // No-sync version (ablation — expected to produce wrong results)
    // ---------------------------------------------------------------------------
    if (run_no_sync) {
        prefix_sum_no_sync<<<grid, block_size, smem>>>(d_in, d_out, N);
        CUDA_KERNEL_CHECK();

        ev_start.record();
        for (int i = 0; i < num_iter; ++i)
            prefix_sum_no_sync<<<grid, block_size, smem>>>(d_in, d_out, N);
        ev_stop.record();
        CUDA_CHECK(cudaEventSynchronize(ev_stop.event));
        float t_nosync_total = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&t_nosync_total, ev_start.event, ev_stop.event));
        float t_nosync = t_nosync_total / num_iter;

        CUDA_CHECK(cudaMemcpy(h_out_gpu.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

        double max_err_nosync = 0.0;
        for (int i = 0; i < N; ++i)
            max_err_nosync = std::max(max_err_nosync,
                                      std::abs(static_cast<double>(h_out_gpu[i] - h_out_cpu[i])));
        oss.str("");
        oss << "No-sync  (broken): " << std::fixed << std::setprecision(3) << t_nosync
            << " ms  max_error=" << max_err_nosync
            << (max_err_nosync < 1e-3 ? "  (accidentally correct this run)" : "  WRONG as expected");
        logger.log_info(oss.str());

        oss.str("");
        oss << "Sync overhead: ~" << std::fixed << std::setprecision(3)
            << (t_sync - t_nosync) << " ms (" << t_sync / t_nosync << "x vs no-sync)";
        logger.log_info(oss.str());

        logger.log_info(
            "Explanation: without __syncthreads(), threads read shared memory "
            "that may not yet reflect writes from other threads in the same warp "
            "or neighboring warps. The race condition produces non-deterministic "
            "results. On H200 the divergence can be subtle due to warp-level "
            "synchrony, but multi-warp blocks always require explicit barriers.");
    }

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    logger.log_info("Tutorial 06 complete.");
    return 0;
}
