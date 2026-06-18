// =============================================================================
// Tutorial 12: Atomic Operations
//
// Concept:
//   Atomics perform read-modify-write operations without race conditions.
//   GPU atomics (atomicAdd, atomicMax, atomicCAS, etc.) serialize conflicting
//   accesses to the same memory location.
//
//   Contention problem:
//     If many threads atomicAdd to the SAME bin, accesses serialize →
//     throughput degrades roughly as N_threads / N_unique_bins.
//
//   Two-phase histogram:
//     Phase 1: Each block accumulates a PRIVATE shared-memory histogram.
//              All N_threads hit num_bins unique bins per block → low contention.
//     Phase 2: Each block atomicAdds its private histogram to global memory.
//              Only block_count threads per bin → much lower contention.
//
// Experiment:
//   Random histogram with N samples, num_bins bins.
//   Variant A: global atomic on every sample
//   Variant B: shared memory private histogram, then merge
// =============================================================================

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
// Kernel A: global atomics — every thread directly increments global histogram
// ---------------------------------------------------------------------------
__global__ void histogram_global_atomic(const int* __restrict__ data,
                                         int*       __restrict__ hist,
                                         int n,
                                         int num_bins) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int bin = data[idx] % num_bins;
        atomicAdd(&hist[bin], 1);
    }
}

// ---------------------------------------------------------------------------
// Kernel B: shared memory atomics + global merge
// ---------------------------------------------------------------------------
__global__ void histogram_smem_atomic(const int* __restrict__ data,
                                       int*       __restrict__ hist,
                                       int n,
                                       int num_bins) {
    extern __shared__ int smem_hist[];  // size = num_bins * sizeof(int)

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Initialize shared histogram to zero
    for (int b = tid; b < num_bins; b += blockDim.x) {
        smem_hist[b] = 0;
    }
    __syncthreads();

    // Phase 1: accumulate in shared memory (low contention within block)
    if (idx < n) {
        int bin = data[idx] % num_bins;
        atomicAdd(&smem_hist[bin], 1);
    }
    __syncthreads();

    // Phase 2: merge block-local histogram into global (fewer global atomics)
    for (int b = tid; b < num_bins; b += blockDim.x) {
        atomicAdd(&hist[b], smem_hist[b]);
    }
}

// ---------------------------------------------------------------------------
// CPU reference histogram
// ---------------------------------------------------------------------------
static std::vector<int> cpu_histogram(const std::vector<int>& data, int num_bins) {
    std::vector<int> hist(num_bins, 0);
    for (int v : data) hist[v % num_bins]++;
    return hist;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    const std::string project_root  = PROJECT_ROOT;
    const std::string config_path   = project_root +
        "/configs/tutorials/12_atomic_operations.yaml";
    const std::string global_config = project_root + "/configs/global.yaml";

    auto cfg = cuda_tutorials::ConfigLoader::from_file(config_path);
    cfg.merge_defaults(global_config);

    auto logger = cuda_tutorials::Logger::create(config_path, project_root,
                                                  "12_atomic_operations");
    logger.log_info("=== Tutorial 12: Atomic Operations ===");

    const int device_id = cfg.get<int>("device", "id");
    const int N         = cfg.get<int>("tutorial", "N");
    const int num_bins  = cfg.get<int>("tutorial", "num_bins");
    const int num_iter  = cfg.get<int>("tutorial", "num_iterations");

    CUDA_CHECK(cudaSetDevice(device_id));
    auto dev_info = cuda_tutorials::get_device_info(device_id);
    cuda_tutorials::print_device_info(logger, dev_info);

    std::ostringstream oss;
    oss << "Config: N=" << N << "  num_bins=" << num_bins << "  num_iter=" << num_iter;
    logger.log_info(oss.str());

    // Generate random data on CPU
    std::vector<int> h_data(N);
    for (int i = 0; i < N; ++i) h_data[i] = rand();

    std::vector<int> h_hist_cpu = cpu_histogram(h_data, num_bins);

    // Device buffers
    int *d_data{}, *d_hist{};
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_hist, num_bins * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), N * sizeof(int), cudaMemcpyHostToDevice));

    const int block_dim = 256;
    const int grid_dim  = (N + block_dim - 1) / block_dim;

    cuda_tutorials::CudaEvent ev_start, ev_stop;

    auto bench = [&](auto fn, const char* label, bool check_correct) -> float {
        CUDA_CHECK(cudaMemset(d_hist, 0, num_bins * sizeof(int)));
        fn(); CUDA_KERNEL_CHECK();  // warm-up

        ev_start.record();
        for (int i = 0; i < num_iter; ++i) {
            CUDA_CHECK(cudaMemset(d_hist, 0, num_bins * sizeof(int)));
            fn();
        }
        ev_stop.record();
        CUDA_CHECK(cudaEventSynchronize(ev_stop.event));
        float t = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&t, ev_start.event, ev_stop.event));
        float mean_ms = t / num_iter;

        double throughput = static_cast<double>(N) / (mean_ms * 1e-3) / 1e9;

        if (check_correct) {
            std::vector<int> h_hist_gpu(num_bins);
            CUDA_CHECK(cudaMemset(d_hist, 0, num_bins * sizeof(int)));
            fn();
            CUDA_KERNEL_CHECK();
            CUDA_CHECK(cudaMemcpy(h_hist_gpu.data(), d_hist,
                                  num_bins * sizeof(int), cudaMemcpyDeviceToHost));
            bool ok = (h_hist_gpu == h_hist_cpu);
            oss.str("");
            oss << label << ": " << std::fixed << std::setprecision(3) << mean_ms
                << " ms  " << std::setprecision(2) << throughput << " Gops/s"
                << (ok ? "  CORRECT" : "  WRONG");
            logger.log_info(oss.str());
        } else {
            oss.str("");
            oss << label << ": " << std::fixed << std::setprecision(3) << mean_ms
                << " ms  " << std::setprecision(2) << throughput << " Gops/s";
            logger.log_info(oss.str());
        }
        return mean_ms;
    };

    float t_global = bench([&]{
        histogram_global_atomic<<<grid_dim, block_dim>>>(d_data, d_hist, N, num_bins);
    }, "Global atomics        ", true);

    float t_smem = bench([&]{
        histogram_smem_atomic<<<grid_dim, block_dim,
                                num_bins * sizeof(int)>>>(d_data, d_hist, N, num_bins);
    }, "Shared mem + merge    ", true);

    oss.str("");
    oss << "Speedup (smem over global): " << std::fixed << std::setprecision(2)
        << (t_global / t_smem) << "x";
    logger.log_info(oss.str());

    logger.log_info(
        "Explanation: global atomics serialize all N threads contending on "
        "num_bins addresses. Shared-memory atomics serialize within each block "
        "only (block_size threads / num_bins bins per block = much less contention). "
        "The final merge only has grid_dim global atomic operations per bin.");

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_hist));

    logger.log_info("Tutorial 12 complete.");
    return 0;
}
