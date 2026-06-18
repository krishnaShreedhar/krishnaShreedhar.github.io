// =============================================================================
// Tutorial 10: Unified Memory
//
// Concept:
//   cudaMallocManaged() allocates memory accessible from both CPU and GPU
//   without explicit memcpy. The CUDA runtime migrates pages on demand.
//
//   Page faults occur when the GPU accesses a page still resident on the CPU
//   (or vice versa). Each fault triggers a page migration — expensive (~µs).
//
//   Optimization strategies:
//     (a) No hints — pages migrate on demand
//     (b) cudaMemPrefetchAsync — proactively move pages to GPU before kernel
//     (c) cudaMemAdvise(ReadMostly) — allow pages to be replicated on both
//
//   cudaMemRangeGetAttribute(CUDA_MANAGED_ATTR_PREFETCHED_LOCATION) can check
//   where pages are; page fault count via driver API (approximate here).
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
// Kernel: vector addition with unified memory pointers
// ---------------------------------------------------------------------------
__global__ void vector_add_um(const float* __restrict__ a,
                               const float* __restrict__ b,
                               float*       __restrict__ c,
                               int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

// ---------------------------------------------------------------------------
// Run one experiment variant, return mean kernel time in ms
// ---------------------------------------------------------------------------
static float run_variant(cuda_tutorials::Logger& logger,
                          float*                  d_a,
                          float*                  d_b,
                          float*                  d_c,
                          int                     n,
                          int                     device_id,
                          int                     num_iter,
                          const std::string&      label) {
    const int block_dim = 256;
    const int grid_dim  = (n + block_dim - 1) / block_dim;

    // Warm-up
    vector_add_um<<<grid_dim, block_dim>>>(d_a, d_b, d_c, n);
    CUDA_KERNEL_CHECK();

    cuda_tutorials::CudaEvent ev_start, ev_stop;
    std::vector<float> times(num_iter);
    for (int i = 0; i < num_iter; ++i) {
        ev_start.record();
        vector_add_um<<<grid_dim, block_dim>>>(d_a, d_b, d_c, n);
        ev_stop.record();
        times[i] = ev_stop.elapsed_ms(ev_start);
    }

    float mean_ms = static_cast<float>(
        std::accumulate(times.begin(), times.end(), 0.0) / num_iter);

    double bw = 3.0 * n * sizeof(float) / (mean_ms * 1e-3) / 1e9;

    std::ostringstream oss;
    oss << label << ": " << std::fixed << std::setprecision(3) << mean_ms
        << " ms  BW=" << std::setprecision(2) << bw << " GB/s";
    logger.log_info(oss.str());

    return mean_ms;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    const std::string project_root  = PROJECT_ROOT;
    const std::string config_path   = project_root + "/configs/tutorials/10_unified_memory.yaml";
    const std::string global_config = project_root + "/configs/global.yaml";

    auto cfg = cuda_tutorials::ConfigLoader::from_file(config_path);
    cfg.merge_defaults(global_config);

    auto logger = cuda_tutorials::Logger::create(config_path, project_root,
                                                  "10_unified_memory");
    logger.log_info("=== Tutorial 10: Unified Memory ===");

    const int  device_id         = cfg.get<int>("device",   "id");
    const int  N                 = cfg.get<int>("tutorial", "N");
    const int  num_iter          = cfg.get<int>("tutorial", "num_iterations");
    const bool prefetch_enabled  = cfg.get<bool>("tutorial", "prefetch_enabled");
    const bool advise_read_mostly = cfg.get<bool>("tutorial", "advise_read_mostly");

    CUDA_CHECK(cudaSetDevice(device_id));
    auto dev_info = cuda_tutorials::get_device_info(device_id);
    cuda_tutorials::print_device_info(logger, dev_info);

    std::ostringstream oss;
    oss << "Config: N=" << N << "  prefetch=" << prefetch_enabled
        << "  read_mostly=" << advise_read_mostly;
    logger.log_info(oss.str());

    size_t bytes = N * sizeof(float);

    // ---------------------------------------------------------------------------
    // (a) No hints — demand paging
    // ---------------------------------------------------------------------------
    {
        float *a{}, *b{}, *c{};
        CUDA_CHECK(cudaMallocManaged(&a, bytes));
        CUDA_CHECK(cudaMallocManaged(&b, bytes));
        CUDA_CHECK(cudaMallocManaged(&c, bytes));

        // Initialize on CPU — pages resident on CPU
        for (int i = 0; i < N; ++i) {
            a[i] = static_cast<float>(i) * 0.001f;
            b[i] = static_cast<float>(N - i) * 0.001f;
        }

        run_variant(logger, a, b, c, N, device_id, num_iter,
                    "(a) No hints (demand paging)      ");

        CUDA_CHECK(cudaFree(a));
        CUDA_CHECK(cudaFree(b));
        CUDA_CHECK(cudaFree(c));
    }

    // ---------------------------------------------------------------------------
    // (b) Prefetch before kernel
    // ---------------------------------------------------------------------------
    if (prefetch_enabled) {
        float *a{}, *b{}, *c{};
        CUDA_CHECK(cudaMallocManaged(&a, bytes));
        CUDA_CHECK(cudaMallocManaged(&b, bytes));
        CUDA_CHECK(cudaMallocManaged(&c, bytes));

        for (int i = 0; i < N; ++i) {
            a[i] = static_cast<float>(i) * 0.001f;
            b[i] = static_cast<float>(N - i) * 0.001f;
        }

        // Proactively migrate pages to GPU before kernel runs
        CUDA_CHECK(cudaMemPrefetchAsync(a, bytes, device_id));
        CUDA_CHECK(cudaMemPrefetchAsync(b, bytes, device_id));
        CUDA_CHECK(cudaMemPrefetchAsync(c, bytes, device_id));
        CUDA_CHECK(cudaDeviceSynchronize());

        run_variant(logger, a, b, c, N, device_id, num_iter,
                    "(b) Prefetch to GPU before kernel  ");

        CUDA_CHECK(cudaFree(a));
        CUDA_CHECK(cudaFree(b));
        CUDA_CHECK(cudaFree(c));
    }

    // ---------------------------------------------------------------------------
    // (c) ReadMostly advice — allows page replication
    // ---------------------------------------------------------------------------
    if (advise_read_mostly) {
        float *a{}, *b{}, *c{};
        CUDA_CHECK(cudaMallocManaged(&a, bytes));
        CUDA_CHECK(cudaMallocManaged(&b, bytes));
        CUDA_CHECK(cudaMallocManaged(&c, bytes));

        for (int i = 0; i < N; ++i) {
            a[i] = static_cast<float>(i) * 0.001f;
            b[i] = static_cast<float>(N - i) * 0.001f;
        }

        // Hint that a and b are read-mostly — GPU may replicate these pages
        CUDA_CHECK(cudaMemAdvise(a, bytes, cudaMemAdviseSetReadMostly, device_id));
        CUDA_CHECK(cudaMemAdvise(b, bytes, cudaMemAdviseSetReadMostly, device_id));
        CUDA_CHECK(cudaMemPrefetchAsync(a, bytes, device_id));
        CUDA_CHECK(cudaMemPrefetchAsync(b, bytes, device_id));
        CUDA_CHECK(cudaDeviceSynchronize());

        run_variant(logger, a, b, c, N, device_id, num_iter,
                    "(c) ReadMostly + prefetch          ");

        CUDA_CHECK(cudaFree(a));
        CUDA_CHECK(cudaFree(b));
        CUDA_CHECK(cudaFree(c));
    }

    logger.log_info(
        "Explanation: demand paging incurs page faults on first kernel access. "
        "Prefetch amortizes migration before the kernel runs. "
        "ReadMostly + prefetch allows the GPU to hold replicated read-only pages, "
        "avoiding eviction when the CPU also accesses the data.");

    logger.log_info("Tutorial 10 complete.");
    return 0;
}
