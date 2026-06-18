// =============================================================================
// Tutorial 04: Memory Coalescing
//
// Concept:
//   Global memory is served in 128-byte cache lines (sm_90). When all 32
//   threads in a warp access consecutive addresses, the hardware merges the
//   32 × 4-byte = 128-byte reads into ONE memory transaction → coalesced.
//
//   If threads access non-consecutive addresses (e.g. each thread accesses
//   a different cache line), up to 32 separate transactions are issued →
//   uncoalesced, catastrophic bandwidth loss.
//
// Experiment:
//   Matrix stored in row-major order (C-order), size M×N.
//   Kernel A: each thread accesses consecutive elements in the same row
//             → threads (t0,t1,...,t31) read columns (0,1,...,31) of same row
//             → coalesced
//   Kernel B: each thread accesses the same column in consecutive rows
//             (stride = N apart) → threads read elements N floats apart
//             → uncoalesced (N floats × 4 bytes >> 128 byte cache line)
//
//   We also sweep an explicit stride parameter from the config.
// =============================================================================

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
// Kernel: coalesced row-major copy
//   Thread t reads element (row, t) — consecutive column addresses per warp.
// ---------------------------------------------------------------------------
__global__ void coalesced_copy(const float* __restrict__ src,
                               float*       __restrict__ dst,
                               int M, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // fast dim
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        dst[row * N + col] = src[row * N + col];
    }
}

// ---------------------------------------------------------------------------
// Kernel: non-coalesced column-major / strided copy
//   Thread t reads element (t, col) — consecutive row addresses, stride=N.
//   Warp reads 32 addresses that are N*4 bytes apart.
// ---------------------------------------------------------------------------
__global__ void strided_copy(const float* __restrict__ src,
                              float*       __restrict__ dst,
                              int M, int N, int stride) {
    // Thread (tx) reads elements stride apart in the flat array
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * stride;
    int total = M * N;
    if (idx < total) {
        dst[idx] = src[idx];
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    const std::string project_root  = PROJECT_ROOT;
    const std::string config_path   = project_root + "/configs/tutorials/04_memory_coalescing.yaml";
    const std::string global_config = project_root + "/configs/global.yaml";

    auto cfg = cuda_tutorials::ConfigLoader::from_file(config_path);
    cfg.merge_defaults(global_config);

    auto logger = cuda_tutorials::Logger::create(config_path, project_root,
                                                  "04_memory_coalescing");
    logger.log_info("=== Tutorial 04: Memory Coalescing ===");

    const int device_id  = cfg.get<int>("device", "id");
    const int M          = cfg.get<int>("tutorial", "M");
    const int N          = cfg.get<int>("tutorial", "N");
    const int num_iter   = cfg.get<int>("tutorial", "num_iterations");
    auto strides         = cfg.get_vector<int>("tutorial", "strides");

    CUDA_CHECK(cudaSetDevice(device_id));
    auto dev_info = cuda_tutorials::get_device_info(device_id);
    cuda_tutorials::print_device_info(logger, dev_info);

    std::ostringstream oss;
    oss << "Config: M=" << M << "  N=" << N << "  num_iter=" << num_iter;
    logger.log_info(oss.str());

    const long long total = static_cast<long long>(M) * N;
    std::vector<float> h_src(total);
    for (long long i = 0; i < total; ++i) h_src[i] = static_cast<float>(i) * 0.001f;

    float *d_src{}, *d_dst{};
    CUDA_CHECK(cudaMalloc(&d_src, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dst, total * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_src, h_src.data(), total * sizeof(float), cudaMemcpyHostToDevice));

    cuda_tutorials::CudaEvent ev_start, ev_stop;

    auto bench = [&](auto kernel_fn, double bytes_transferred) -> double {
        kernel_fn(); // warm-up
        CUDA_KERNEL_CHECK();
        ev_start.record();
        for (int i = 0; i < num_iter; ++i) kernel_fn();
        ev_stop.record();
        CUDA_CHECK(cudaEventSynchronize(ev_stop.event));
        float total_ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&total_ms, ev_start.event, ev_stop.event));
        float mean_ms = total_ms / num_iter;
        return bytes_transferred / (mean_ms * 1e-3) / 1e9;
    };

    // ---------------------------------------------------------------------------
    // (a) Coalesced: 2D grid, consecutive threads → consecutive columns
    // ---------------------------------------------------------------------------
    {
        dim3 block(32, 8);
        dim3 grid((N + 31) / 32, (M + 7) / 8);
        double bw = bench([&]{
            coalesced_copy<<<grid, block>>>(d_src, d_dst, M, N);
        }, 2.0 * total * sizeof(float));

        oss.str("");
        oss << "Coalesced (row-major)  : "
            << std::fixed << std::setprecision(2) << bw << " GB/s";
        logger.log_info(oss.str());
    }

    // ---------------------------------------------------------------------------
    // (b) Strided (uncoalesced): sweep strides
    // ---------------------------------------------------------------------------
    logger.log_info("--- Strided access sweep ---");
    for (int s : strides) {
        int eff_n = static_cast<int>(total / s);
        if (eff_n == 0) continue;
        int block_dim = 256;
        int grid_dim  = (eff_n + block_dim - 1) / block_dim;

        double bw = bench([&]{
            strided_copy<<<grid_dim, block_dim>>>(d_src, d_dst, M, N, s);
        }, 2.0 * eff_n * sizeof(float));

        oss.str("");
        oss << "Strided stride=" << std::setw(3) << s
            << "           : "
            << std::fixed << std::setprecision(2) << bw << " GB/s";
        logger.log_info(oss.str());
    }

    // Explanation
    logger.log_info(
        "Observation: coalesced access saturates HBM bandwidth. "
        "As stride increases, each warp issues more cache line fetches "
        "for fewer useful bytes, sharply degrading effective bandwidth.");

    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));

    logger.log_info("Tutorial 04 complete.");
    return 0;
}
