// =============================================================================
// Tutorial 05: Shared Memory & Bank Conflicts
//
// Concept:
//   Shared memory on H200 is divided into 32 banks (one per warp lane),
//   each bank is 4 bytes wide, cycling every 128 bytes.
//   Bank for element i: bank = (i * sizeof(float) / 4) % 32  → bank = i % 32
//
//   A BANK CONFLICT occurs when two or more threads in a warp access
//   different addresses that map to the SAME bank — the hardware serializes
//   those accesses.
//
// Matrix Transpose experiment:
//   We transpose an M×M matrix using shared memory tiles.
//
//   Naive: tile[threadIdx.y][threadIdx.x]
//     → When writing transposed, threads in a warp all hit bank
//       (threadIdx.y % 32) — a 32-way conflict!
//
//   Padded: tile[threadIdx.y][threadIdx.x + PADDING]
//     → Padding shifts the column index, spreading accesses across banks.
//       With padding=1, bank = (col + 1) % 32 → no two warps share a bank.
//
// Config: tile_size (must be ≤ 32 for simplicity), padding (0 or 1)
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
// Kernel: naive transpose — bank conflicts on shared memory write
// ---------------------------------------------------------------------------
template <int TILE>
__global__ void transpose_naive(const float* __restrict__ src,
                                float*       __restrict__ dst,
                                int N) {
    // Shared memory tile (no padding)
    __shared__ float tile[TILE][TILE];

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;

    // Load: coalesced reads from global (threads read consecutive cols)
    if (x < N && y < N) tile[threadIdx.y][threadIdx.x] = src[y * N + x];
    __syncthreads();

    // Swap x/y indices for transposition
    int nx = blockIdx.y * TILE + threadIdx.x;
    int ny = blockIdx.x * TILE + threadIdx.y;

    // Write: read tile[threadIdx.x][threadIdx.y]
    // threadIdx.x varies → accesses column threadIdx.x of the same row threadIdx.y
    // Bank = threadIdx.y % 32 → all 32 threads in a warp hit the same bank!
    if (nx < N && ny < N) dst[ny * N + nx] = tile[threadIdx.x][threadIdx.y];
}

// ---------------------------------------------------------------------------
// Kernel: padded transpose — bank-conflict-free
// ---------------------------------------------------------------------------
template <int TILE, int PAD>
__global__ void transpose_padded(const float* __restrict__ src,
                                 float*       __restrict__ dst,
                                 int N) {
    // Extra PAD column per row breaks the bank-conflict pattern
    __shared__ float tile[TILE][TILE + PAD];

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;

    if (x < N && y < N) tile[threadIdx.y][threadIdx.x] = src[y * N + x];
    __syncthreads();

    int nx = blockIdx.y * TILE + threadIdx.x;
    int ny = blockIdx.x * TILE + threadIdx.y;

    // Now tile[row][col+PAD] — the extra PAD ensures different bank mapping
    if (nx < N && ny < N) dst[ny * N + nx] = tile[threadIdx.x][threadIdx.y];
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    const std::string project_root  = PROJECT_ROOT;
    const std::string config_path   = project_root +
        "/configs/tutorials/05_shared_memory_bank_conflicts.yaml";
    const std::string global_config = project_root + "/configs/global.yaml";

    auto cfg = cuda_tutorials::ConfigLoader::from_file(config_path);
    cfg.merge_defaults(global_config);

    auto logger = cuda_tutorials::Logger::create(config_path, project_root,
                                                  "05_shared_mem_bank_conflicts");
    logger.log_info("=== Tutorial 05: Shared Memory & Bank Conflicts ===");

    const int device_id  = cfg.get<int>("device", "id");
    const int mat_size   = cfg.get<int>("tutorial", "matrix_size");
    const int num_iter   = cfg.get<int>("tutorial", "num_iterations");

    CUDA_CHECK(cudaSetDevice(device_id));
    auto dev_info = cuda_tutorials::get_device_info(device_id);
    cuda_tutorials::print_device_info(logger, dev_info);

    std::ostringstream oss;
    oss << "Config: matrix=" << mat_size << "x" << mat_size
        << "  tile=32  padding=1  num_iter=" << num_iter;
    logger.log_info(oss.str());

    // Bank explanation
    logger.log_info(
        "Bank conflict theory: shared mem has 32 banks (4B each). "
        "Bank for element i = i % 32. "
        "In naive transpose, reading tile[threadIdx.x][threadIdx.y] means "
        "all threads in a warp share bank=(threadIdx.y % 32) => 32-way conflict. "
        "Padding shifts indices so each warp lane hits a unique bank.");

    const long long total = static_cast<long long>(mat_size) * mat_size;
    std::vector<float> h_src(total), h_dst(total);
    for (long long i = 0; i < total; ++i) h_src[i] = static_cast<float>(i);

    float *d_src{}, *d_dst{};
    CUDA_CHECK(cudaMalloc(&d_src, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dst, total * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_src, h_src.data(), total * sizeof(float), cudaMemcpyHostToDevice));

    static constexpr int TILE = 32;
    static constexpr int PAD  = 1;
    dim3 block(TILE, TILE);
    dim3 grid((mat_size + TILE - 1) / TILE, (mat_size + TILE - 1) / TILE);

    cuda_tutorials::CudaEvent ev_start, ev_stop;

    auto bench = [&](auto fn, const char* label) {
        fn(); CUDA_KERNEL_CHECK();
        ev_start.record();
        for (int i = 0; i < num_iter; ++i) fn();
        ev_stop.record();
        CUDA_CHECK(cudaEventSynchronize(ev_stop.event));
        float t = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&t, ev_start.event, ev_stop.event));
        float mean_ms = t / num_iter;
        double bw = 2.0 * total * sizeof(float) / (mean_ms * 1e-3) / 1e9;
        oss.str("");
        oss << label << ": " << std::fixed << std::setprecision(3) << mean_ms
            << " ms  " << std::setprecision(2) << bw << " GB/s";
        logger.log_info(oss.str());
        return mean_ms;
    };

    float t_naive   = bench([&]{ transpose_naive<TILE><<<grid, block>>>(d_src, d_dst, mat_size); },
                            "Naive (bank conflicts)  ");
    float t_padded  = bench([&]{ transpose_padded<TILE, PAD><<<grid, block>>>(d_src, d_dst, mat_size); },
                            "Padded (conflict-free)  ");

    oss.str("");
    oss << "Speedup from padding: " << std::fixed << std::setprecision(2)
        << (t_naive / t_padded) << "x";
    logger.log_info(oss.str());

    // Correctness: compare naive and padded outputs
    std::vector<float> h_out_naive(total), h_out_padded(total);
    transpose_naive<TILE><<<grid, block>>>(d_src, d_dst, mat_size);
    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaMemcpy(h_out_naive.data(), d_dst, total * sizeof(float), cudaMemcpyDeviceToHost));

    transpose_padded<TILE, PAD><<<grid, block>>>(d_src, d_dst, mat_size);
    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaMemcpy(h_out_padded.data(), d_dst, total * sizeof(float), cudaMemcpyDeviceToHost));

    double max_err = 0.0;
    for (long long i = 0; i < total; ++i)
        max_err = std::max(max_err, std::abs(static_cast<double>(h_out_naive[i] - h_out_padded[i])));
    oss.str("");
    oss << "Correctness (naive vs padded max diff): " << max_err
        << (max_err < 1e-6 ? "  PASS" : "  FAIL");
    logger.log_info(oss.str());

    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));

    logger.log_info("Tutorial 05 complete.");
    return 0;
}
