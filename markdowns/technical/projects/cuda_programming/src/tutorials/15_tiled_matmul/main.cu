// =============================================================================
// Tutorial 15: Tiled Matrix Multiplication
//
// Concept:
//   GEMM: C[M×N] = A[M×K] × B[K×N]
//
//   Naive GEMM: each thread computes one C[i,j] by looping over K.
//   Each thread fetches K elements from A and K from B → 2MNK global reads.
//
//   Tiled GEMM (shared memory):
//   The K dimension is split into tiles of size TILE.
//   Each block loads one tile of A and one tile of B into shared memory,
//   computes the partial dot product, then advances to the next tile.
//   Each global load is reused TILE times within the block.
//   Memory traffic: 2MNK/TILE reads → TILE× reduction in global memory access.
//
//   For TILE=32, this gives 32× fewer global reads at the cost of shared memory.
//   Peak GEMM efficiency requires register blocking (outer-product style),
//   which cuBLAS and CUTLASS implement. This tutorial shows the tiled version.
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
// Kernel A: Naive GEMM — one thread per output element
// ---------------------------------------------------------------------------
__global__ void gemm_naive(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float*       __restrict__ C,
                            int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float acc = 0.f;
    for (int k = 0; k < K; ++k) {
        acc += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = acc;
}

// ---------------------------------------------------------------------------
// Kernel B: Tiled GEMM with shared memory
// Template param TILE must equal blockDim.x = blockDim.y.
// ---------------------------------------------------------------------------
template <int TILE>
__global__ void gemm_tiled(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float*       __restrict__ C,
                            int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.f;

    // Loop over tiles along the K dimension
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        // Load A tile: row × [t*TILE .. t*TILE+TILE-1]
        int a_col = t * TILE + threadIdx.x;
        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K)
                                        ? A[row * K + a_col]
                                        : 0.f;
        // Load B tile: [t*TILE .. t*TILE+TILE-1] × col
        int b_row = t * TILE + threadIdx.y;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N)
                                        ? B[b_row * N + col]
                                        : 0.f;
        __syncthreads();

        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) C[row * N + col] = acc;
}

// ---------------------------------------------------------------------------
// GFLOPS computation: 2MNK FLOPs (multiply + add per element of K)
// ---------------------------------------------------------------------------
static double gflops(int M, int N, int K, float ms) {
    return 2.0 * M * N * K / (ms * 1e-3) / 1e9;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    const std::string project_root  = PROJECT_ROOT;
    const std::string config_path   = project_root + "/configs/tutorials/15_tiled_matmul.yaml";
    const std::string global_config = project_root + "/configs/global.yaml";

    auto cfg = cuda_tutorials::ConfigLoader::from_file(config_path);
    cfg.merge_defaults(global_config);

    auto logger = cuda_tutorials::Logger::create(config_path, project_root,
                                                  "15_tiled_matmul");
    logger.log_info("=== Tutorial 15: Tiled Matrix Multiplication ===");

    const int device_id = cfg.get<int>("device", "id");
    const int M         = cfg.get<int>("tutorial", "M");
    const int N         = cfg.get<int>("tutorial", "N");
    const int K         = cfg.get<int>("tutorial", "K");
    const int num_iter  = cfg.get<int>("tutorial", "num_iterations");

    CUDA_CHECK(cudaSetDevice(device_id));
    auto dev_info = cuda_tutorials::get_device_info(device_id);
    cuda_tutorials::print_device_info(logger, dev_info);

    std::ostringstream oss;
    oss << "Config: M=" << M << "  N=" << N << "  K=" << K << "  num_iter=" << num_iter;
    logger.log_info(oss.str());

    // Allocate host matrices
    std::vector<float> h_A(M * K), h_B(K * N), h_C_naive(M * N), h_C_tiled(M * N);
    for (int i = 0; i < M * K; ++i) h_A[i] = static_cast<float>(i % 100) * 0.01f;
    for (int i = 0; i < K * N; ++i) h_B[i] = static_cast<float>(i % 100) * 0.01f;

    float *d_A{}, *d_B{}, *d_C{};
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));

    static constexpr int TILE = 32;
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    cuda_tutorials::CudaEvent ev_start, ev_stop;

    auto bench = [&](auto fn, const char* label) -> float {
        fn(); CUDA_KERNEL_CHECK();
        ev_start.record();
        for (int i = 0; i < num_iter; ++i) fn();
        ev_stop.record();
        CUDA_CHECK(cudaEventSynchronize(ev_stop.event));
        float t = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&t, ev_start.event, ev_stop.event));
        float ms = t / num_iter;
        double gf = gflops(M, N, K, ms);
        oss.str("");
        oss << label << ": " << std::fixed << std::setprecision(3) << ms
            << " ms  " << std::setprecision(2) << gf << " GFLOPS";
        logger.log_info(oss.str());
        return ms;
    };

    // Naive GEMM
    float t_naive = bench([&]{
        // For naive, use 2D block of (16,16) to avoid exceeding 1024 threads/block
        dim3 block_naive(16, 16);
        dim3 grid_naive((N + 15) / 16, (M + 15) / 16);
        gemm_naive<<<grid_naive, block_naive>>>(d_A, d_B, d_C, M, N, K);
    }, "Naive GEMM            ");
    CUDA_CHECK(cudaMemcpy(h_C_naive.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Tiled GEMM
    float t_tiled = bench([&]{
        gemm_tiled<TILE><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }, "Tiled GEMM (tile=32)  ");
    CUDA_CHECK(cudaMemcpy(h_C_tiled.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Correctness
    double max_err = 0.0;
    for (int i = 0; i < M * N; ++i)
        max_err = std::max(max_err, std::abs(static_cast<double>(h_C_naive[i] - h_C_tiled[i])));
    oss.str("");
    oss << "Correctness (naive vs tiled): max_diff=" << max_err
        << (max_err < 1e-3 ? "  PASS" : "  FAIL");
    logger.log_info(oss.str());

    oss.str("");
    oss << "Tiled speedup over naive: " << std::fixed << std::setprecision(2)
        << (t_naive / t_tiled) << "x";
    logger.log_info(oss.str());

    logger.log_info(
        "Analysis: naive GEMM fetches A[row,k] and B[k,col] fresh from global "
        "memory for each k. Tiled GEMM loads TILE columns of A and TILE rows of B "
        "into shared memory once per tile, reusing each element TILE times. "
        "Next level of optimization: register-tiled (outer-product) GEMM as in cuBLAS.");

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    logger.log_info("Tutorial 15 complete.");
    return 0;
}
