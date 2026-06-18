// =============================================================================
// Tutorial 16: Tensor Cores & WMMA API
//
// Concept:
//   Tensor Cores are specialized compute units on Volta+ GPUs that perform
//   matrix multiply-accumulate (MMA) operations at high throughput.
//
//   H200 sm_90 supports:
//     FP16×FP16 → FP32 (or FP16): 1979 TFLOPS (sparse)
//     BF16×BF16 → FP32: equivalent TFLOPS
//     FP8×FP8 → FP32/FP16: 3958 TFLOPS (sparse)
//     INT8 and INT4 variants
//
//   WMMA API (nvcuda::wmma):
//     Each warp computes a 16×16×16 matrix fragment MMA.
//     fragment<matrix_a, 16, 16, 16, half, row_major>
//     fragment<matrix_b, 16, 16, 16, half, col_major>
//     fragment<accumulator, 16, 16, 16, float>
//
//     load_matrix_sync()  — load tile from shared/global memory
//     mma_sync()          — compute MMA
//     store_matrix_sync() — store result
//
// Experiment:
//   (a) FP32 CUDA cores GEMM (tiled, 32×32 tile)
//   (b) FP16 Tensor Cores GEMM using wmma (16×16×16 fragments)
//
//   Compare GFLOPS and numerical error of FP16 vs FP32.
// =============================================================================

#include <cmath>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>  // nvcuda::wmma

#include "../../common/config_loader.hpp"
#include "../../common/cuda_utils.hpp"
#include "../../common/logger.hpp"

using namespace nvcuda;

// ---------------------------------------------------------------------------
// Kernel A: FP32 tiled GEMM (reference)
// ---------------------------------------------------------------------------
template <int TILE>
__global__ void gemm_fp32_tiled(const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float*       __restrict__ C,
                                 int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.f;

    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int a_col = t * TILE + threadIdx.x;
        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.f;
        int b_row = t * TILE + threadIdx.y;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.f;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE; ++k) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

// ---------------------------------------------------------------------------
// Kernel B: FP16 GEMM using WMMA (Tensor Cores)
//   Each warp handles a 16×16 output tile.
//   Block: 128 threads = 4 warps → each warp takes one 16×16 tile.
//   Grid: ceil(M/16) × ceil(N/16) / warps_per_block in x direction.
// ---------------------------------------------------------------------------
static constexpr int WMMA_M = 16;
static constexpr int WMMA_N = 16;
static constexpr int WMMA_K = 16;

__global__ void gemm_fp16_wmma(const half* __restrict__ A,   // M×K row-major
                                const half* __restrict__ B,   // K×N row-major
                                float*      __restrict__ C,   // M×N row-major
                                int M, int N, int K) {
    // Warp ID within block (4 warps of 32 threads)
    int warp_id = threadIdx.x / 32;

    // Each warp computes one WMMA_M × WMMA_N = 16×16 output tile.
    // Block covers WMMA_M rows and (warps_per_block * WMMA_N) cols.
    int warps_per_row = blockDim.x / 32;  // = warpSize / warpSize = 1 if blockDim=32
    // We use blockDim.x=128 → 4 warps per block, each handles different column tile
    int warp_row = blockIdx.y * WMMA_M;
    int warp_col = (blockIdx.x * warps_per_row + warp_id) * WMMA_N;

    if (warp_row >= M || warp_col >= N) return;

    // Declare accumulators
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    // Loop over K tiles
    for (int k = 0; k < K; k += WMMA_K) {
        if (k + WMMA_K > K) break;  // skip partial tiles (K must be multiple of 16)

        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;

        // Load A tile: rows [warp_row .. warp_row+15], cols [k .. k+15]
        wmma::load_matrix_sync(a_frag, A + warp_row * K + k, K);
        // Load B tile: rows [k .. k+15], cols [warp_col .. warp_col+15]
        wmma::load_matrix_sync(b_frag, B + k * N + warp_col, N);

        // Tensor Core MMA
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    // Store result to C
    wmma::store_matrix_sync(C + warp_row * N + warp_col, acc_frag, N, wmma::mem_row_major);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    const std::string project_root  = PROJECT_ROOT;
    const std::string config_path   = project_root +
        "/configs/tutorials/16_tensor_cores_wmma.yaml";
    const std::string global_config = project_root + "/configs/global.yaml";

    auto cfg = cuda_tutorials::ConfigLoader::from_file(config_path);
    cfg.merge_defaults(global_config);

    auto logger = cuda_tutorials::Logger::create(config_path, project_root,
                                                  "16_tensor_cores_wmma");
    logger.log_info("=== Tutorial 16: Tensor Cores & WMMA ===");

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

    if (M % 16 != 0 || N % 16 != 0 || K % 16 != 0) {
        throw std::runtime_error("M, N, K must all be multiples of 16 for WMMA.");
    }

    // Host data
    std::vector<float> h_A_fp32(M * K), h_B_fp32(K * N);
    for (int i = 0; i < M * K; ++i) h_A_fp32[i] = static_cast<float>(i % 100) * 0.01f;
    for (int i = 0; i < K * N; ++i) h_B_fp32[i] = static_cast<float>(i % 100) * 0.01f;

    // Convert A and B to FP16 for tensor core kernel
    std::vector<half> h_A_fp16(M * K), h_B_fp16(K * N);
    for (int i = 0; i < M * K; ++i) h_A_fp16[i] = __float2half(h_A_fp32[i]);
    for (int i = 0; i < K * N; ++i) h_B_fp16[i] = __float2half(h_B_fp32[i]);

    // Device allocations
    float *d_A_fp32{}, *d_B_fp32{}, *d_C_fp32{}, *d_C_tc{};
    half  *d_A_fp16{}, *d_B_fp16{};

    CUDA_CHECK(cudaMalloc(&d_A_fp32, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B_fp32, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C_fp32, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C_tc,   M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_A_fp16, M * K * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B_fp16, K * N * sizeof(half)));

    CUDA_CHECK(cudaMemcpy(d_A_fp32, h_A_fp32.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_fp32, h_B_fp32.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_A_fp16, h_A_fp16.data(), M * K * sizeof(half),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_fp16, h_B_fp16.data(), K * N * sizeof(half),  cudaMemcpyHostToDevice));

    cuda_tutorials::CudaEvent ev_start, ev_stop;
    double flops = 2.0 * M * N * K;

    auto bench = [&](auto fn, const char* label) -> float {
        fn(); CUDA_KERNEL_CHECK();
        ev_start.record();
        for (int i = 0; i < num_iter; ++i) fn();
        ev_stop.record();
        CUDA_CHECK(cudaEventSynchronize(ev_stop.event));
        float t = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&t, ev_start.event, ev_stop.event));
        float ms = t / num_iter;
        double gf = flops / (ms * 1e-3) / 1e9;
        oss.str("");
        oss << label << ": " << std::fixed << std::setprecision(3) << ms
            << " ms  " << std::setprecision(1) << gf << " GFLOPS";
        logger.log_info(oss.str());
        return ms;
    };

    // FP32 tiled GEMM
    static constexpr int TILE = 16;
    dim3 block_fp32(TILE, TILE);
    dim3 grid_fp32((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    float t_fp32 = bench([&]{
        gemm_fp32_tiled<TILE><<<grid_fp32, block_fp32>>>(d_A_fp32, d_B_fp32, d_C_fp32, M, N, K);
    }, "FP32 tiled GEMM (CUDA cores)");

    // FP16 Tensor Core GEMM
    // 4 warps per block, each handles 16×16 tile
    int warps_per_block = 4;
    dim3 block_tc(warps_per_block * 32);
    dim3 grid_tc((N / WMMA_N + warps_per_block - 1) / warps_per_block,
                 (M + WMMA_M - 1) / WMMA_M);
    float t_tc = bench([&]{
        gemm_fp16_wmma<<<grid_tc, block_tc>>>(d_A_fp16, d_B_fp16, d_C_tc, M, N, K);
    }, "FP16 wmma GEMM (Tensor Cores)");

    // Numerical comparison
    std::vector<float> h_C_fp32(M * N), h_C_tc(M * N);
    CUDA_CHECK(cudaMemcpy(h_C_fp32.data(), d_C_fp32, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_C_tc.data(),   d_C_tc,   M * N * sizeof(float), cudaMemcpyDeviceToHost));

    double max_err = 0.0, mean_err = 0.0;
    for (int i = 0; i < M * N; ++i) {
        double e = std::abs(static_cast<double>(h_C_fp32[i] - h_C_tc[i]));
        max_err  = std::max(max_err, e);
        mean_err += e;
    }
    mean_err /= (M * N);

    oss.str("");
    oss << "Numerical error (FP16 TC vs FP32): max=" << std::scientific << max_err
        << "  mean=" << mean_err;
    logger.log_info(oss.str());

    oss.str("");
    oss << "Tensor Core speedup: " << std::fixed << std::setprecision(2)
        << (t_fp32 / t_tc) << "x";
    logger.log_info(oss.str());

    logger.log_info(
        "Tensor Cores execute a 16×16×16 MMA in a single warp instruction. "
        "FP16 inputs allow 2× more data per memory transaction. "
        "Small numerical error is expected: FP16 has 10-bit mantissa vs FP32's 23-bit.");

    CUDA_CHECK(cudaFree(d_A_fp32)); CUDA_CHECK(cudaFree(d_B_fp32));
    CUDA_CHECK(cudaFree(d_C_fp32)); CUDA_CHECK(cudaFree(d_C_tc));
    CUDA_CHECK(cudaFree(d_A_fp16)); CUDA_CHECK(cudaFree(d_B_fp16));

    logger.log_info("Tutorial 16 complete.");
    return 0;
}
