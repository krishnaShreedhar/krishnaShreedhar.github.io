// =============================================================================
// Tutorial 20: cuBLAS
//
// Concept:
//   cuBLAS is NVIDIA's GPU-accelerated BLAS library. It provides:
//     Level 1: vector ops (dot, axpy, scal)
//     Level 2: matrix-vector ops (gemv)
//     Level 3: matrix-matrix ops (gemm, syrk, trsm)
//
//   Key APIs:
//     cublasSgemm()   — FP32 GEMM
//     cublasGemmEx()  — extended GEMM with mixed precision and algorithm hints
//     cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH) — enable tensor cores
//
//   cuBLAS expects column-major layout by default (Fortran convention).
//   For row-major C arrays, use the identity:
//     C_col = (B^T × A^T)^T  → cublasS/GemmEx with CUBLAS_OP_N/T swapped
//
// Experiment:
//   (a) cublasSgemm: FP32 GEMM
//   (b) cublasGemmEx with CUBLAS_COMPUTE_32F_FAST_TF32: Tensor Core GEMM
//   (c) Manual tiled GEMM for comparison
// =============================================================================

#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../../common/config_loader.hpp"
#include "../../common/cuda_utils.hpp"
#include "../../common/logger.hpp"

// ---------------------------------------------------------------------------
// Reference tiled GEMM for comparison
// ---------------------------------------------------------------------------
template <int TILE>
__global__ void gemm_tiled_ref(const float* __restrict__ A,
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
// main
// ---------------------------------------------------------------------------
int main() {
    const std::string project_root  = PROJECT_ROOT;
    const std::string config_path   = project_root + "/configs/tutorials/20_cublas.yaml";
    const std::string global_config = project_root + "/configs/global.yaml";

    auto cfg = cuda_tutorials::ConfigLoader::from_file(config_path);
    cfg.merge_defaults(global_config);

    auto logger = cuda_tutorials::Logger::create(config_path, project_root, "20_cublas");
    logger.log_info("=== Tutorial 20: cuBLAS ===");

    const int   device_id      = cfg.get<int>("device",   "id");
    const int   M              = cfg.get<int>("tutorial", "M");
    const int   N              = cfg.get<int>("tutorial", "N");
    const int   K              = cfg.get<int>("tutorial", "K");
    const float alpha          = cfg.get<float>("tutorial", "alpha");
    const float beta           = cfg.get<float>("tutorial", "beta");
    const bool  use_tc         = cfg.get<bool>("tutorial", "use_tensor_cores");
    const int   num_iter       = cfg.get<int>("tutorial", "num_iterations");

    CUDA_CHECK(cudaSetDevice(device_id));
    auto dev_info = cuda_tutorials::get_device_info(device_id);
    cuda_tutorials::print_device_info(logger, dev_info);

    std::ostringstream oss;
    oss << "Config: M=" << M << "  N=" << N << "  K=" << K
        << "  alpha=" << alpha << "  beta=" << beta
        << "  use_tensor_cores=" << use_tc;
    logger.log_info(oss.str());

    // Create cuBLAS handle
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    if (use_tc) {
        // TF32 Tensor Core math mode (H100/H200: uses TF32 for FP32 inputs)
        CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));
        logger.log_info("Tensor Core math mode: CUBLAS_TF32_TENSOR_OP_MATH enabled");
    } else {
        CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
        logger.log_info("Math mode: CUBLAS_DEFAULT_MATH (no Tensor Cores)");
    }

    // Allocate device memory
    float *d_A{}, *d_B{}, *d_C{}, *d_C_tiled{};
    CUDA_CHECK(cudaMalloc(&d_A,      M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B,      K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C,      M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C_tiled, M * N * sizeof(float)));

    std::vector<float> h_A(M * K), h_B(K * N);
    for (int i = 0; i < M * K; ++i) h_A[i] = static_cast<float>(i % 100) * 0.01f;
    for (int i = 0; i < K * N; ++i) h_B[i] = static_cast<float>(i % 100) * 0.01f;
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, M * N * sizeof(float)));

    cuda_tutorials::CudaEvent ev_start, ev_stop;
    double flops = 2.0 * M * N * K;

    // ---------------------------------------------------------------------------
    // cuBLAS SGEMM (row-major trick: compute C^T = B^T × A^T)
    // cuBLAS is column-major, so:
    //   C (M×N, row-major) → C^T (N×M, col-major)
    //   A (M×K, row-major) → A^T (K×M, col-major)
    //   B (K×N, row-major) → B^T (N×K, col-major)
    // cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N)
    // ---------------------------------------------------------------------------
    {
        CUBLAS_CHECK(cublasSgemm(handle,
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, M, K,
                                 &alpha,
                                 d_B, N,
                                 d_A, K,
                                 &beta,
                                 d_C, N));
        CUDA_KERNEL_CHECK();

        ev_start.record();
        for (int i = 0; i < num_iter; ++i) {
            CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                     N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N));
        }
        ev_stop.record();
        CUDA_CHECK(cudaEventSynchronize(ev_stop.event));
        float t = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&t, ev_start.event, ev_stop.event));
        float ms = t / num_iter;
        double gf = flops / (ms * 1e-3) / 1e9;

        oss.str("");
        oss << "cublasSgemm: " << std::fixed << std::setprecision(3) << ms
            << " ms  " << std::setprecision(1) << gf << " GFLOPS";
        logger.log_info(oss.str());
    }

    // ---------------------------------------------------------------------------
    // Manual tiled GEMM
    // ---------------------------------------------------------------------------
    {
        static constexpr int TILE = 32;
        dim3 block(TILE, TILE);
        dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
        gemm_tiled_ref<TILE><<<grid, block>>>(d_A, d_B, d_C_tiled, M, N, K);
        CUDA_KERNEL_CHECK();

        ev_start.record();
        for (int i = 0; i < num_iter; ++i)
            gemm_tiled_ref<TILE><<<grid, block>>>(d_A, d_B, d_C_tiled, M, N, K);
        ev_stop.record();
        CUDA_CHECK(cudaEventSynchronize(ev_stop.event));
        float t = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&t, ev_start.event, ev_stop.event));
        float ms = t / num_iter;
        double gf = flops / (ms * 1e-3) / 1e9;

        oss.str("");
        oss << "Manual tiled GEMM: " << std::fixed << std::setprecision(3) << ms
            << " ms  " << std::setprecision(1) << gf << " GFLOPS";
        logger.log_info(oss.str());
    }

    logger.log_info(
        "cuBLAS achieves peak GFLOPS through:\n"
        "  - Warp-specialized threadblocks with persistent kernels\n"
        "  - TF32 Tensor Cores (drops mantissa to 10 bits for FP32 GEMM)\n"
        "  - Multi-stage async pipelining (cp.async + wgmma)\n"
        "  - Kernel autotuning across hundreds of tile configurations");

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C)); CUDA_CHECK(cudaFree(d_C_tiled));

    logger.log_info("Tutorial 20 complete.");
    return 0;
}
