// =============================================================================
// Tutorial 17: CUTLASS
//
// Concept:
//   CUTLASS (CUDA Templates for Linear Algebra Subroutines and Solvers) is
//   NVIDIA's C++ template library for high-performance GEMM and convolutions.
//
//   It exposes a hierarchical abstraction:
//     Device level    → cutlass::gemm::device::Gemm<...>
//     Threadblock     → MmaPipelined, MmaMultistage
//     Warp            → MmaTensorOp (Tensor Core MMA)
//     Thread/Warp     → Register-level epilogue
//
//   CUTLASS achieves near-cuBLAS performance through:
//     - Multi-stage async software pipelining (cp.async)
//     - Tensor Core MMA via WMMA / mma.sync PTX
//     - Warp-specialized threadblocks (H100+)
//
//   Key template parameters:
//     ElementA, ElementB, ElementC  — data types
//     LayoutA, LayoutB, LayoutC     — matrix layouts (row/col major)
//     MmaShape (tile_M, tile_N, tile_K) — threadblock tile dimensions
//
// Experiment:
//   CUTLASS GEMM vs naive tiled GEMM — GFLOPS comparison.
// =============================================================================

#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// CUTLASS headers
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/host/gemm.h>

#include "../../common/config_loader.hpp"
#include "../../common/cuda_utils.hpp"
#include "../../common/logger.hpp"

// ---------------------------------------------------------------------------
// Reference naive GEMM for correctness check
// ---------------------------------------------------------------------------
__global__ void gemm_naive_ref(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float*       __restrict__ C,
                                int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float acc = 0.f;
    for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = acc;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    const std::string project_root  = PROJECT_ROOT;
    const std::string config_path   = project_root + "/configs/tutorials/17_cutlass.yaml";
    const std::string global_config = project_root + "/configs/global.yaml";

    auto cfg = cuda_tutorials::ConfigLoader::from_file(config_path);
    cfg.merge_defaults(global_config);

    auto logger = cuda_tutorials::Logger::create(config_path, project_root, "17_cutlass");
    logger.log_info("=== Tutorial 17: CUTLASS ===");

    const int    device_id = cfg.get<int>("device",   "id");
    const int    M         = cfg.get<int>("tutorial", "M");
    const int    N         = cfg.get<int>("tutorial", "N");
    const int    K         = cfg.get<int>("tutorial", "K");
    const float  alpha     = cfg.get<float>("tutorial", "alpha");
    const float  beta      = cfg.get<float>("tutorial", "beta");
    const int    num_iter  = cfg.get<int>("tutorial", "num_iterations");

    CUDA_CHECK(cudaSetDevice(device_id));
    auto dev_info = cuda_tutorials::get_device_info(device_id);
    cuda_tutorials::print_device_info(logger, dev_info);

    std::ostringstream oss;
    oss << "Config: M=" << M << "  N=" << N << "  K=" << K
        << "  alpha=" << alpha << "  beta=" << beta;
    logger.log_info(oss.str());

    // ---------------------------------------------------------------------------
    // Define CUTLASS GEMM type: FP32, row-major × row-major → row-major
    // Threadblock tile: 128 × 256 × 32 (from config hints, hardcoded for this type)
    // ---------------------------------------------------------------------------
    using ColumnMajor   = cutlass::layout::ColumnMajor;
    using RowMajor      = cutlass::layout::RowMajor;

    using CutlassGemm = cutlass::gemm::device::Gemm<
        float, RowMajor,   // ElementA, LayoutA
        float, RowMajor,   // ElementB, LayoutB
        float, RowMajor,   // ElementC, LayoutC
        float              // ElementAccumulator
    >;

    oss.str("");
    oss << "CUTLASS GEMM type: FP32 row×row→row"
        << "  default threadblock tile (128,256,8)";
    logger.log_info(oss.str());

    // Allocate device memory
    float *d_A{}, *d_B{}, *d_C_cutlass{}, *d_C_naive{};
    CUDA_CHECK(cudaMalloc(&d_A,         M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B,         K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C_cutlass, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C_naive,   M * N * sizeof(float)));

    // Initialize host data and upload
    std::vector<float> h_A(M * K), h_B(K * N);
    for (int i = 0; i < M * K; ++i) h_A[i] = static_cast<float>(i % 100) * 0.01f;
    for (int i = 0; i < K * N; ++i) h_B[i] = static_cast<float>(i % 100) * 0.01f;
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C_cutlass, 0, M * N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_C_naive,   0, M * N * sizeof(float)));

    // Construct CUTLASS problem size and arguments
    cutlass::gemm::GemmCoord problem_size(M, N, K);

    typename CutlassGemm::Arguments args{
        problem_size,
        {d_A, K},      // TensorRef A
        {d_B, N},      // TensorRef B
        {d_C_cutlass, N}, // TensorRef C (source)
        {d_C_cutlass, N}, // TensorRef D (destination)
        {alpha, beta}
    };

    CutlassGemm gemm_op;
    cutlass::Status status = gemm_op.initialize(args);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("CUTLASS GEMM initialization failed: " +
                                 std::string(cutlassGetStatusString(status)));
    }

    logger.log_info("CUTLASS GEMM initialized successfully.");

    cuda_tutorials::CudaEvent ev_start, ev_stop;
    double flops = 2.0 * M * N * K;

    // --- CUTLASS benchmark ---
    {
        status = gemm_op(); // warm-up
        if (status != cutlass::Status::kSuccess)
            throw std::runtime_error("CUTLASS run failed");
        CUDA_KERNEL_CHECK();

        ev_start.record();
        for (int i = 0; i < num_iter; ++i) gemm_op();
        ev_stop.record();
        CUDA_CHECK(cudaEventSynchronize(ev_stop.event));
        float t = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&t, ev_start.event, ev_stop.event));
        float ms = t / num_iter;
        double gf = flops / (ms * 1e-3) / 1e9;

        oss.str("");
        oss << "CUTLASS FP32 GEMM: " << std::fixed << std::setprecision(3) << ms
            << " ms  " << std::setprecision(1) << gf << " GFLOPS";
        logger.log_info(oss.str());
    }

    // --- Naive reference benchmark ---
    {
        dim3 blk(16, 16);
        dim3 grd((N + 15) / 16, (M + 15) / 16);
        gemm_naive_ref<<<grd, blk>>>(d_A, d_B, d_C_naive, M, N, K);
        CUDA_KERNEL_CHECK();

        ev_start.record();
        for (int i = 0; i < num_iter; ++i)
            gemm_naive_ref<<<grd, blk>>>(d_A, d_B, d_C_naive, M, N, K);
        ev_stop.record();
        CUDA_CHECK(cudaEventSynchronize(ev_stop.event));
        float t = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&t, ev_start.event, ev_stop.event));
        float ms = t / num_iter;
        double gf = flops / (ms * 1e-3) / 1e9;

        oss.str("");
        oss << "Naive FP32 GEMM : " << std::fixed << std::setprecision(3) << ms
            << " ms  " << std::setprecision(1) << gf << " GFLOPS";
        logger.log_info(oss.str());
    }

    // --- Correctness ---
    std::vector<float> h_cutlass(M * N), h_naive(M * N);
    CUDA_CHECK(cudaMemcpy(h_cutlass.data(), d_C_cutlass, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_naive.data(),   d_C_naive,   M * N * sizeof(float), cudaMemcpyDeviceToHost));
    double max_err = 0.0;
    for (int i = 0; i < M * N; ++i)
        max_err = std::max(max_err, std::abs(static_cast<double>(h_cutlass[i] - h_naive[i])));
    oss.str("");
    oss << "Correctness (CUTLASS vs naive): max_diff=" << max_err
        << (max_err < 1.0 ? "  PASS" : "  FAIL (check alpha/beta)");
    logger.log_info(oss.str());

    logger.log_info(
        "CUTLASS achieves high GFLOPS through multi-stage pipelining: "
        "while tensor cores compute one tile, cp.async prefetches the next. "
        "Warp-specialized threadblocks (H100 sm_90) assign some warps as "
        "'producers' (data loading) and others as 'consumers' (MMA compute).");

    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C_cutlass)); CUDA_CHECK(cudaFree(d_C_naive));

    logger.log_info("Tutorial 17 complete.");
    return 0;
}
