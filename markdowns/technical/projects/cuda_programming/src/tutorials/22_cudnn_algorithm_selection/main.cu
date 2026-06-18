// =============================================================================
// Tutorial 22: cuDNN Algorithm Selection
//
// Concept:
//   cuDNN provides multiple convolution algorithms, each with different
//   speed/workspace trade-offs:
//
//   CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM       — no workspace, slow
//   CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM — small workspace
//   CUDNN_CONVOLUTION_FWD_ALGO_GEMM                — col2im + GEMM
//   CUDNN_CONVOLUTION_FWD_ALGO_DIRECT              — direct conv
//   CUDNN_CONVOLUTION_FWD_ALGO_FFT                 — FFT-based (large kernels)
//   CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING          — tiled FFT
//   CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD            — Winograd (3×3 favored)
//   CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED   — non-fused Winograd
//
//   cudnnFindConvolutionForwardAlgorithm() benchmarks all algorithms and
//   returns them sorted by speed. This tutorial exposes all results.
// =============================================================================

#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cudnn.h>

#include "../../common/config_loader.hpp"
#include "../../common/cuda_utils.hpp"
#include "../../common/logger.hpp"

static const char* algo_name(cudnnConvolutionFwdAlgo_t algo) {
    switch (algo) {
        case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM:         return "IMPLICIT_GEMM";
        case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM: return "IMPLICIT_PRECOMP_GEMM";
        case CUDNN_CONVOLUTION_FWD_ALGO_GEMM:                  return "GEMM";
        case CUDNN_CONVOLUTION_FWD_ALGO_DIRECT:                return "DIRECT";
        case CUDNN_CONVOLUTION_FWD_ALGO_FFT:                   return "FFT";
        case CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING:            return "FFT_TILING";
        case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD:              return "WINOGRAD";
        case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED:     return "WINOGRAD_NONFUSED";
        default:                                                return "UNKNOWN";
    }
}

static int conv_output_dim(int input, int pad, int kernel, int stride, int dilation) {
    return (input + 2 * pad - dilation * (kernel - 1) - 1) / stride + 1;
}

int main() {
    const std::string project_root  = PROJECT_ROOT;
    const std::string config_path   = project_root +
        "/configs/tutorials/22_cudnn_algorithm_selection.yaml";
    const std::string global_config = project_root + "/configs/global.yaml";

    auto cfg = cuda_tutorials::ConfigLoader::from_file(config_path);
    cfg.merge_defaults(global_config);

    auto logger = cuda_tutorials::Logger::create(config_path, project_root,
                                                  "22_cudnn_algorithm_selection");
    logger.log_info("=== Tutorial 22: cuDNN Algorithm Selection ===");

    const int   device_id    = cfg.get<int>("device",   "id");
    const int   N            = cfg.get<int>("tutorial", "N");
    const int   C            = cfg.get<int>("tutorial", "C");
    const int   H            = cfg.get<int>("tutorial", "H");
    const int   W            = cfg.get<int>("tutorial", "W");
    const int   K            = cfg.get<int>("tutorial", "K");
    const int   kH           = cfg.get<int>("tutorial", "kH");
    const int   kW           = cfg.get<int>("tutorial", "kW");
    const int   pad_h        = cfg.get<int>("tutorial", "pad_h");
    const int   pad_w        = cfg.get<int>("tutorial", "pad_w");
    const int   stride_h     = cfg.get<int>("tutorial", "stride_h");
    const int   stride_w     = cfg.get<int>("tutorial", "stride_w");
    const int   dil_h        = cfg.get<int>("tutorial", "dilation_h");
    const int   dil_w        = cfg.get<int>("tutorial", "dilation_w");
    const int   perf_runs    = cfg.get<int>("tutorial", "num_algo_perf_runs");

    CUDA_CHECK(cudaSetDevice(device_id));
    auto dev_info = cuda_tutorials::get_device_info(device_id);
    cuda_tutorials::print_device_info(logger, dev_info);

    int oH = conv_output_dim(H, pad_h, kH, stride_h, dil_h);
    int oW = conv_output_dim(W, pad_w, kW, stride_w, dil_w);

    std::ostringstream oss;
    oss << "Conv: N=" << N << " C=" << C << " H=" << H << " W=" << W
        << " K=" << K << " kH=" << kH << " kW=" << kW
        << " → oH=" << oH << " oW=" << oW;
    logger.log_info(oss.str());

    cudnnHandle_t handle;
    CUDNN_CHECK(cudnnCreate(&handle));

    cudnnTensorDescriptor_t in_desc, out_desc;
    cudnnFilterDescriptor_t flt_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&in_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&out_desc));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&flt_desc));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW,
                                            CUDNN_DATA_FLOAT, N, C, H, W));
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(flt_desc, CUDNN_DATA_FLOAT,
                                            CUDNN_TENSOR_NCHW, K, C, kH, kW));
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc, pad_h, pad_w,
                                                 stride_h, stride_w, dil_h, dil_w,
                                                 CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
    CUDNN_CHECK(cudnnSetConvolutionMathType(conv_desc, CUDNN_TENSOR_OP_MATH));

    int on, oc, oh, ow;
    CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(conv_desc, in_desc, flt_desc,
                                                       &on, &oc, &oh, &ow));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW,
                                            CUDNN_DATA_FLOAT, on, oc, oh, ow));

    // Allocate device tensors
    float *d_in{}, *d_flt{}, *d_out{};
    CUDA_CHECK(cudaMalloc(&d_in,  N * C * H * W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_flt, K * C * kH * kW * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, on * oc * oh * ow * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_in,  0x3f, N * C * H * W * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_flt, 0x3f, K * C * kH * kW * sizeof(float)));

    // ---------------------------------------------------------------------------
    // cudnnFindConvolutionForwardAlgorithm — benchmarks all supported algorithms
    // Returns results sorted by time (fastest first)
    // ---------------------------------------------------------------------------
    static constexpr int kMaxAlgos = 8;
    cudnnConvolutionFwdAlgoPerf_t perf_results[kMaxAlgos];
    int num_returned = 0;

    // First query: find what algorithms are available (no time limit)
    CUDNN_CHECK(cudnnFindConvolutionForwardAlgorithmEx(
        handle,
        in_desc,  d_in,
        flt_desc, d_flt,
        conv_desc,
        out_desc, d_out,
        kMaxAlgos,
        &num_returned,
        perf_results,
        nullptr,  // let cuDNN allocate workspace
        0         // 0 = no workspace limit
    ));

    logger.log_info("--- Algorithm benchmark results (fastest first) ---");
    for (int i = 0; i < num_returned; ++i) {
        auto& r = perf_results[i];
        if (r.status != CUDNN_STATUS_SUCCESS) {
            oss.str("");
            oss << "  [" << std::setw(2) << i << "] " << std::setw(28) << algo_name(r.algo)
                << "  FAILED: " << cudnnGetErrorString(r.status);
            logger.log_info(oss.str());
            continue;
        }
        oss.str("");
        oss << "  [" << std::setw(2) << i << "] " << std::setw(28) << algo_name(r.algo)
            << "  time=" << std::fixed << std::setprecision(3) << r.time << " ms"
            << "  workspace=" << std::setw(8)
            << static_cast<double>(r.memory) / (1024.0 * 1024.0) << " MB"
            << "  deterministic=" << r.determinism;
        logger.log_info(oss.str());
    }

    if (num_returned > 0 && perf_results[0].status == CUDNN_STATUS_SUCCESS) {
        oss.str("");
        oss << "Best algorithm: " << algo_name(perf_results[0].algo)
            << " (" << perf_results[0].time << " ms)";
        logger.log_info(oss.str());
    }

    // Cleanup
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(in_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(out_desc));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(flt_desc));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc));
    CUDNN_CHECK(cudnnDestroy(handle));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_flt));
    CUDA_CHECK(cudaFree(d_out));

    logger.log_info("Tutorial 22 complete.");
    return 0;
}
