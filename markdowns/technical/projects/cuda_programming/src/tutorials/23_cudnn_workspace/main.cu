// =============================================================================
// Tutorial 23: cuDNN Workspace Memory
//
// Concept:
//   Many cuDNN algorithms require temporary workspace memory on the device.
//   The workspace enables faster algorithms (Winograd, FFT, im2col+GEMM) at
//   the cost of additional VRAM.
//
//   Trade-off:
//     No workspace → CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM (slow, ~correct)
//     Small workspace → precomputed GEMM or simple Winograd
//     Large workspace → full Winograd or FFT (fastest for most shapes)
//
//   Ablation: query the best algorithm under each workspace cap and compare
//   performance. This simulates environments where VRAM is shared (training +
//   inference co-located) and workspace budget must be controlled.
//
//   Key insight: Framework memory managers (PyTorch, TF) pre-allocate a
//   workspace pool and pass its size to cuDNN. Larger pools = better algos.
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

static int conv_out(int in, int pad, int k, int s, int d) {
    return (in + 2 * pad - d * (k - 1) - 1) / s + 1;
}

int main() {
    const std::string project_root  = PROJECT_ROOT;
    const std::string config_path   = project_root +
        "/configs/tutorials/23_cudnn_workspace.yaml";
    const std::string global_config = project_root + "/configs/global.yaml";

    auto cfg = cuda_tutorials::ConfigLoader::from_file(config_path);
    cfg.merge_defaults(global_config);

    auto logger = cuda_tutorials::Logger::create(config_path, project_root,
                                                  "23_cudnn_workspace");
    logger.log_info("=== Tutorial 23: cuDNN Workspace Memory ===");

    const int device_id  = cfg.get<int>("device",   "id");
    const int N          = cfg.get<int>("tutorial", "N");
    const int C          = cfg.get<int>("tutorial", "C");
    const int H          = cfg.get<int>("tutorial", "H");
    const int W          = cfg.get<int>("tutorial", "W");
    const int K          = cfg.get<int>("tutorial", "K");
    const int kH         = cfg.get<int>("tutorial", "kH");
    const int kW         = cfg.get<int>("tutorial", "kW");
    const int pad_h      = cfg.get<int>("tutorial", "pad_h");
    const int pad_w      = cfg.get<int>("tutorial", "pad_w");
    const int stride_h   = cfg.get<int>("tutorial", "stride_h");
    const int stride_w   = cfg.get<int>("tutorial", "stride_w");
    const int dil_h      = cfg.get<int>("tutorial", "dilation_h");
    const int dil_w      = cfg.get<int>("tutorial", "dilation_w");
    const int num_iter   = cfg.get<int>("tutorial", "num_iterations");
    auto ws_mb_values    = cfg.get_vector<int>("tutorial", "max_workspace_mb_values");

    CUDA_CHECK(cudaSetDevice(device_id));
    auto dev_info = cuda_tutorials::get_device_info(device_id);
    cuda_tutorials::print_device_info(logger, dev_info);

    int oH = conv_out(H, pad_h, kH, stride_h, dil_h);
    int oW = conv_out(W, pad_w, kW, stride_w, dil_w);

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

    float *d_in{}, *d_flt{}, *d_out{};
    CUDA_CHECK(cudaMalloc(&d_in,  N * C * H * W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_flt, K * C * kH * kW * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, on * oc * oh * ow * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_in,  0x3f, N * C * H * W * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_flt, 0x3f, K * C * kH * kW * sizeof(float)));

    cuda_tutorials::CudaEvent ev_start, ev_stop;

    logger.log_info("--- Workspace cap ablation ---");
    logger.log_info("  WS_cap(MB)  | Algorithm               | WS_required(MB) | Time(ms)");

    for (int ws_mb : ws_mb_values) {
        size_t ws_limit = static_cast<size_t>(ws_mb) * 1024 * 1024;

        // Find best algorithm within workspace limit
        cudnnConvolutionFwdAlgo_t algo;
        CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(
            handle, in_desc, flt_desc, conv_desc, out_desc,
            CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
            ws_limit, &algo));

        // Get actual workspace required by selected algo
        size_t ws_required = 0;
        CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
            handle, in_desc, flt_desc, conv_desc, out_desc, algo, &ws_required));

        // Allocate workspace (actual requirement, bounded by cap)
        float* d_ws = nullptr;
        size_t ws_alloc = std::min(ws_required, ws_limit);
        if (ws_alloc > 0) CUDA_CHECK(cudaMalloc(&d_ws, ws_alloc));

        // Benchmark
        const float alpha_v = 1.f, beta_v = 0.f;
        CUDNN_CHECK(cudnnConvolutionForward(handle, &alpha_v,
                                            in_desc,  d_in,
                                            flt_desc, d_flt,
                                            conv_desc, algo, d_ws, ws_alloc,
                                            &beta_v, out_desc, d_out));
        CUDA_CHECK(cudaDeviceSynchronize());

        ev_start.record();
        for (int i = 0; i < num_iter; ++i) {
            CUDNN_CHECK(cudnnConvolutionForward(handle, &alpha_v,
                                                in_desc,  d_in,
                                                flt_desc, d_flt,
                                                conv_desc, algo, d_ws, ws_alloc,
                                                &beta_v, out_desc, d_out));
        }
        ev_stop.record();
        CUDA_CHECK(cudaEventSynchronize(ev_stop.event));
        float t = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&t, ev_start.event, ev_stop.event));
        float ms = t / num_iter;

        oss.str("");
        oss << "  " << std::setw(10) << ws_mb
            << "  | " << std::setw(23) << algo_name(algo)
            << " | " << std::setw(15) << std::fixed << std::setprecision(2)
            << static_cast<double>(ws_required) / (1024.0 * 1024.0)
            << " | " << std::setprecision(3) << ms;
        logger.log_info(oss.str());

        if (d_ws) CUDA_CHECK(cudaFree(d_ws));
    }

    CUDNN_CHECK(cudnnDestroyTensorDescriptor(in_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(out_desc));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(flt_desc));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc));
    CUDNN_CHECK(cudnnDestroy(handle));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_flt));
    CUDA_CHECK(cudaFree(d_out));

    logger.log_info(
        "Observation: as workspace cap increases, cuDNN selects more capable "
        "algorithms (Winograd → FFT) which run faster. A few MB of workspace "
        "can yield 2–5× speedup for 3×3 convolutions via Winograd F(6,3).");

    logger.log_info("Tutorial 23 complete.");
    return 0;
}
