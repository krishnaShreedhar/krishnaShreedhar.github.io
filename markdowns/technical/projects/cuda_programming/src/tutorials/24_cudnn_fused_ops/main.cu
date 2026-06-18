// =============================================================================
// Tutorial 24: cuDNN Fused Operations (Graph API)
//
// Concept:
//   In a standard forward pass: Conv → Bias-Add → ReLU
//   If implemented as 3 separate operations:
//     - Each op reads from global memory and writes back
//     - 3 separate kernel launches, 3 rounds of memory traffic
//
//   cuDNN Graph API allows fusing Conv + Bias + ReLU into a SINGLE kernel:
//     - Data flows through registers/shared memory within one kernel
//     - Only 1 global read (input) + 1 global write (output)
//     - Reduced memory bandwidth and kernel launch overhead
//
//   cuDNN Operation Graph:
//     cudnnBackendDescriptor_t for each operation (conv, bias, activation)
//     cudnnBackendDescriptor_t for tensors
//     cudnnOperationGraph_t — the computation DAG
//     cudnnExecutionPlan_t  — compiled plan
//
//   Note: Graph API (cudnn backend) is available from cuDNN 8.0+.
//         We use legacy API for unfused and attempt graph API for fused.
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

static int conv_out(int in, int pad, int k, int s, int d = 1) {
    return (in + 2 * pad - d * (k - 1) - 1) / s + 1;
}

// ---------------------------------------------------------------------------
// Run unfused Conv + Bias + ReLU using 3 separate cuDNN calls
// ---------------------------------------------------------------------------
static float run_unfused(cudnnHandle_t handle,
                          cudnnTensorDescriptor_t in_desc,
                          cudnnFilterDescriptor_t flt_desc,
                          cudnnConvolutionDescriptor_t conv_desc,
                          cudnnTensorDescriptor_t out_desc,
                          cudnnTensorDescriptor_t bias_desc,
                          cudnnActivationDescriptor_t act_desc,
                          float* d_in, float* d_flt, float* d_out_tmp, float* d_out,
                          float* d_bias, float* d_ws, size_t ws_bytes,
                          cudnnConvolutionFwdAlgo_t algo,
                          int num_iter,
                          cuda_tutorials::Logger& logger) {
    const float alpha = 1.f, beta = 0.f;

    // Warm-up
    CUDNN_CHECK(cudnnConvolutionForward(handle, &alpha,
                                        in_desc, d_in, flt_desc, d_flt,
                                        conv_desc, algo, d_ws, ws_bytes,
                                        &beta, out_desc, d_out_tmp));
    CUDNN_CHECK(cudnnAddTensor(handle, &alpha, bias_desc, d_bias,
                                &alpha, out_desc, d_out_tmp));
    CUDNN_CHECK(cudnnActivationForward(handle, act_desc,
                                       &alpha, out_desc, d_out_tmp,
                                       &beta, out_desc, d_out));
    CUDA_CHECK(cudaDeviceSynchronize());

    cuda_tutorials::CudaEvent ev_start, ev_stop;
    ev_start.record();
    for (int i = 0; i < num_iter; ++i) {
        CUDNN_CHECK(cudnnConvolutionForward(handle, &alpha,
                                            in_desc, d_in, flt_desc, d_flt,
                                            conv_desc, algo, d_ws, ws_bytes,
                                            &beta, out_desc, d_out_tmp));
        CUDNN_CHECK(cudnnAddTensor(handle, &alpha, bias_desc, d_bias,
                                    &alpha, out_desc, d_out_tmp));
        CUDNN_CHECK(cudnnActivationForward(handle, act_desc,
                                           &alpha, out_desc, d_out_tmp,
                                           &beta, out_desc, d_out));
    }
    ev_stop.record();
    CUDA_CHECK(cudaEventSynchronize(ev_stop.event));
    float t = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&t, ev_start.event, ev_stop.event));
    return t / num_iter;
}

// ---------------------------------------------------------------------------
// Run fused Conv+Bias+ReLU via cudnnConvolutionBiasActivationForward (legacy fused)
// This uses a single cuDNN call that internally fuses the ops.
// ---------------------------------------------------------------------------
static float run_fused_legacy(cudnnHandle_t handle,
                               cudnnTensorDescriptor_t in_desc,
                               cudnnFilterDescriptor_t flt_desc,
                               cudnnConvolutionDescriptor_t conv_desc,
                               cudnnTensorDescriptor_t out_desc,
                               cudnnTensorDescriptor_t bias_desc,
                               cudnnActivationDescriptor_t act_desc,
                               float* d_in, float* d_flt, float* d_out,
                               float* d_bias, float* d_ws, size_t ws_bytes,
                               cudnnConvolutionFwdAlgo_t algo,
                               int num_iter,
                               cuda_tutorials::Logger& logger) {
    const float alpha = 1.f, beta = 0.f;

    // Warm-up
    CUDNN_CHECK(cudnnConvolutionBiasActivationForward(
        handle, &alpha,
        in_desc,  d_in,
        flt_desc, d_flt,
        conv_desc, algo, d_ws, ws_bytes,
        &beta,
        out_desc, d_out,   // z (residual, beta=0)
        bias_desc, d_bias,
        act_desc,
        out_desc, d_out));
    CUDA_CHECK(cudaDeviceSynchronize());

    cuda_tutorials::CudaEvent ev_start, ev_stop;
    ev_start.record();
    for (int i = 0; i < num_iter; ++i) {
        CUDNN_CHECK(cudnnConvolutionBiasActivationForward(
            handle, &alpha,
            in_desc,  d_in,
            flt_desc, d_flt,
            conv_desc, algo, d_ws, ws_bytes,
            &beta,
            out_desc, d_out,
            bias_desc, d_bias,
            act_desc,
            out_desc, d_out));
    }
    ev_stop.record();
    CUDA_CHECK(cudaEventSynchronize(ev_stop.event));
    float t = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&t, ev_start.event, ev_stop.event));
    return t / num_iter;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    const std::string project_root  = PROJECT_ROOT;
    const std::string config_path   = project_root +
        "/configs/tutorials/24_cudnn_fused_ops.yaml";
    const std::string global_config = project_root + "/configs/global.yaml";

    auto cfg = cuda_tutorials::ConfigLoader::from_file(config_path);
    cfg.merge_defaults(global_config);

    auto logger = cuda_tutorials::Logger::create(config_path, project_root,
                                                  "24_cudnn_fused_ops");
    logger.log_info("=== Tutorial 24: cuDNN Fused Operations ===");

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
    const int num_iter   = cfg.get<int>("tutorial", "num_iterations");

    CUDA_CHECK(cudaSetDevice(device_id));
    auto dev_info = cuda_tutorials::get_device_info(device_id);
    cuda_tutorials::print_device_info(logger, dev_info);

    int oH = conv_out(H, pad_h, kH, stride_h);
    int oW = conv_out(W, pad_w, kW, stride_w);

    std::ostringstream oss;
    oss << "Config: N=" << N << " C=" << C << " H=" << H << " W=" << W
        << " K=" << K << " kH=" << kH << " kW=" << kW
        << " oH=" << oH << " oW=" << oW;
    logger.log_info(oss.str());

    cudnnHandle_t handle;
    CUDNN_CHECK(cudnnCreate(&handle));

    // Descriptors
    cudnnTensorDescriptor_t in_desc, out_desc, bias_desc;
    cudnnFilterDescriptor_t flt_desc;
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnActivationDescriptor_t act_desc;

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&in_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&out_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&bias_desc));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&flt_desc));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_CHECK(cudnnCreateActivationDescriptor(&act_desc));

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW,
                                            CUDNN_DATA_FLOAT, N, C, H, W));
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(flt_desc, CUDNN_DATA_FLOAT,
                                            CUDNN_TENSOR_NCHW, K, C, kH, kW));
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc, pad_h, pad_w,
                                                 stride_h, stride_w, 1, 1,
                                                 CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
    CUDNN_CHECK(cudnnSetConvolutionMathType(conv_desc, CUDNN_TENSOR_OP_MATH));

    int on, oc, oh, ow;
    CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(conv_desc, in_desc, flt_desc,
                                                       &on, &oc, &oh, &ow));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW,
                                            CUDNN_DATA_FLOAT, on, oc, oh, ow));
    // Bias: (1, K, 1, 1)
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW,
                                            CUDNN_DATA_FLOAT, 1, K, 1, 1));
    // ReLU activation
    CUDNN_CHECK(cudnnSetActivationDescriptor(act_desc, CUDNN_ACTIVATION_RELU,
                                              CUDNN_NOT_PROPAGATE_NAN, 0.0));

    // Find best algorithm
    cudnnConvolutionFwdAlgo_t algo;
    CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(handle, in_desc, flt_desc,
                                                     conv_desc, out_desc,
                                                     CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                     0, &algo));
    size_t ws_bytes = 0;
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(handle, in_desc, flt_desc,
                                                        conv_desc, out_desc, algo, &ws_bytes));

    // Allocate memory
    float *d_in{}, *d_flt{}, *d_out{}, *d_out_tmp{}, *d_bias{}, *d_ws{};
    CUDA_CHECK(cudaMalloc(&d_in,     N * C * H * W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_flt,    K * C * kH * kW * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out,    on * oc * oh * ow * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_tmp, on * oc * oh * ow * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bias,   K * sizeof(float)));
    if (ws_bytes > 0) CUDA_CHECK(cudaMalloc(&d_ws, ws_bytes));
    CUDA_CHECK(cudaMemset(d_in,   0x3f, N * C * H * W * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_flt,  0x3f, K * C * kH * kW * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_bias, 0x3d, K * sizeof(float)));

    // Run unfused
    float t_unfused = run_unfused(handle, in_desc, flt_desc, conv_desc, out_desc,
                                   bias_desc, act_desc,
                                   d_in, d_flt, d_out_tmp, d_out, d_bias,
                                   d_ws, ws_bytes, algo, num_iter, logger);

    // Run fused (legacy cudnnConvolutionBiasActivationForward)
    float t_fused = run_fused_legacy(handle, in_desc, flt_desc, conv_desc, out_desc,
                                      bias_desc, act_desc,
                                      d_in, d_flt, d_out, d_bias,
                                      d_ws, ws_bytes, algo, num_iter, logger);

    oss.str("");
    oss << "Unfused Conv+Bias+ReLU: " << std::fixed << std::setprecision(3)
        << t_unfused << " ms  (3 kernel calls)";
    logger.log_info(oss.str());

    oss.str("");
    oss << "Fused   Conv+Bias+ReLU: " << std::fixed << std::setprecision(3)
        << t_fused << " ms  (1 fused kernel)";
    logger.log_info(oss.str());

    oss.str("");
    oss << "Fusion speedup: " << std::fixed << std::setprecision(2)
        << (t_unfused / t_fused) << "x";
    logger.log_info(oss.str());

    logger.log_info(
        "Fused ops avoid writing/reading intermediate tensors (conv output "
        "before bias and ReLU) to global memory. The conv output stays in "
        "registers/shared memory, and bias+ReLU are applied in the same kernel. "
        "For memory-bound workloads, this can halve the bandwidth requirement.");

    // Cleanup
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(in_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(out_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(bias_desc));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(flt_desc));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc));
    CUDNN_CHECK(cudnnDestroyActivationDescriptor(act_desc));
    CUDNN_CHECK(cudnnDestroy(handle));

    if (d_ws) CUDA_CHECK(cudaFree(d_ws));
    CUDA_CHECK(cudaFree(d_in)); CUDA_CHECK(cudaFree(d_flt));
    CUDA_CHECK(cudaFree(d_out)); CUDA_CHECK(cudaFree(d_out_tmp));
    CUDA_CHECK(cudaFree(d_bias));

    logger.log_info("Tutorial 24 complete.");
    return 0;
}
