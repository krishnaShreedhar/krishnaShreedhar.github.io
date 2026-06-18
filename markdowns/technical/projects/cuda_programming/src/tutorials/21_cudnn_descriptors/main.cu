// =============================================================================
// Tutorial 21: cuDNN Handles & Descriptors
//
// Concept:
//   cuDNN operations are configured through a system of opaque descriptor
//   objects that describe tensor shapes, data types, and operation parameters.
//
//   Core descriptor types:
//     cudnnHandle_t            — library context (one per thread or shared)
//     cudnnTensorDescriptor_t  — describes N-D tensor (shape, stride, dtype)
//     cudnnFilterDescriptor_t  — describes conv filter (K, C, H, W)
//     cudnnConvolutionDescriptor_t — describes conv params (pad, stride, dilation)
//
//   Workflow:
//     1. Create handle: cudnnCreate(&handle)
//     2. Set tensor descriptors for input, output, filter
//     3. Set convolution descriptor
//     4. Query workspace size
//     5. Allocate workspace
//     6. Find best algorithm
//     7. Execute cudnnConvolutionForward
//     8. Destroy descriptors and handle
//
// Experiment:
//   Forward convolution: Input (N,C,H,W) × Filter (K,C,kH,kW) → Output (N,K,oH,oW)
//   Config supports NCHW and NHWC layouts.
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

// ---------------------------------------------------------------------------
// Helper: compute convolution output spatial size
// ---------------------------------------------------------------------------
static int conv_output_dim(int input, int pad, int kernel, int stride, int dilation) {
    int effective_kernel = dilation * (kernel - 1) + 1;
    return (input + 2 * pad - effective_kernel) / stride + 1;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    const std::string project_root  = PROJECT_ROOT;
    const std::string config_path   = project_root +
        "/configs/tutorials/21_cudnn_descriptors.yaml";
    const std::string global_config = project_root + "/configs/global.yaml";

    auto cfg = cuda_tutorials::ConfigLoader::from_file(config_path);
    cfg.merge_defaults(global_config);

    auto logger = cuda_tutorials::Logger::create(config_path, project_root,
                                                  "21_cudnn_descriptors");
    logger.log_info("=== Tutorial 21: cuDNN Handles & Descriptors ===");

    const int device_id   = cfg.get<int>("device",   "id");
    const int N           = cfg.get<int>("tutorial", "N");
    const int C           = cfg.get<int>("tutorial", "C");
    const int H           = cfg.get<int>("tutorial", "H");
    const int W           = cfg.get<int>("tutorial", "W");
    const int K           = cfg.get<int>("tutorial", "K");
    const int kH          = cfg.get<int>("tutorial", "kH");
    const int kW          = cfg.get<int>("tutorial", "kW");
    const int pad_h       = cfg.get<int>("tutorial", "pad_h");
    const int pad_w       = cfg.get<int>("tutorial", "pad_w");
    const int stride_h    = cfg.get<int>("tutorial", "stride_h");
    const int stride_w    = cfg.get<int>("tutorial", "stride_w");
    const int dil_h       = cfg.get<int>("tutorial", "dilation_h");
    const int dil_w       = cfg.get<int>("tutorial", "dilation_w");
    const std::string layout = cfg.get<std::string>("tutorial", "layout");
    const int num_iter    = cfg.get<int>("tutorial", "num_iterations");

    CUDA_CHECK(cudaSetDevice(device_id));
    auto dev_info = cuda_tutorials::get_device_info(device_id);
    cuda_tutorials::print_device_info(logger, dev_info);

    std::ostringstream oss;
    oss << "Conv config: N=" << N << " C=" << C << " H=" << H << " W=" << W
        << " K=" << K << " kH=" << kH << " kW=" << kW
        << " pad=(" << pad_h << "," << pad_w << ")"
        << " stride=(" << stride_h << "," << stride_w << ")"
        << " dilation=(" << dil_h << "," << dil_w << ")"
        << " layout=" << layout;
    logger.log_info(oss.str());

    // Compute output dimensions
    int oH = conv_output_dim(H, pad_h, kH, stride_h, dil_h);
    int oW = conv_output_dim(W, pad_w, kW, stride_w, dil_w);
    oss.str("");
    oss << "Output spatial: " << oH << " × " << oW;
    logger.log_info(oss.str());

    // Choose layout format
    cudnnTensorFormat_t tensor_fmt = (layout == "NHWC")
        ? CUDNN_TENSOR_NHWC
        : CUDNN_TENSOR_NCHW;

    // Create cuDNN handle
    cudnnHandle_t handle;
    CUDNN_CHECK(cudnnCreate(&handle));
    logger.log_info("cuDNN handle created.");

    // ---------------------------------------------------------------------------
    // Descriptors
    // ---------------------------------------------------------------------------
    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));

    // Input: (N, C, H, W)
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(input_desc, tensor_fmt,
                                            CUDNN_DATA_FLOAT, N, C, H, W));
    // Filter: (K, C, kH, kW)
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT,
                                            tensor_fmt, K, C, kH, kW));
    // Convolution
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc,
                                                 pad_h, pad_w,
                                                 stride_h, stride_w,
                                                 dil_h, dil_w,
                                                 CUDNN_CROSS_CORRELATION,
                                                 CUDNN_DATA_FLOAT));
    // Allow tensor core math
    CUDNN_CHECK(cudnnSetConvolutionMathType(conv_desc, CUDNN_TENSOR_OP_MATH));

    // Query output descriptor from cuDNN
    int on, oc, oh, ow;
    CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(conv_desc, input_desc,
                                                       filter_desc, &on, &oc, &oh, &ow));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(output_desc, tensor_fmt,
                                            CUDNN_DATA_FLOAT, on, oc, oh, ow));

    oss.str("");
    oss << "cuDNN output descriptor: N=" << on << " K=" << oc
        << " oH=" << oh << " oW=" << ow;
    logger.log_info(oss.str());

    // Find best algorithm
    cudnnConvolutionFwdAlgo_t algo;
    CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(handle,
                                                     input_desc, filter_desc,
                                                     conv_desc, output_desc,
                                                     CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                     0, &algo));

    // Query workspace size
    size_t workspace_bytes = 0;
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(handle,
                                                        input_desc, filter_desc,
                                                        conv_desc, output_desc,
                                                        algo, &workspace_bytes));

    oss.str("");
    oss << "Selected algorithm: " << algo
        << "  workspace: " << workspace_bytes / (1024.0 * 1024.0) << " MB";
    logger.log_info(oss.str());

    // Allocate device memory
    size_t input_bytes   = N * C * H * W * sizeof(float);
    size_t filter_bytes  = K * C * kH * kW * sizeof(float);
    size_t output_bytes  = on * oc * oh * ow * sizeof(float);

    float *d_input{}, *d_filter{}, *d_output{}, *d_workspace{};
    CUDA_CHECK(cudaMalloc(&d_input,     input_bytes));
    CUDA_CHECK(cudaMalloc(&d_filter,    filter_bytes));
    CUDA_CHECK(cudaMalloc(&d_output,    output_bytes));
    if (workspace_bytes > 0)
        CUDA_CHECK(cudaMalloc(&d_workspace, workspace_bytes));
    CUDA_CHECK(cudaMemset(d_input,  0x3f, input_bytes));
    CUDA_CHECK(cudaMemset(d_filter, 0x3f, filter_bytes));

    // Run convolution
    const float alpha_val = 1.0f, beta_val = 0.0f;
    cuda_tutorials::CudaEvent ev_start, ev_stop;

    // Warm-up
    CUDNN_CHECK(cudnnConvolutionForward(handle,
                                        &alpha_val,
                                        input_desc,  d_input,
                                        filter_desc, d_filter,
                                        conv_desc,   algo,
                                        d_workspace, workspace_bytes,
                                        &beta_val,
                                        output_desc, d_output));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed iterations
    ev_start.record();
    for (int i = 0; i < num_iter; ++i) {
        CUDNN_CHECK(cudnnConvolutionForward(handle,
                                            &alpha_val,
                                            input_desc,  d_input,
                                            filter_desc, d_filter,
                                            conv_desc,   algo,
                                            d_workspace, workspace_bytes,
                                            &beta_val,
                                            output_desc, d_output));
    }
    ev_stop.record();
    CUDA_CHECK(cudaEventSynchronize(ev_stop.event));
    float t = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&t, ev_start.event, ev_stop.event));
    float ms = t / num_iter;

    double gflops = 2.0 * on * oc * oh * ow * C * kH * kW / (ms * 1e-3) / 1e9;
    oss.str("");
    oss << "Conv forward: " << std::fixed << std::setprecision(3) << ms
        << " ms  " << std::setprecision(1) << gflops << " GFLOPS";
    logger.log_info(oss.str());

    // Cleanup
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(output_desc));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(filter_desc));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc));
    CUDNN_CHECK(cudnnDestroy(handle));

    if (d_workspace) CUDA_CHECK(cudaFree(d_workspace));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_filter));
    CUDA_CHECK(cudaFree(d_output));

    logger.log_info("Tutorial 21 complete.");
    return 0;
}
