// =============================================================================
// Tutorial 25: CUDA Graphs
//
// Concept:
//   Each cudaLaunchKernel() call has CPU-side launch overhead (~5–20 µs):
//     - API call through the CUDA driver
//     - Command encoding into the GPU command buffer
//     - Submission to the GPU hardware queue
//
//   For workloads with many small kernels (inference, step functions),
//   this overhead accumulates significantly.
//
//   CUDA Graphs solve this by recording a sequence of CUDA operations
//   (kernels, memcpy, memset) into a GRAPH OBJECT, then launching the
//   entire graph with a single API call: cudaGraphLaunch().
//
//   Benefits:
//     - Amortized CPU launch overhead (one call instead of N_kernels)
//     - GPU-side optimization: kernel fusion, dependency analysis
//     - Deterministic replay: same execution graph every invocation
//
//   API:
//     cudaStreamBeginCapture(stream, mode)  — start recording
//     <issue kernels and memcpy on stream>
//     cudaStreamEndCapture(stream, &graph)  — stop recording
//     cudaGraphInstantiate(&exec, graph, ...) — compile graph
//     cudaGraphLaunch(exec, stream)         — launch entire graph
//
// Experiment:
//   Run a pipeline of num_kernels simple kernels.
//   Mode A: Regular loop (N separate kernel launches)
//   Mode B: CUDA graph (record once, replay num_replay_iterations times)
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
// Simple kernel: scale + offset (lightweight, launch-overhead dominates)
// ---------------------------------------------------------------------------
__global__ void step_kernel(float* __restrict__ data, int n, float scale, float offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = data[idx] * scale + offset;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    const std::string project_root  = PROJECT_ROOT;
    const std::string config_path   = project_root + "/configs/tutorials/25_cuda_graphs.yaml";
    const std::string global_config = project_root + "/configs/global.yaml";

    auto cfg = cuda_tutorials::ConfigLoader::from_file(config_path);
    cfg.merge_defaults(global_config);

    auto logger = cuda_tutorials::Logger::create(config_path, project_root,
                                                  "25_cuda_graphs");
    logger.log_info("=== Tutorial 25: CUDA Graphs ===");

    const int device_id     = cfg.get<int>("device",   "id");
    const int num_kernels   = cfg.get<int>("tutorial", "num_kernels");
    const int N             = cfg.get<int>("tutorial", "N");
    const int num_replay    = cfg.get<int>("tutorial", "num_replay_iterations");

    CUDA_CHECK(cudaSetDevice(device_id));
    auto dev_info = cuda_tutorials::get_device_info(device_id);
    cuda_tutorials::print_device_info(logger, dev_info);

    std::ostringstream oss;
    oss << "Config: num_kernels=" << num_kernels
        << "  N=" << N
        << "  num_replay_iterations=" << num_replay;
    logger.log_info(oss.str());

    // Allocate device buffer
    float* d_data{};
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_data, 0x3f, N * sizeof(float)));

    const int block_dim = 256;
    const int grid_dim  = (N + block_dim - 1) / block_dim;

    // Create a non-default stream for graph capture
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // =========================================================================
    // Mode A: Regular kernel launch loop (repeated num_replay times)
    // =========================================================================
    logger.log_info("--- Mode A: Regular kernel launch loop ---");

    // Warm-up
    for (int k = 0; k < num_kernels; ++k) {
        step_kernel<<<grid_dim, block_dim, 0, stream>>>(
            d_data, N, 1.0001f, static_cast<float>(k) * 0.0001f);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cuda_tutorials::CudaEvent ev_start, ev_stop;
    ev_start.record(stream);

    for (int r = 0; r < num_replay; ++r) {
        for (int k = 0; k < num_kernels; ++k) {
            step_kernel<<<grid_dim, block_dim, 0, stream>>>(
                d_data, N, 1.0001f, static_cast<float>(k) * 0.0001f);
        }
    }
    ev_stop.record(stream);
    float t_regular_total = ev_stop.elapsed_ms(ev_start);
    float t_regular_per_iter = t_regular_total / num_replay;
    float t_regular_per_kernel = t_regular_per_iter / num_kernels;

    oss.str("");
    oss << "Regular: total=" << std::fixed << std::setprecision(3) << t_regular_total
        << " ms  per_iter=" << t_regular_per_iter
        << " ms  per_kernel=" << t_regular_per_kernel << " ms";
    logger.log_info(oss.str());

    // =========================================================================
    // Mode B: CUDA Graph — record once, replay num_replay times
    // =========================================================================
    logger.log_info("--- Mode B: CUDA Graph capture + replay ---");

    CUDA_CHECK(cudaMemset(d_data, 0x3f, N * sizeof(float)));

    // Begin graph capture
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

    // Record the kernel sequence into the graph
    for (int k = 0; k < num_kernels; ++k) {
        step_kernel<<<grid_dim, block_dim, 0, stream>>>(
            d_data, N, 1.0001f, static_cast<float>(k) * 0.0001f);
    }

    // End capture — creates a cudaGraph_t object
    cudaGraph_t graph;
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));

    // Query graph node count
    size_t num_nodes = 0;
    CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &num_nodes));
    oss.str("");
    oss << "Graph captured with " << num_nodes << " nodes ("
        << num_kernels << " kernels)";
    logger.log_info(oss.str());

    // Instantiate (compile) the graph
    cudaGraphExec_t graph_exec;
    CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
    logger.log_info("Graph instantiated (compiled for execution).");

    // Warm-up replay
    CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Timed replays
    ev_start.record(stream);
    for (int r = 0; r < num_replay; ++r) {
        CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
    }
    ev_stop.record(stream);
    float t_graph_total = ev_stop.elapsed_ms(ev_start);
    float t_graph_per_iter   = t_graph_total / num_replay;
    float t_graph_per_kernel = t_graph_per_iter / num_kernels;

    oss.str("");
    oss << "Graph  : total=" << std::fixed << std::setprecision(3) << t_graph_total
        << " ms  per_iter=" << t_graph_per_iter
        << " ms  per_kernel=" << t_graph_per_kernel << " ms";
    logger.log_info(oss.str());

    // =========================================================================
    // Comparison
    // =========================================================================
    double speedup = t_regular_per_iter / t_graph_per_iter;
    double overhead_saved_us = (t_regular_per_kernel - t_graph_per_kernel) * 1000.0;

    oss.str("");
    oss << "Graph speedup: " << std::fixed << std::setprecision(2) << speedup << "x";
    logger.log_info(oss.str());

    oss.str("");
    oss << "Launch overhead saved per kernel: ~" << std::setprecision(1)
        << overhead_saved_us << " µs";
    logger.log_info(oss.str());

    logger.log_info(
        "Explanation: regular kernel launches pay ~5–20 µs CPU overhead each. "
        "With " + std::to_string(num_kernels) + " kernels, this is " +
        std::to_string(num_kernels) + "× that overhead per iteration. "
        "CUDA Graphs reduce this to a single cudaGraphLaunch() call, with "
        "the GPU executing all " + std::to_string(num_kernels) + " kernels "
        "from its prefetched command buffer — no per-kernel CPU round-trip.");

    // Cleanup
    CUDA_CHECK(cudaGraphExecDestroy(graph_exec));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_data));

    logger.log_info("Tutorial 25 complete.");
    return 0;
}
