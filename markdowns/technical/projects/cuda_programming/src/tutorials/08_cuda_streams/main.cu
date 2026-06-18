// =============================================================================
// Tutorial 08: CUDA Streams
//
// Concept:
//   A CUDA STREAM is an ordered sequence of CUDA operations (memcpy, kernel)
//   that execute in order within the stream. Operations in DIFFERENT streams
//   can overlap with each other if:
//     1. The GPU has separate copy engines (H2D, D2H, and compute).
//     2. Resources are not mutually exclusive.
//
//   H200 has:
//     - 1 H2D copy engine
//     - 1 D2H copy engine
//     - Many compute engines (SMs)
//   → True triple overlap: H2D + compute + D2H in flight simultaneously.
//
// Experiment:
//   Process num_chunks data chunks, each of size chunk_size (floats).
//   Total data = num_chunks * chunk_size floats.
//
//   Mode A: single stream — H2D(0), kernel(0), D2H(0), H2D(1), kernel(1), ...
//             All operations serialized.
//   Mode B: num_streams streams — chunk i goes to stream (i % num_streams)
//             Copy and compute overlap across different streams.
//
//   The kernel is a simple scale-by-2 operation (compute is light, so the
//   overlap benefit is mostly on the copy/compute pipeline).
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
// Kernel: scale by 2 (trivial compute, lets memory copies dominate)
// ---------------------------------------------------------------------------
__global__ void scale_kernel(float* __restrict__ data, int n, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] *= scale;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    const std::string project_root  = PROJECT_ROOT;
    const std::string config_path   = project_root + "/configs/tutorials/08_cuda_streams.yaml";
    const std::string global_config = project_root + "/configs/global.yaml";

    auto cfg = cuda_tutorials::ConfigLoader::from_file(config_path);
    cfg.merge_defaults(global_config);

    auto logger = cuda_tutorials::Logger::create(config_path, project_root,
                                                  "08_cuda_streams");
    logger.log_info("=== Tutorial 08: CUDA Streams ===");

    const int device_id   = cfg.get<int>("device", "id");
    const int num_streams = cfg.get<int>("tutorial", "num_streams");
    const int chunk_size  = cfg.get<int>("tutorial", "chunk_size");
    const int num_chunks  = cfg.get<int>("tutorial", "num_chunks");
    const int num_iter    = cfg.get<int>("tutorial", "num_iterations");

    CUDA_CHECK(cudaSetDevice(device_id));
    auto dev_info = cuda_tutorials::get_device_info(device_id);
    cuda_tutorials::print_device_info(logger, dev_info);

    std::ostringstream oss;
    oss << "Config: num_streams=" << num_streams
        << "  chunk_size=" << chunk_size
        << "  num_chunks=" << num_chunks;
    logger.log_info(oss.str());

    const int total_elems = chunk_size * num_chunks;
    double total_bytes    = static_cast<double>(total_elems) * sizeof(float);

    // Pinned host memory (required for async memcpy overlap)
    float* h_in{};
    float* h_out{};
    CUDA_CHECK(cudaMallocHost(&h_in,  total_elems * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_out, total_elems * sizeof(float)));
    for (int i = 0; i < total_elems; ++i) h_in[i] = static_cast<float>(i) * 0.001f;

    // Device buffer: one full allocation, partition into chunks
    float* d_buf{};
    CUDA_CHECK(cudaMalloc(&d_buf, total_elems * sizeof(float)));

    const int   block_dim = 256;
    const int   chunk_grid = (chunk_size + block_dim - 1) / block_dim;

    // Create streams
    std::vector<cudaStream_t> streams(num_streams);
    for (auto& s : streams) CUDA_CHECK(cudaStreamCreate(&s));

    // CUDA events for wall-time measurement
    cuda_tutorials::CudaEvent ev_start, ev_stop;

    // =========================================================================
    // Mode A: Single stream (serialized pipeline)
    // =========================================================================
    logger.log_info("--- Mode A: single stream ---");
    float best_single = 1e9f;
    for (int iter = 0; iter < num_iter; ++iter) {
        ev_start.record(streams[0]);

        for (int c = 0; c < num_chunks; ++c) {
            float* h_in_c  = h_in  + c * chunk_size;
            float* h_out_c = h_out + c * chunk_size;
            float* d_c     = d_buf + c * chunk_size;

            // H2D
            CUDA_CHECK(cudaMemcpyAsync(d_c, h_in_c,
                                       chunk_size * sizeof(float),
                                       cudaMemcpyHostToDevice, streams[0]));
            // Compute
            scale_kernel<<<chunk_grid, block_dim, 0, streams[0]>>>(d_c, chunk_size, 2.0f);

            // D2H
            CUDA_CHECK(cudaMemcpyAsync(h_out_c, d_c,
                                       chunk_size * sizeof(float),
                                       cudaMemcpyDeviceToHost, streams[0]));
        }

        ev_stop.record(streams[0]);
        float ms = ev_stop.elapsed_ms(ev_start);
        if (ms < best_single) best_single = ms;
    }

    oss.str("");
    oss << "Single stream best: " << std::fixed << std::setprecision(3) << best_single
        << " ms  effective BW=" << std::setprecision(2)
        << (2.0 * total_bytes / (best_single * 1e-3) / 1e9) << " GB/s";
    logger.log_info(oss.str());

    // =========================================================================
    // Mode B: Multi-stream (overlapping pipeline)
    // =========================================================================
    logger.log_info("--- Mode B: " + std::to_string(num_streams) + " streams ---");

    float best_multi = 1e9f;
    for (int iter = 0; iter < num_iter; ++iter) {
        // Record on default stream (synchronizes before we start)
        CUDA_CHECK(cudaDeviceSynchronize());
        ev_start.record();

        for (int c = 0; c < num_chunks; ++c) {
            cudaStream_t s = streams[c % num_streams];
            float* h_in_c  = h_in  + c * chunk_size;
            float* h_out_c = h_out + c * chunk_size;
            float* d_c     = d_buf + c * chunk_size;

            CUDA_CHECK(cudaMemcpyAsync(d_c, h_in_c,
                                       chunk_size * sizeof(float),
                                       cudaMemcpyHostToDevice, s));
            scale_kernel<<<chunk_grid, block_dim, 0, s>>>(d_c, chunk_size, 2.0f);
            CUDA_CHECK(cudaMemcpyAsync(h_out_c, d_c,
                                       chunk_size * sizeof(float),
                                       cudaMemcpyDeviceToHost, s));
        }

        // Sync all streams
        for (auto& s : streams) CUDA_CHECK(cudaStreamSynchronize(s));
        ev_stop.record();
        float ms = ev_stop.elapsed_ms(ev_start);
        if (ms < best_multi) best_multi = ms;
    }

    oss.str("");
    oss << "Multi-stream best : " << std::fixed << std::setprecision(3) << best_multi
        << " ms  effective BW=" << std::setprecision(2)
        << (2.0 * total_bytes / (best_multi * 1e-3) / 1e9) << " GB/s";
    logger.log_info(oss.str());

    double speedup      = best_single / best_multi;
    double overlap_pct  = (1.0 - best_multi / best_single) * 100.0;
    oss.str("");
    oss << "Speedup: " << std::fixed << std::setprecision(2) << speedup
        << "x  Overlap efficiency: " << overlap_pct << "%";
    logger.log_info(oss.str());

    logger.log_info(
        "Explanation: with num_streams > 1, while stream[0] is waiting for "
        "D2H of chunk[0], stream[1] can be issuing H2D for chunk[1] on the "
        "separate copy engine, and the compute engine executes kernels. "
        "This triple overlap (H2D + compute + D2H) reduces total wall time.");

    // Cleanup
    for (auto& s : streams) CUDA_CHECK(cudaStreamDestroy(s));
    CUDA_CHECK(cudaFreeHost(h_in));
    CUDA_CHECK(cudaFreeHost(h_out));
    CUDA_CHECK(cudaFree(d_buf));

    logger.log_info("Tutorial 08 complete.");
    return 0;
}
