// =============================================================================
// Tutorial 11: Pinned (Page-Locked) Memory
//
// Concept:
//   Normal malloc() returns pageable memory — the OS may swap pages out.
//   CUDA H2D/D2H copies with pageable memory require an intermediate bounce
//   buffer in pinned memory, adding a CPU-side memcpy before the DMA transfer.
//
//   cudaMallocHost() (aka pinned/page-locked memory):
//     - Pages are locked in physical RAM — no OS swapping allowed
//     - DMA engine can transfer directly without bounce buffer
//     - Higher H2D/D2H bandwidth (often 2–3× improvement on PCIe/NVLink)
//     - Overhead: reduces available OS physical memory
//
//   NVLink bandwidth (H200 SXM): ~900 GB/s bidirectional
//   PCIe Gen5 x16: ~64 GB/s each direction (if using PCIe H200)
// =============================================================================

#include <chrono>
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
// main
// ---------------------------------------------------------------------------
int main() {
    const std::string project_root  = PROJECT_ROOT;
    const std::string config_path   = project_root + "/configs/tutorials/11_pinned_memory.yaml";
    const std::string global_config = project_root + "/configs/global.yaml";

    auto cfg = cuda_tutorials::ConfigLoader::from_file(config_path);
    cfg.merge_defaults(global_config);

    auto logger = cuda_tutorials::Logger::create(config_path, project_root,
                                                  "11_pinned_memory");
    logger.log_info("=== Tutorial 11: Pinned Memory ===");

    const int device_id = cfg.get<int>("device", "id");
    const int num_iter  = cfg.get<int>("tutorial", "num_iterations");
    auto sizes_mb       = cfg.get_vector<int>("tutorial", "sizes_mb");

    CUDA_CHECK(cudaSetDevice(device_id));
    auto dev_info = cuda_tutorials::get_device_info(device_id);
    cuda_tutorials::print_device_info(logger, dev_info);

    std::ostringstream oss;

    // CUDA events for GPU-side transfer timing
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    logger.log_info("--- H2D bandwidth: pageable vs pinned ---");

    for (int size_mb : sizes_mb) {
        size_t bytes = static_cast<size_t>(size_mb) * 1024 * 1024;
        size_t n     = bytes / sizeof(float);

        // Pageable memory
        std::vector<float> h_pageable(n, 1.0f);

        // Pinned memory
        float* h_pinned{};
        CUDA_CHECK(cudaMallocHost(&h_pinned, bytes));
        for (size_t i = 0; i < n; ++i) h_pinned[i] = 1.0f;

        // Device buffer
        float* d_buf{};
        CUDA_CHECK(cudaMalloc(&d_buf, bytes));

        // --- Pageable H2D ---
        CUDA_CHECK(cudaEventRecord(ev_start));
        for (int i = 0; i < num_iter; ++i) {
            CUDA_CHECK(cudaMemcpy(d_buf, h_pageable.data(), bytes, cudaMemcpyHostToDevice));
        }
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        float t_pg_h2d = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&t_pg_h2d, ev_start, ev_stop));
        t_pg_h2d /= num_iter;
        double bw_pg_h2d = bytes / (t_pg_h2d * 1e-3) / 1e9;

        // --- Pinned H2D ---
        CUDA_CHECK(cudaEventRecord(ev_start));
        for (int i = 0; i < num_iter; ++i) {
            CUDA_CHECK(cudaMemcpy(d_buf, h_pinned, bytes, cudaMemcpyHostToDevice));
        }
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        float t_pin_h2d = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&t_pin_h2d, ev_start, ev_stop));
        t_pin_h2d /= num_iter;
        double bw_pin_h2d = bytes / (t_pin_h2d * 1e-3) / 1e9;

        // --- Pinned D2H ---
        CUDA_CHECK(cudaEventRecord(ev_start));
        for (int i = 0; i < num_iter; ++i) {
            CUDA_CHECK(cudaMemcpy(h_pinned, d_buf, bytes, cudaMemcpyDeviceToHost));
        }
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        float t_pin_d2h = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&t_pin_d2h, ev_start, ev_stop));
        t_pin_d2h /= num_iter;
        double bw_pin_d2h = bytes / (t_pin_d2h * 1e-3) / 1e9;

        double speedup_h2d = bw_pin_h2d / bw_pg_h2d;

        oss.str("");
        oss << std::setw(4) << size_mb << " MB"
            << "  pageable H2D=" << std::fixed << std::setprecision(2)
            << std::setw(7) << bw_pg_h2d << " GB/s"
            << "  pinned H2D=" << std::setw(7) << bw_pin_h2d << " GB/s"
            << "  pinned D2H=" << std::setw(7) << bw_pin_d2h << " GB/s"
            << "  speedup=" << std::setprecision(2) << speedup_h2d << "x";
        logger.log_info(oss.str());

        CUDA_CHECK(cudaFreeHost(h_pinned));
        CUDA_CHECK(cudaFree(d_buf));
    }

    logger.log_info(
        "Explanation: pageable H2D requires copying data to a staging "
        "pinned buffer first (one extra CPU memcpy), then DMA. "
        "Pinned memory skips the staging copy, directly DMA-ing from "
        "the locked page. Speedup is most visible at large transfer sizes "
        "where DMA efficiency dominates.");

    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));

    logger.log_info("Tutorial 11 complete.");
    return 0;
}
