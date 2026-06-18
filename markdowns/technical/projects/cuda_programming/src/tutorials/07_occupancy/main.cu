// =============================================================================
// Tutorial 07: Occupancy
//
// Concept:
//   Occupancy = (active warps on SM) / (max warps SM can hold)
//
//   The SM has limited resources shared among all resident blocks:
//     - Registers (65536 per SM on sm_90)
//     - Shared memory (228 KB max per SM on H200)
//     - Max resident warps (64 per SM on sm_90)
//     - Max resident blocks (32 per SM)
//
//   Higher occupancy helps hide memory latency by having more warps ready
//   to execute when others are waiting for memory. However, kernel code
//   that maximizes register use may achieve better IPC at lower occupancy.
//
// Experiment:
//   Three kernel variants:
//     light: few registers, small shared mem → high occupancy
//     heavy: many register-pressure ops   → register-limited occupancy
//     smem:  large shared memory allocation → shared-mem-limited occupancy
//
//   We use cudaOccupancyMaxActiveBlocksPerMultiprocessor() to compute
//   theoretical occupancy without launching, then measure actual throughput.
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
// Kernel: light — few registers, minimal shared memory
// ---------------------------------------------------------------------------
__global__ void kernel_light(const float* __restrict__ in,
                              float*       __restrict__ out,
                              int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = in[idx] * 2.0f + 1.0f;
}

// ---------------------------------------------------------------------------
// Kernel: heavy — register-pressure via many locals
// ---------------------------------------------------------------------------
__global__ void kernel_heavy(const float* __restrict__ in,
                              float*       __restrict__ out,
                              int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Force compiler to keep many live registers
    float a0 = in[idx];
    float a1 = a0 * 1.1f; float a2 = a1 + 0.2f; float a3 = a2 * 1.3f;
    float a4 = a3 - 0.4f; float a5 = a4 * 1.5f; float a6 = a5 + 0.6f;
    float a7 = a6 * 1.7f; float a8 = a7 - 0.8f; float a9 = a8 * 1.9f;
    float b0 = a9 + a0;   float b1 = b0 * a1;   float b2 = b1 - a2;
    float b3 = b2 * a3;   float b4 = b3 + a4;   float b5 = b4 - a5;
    float b6 = b5 * a6;   float b7 = b6 + a7;   float b8 = b7 - a8;
    float b9 = b8 * a9 + b0 - b1 + b2 - b3 + b4 - b5 + b6 - b7 + b8;

    out[idx] = b9;
}

// ---------------------------------------------------------------------------
// Kernel: smem — large shared memory allocation limits blocks per SM
// ---------------------------------------------------------------------------
__global__ void kernel_smem(const float* __restrict__ in,
                             float*       __restrict__ out,
                             int n) {
    // Dynamic shared memory allocated at launch; size from config.
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    smem[tid] = (idx < n) ? in[idx] : 0.0f;
    __syncthreads();

    // Simple op in shared memory
    smem[tid] = smem[tid] * 2.0f;
    __syncthreads();

    if (idx < n) out[idx] = smem[tid];
}

// ---------------------------------------------------------------------------
// Compute and log theoretical occupancy for a given kernel and block size
// ---------------------------------------------------------------------------
template <typename KernelFn>
static void log_occupancy(cuda_tutorials::Logger&       logger,
                           KernelFn                      kernel_fn,
                           int                           block_size,
                           size_t                        smem_bytes,
                           const cuda_tutorials::DeviceInfo& dev,
                           const std::string&            kernel_name) {
    int max_blocks_per_sm = 0;
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm, kernel_fn, block_size, smem_bytes);
    if (err != cudaSuccess) {
        logger.log_warn(kernel_name + ": occupancy query failed: " +
                        std::string(cudaGetErrorString(err)));
        return;
    }

    int max_warps_per_sm = dev.max_threads_per_sm / dev.warp_size;
    int active_warps     = max_blocks_per_sm * (block_size / dev.warp_size);
    double occupancy_pct = 100.0 * active_warps / max_warps_per_sm;

    std::ostringstream oss;
    oss << kernel_name
        << "  block_size=" << std::setw(5) << block_size
        << "  active_blocks/SM=" << max_blocks_per_sm
        << "  active_warps/SM=" << std::setw(3) << active_warps
        << "/" << max_warps_per_sm
        << "  occupancy=" << std::fixed << std::setprecision(1) << occupancy_pct << "%";
    logger.log_info(oss.str());
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    const std::string project_root  = PROJECT_ROOT;
    const std::string config_path   = project_root + "/configs/tutorials/07_occupancy.yaml";
    const std::string global_config = project_root + "/configs/global.yaml";

    auto cfg = cuda_tutorials::ConfigLoader::from_file(config_path);
    cfg.merge_defaults(global_config);

    auto logger = cuda_tutorials::Logger::create(config_path, project_root,
                                                  "07_occupancy");
    logger.log_info("=== Tutorial 07: Occupancy ===");

    const int device_id          = cfg.get<int>("device", "id");
    const int N                  = cfg.get<int>("tutorial", "N");
    const int num_iter           = cfg.get<int>("tutorial", "num_iterations");
    const int smem_per_block     = cfg.get<int>("tutorial", "shared_mem_per_block");
    auto block_sizes             = cfg.get_vector<int>("tutorial", "block_sizes");
    auto kernel_types            = cfg.get_vector<std::string>("tutorial", "kernel_types");

    CUDA_CHECK(cudaSetDevice(device_id));
    auto dev_info = cuda_tutorials::get_device_info(device_id);
    cuda_tutorials::print_device_info(logger, dev_info);

    std::ostringstream oss;
    oss << "Config: N=" << N << "  smem_per_block=" << smem_per_block << "B";
    logger.log_info(oss.str());

    float *d_in{}, *d_out{};
    CUDA_CHECK(cudaMalloc(&d_in,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_in, 0x3f, N * sizeof(float)));  // fill with ~0.5

    cuda_tutorials::CudaEvent ev_start, ev_stop;

    logger.log_info("--- Theoretical occupancy sweep ---");

    for (int bs : block_sizes) {
        log_occupancy(logger, kernel_light,    bs, 0,              dev_info, "light");
        log_occupancy(logger, kernel_heavy,    bs, 0,              dev_info, "heavy");
        log_occupancy(logger, kernel_smem,     bs, bs*sizeof(float), dev_info, "smem ");
        logger.log_debug("---");
    }

    // ---------------------------------------------------------------------------
    // Actual throughput at block_sizes[2] (representative middle value)
    // ---------------------------------------------------------------------------
    int bench_bs = block_sizes.size() >= 3 ? block_sizes[2] : block_sizes.back();
    int grid = (N + bench_bs - 1) / bench_bs;

    logger.log_info("--- Throughput benchmark at block_size=" + std::to_string(bench_bs) + " ---");

    auto bench = [&](auto fn_launch, const char* label) {
        fn_launch(); CUDA_KERNEL_CHECK();
        ev_start.record();
        for (int i = 0; i < num_iter; ++i) fn_launch();
        ev_stop.record();
        CUDA_CHECK(cudaEventSynchronize(ev_stop.event));
        float t = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&t, ev_start.event, ev_stop.event));
        float ms  = t / num_iter;
        double bw = 2.0 * N * sizeof(float) / (ms * 1e-3) / 1e9;
        oss.str("");
        oss << label << ": " << std::fixed << std::setprecision(3) << ms
            << " ms  " << std::setprecision(2) << bw << " GB/s";
        logger.log_info(oss.str());
    };

    bench([&]{ kernel_light<<<grid, bench_bs>>>(d_in, d_out, N); }, "light");
    bench([&]{ kernel_heavy<<<grid, bench_bs>>>(d_in, d_out, N); }, "heavy");
    bench([&]{ kernel_smem<<<grid, bench_bs, bench_bs*sizeof(float)>>>(d_in, d_out, N); }, "smem ");

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    logger.log_info("Tutorial 07 complete.");
    return 0;
}
