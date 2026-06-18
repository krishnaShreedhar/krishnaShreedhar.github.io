// =============================================================================
// Tutorial 19: Nsight Profiling & NVTX
//
// Concept:
//   NVTX (NVIDIA Tools Extension Library) allows you to annotate your code
//   with named ranges and markers. These appear in Nsight Systems and Nsight
//   Compute timelines for visual correlation of CPU activity and GPU kernels.
//
//   NVTX API:
//     nvtxRangePushA("name")  — push a CPU-side named range onto the stack
//     nvtxRangePop()          — pop the innermost range
//     nvtxMark("label")       — instant marker in the timeline
//
//   To profile with Nsight Systems:
//     nsys profile --trace=cuda,nvtx ./19_nsight_profiling
//   To profile with Nsight Compute:
//     ncu --set full ./19_nsight_profiling
//
//   Metrics to examine in Nsight Compute:
//     - SM utilization (sm__throughput.avg.pct_of_peak_sustained_active)
//     - Memory bandwidth (l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum)
//     - Warp stall reasons (smsp__warp_issue_stalled_*)
//     - Achieved occupancy (sm__warps_active.avg.pct_of_peak_sustained_active)
// =============================================================================

#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

// NVTX v3 header — included via cmake target nvtx3
#include <nvtx3/nvToolsExt.h>

#include "../../common/config_loader.hpp"
#include "../../common/cuda_utils.hpp"
#include "../../common/logger.hpp"

// ---------------------------------------------------------------------------
// Kernels for each phase of the workload
// ---------------------------------------------------------------------------
__global__ void phase_load(const float* __restrict__ src,
                            float*       __restrict__ tmp,
                            int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) tmp[idx] = src[idx];
}

__global__ void phase_compute(const float* __restrict__ in,
                               float*       __restrict__ out,
                               int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = in[idx];
        // Simulate multi-step compute
        for (int i = 0; i < 10; ++i) v = v * 1.001f + 0.001f;
        out[idx] = v;
    }
}

__global__ void phase_reduce(const float* __restrict__ in,
                              float*       __restrict__ out,
                              int n) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    smem[tid] = (idx < n) ? in[idx] : 0.f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = smem[0];
}

__global__ void phase_store(const float* __restrict__ src,
                             float*       __restrict__ dst,
                             int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = src[idx] * 2.0f;
}

// ---------------------------------------------------------------------------
// NVTX color palette for visual distinction in timeline
// ---------------------------------------------------------------------------
static const uint32_t NVTX_COLORS[] = {
    0xFF0000FF,  // blue
    0xFF00FF00,  // green
    0xFFFF0000,  // red
    0xFFFF8000,  // orange
};
static const char* PHASE_NAMES[] = {"Load", "Compute", "Reduce", "Store"};

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    const std::string project_root  = PROJECT_ROOT;
    const std::string config_path   = project_root +
        "/configs/tutorials/19_nsight_profiling.yaml";
    const std::string global_config = project_root + "/configs/global.yaml";

    auto cfg = cuda_tutorials::ConfigLoader::from_file(config_path);
    cfg.merge_defaults(global_config);

    auto logger = cuda_tutorials::Logger::create(config_path, project_root,
                                                  "19_nsight_profiling");
    logger.log_info("=== Tutorial 19: Nsight Profiling & NVTX ===");

    const int device_id  = cfg.get<int>("device",   "id");
    const int N          = cfg.get<int>("tutorial", "N");
    const int num_phases = cfg.get<int>("tutorial", "num_phases");
    const int num_iter   = cfg.get<int>("tutorial", "num_iterations");

    CUDA_CHECK(cudaSetDevice(device_id));
    auto dev_info = cuda_tutorials::get_device_info(device_id);
    cuda_tutorials::print_device_info(logger, dev_info);

    std::ostringstream oss;
    oss << "Config: N=" << N << "  num_phases=" << num_phases
        << "  num_iter=" << num_iter;
    logger.log_info(oss.str());

    logger.log_info(
        "To profile: nsys profile --trace=cuda,nvtx ./19_nsight_profiling "
        "  or: ncu --set full ./19_nsight_profiling");

    const int block_dim = 256;
    const int grid_dim  = (N + block_dim - 1) / block_dim;

    float *d_src{}, *d_tmp{}, *d_out{}, *d_partial{};
    CUDA_CHECK(cudaMalloc(&d_src,     N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tmp,     N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out,     N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_partial, grid_dim * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_src, 0x3f, N * sizeof(float)));

    cuda_tutorials::CudaEvent ev_start, ev_stop;

    // Main timed loop — annotated with NVTX
    nvtxRangePushA("tutorial_19_main_loop");

    for (int iter = 0; iter < num_iter; ++iter) {
        oss.str("");
        oss << "iteration_" << iter;
        nvtxRangePushA(oss.str().c_str());

        // Phase 0: Load
        {
            nvtxEventAttributes_t attr = {};
            attr.version   = NVTX_VERSION;
            attr.size      = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
            attr.colorType = NVTX_COLOR_ARGB;
            attr.color     = NVTX_COLORS[0];
            attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
            attr.message.ascii = PHASE_NAMES[0];
            nvtxRangePushEx(&attr);

            ev_start.record();
            phase_load<<<grid_dim, block_dim>>>(d_src, d_tmp, N);
            ev_stop.record();
            float ms = ev_stop.elapsed_ms(ev_start);

            nvtxRangePop();
            oss.str("");
            oss << "iter=" << iter << "  Phase[Load]    : " << std::fixed
                << std::setprecision(3) << ms << " ms";
            logger.log_debug(oss.str());
        }

        // Phase 1: Compute
        {
            nvtxEventAttributes_t attr = {};
            attr.version   = NVTX_VERSION;
            attr.size      = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
            attr.colorType = NVTX_COLOR_ARGB;
            attr.color     = NVTX_COLORS[1];
            attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
            attr.message.ascii = PHASE_NAMES[1];
            nvtxRangePushEx(&attr);

            ev_start.record();
            phase_compute<<<grid_dim, block_dim>>>(d_tmp, d_out, N);
            ev_stop.record();
            float ms = ev_stop.elapsed_ms(ev_start);

            nvtxRangePop();
            oss.str("");
            oss << "iter=" << iter << "  Phase[Compute] : " << std::fixed
                << std::setprecision(3) << ms << " ms";
            logger.log_debug(oss.str());
        }

        // Phase 2: Reduce
        if (num_phases >= 3) {
            nvtxEventAttributes_t attr = {};
            attr.version     = NVTX_VERSION;
            attr.size        = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
            attr.colorType   = NVTX_COLOR_ARGB;
            attr.color       = NVTX_COLORS[2];
            attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
            attr.message.ascii = PHASE_NAMES[2];
            nvtxRangePushEx(&attr);

            ev_start.record();
            phase_reduce<<<grid_dim, block_dim, block_dim * sizeof(float)>>>(
                d_out, d_partial, N);
            ev_stop.record();
            float ms = ev_stop.elapsed_ms(ev_start);

            nvtxRangePop();
            oss.str("");
            oss << "iter=" << iter << "  Phase[Reduce]  : " << ms << " ms";
            logger.log_debug(oss.str());
        }

        // Phase 3: Store
        if (num_phases >= 4) {
            nvtxEventAttributes_t attr = {};
            attr.version     = NVTX_VERSION;
            attr.size        = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
            attr.colorType   = NVTX_COLOR_ARGB;
            attr.color       = NVTX_COLORS[3];
            attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
            attr.message.ascii = PHASE_NAMES[3];
            nvtxRangePushEx(&attr);

            ev_start.record();
            phase_store<<<grid_dim, block_dim>>>(d_out, d_tmp, N);
            ev_stop.record();
            float ms = ev_stop.elapsed_ms(ev_start);

            nvtxRangePop();
            oss.str("");
            oss << "iter=" << iter << "  Phase[Store]   : " << ms << " ms";
            logger.log_debug(oss.str());
        }

        nvtxMark("iteration_complete");
        nvtxRangePop();  // iteration range
    }

    nvtxRangePop();  // main_loop range

    CUDA_CHECK(cudaDeviceSynchronize());

    logger.log_info("NVTX ranges logged. Run with nsys/ncu to visualize.");
    logger.log_info(
        "Key Nsight Compute metrics to check:\n"
        "  sm__throughput.avg.pct_of_peak_sustained_active  (SM utilization)\n"
        "  l1tex__t_bytes.sum (L1/shared memory traffic)\n"
        "  dram__bytes.sum (HBM traffic)\n"
        "  smsp__warp_issue_stalled_long_scoreboard_per_warp (memory stalls)");

    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_tmp));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_partial));

    logger.log_info("Tutorial 19 complete.");
    return 0;
}
