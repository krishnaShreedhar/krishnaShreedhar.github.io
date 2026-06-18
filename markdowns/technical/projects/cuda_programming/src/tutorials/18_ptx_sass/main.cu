// =============================================================================
// Tutorial 18: PTX & SASS
//
// Concept:
//   PTX (Parallel Thread eXecution) is NVIDIA's virtual ISA — an intermediate
//   representation that nvcc compiles CUDA C++ into. PTX is then compiled by
//   ptxas to SASS (Streaming ASsembler), the actual GPU machine code.
//
//   Inline PTX via asm volatile:
//     asm volatile("add.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
//
//   Common PTX instructions:
//     ld.global.f32  — load from global memory
//     st.global.f32  — store to global memory
//     fma.rn.f32     — fused multiply-add (round to nearest)
//     mov.u32        — move (register assign)
//     @p bra         — predicated branch
//
//   Optimization hints:
//     #pragma unroll N  — fully/partially unroll the next loop
//     __builtin_assume_aligned(ptr, N) — hint that ptr is N-byte aligned
//     __restrict__      — no aliasing, enables more aggressive optimization
//
// Experiment:
//   Simple dot-product kernel, three variants:
//     (a) Plain C++ loop
//     (b) With #pragma unroll
//     (c) With inline PTX fma.rn
//   Compare throughput and inspect register counts via cudaFuncGetAttributes.
// =============================================================================

#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "../../common/config_loader.hpp"
#include "../../common/cuda_utils.hpp"
#include "../../common/logger.hpp"

// ---------------------------------------------------------------------------
// Kernel A: plain C++ loop (no unroll hint)
// ---------------------------------------------------------------------------
__global__ void kernel_plain(const float* __restrict__ a,
                              const float* __restrict__ b,
                              float*       __restrict__ c,
                              int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    // Load from global, compute, store — no special hints
    c[idx] = a[idx] * b[idx] + a[idx];
}

// ---------------------------------------------------------------------------
// Kernel B: with loop unroll pragma (helps for reduction-like inner loops)
// ---------------------------------------------------------------------------
__global__ void kernel_unroll(const float* __restrict__ a,
                               const float* __restrict__ b,
                               float*       __restrict__ c,
                               int n,
                               int unroll_factor) {
    int base = (blockIdx.x * blockDim.x + threadIdx.x) * unroll_factor;

    // Hint to compiler: unroll the next loop 8 times
    // This exposes more instruction-level parallelism (ILP) to the GPU
    float acc[8] = {};
    #pragma unroll 8
    for (int i = 0; i < unroll_factor && base + i < n; ++i) {
        acc[i] = a[base + i] * b[base + i] + a[base + i];
    }
    #pragma unroll 8
    for (int i = 0; i < unroll_factor && base + i < n; ++i) {
        c[base + i] = acc[i];
    }
}

// ---------------------------------------------------------------------------
// Kernel C: inline PTX using fma.rn (fused multiply-add, round nearest)
//   This bypasses the C++ compiler's FP contraction rules and explicitly
//   uses the hardware FMA instruction.
// ---------------------------------------------------------------------------
__global__ void kernel_ptx_fma(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float*       __restrict__ c,
                                int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float val_a = a[idx];
    float val_b = b[idx];
    float result;

    // Inline PTX: result = val_a * val_b + val_a
    // fma.rn.f32 dst, src1, src2, src3  → dst = src1 * src2 + src3
    asm volatile(
        "fma.rn.f32 %0, %1, %2, %3;"
        : "=f"(result)
        : "f"(val_a), "f"(val_b), "f"(val_a)
    );

    c[idx] = result;
}

// ---------------------------------------------------------------------------
// Print kernel register and shared memory usage
// ---------------------------------------------------------------------------
template <typename KernelT>
static void log_func_attrs(cuda_tutorials::Logger& logger,
                            KernelT* kernel_fn,
                            const char* kernel_name) {
    cudaFuncAttributes attrs{};
    CUDA_CHECK(cudaFuncGetAttributes(&attrs, kernel_fn));
    std::ostringstream oss;
    oss << kernel_name
        << "  regs=" << attrs.numRegs
        << "  const_mem=" << attrs.constSizeBytes << "B"
        << "  local_mem=" << attrs.localSizeBytes << "B"
        << "  shared_mem=" << attrs.sharedSizeBytes << "B"
        << "  max_threads=" << attrs.maxThreadsPerBlock;
    logger.log_info(oss.str());
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    const std::string project_root  = PROJECT_ROOT;
    const std::string config_path   = project_root + "/configs/tutorials/18_ptx_sass.yaml";
    const std::string global_config = project_root + "/configs/global.yaml";

    auto cfg = cuda_tutorials::ConfigLoader::from_file(config_path);
    cfg.merge_defaults(global_config);

    auto logger = cuda_tutorials::Logger::create(config_path, project_root, "18_ptx_sass");
    logger.log_info("=== Tutorial 18: PTX & SASS ===");

    const int device_id     = cfg.get<int>("device",   "id");
    const int N             = cfg.get<int>("tutorial", "N");
    const int unroll_factor = cfg.get<int>("tutorial", "unroll_factor");
    const int num_iter      = cfg.get<int>("tutorial", "num_iterations");

    CUDA_CHECK(cudaSetDevice(device_id));
    auto dev_info = cuda_tutorials::get_device_info(device_id);
    cuda_tutorials::print_device_info(logger, dev_info);

    std::ostringstream oss;
    oss << "Config: N=" << N << "  unroll_factor=" << unroll_factor;
    logger.log_info(oss.str());

    // Log register counts from kernel metadata
    logger.log_info("--- Kernel attributes (registers, memory) ---");
    log_func_attrs(logger, kernel_plain,   "kernel_plain  ");
    log_func_attrs(logger, kernel_unroll,  "kernel_unroll ");
    log_func_attrs(logger, kernel_ptx_fma, "kernel_ptx_fma");

    float *d_a{}, *d_b{}, *d_c{};
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_a, 0x3f, N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_b, 0x3f, N * sizeof(float)));

    const int block_dim = 256;
    const int grid_dim  = (N + block_dim - 1) / block_dim;

    cuda_tutorials::CudaEvent ev_start, ev_stop;

    auto bench = [&](auto fn, const char* label) -> float {
        fn(); CUDA_KERNEL_CHECK();
        ev_start.record();
        for (int i = 0; i < num_iter; ++i) fn();
        ev_stop.record();
        CUDA_CHECK(cudaEventSynchronize(ev_stop.event));
        float t = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&t, ev_start.event, ev_stop.event));
        float ms = t / num_iter;
        double bw = 3.0 * N * sizeof(float) / (ms * 1e-3) / 1e9;
        oss.str("");
        oss << label << ": " << std::fixed << std::setprecision(3) << ms
            << " ms  " << std::setprecision(2) << bw << " GB/s";
        logger.log_info(oss.str());
        return ms;
    };

    bench([&]{ kernel_plain<<<grid_dim, block_dim>>>(d_a, d_b, d_c, N); },
          "Plain (no hints)  ");

    int unroll_grid = (N / unroll_factor + block_dim - 1) / block_dim;
    bench([&]{ kernel_unroll<<<unroll_grid, block_dim>>>(d_a, d_b, d_c, N, unroll_factor); },
          "Unrolled          ");

    bench([&]{ kernel_ptx_fma<<<grid_dim, block_dim>>>(d_a, d_b, d_c, N); },
          "Inline PTX fma    ");

    logger.log_info(
        "PTX tip: compile with '-ptx' to dump PTX, then 'ptxas -v' for register info. "
        "View SASS with: cuobjdump --dump-sass <binary>. "
        "Unrolling exposes ILP — the GPU can issue multiple independent instructions "
        "per cycle if they don't depend on each other. "
        "fma.rn combines multiply+add in one instruction, reducing instruction count "
        "and improving throughput for compute-bound code.");

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    logger.log_info("Tutorial 18 complete.");
    return 0;
}
