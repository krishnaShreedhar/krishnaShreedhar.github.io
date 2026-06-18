#pragma once
// =============================================================================
// cuda_utils.hpp — CUDA/cuDNN/cuBLAS error checking and device helpers
//
// Macros:
//   CUDA_CHECK(call)         — wraps any CUDA runtime call
//   CUDA_KERNEL_CHECK()      — call after kernel launches
//   cudnn_check(status)      — wraps cuDNN status
//   cublas_check(status)     — wraps cuBLAS status
//
// Structs:
//   DeviceInfo               — device properties summary
//
// Functions:
//   get_device_info(id)      — returns DeviceInfo for given device
//   print_device_info(log, info) — logs device info at INFO level
// =============================================================================

#include <sstream>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>

#include "logger.hpp"

namespace cuda_tutorials {

// ---------------------------------------------------------------------------
// CUDA runtime error check
// ---------------------------------------------------------------------------
inline void cuda_check_impl(cudaError_t err,
                             const char* file,
                             int line,
                             const char* call) {
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "CUDA error at " << file << ":" << line
            << " — " << call
            << " → " << cudaGetErrorName(err)
            << ": " << cudaGetErrorString(err);
        throw std::runtime_error(oss.str());
    }
}

#define CUDA_CHECK(call) \
    ::cuda_tutorials::cuda_check_impl((call), __FILE__, __LINE__, #call)

// After kernel launches (checks cudaGetLastError + device sync)
#define CUDA_KERNEL_CHECK() \
    ::cuda_tutorials::cuda_check_impl( \
        cudaGetLastError(), __FILE__, __LINE__, "kernel launch"); \
    ::cuda_tutorials::cuda_check_impl( \
        cudaDeviceSynchronize(), __FILE__, __LINE__, "cudaDeviceSynchronize")

// ---------------------------------------------------------------------------
// cuDNN status check
// ---------------------------------------------------------------------------
inline void cudnn_check_impl(cudnnStatus_t status,
                              const char* file,
                              int line,
                              const char* call) {
    if (status != CUDNN_STATUS_SUCCESS) {
        std::ostringstream oss;
        oss << "cuDNN error at " << file << ":" << line
            << " — " << call
            << " → " << cudnnGetErrorString(status);
        throw std::runtime_error(oss.str());
    }
}

#define CUDNN_CHECK(call) \
    ::cuda_tutorials::cudnn_check_impl((call), __FILE__, __LINE__, #call)

// ---------------------------------------------------------------------------
// cuBLAS status check
// ---------------------------------------------------------------------------
inline const char* cublas_status_string(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS:          return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:  return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:     return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:    return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:    return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:    return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:   return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED:    return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR:    return "CUBLAS_STATUS_LICENSE_ERROR";
        default:                             return "CUBLAS_STATUS_UNKNOWN";
    }
}

inline void cublas_check_impl(cublasStatus_t status,
                               const char* file,
                               int line,
                               const char* call) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::ostringstream oss;
        oss << "cuBLAS error at " << file << ":" << line
            << " — " << call
            << " → " << cublas_status_string(status);
        throw std::runtime_error(oss.str());
    }
}

#define CUBLAS_CHECK(call) \
    ::cuda_tutorials::cublas_check_impl((call), __FILE__, __LINE__, #call)

// ---------------------------------------------------------------------------
// Device information
// ---------------------------------------------------------------------------
struct DeviceInfo {
    std::string name;
    int         compute_capability_major;
    int         compute_capability_minor;
    int         sm_count;
    size_t      global_mem_bytes;
    size_t      shared_mem_per_block;
    int         warp_size;
    int         max_threads_per_block;
    int         max_threads_per_sm;
    int         clock_rate_khz;
    int         memory_clock_rate_khz;
    int         memory_bus_width_bits;
    size_t      l2_cache_size;
};

inline DeviceInfo get_device_info(int device_id) {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

    DeviceInfo info{};
    info.name                    = prop.name;
    info.compute_capability_major = prop.major;
    info.compute_capability_minor = prop.minor;
    info.sm_count                = prop.multiProcessorCount;
    info.global_mem_bytes        = prop.totalGlobalMem;
    info.shared_mem_per_block    = prop.sharedMemPerBlock;
    info.warp_size               = prop.warpSize;
    info.max_threads_per_block   = prop.maxThreadsPerBlock;
    info.max_threads_per_sm      = prop.maxThreadsPerMultiProcessor;
    info.clock_rate_khz          = prop.clockRate;
    info.memory_clock_rate_khz   = prop.memoryClockRate;
    info.memory_bus_width_bits   = prop.memoryBusWidth;
    info.l2_cache_size           = static_cast<size_t>(prop.l2CacheSize);
    return info;
}

inline void print_device_info(Logger& logger, const DeviceInfo& info) {
    auto mb  = [](size_t b) { return static_cast<double>(b) / (1024.0 * 1024.0); };
    auto gb  = [](size_t b) { return static_cast<double>(b) / (1024.0 * 1024.0 * 1024.0); };

    // Peak memory bandwidth = 2 * mem_clock_rate(Hz) * bus_width(bytes)
    double peak_bw_gbs = 2.0 * (info.memory_clock_rate_khz * 1e3) *
                         (info.memory_bus_width_bits / 8.0) / 1e9;

    std::ostringstream oss;
    oss << "=== Device Info ===\n"
        << "  Name                : " << info.name << "\n"
        << "  Compute capability  : sm_"
            << info.compute_capability_major
            << info.compute_capability_minor << "\n"
        << "  SM count            : " << info.sm_count << "\n"
        << "  Global memory       : " << std::fixed << std::setprecision(2)
            << gb(info.global_mem_bytes) << " GB\n"
        << "  Shared mem/block    : " << mb(info.shared_mem_per_block) * 1024.0
            << " KB\n"
        << "  Warp size           : " << info.warp_size << "\n"
        << "  Max threads/block   : " << info.max_threads_per_block << "\n"
        << "  Max threads/SM      : " << info.max_threads_per_sm << "\n"
        << "  Core clock          : " << info.clock_rate_khz / 1000 << " MHz\n"
        << "  Memory clock        : " << info.memory_clock_rate_khz / 1000 << " MHz\n"
        << "  Memory bus width    : " << info.memory_bus_width_bits << " bits\n"
        << "  L2 cache size       : " << mb(info.l2_cache_size) << " MB\n"
        << "  Peak mem bandwidth  : " << peak_bw_gbs << " GB/s\n"
        << "===================";
    logger.log_info(oss.str());
}

// ---------------------------------------------------------------------------
// CUDA event RAII wrapper — simplifies timing
// ---------------------------------------------------------------------------
struct CudaEvent {
    cudaEvent_t event{};
    CudaEvent()  { CUDA_CHECK(cudaEventCreate(&event)); }
    ~CudaEvent() { cudaEventDestroy(event); }
    void record(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(event, stream));
    }
    // Returns elapsed milliseconds between this event and 'start'
    float elapsed_ms(const CudaEvent& start) const {
        float ms = 0.f;
        CUDA_CHECK(cudaEventSynchronize(event));
        CUDA_CHECK(cudaEventElapsedTime(&ms, start.event, event));
        return ms;
    }
};

} // namespace cuda_tutorials
