cmake_minimum_required(VERSION 3.24)
project(cuda_tutorials LANGUAGES CXX CUDA)

# ===========================================================================
# C++ / CUDA standards
# ===========================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ===========================================================================
# GPU target: H200 = sm_90
# ===========================================================================
set(CMAKE_CUDA_ARCHITECTURES 90)

# Generate device debug info (line-level PTX mapping for Nsight)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -lineinfo")

# ===========================================================================
# Project root (injected into each tutorial as a compile definition)
# ===========================================================================
set(PROJECT_ROOT "${CMAKE_SOURCE_DIR}" CACHE STRING "Project root directory")

# ===========================================================================
# Find required packages
# ===========================================================================
find_package(CUDAToolkit REQUIRED)
find_package(yaml-cpp REQUIRED)

# cuDNN — search in common installation locations
find_path(CUDNN_INCLUDE_DIR cudnn.h
    HINTS ${CUDAToolkit_INCLUDE_DIRS} /usr/include /usr/local/cuda/include
    PATH_SUFFIXES ""
)
find_library(CUDNN_LIBRARY cudnn
    HINTS ${CUDAToolkit_LIBRARY_DIR} /usr/lib /usr/lib64 /usr/local/cuda/lib64
)
if(NOT CUDNN_INCLUDE_DIR OR NOT CUDNN_LIBRARY)
    message(FATAL_ERROR "cuDNN not found. Install cudnn development headers.")
endif()
message(STATUS "cuDNN: ${CUDNN_INCLUDE_DIR}  ${CUDNN_LIBRARY}")

# CUTLASS — expected at /workspace/cutlass or provided via -DCUTLASS_DIR
if(NOT DEFINED CUTLASS_DIR)
    set(CUTLASS_DIR "/workspace/cutlass")
endif()
if(NOT EXISTS "${CUTLASS_DIR}/include/cutlass/cutlass.h")
    message(WARNING "CUTLASS not found at ${CUTLASS_DIR}. Tutorial 17 may not compile.")
endif()

# NVTX3 — part of CUDA toolkit
find_path(NVTX3_INCLUDE_DIR nvtx3/nvToolsExt.h
    HINTS ${CUDAToolkit_INCLUDE_DIRS}
    NO_DEFAULT_PATH
)

# ===========================================================================
# Common include interface
# ===========================================================================
add_library(tutorial_common INTERFACE)
target_include_directories(tutorial_common INTERFACE
    ${CMAKE_SOURCE_DIR}/src/common
    ${CUDNN_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(tutorial_common INTERFACE
    CUDA::cudart
    CUDA::cublas
    yaml-cpp
    ${CUDNN_LIBRARY}
)
target_compile_definitions(tutorial_common INTERFACE
    PROJECT_ROOT="${PROJECT_ROOT}"
)

# ===========================================================================
# Helper macro: add_tutorial(NUMBER NAME [EXTRA_LIBS])
# ===========================================================================
macro(add_tutorial NUMBER NAME)
    set(_extra_libs ${ARGN})
    set(_target "tutorial_${NUMBER}_${NAME}")
    set(_src "${CMAKE_SOURCE_DIR}/src/tutorials/${NUMBER}_${NAME}/main.cu")

    if(EXISTS "${_src}")
        add_executable(${_target} "${_src}")
        set_target_properties(${_target} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        )
        target_link_libraries(${_target} PRIVATE tutorial_common ${_extra_libs})
        target_compile_options(${_target} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:
                -arch=sm_90
                --ptxas-options=-v
                -Xcompiler=-Wall
            >
        )
        message(STATUS "Tutorial target: ${_target}")
    else()
        message(WARNING "Source not found, skipping: ${_src}")
    endif()
endmacro()

# ===========================================================================
# Tutorial targets
# ===========================================================================
add_tutorial(01 threads_blocks_grids)
add_tutorial(02 warps_simt)
add_tutorial(03 cuda_memory_types)
add_tutorial(04 memory_coalescing)
add_tutorial(05 shared_memory_bank_conflicts)
add_tutorial(06 thread_synchronization)
add_tutorial(07 occupancy)
add_tutorial(08 cuda_streams)
add_tutorial(09 cuda_events_timing)
add_tutorial(10 unified_memory)
add_tutorial(11 pinned_memory)
add_tutorial(12 atomic_operations)
add_tutorial(13 parallel_reduction)
add_tutorial(14 warp_shuffle)
add_tutorial(15 tiled_matmul)
add_tutorial(16 tensor_cores_wmma)

# Tutorial 17: CUTLASS requires its include directory
if(EXISTS "${CUTLASS_DIR}/include/cutlass/cutlass.h")
    add_executable(tutorial_17_cutlass
        ${CMAKE_SOURCE_DIR}/src/tutorials/17_cutlass/main.cu)
    set_target_properties(tutorial_17_cutlass PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    target_include_directories(tutorial_17_cutlass PRIVATE
        ${CUTLASS_DIR}/include
        ${CUTLASS_DIR}/tools/util/include
    )
    target_link_libraries(tutorial_17_cutlass PRIVATE tutorial_common)
    target_compile_options(tutorial_17_cutlass PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_90 --ptxas-options=-v>
    )
    message(STATUS "Tutorial target: tutorial_17_cutlass (with CUTLASS)")
else()
    message(WARNING "Skipping tutorial 17 (CUTLASS not found at ${CUTLASS_DIR})")
endif()

add_tutorial(18 ptx_sass)

# Tutorial 19: Nsight / NVTX
if(NVTX3_INCLUDE_DIR)
    add_executable(tutorial_19_nsight_profiling
        ${CMAKE_SOURCE_DIR}/src/tutorials/19_nsight_profiling/main.cu)
    set_target_properties(tutorial_19_nsight_profiling PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    target_include_directories(tutorial_19_nsight_profiling PRIVATE
        ${NVTX3_INCLUDE_DIR}
    )
    target_link_libraries(tutorial_19_nsight_profiling PRIVATE tutorial_common
        CUDA::nvToolsExt)
    target_compile_options(tutorial_19_nsight_profiling PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_90>
    )
    message(STATUS "Tutorial target: tutorial_19_nsight_profiling")
else()
    message(WARNING "NVTX3 headers not found, skipping tutorial 19")
endif()

add_tutorial(20 cublas)
add_tutorial(21 cudnn_descriptors)
add_tutorial(22 cudnn_algorithm_selection)
add_tutorial(23 cudnn_workspace)
add_tutorial(24 cudnn_fused_ops)
add_tutorial(25 cuda_graphs)

# ===========================================================================
# All-tutorials convenience target
# ===========================================================================
add_custom_target(all_tutorials
    DEPENDS
        tutorial_01_threads_blocks_grids
        tutorial_02_warps_simt
        tutorial_03_cuda_memory_types
        tutorial_04_memory_coalescing
        tutorial_05_shared_memory_bank_conflicts
        tutorial_06_thread_synchronization
        tutorial_07_occupancy
        tutorial_08_cuda_streams
        tutorial_09_cuda_events_timing
        tutorial_10_unified_memory
        tutorial_11_pinned_memory
        tutorial_12_atomic_operations
        tutorial_13_parallel_reduction
        tutorial_14_warp_shuffle
        tutorial_15_tiled_matmul
        tutorial_16_tensor_cores_wmma
        tutorial_18_ptx_sass
        tutorial_20_cublas
        tutorial_21_cudnn_descriptors
        tutorial_22_cudnn_algorithm_selection
        tutorial_23_cudnn_workspace
        tutorial_24_cudnn_fused_ops
        tutorial_25_cuda_graphs
    COMMENT "Building all tutorials (17 and 19 depend on CUTLASS/NVTX availability)"
)

message(STATUS "")
message(STATUS "=== Build configuration ===")
message(STATUS "  CUDA architectures : ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  C++ standard       : ${CMAKE_CXX_STANDARD}")
message(STATUS "  Project root       : ${PROJECT_ROOT}")
message(STATUS "  CUTLASS dir        : ${CUTLASS_DIR}")
message(STATUS "  Build dir          : ${CMAKE_BINARY_DIR}")
message(STATUS "===========================")
