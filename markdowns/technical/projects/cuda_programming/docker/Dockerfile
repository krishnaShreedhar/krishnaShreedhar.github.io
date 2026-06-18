FROM nvidia/cuda:13.0.1-devel-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# ---------------------------------------------------------------------------
# System build tools
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    git \
    wget \
    curl \
    ca-certificates \
    libgtest-dev \
    # yaml-cpp development headers
    libyaml-cpp-dev \
    # Python for notebooks
    python3 \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# CUTLASS — clone from NVIDIA GitHub (tag for CUDA 12/13 compatibility)
# ---------------------------------------------------------------------------
ARG CUTLASS_TAG=v3.5.1
RUN git clone --depth 1 --branch ${CUTLASS_TAG} \
    https://github.com/NVIDIA/cutlass.git /workspace/cutlass && \
    echo "CUTLASS ${CUTLASS_TAG} cloned to /workspace/cutlass"

# CUTLASS is header-only — no build step needed for the library itself.

# ---------------------------------------------------------------------------
# NVTX3 — already in CUDA toolkit headers at /usr/local/cuda/include/nvtx3/
# Verify presence
# ---------------------------------------------------------------------------
RUN ls /usr/local/cuda/include/nvtx3/ && \
    echo "NVTX3 headers present"

# ---------------------------------------------------------------------------
# cuDNN — installed via CUDA base image or via apt
# Check if available
# ---------------------------------------------------------------------------
RUN ldconfig && ldconfig -p | grep cudnn || \
    echo "WARNING: cuDNN not found in ldconfig. Install separately if needed."

# ---------------------------------------------------------------------------
# Python dependencies for notebooks
# ---------------------------------------------------------------------------
RUN pip3 install --no-cache-dir \
    jupyterlab \
    numpy \
    matplotlib \
    pandas

# ---------------------------------------------------------------------------
# Working directory
# ---------------------------------------------------------------------------
WORKDIR /workspace

# Copy the project source into the container
COPY . /workspace/cuda_programming/

# ---------------------------------------------------------------------------
# Build all tutorials
# ---------------------------------------------------------------------------
RUN mkdir -p /workspace/cuda_programming/build && \
    cmake \
        -S /workspace/cuda_programming \
        -B /workspace/cuda_programming/build \
        -G Ninja \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DCUTLASS_DIR=/workspace/cutlass \
        -DPROJECT_ROOT=/workspace/cuda_programming \
    && cmake --build /workspace/cuda_programming/build --target all_tutorials -j$(nproc)

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
ENV PATH="/workspace/cuda_programming/build/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Logs directory inside container maps to host via docker-compose volume
RUN mkdir -p /workspace/cuda_programming/logs

CMD ["/bin/bash"]
