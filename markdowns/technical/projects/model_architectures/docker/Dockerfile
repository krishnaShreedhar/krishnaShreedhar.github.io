FROM nvidia/cuda:13.0.1-devel-ubuntu22.04

# ---- System dependencies ----
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3-pip \
        git \
        curl \
        wget \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
 && update-alternatives --set python3 /usr/bin/python3.11

# ---- Install uv (fast Python package manager) ----
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /workspace

# ---- Install Python dependencies via uv ----
COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen || uv sync

# ---- Copy project source ----
COPY . .

# ---- Environment variables ----
ENV PYTHONPATH=/workspace
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Use nccl for multi-GPU communication
ENV NCCL_DEBUG=INFO
ENV NCCL_IB_DISABLE=0

# ---- Default command: show help ----
CMD ["python3", "-c", "import src; print('SmallVLM environment ready.')"]
