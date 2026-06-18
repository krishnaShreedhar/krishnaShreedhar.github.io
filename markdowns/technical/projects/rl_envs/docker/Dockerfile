FROM nvidia/cuda:13.0.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-venv python3-pip \
    git wget curl ca-certificates \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# uv for fast package management
RUN pip3 install uv

WORKDIR /workspace/rl_envs

COPY pyproject.toml ./
RUN uv pip install --system -e .

COPY . /workspace/rl_envs/

# Multi-GPU env vars
ENV CUDA_VISIBLE_DEVICES=0,1
ENV PYTHONPATH=/workspace/rl_envs/src

RUN mkdir -p /workspace/rl_envs/logs /workspace/rl_envs/checkpoints

CMD ["/bin/bash"]
