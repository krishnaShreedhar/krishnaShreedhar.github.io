FROM nvidia/cuda:13.2.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3-pip \
    curl git build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /workspace

COPY pyproject.toml ./
RUN uv pip install -e ".[dev]" 2>/dev/null || uv pip install \
    "torch>=2.4.0" \
    "torchvision>=0.19.0" \
    "numpy>=1.26.0" \
    "pyyaml>=6.0.1" \
    "psutil>=5.9.0" \
    "jupyter>=1.0.0" \
    "matplotlib>=3.8.0" \
    "nvidia-ml-py>=12.535.77"

COPY . .

RUN mkdir -p logs

CMD ["python", "-m", "src.transformer_training.main"]
