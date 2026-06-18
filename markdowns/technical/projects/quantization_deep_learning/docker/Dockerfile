FROM python:3.11-slim

LABEL maintainer="Shreedhar Kodate"
LABEL description="Deep Learning Quantization: PTQ, QAT, calibration, and mixed-precision"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside container
WORKDIR /workspace

# Install Python dependencies first (leverages Docker layer caching)
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /workspace/requirements.txt

# Copy project source
COPY ../ /workspace/

# Create required directories
RUN mkdir -p /workspace/logs \
             /workspace/outputs/models \
             /workspace/outputs/models/weight_analysis \
             /workspace/outputs/models/fake_quant \
             /workspace/outputs/models/ste_demo \
             /workspace/outputs/models/calibration \
             /workspace/outputs/models/benchmark \
             /workspace/outputs/models/error_metrics \
             /workspace/outputs/models/sensitivity

# Set PYTHONPATH so imports resolve correctly
ENV PYTHONPATH=/workspace

# Expose Jupyter Lab port
EXPOSE 8888

# Default command: start Jupyter Lab
CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--allow-root", \
     "--notebook-dir=/workspace", \
     "--ServerApp.token=''", \
     "--ServerApp.password=''"]
