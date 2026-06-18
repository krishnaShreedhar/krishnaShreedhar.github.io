FROM python:3.11-slim

WORKDIR /app

# Install system dependencies needed by Ray and PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create the logs directory expected by the logging config
RUN mkdir -p /app/logs

ENV PYTHONPATH=/app/src

CMD ["python", "src/core_primitives/remote_functions.py"]
