FROM python:3.11-slim

LABEL maintainer="Shreedhar Kodate"
LABEL description="Python Engineering Patterns demo environment"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first (layer cache optimisation)
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY ../src ./src
COPY ../config.yaml ./config.yaml
COPY ../pyproject.toml ./pyproject.toml

# Create runtime directories
RUN mkdir -p logs

# Set Python path so modules can import siblings
ENV PYTHONPATH="/app/src/collections_itertools:/app/src/logging_patterns:/app/src/concurrency:/app/src/design_patterns:/app/src/production_template"
ENV PYTHONUNBUFFERED=1

# Default: run the production service demo
CMD ["python", "src/production_template/service.py"]
