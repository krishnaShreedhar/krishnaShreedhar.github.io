FROM python:3.11-slim

LABEL maintainer="probability-statistics-project"
LABEL description="Probability & Statistics educational project with DuckDB SQL analytics"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY .. /app/

# Create required directories
RUN mkdir -p /app/logs /app/outputs

# Set PYTHONPATH so imports from src/ work
ENV PYTHONPATH="/app/src:/app"

# Default command: run the SQL statistics demo
CMD ["python", "src/sql_analytics/sql_statistics.py"]
