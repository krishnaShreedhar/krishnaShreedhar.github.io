# Base: Python 3.11 slim — add OpenJDK 17 for PySpark
FROM python:3.11-slim

LABEL maintainer="pandas-pyspark-demo"
LABEL description="Pandas & PySpark learning environment"

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        openjdk-17-jre-headless \
        curl \
        wget \
        procps \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Java environment for PySpark ─────────────────────────────────────────────
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# ── Python dependencies ───────────────────────────────────────────────────────
WORKDIR /app
COPY requirements.txt /app/docker/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/docker/requirements.txt

# ── Copy project ──────────────────────────────────────────────────────────────
COPY .. /app/

# ── Create required runtime directories ───────────────────────────────────────
RUN mkdir -p /app/logs /app/outputs

# ── Environment variables ─────────────────────────────────────────────────────
ENV PYTHONPATH="/app/src:${PYTHONPATH}"
ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3

# ── Default: launch JupyterLab ────────────────────────────────────────────────
EXPOSE 8888
CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--allow-root", \
     "--notebook-dir=/app/src/notebooks", \
     "--ServerApp.token=''", \
     "--ServerApp.password=''"]
