FROM python:3.11-slim

# ---- System dependencies ----
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        wget \
        build-essential \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/*

# ---- Working directory ----
WORKDIR /workspace

# ---- Install Python dependencies ----
COPY requirements.txt ./
RUN pip install --upgrade pip --no-cache-dir \
 && pip install --no-cache-dir -r requirements.txt

# ---- Copy project source ----
COPY .. .

# ---- Create required runtime directories ----
RUN mkdir -p logs outputs/models

# ---- Environment variables ----
ENV PYTHONPATH=/workspace
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ---- Expose Jupyter port ----
EXPOSE 8888

# ---- Default command: start Jupyter ----
CMD ["jupyter", "notebook", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--allow-root", \
     "--notebook-dir=/workspace", \
     "--NotebookApp.token=''", \
     "--NotebookApp.password=''"]
