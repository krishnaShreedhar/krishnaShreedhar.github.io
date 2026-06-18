FROM python:3.11-slim

LABEL maintainer="kafka-event-driven"
LABEL description="Kafka and event-driven architecture demonstration (mock broker)"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    librdkafka-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY docker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY . .

# Add src to Python path so modules can be imported without package installation
ENV PYTHONPATH=/app/src

# Create necessary directories
RUN mkdir -p logs outputs

# Default: run the streaming analytics demo
CMD ["python", "-c", "from streaming_analytics.event_processor import main; main()"]
