# Kafka Event-Driven Architecture — Demonstration Project

A self-contained, runnable illustration of Apache Kafka concepts and
event-driven design patterns.  Every module uses an in-memory mock broker
(`MockKafkaBroker`) so no external services are required.

---

## Kafka Concepts Covered

| Concept | Where demonstrated |
|---------|-------------------|
| Topics, partitions, offsets | `src/kafka_core/mock_kafka.py` |
| Producer with delivery callbacks | `src/kafka_core/producer.py` |
| Consumer group offset management | `src/kafka_core/consumer.py` |
| Manual commit / at-least-once delivery | `src/kafka_core/consumer.py` |
| Consumer lag computation | `src/kafka_core/mock_kafka.py` |
| Topic administration | `src/kafka_core/topic_manager.py` |
| Dead Letter Queue (DLQ) | `src/event_patterns/dlq_handler.py` |
| In-Sync Replicas (ISR) concept | `docs/architecture.md` |
| Idempotent producers | `config.yaml` (`enable_idempotence: true`) |

## Event-Driven Patterns

| Pattern | Module |
|---------|--------|
| Event Sourcing | `src/event_patterns/event_sourcing.py` |
| Saga (with compensation) | `src/event_patterns/saga_orchestrator.py` |
| Transactional Outbox | `src/event_patterns/outbox_pattern.py` |
| Dead Letter Queue + Exponential Backoff | `src/event_patterns/dlq_handler.py` |

## ML Pipeline Components

| Component | Module |
|-----------|--------|
| Real-time feature computation | `src/ml_pipeline/feature_pipeline.py` |
| Online model inference | `src/ml_pipeline/model_inference.py` |
| Hash-based A/B routing | `src/ml_pipeline/ab_routing.py` |
| PSI drift detection | `src/ml_pipeline/monitoring.py` |

## Streaming Analytics

| Component | Module |
|-----------|--------|
| Tumbling windows | `src/streaming_analytics/windowed_aggregation.py` |
| Session windows | `src/streaming_analytics/windowed_aggregation.py` |
| Welford online statistics | `src/streaming_analytics/event_processor.py` |
| Consumer lag tracking | `src/streaming_analytics/event_processor.py` |

---

## Project Structure

```
kafka_event_driven/
├── config.yaml                        # All constants and hyperparameters
├── pyproject.toml
├── README.md
├── src/
│   ├── kafka_core/
│   │   ├── mock_kafka.py              # In-memory broker (topics/partitions/offsets)
│   │   ├── producer.py                # Serialisation, delivery callbacks
│   │   ├── consumer.py                # Poll loop, manual commit, DLQ routing
│   │   └── topic_manager.py           # Topic lifecycle management
│   ├── event_patterns/
│   │   ├── event_sourcing.py          # Append-only log, state reconstruction
│   │   ├── saga_orchestrator.py       # Choreography Saga + compensation
│   │   ├── outbox_pattern.py          # Transactional outbox relay
│   │   └── dlq_handler.py             # DLQ with exponential backoff
│   ├── ml_pipeline/
│   │   ├── feature_pipeline.py        # Kafka → rolling features → feature store
│   │   ├── model_inference.py         # Features → model → predictions topic
│   │   ├── ab_routing.py              # Hash-based A/B model routing
│   │   └── monitoring.py              # PSI drift detection on predictions
│   ├── streaming_analytics/
│   │   ├── windowed_aggregation.py    # Tumbling and session windows
│   │   └── event_processor.py        # Welford stats, consumer lag tracking
│   └── notebooks/                     # Jupyter notebooks (empty placeholder)
├── docs/
│   ├── architecture.md                # Mermaid diagrams, concept explanations
│   └── flow_diagrams.md               # Detailed flow diagrams
└── docker/
    ├── Dockerfile
    ├── docker-compose.yml             # Zookeeper + Kafka + app services
    └── requirements.txt
```

---

## Running with Mock Broker (default, no dependencies)

The mock broker is enabled by default (`kafka.use_mock: true` in `config.yaml`).
Simply add the `src/` directory to `PYTHONPATH` and run any module directly.

```bash
cd projects/kafka_event_driven
export PYTHONPATH=$PWD/src

# Run individual module demos
python src/kafka_core/mock_kafka.py
python src/kafka_core/producer.py
python src/kafka_core/consumer.py
python src/kafka_core/topic_manager.py

python src/event_patterns/event_sourcing.py
python src/event_patterns/saga_orchestrator.py
python src/event_patterns/outbox_pattern.py
python src/event_patterns/dlq_handler.py

python src/ml_pipeline/feature_pipeline.py
python src/ml_pipeline/model_inference.py
python src/ml_pipeline/ab_routing.py
python src/ml_pipeline/monitoring.py

python src/streaming_analytics/windowed_aggregation.py
python src/streaming_analytics/event_processor.py
```

Logs are written to `logs/kafka_event_driven.log` (JSON format, rotating).

---

## Running with Real Kafka

1. Set `kafka.use_mock: false` in `config.yaml`.
2. Start Kafka with Docker Compose:

```bash
cd projects/kafka_event_driven
docker compose -f docker/docker-compose.yml up -d zookeeper kafka
```

3. Wait for Kafka to be healthy, then run the app:

```bash
export PYTHONPATH=$PWD/src
python src/streaming_analytics/event_processor.py
```

Or run the full Docker stack:

```bash
docker compose -f docker/docker-compose.yml up --build
```

---

## Configuration

All tuneable parameters live in `config.yaml`.  Key settings:

```yaml
kafka:
  use_mock: true          # Switch to false for real Kafka

ml_pipeline:
  model_v2_traffic_pct: 20    # % of users routed to challenger model

event_patterns:
  dlq_max_retries: 3          # Max retries before quarantine
  dlq_backoff_base_s: 2.0     # Exponential backoff base (2^retry_count seconds)

streaming:
  lag_alert_threshold: 1000   # Consumer lag count triggering warning
  num_events_to_generate: 200 # Events produced in streaming demo
```

---

## Logging

All modules write structured JSON logs to `logs/kafka_event_driven.log` using
a `RotatingFileHandler` (100 MB per file, 5 backup files).  Console output
mirrors the file.

Log record format:
```json
{
  "timestamp": "2026-06-04T12:00:00",
  "level": "INFO",
  "logger": "kafka_core.mock_kafka",
  "message": "Topic created: name='user_events', partitions=4"
}
```

Log level is configurable via `logging.level` in `config.yaml` (DEBUG, INFO, WARNING, ERROR).

---

## Design Principles

- **SOLID**: Each class has one responsibility; dependencies are injected via constructor.
- **No fallbacks**: If a topic does not exist, an exception is raised rather than silently creating it.
- **YAML-driven configuration**: No hardcoded constants in source files.
- **Extensive logging**: Every state change, produce, consume, commit, and error is logged.
- **Standalone**: Zero imports from other projects in this repository.
