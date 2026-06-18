# Python Engineering Patterns

A curated collection of minimal, runnable examples illustrating production-grade Python
engineering patterns. Every module is self-contained, reads its configuration from
`config.yaml`, and writes structured logs under `logs/`.

## Patterns Covered

| Area | Concepts |
|------|----------|
| **Collections / itertools** | `defaultdict`, `Counter`, `deque`, `namedtuple`, `ChainMap`, `itertools` combinatorics, `functools.lru_cache`, `singledispatch`, generators |
| **Logging** | JSON formatter, rotating file handler, `LoggerAdapter` context, `QueueHandler` async logging, `dictConfig` |
| **Concurrency** | `ThreadPoolExecutor`, thread-safe `Queue` / `Lock` / `Semaphore` / `Event` / `Barrier`, `ProcessPoolExecutor`, shared memory, `asyncio` producer-consumer, `TaskGroup`, timeout/cancellation |
| **Design Patterns** | Singleton (metaclass, thread-safe), Observer / EventBus, Strategy, Builder, Retry decorator (exp backoff + jitter), context managers, TTL cache |
| **Production Template** | Graceful shutdown, batch processing with GC control, `stream_jsonl`, in-process metrics (counter + histogram), exception hierarchy, global exception handler |

## Repository Layout

```
python_engg_tmplates/
├── config.yaml                          # All constants and hyperparameters
├── pyproject.toml
├── README.md
├── logs/                                # Runtime log files (auto-created)
├── docs/
│   ├── concepts.md                      # Pattern explanations + mermaid diagrams
│   └── flow_diagrams.md                 # Execution flow diagrams
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── requirements.txt
└── src/
    ├── collections_itertools/
    │   ├── collections_demo.py
    │   ├── itertools_demo.py
    │   └── functools_demo.py
    ├── logging_patterns/
    │   ├── json_logger.py
    │   ├── async_logger.py
    │   └── logger_factory.py
    ├── concurrency/
    │   ├── threading_patterns.py
    │   ├── multiprocessing_patterns.py
    │   └── async_patterns.py
    ├── design_patterns/
    │   ├── creational_patterns.py
    │   ├── behavioral_patterns.py
    │   └── structural_patterns.py
    ├── production_template/
    │   ├── service.py
    │   ├── metrics.py
    │   └── exception_hierarchy.py
    └── notebooks/
        └── python_patterns_demo.ipynb
```

## Quick Start

### Local

```bash
# Install dependencies
pip install -e ".[dev]"

# Create logs directory
mkdir -p logs

# Run any module directly (each has __main__ block)
python src/collections_itertools/collections_demo.py
python src/logging_patterns/json_logger.py
python src/concurrency/threading_patterns.py
python src/design_patterns/creational_patterns.py
python src/production_template/service.py

# Launch notebook
jupyter notebook src/notebooks/python_patterns_demo.ipynb
```

### Docker

```bash
cd docker
docker-compose up --build
```

## Configuration

All tuneable knobs live in `config.yaml`. No CLI flags are used anywhere in this project.
Edit `config.yaml` and re-run any module to change behaviour.

## Logging

Every module writes to `logs/python_engg.log` (rotating, JSON-structured) and also
emits to stdout. The log level is controlled by `logging.level` in `config.yaml`.
