# Python Engineering Patterns – Concept Reference

This document explains when and why to use each pattern, illustrated with
Mermaid diagrams.

---

## 1. Python Concurrency Models

Python offers three concurrency primitives. The right choice depends on
whether your bottleneck is I/O, CPU, or both.

```mermaid
flowchart TD
    Start([Bottleneck type?])

    Start -->|"I/O-bound\n(network, disk, DB)"| IO[asyncio / Threading]
    Start -->|"CPU-bound\n(compute, numpy)"| CPU[Multiprocessing]
    Start -->|"Mixed"| Mixed[Multiprocessing + asyncio workers]

    IO --> AsyncIO["asyncio\n• Single thread\n• Cooperative multitasking\n• Best for 1000s of tasks\n• Zero OS overhead"]
    IO --> Threads["threading\n• OS threads\n• GIL released during I/O\n• Simpler mental model\n• Best for 10-100 tasks"]

    CPU --> MP["multiprocessing\n• Separate processes\n• True parallelism\n• Bypasses GIL\n• Higher memory cost"]

    AsyncIO --> GIL["GIL (Global Interpreter Lock)\nOne Python bytecode at a time\nReleased during I/O syscalls\nNot released for pure Python compute"]
    Threads --> GIL
    MP --> NoGIL["No GIL constraint\n(each process has its own GIL)"]
```

**When to use what:**

| Workload | Best tool | Why |
|----------|-----------|-----|
| HTTP requests, DB queries | `asyncio` | Cooperative, zero thread overhead |
| File I/O, subprocess | `threading` or `asyncio` | GIL released during I/O |
| Numerical computation | `multiprocessing` | True CPU parallelism |
| ML inference batch | `ProcessPoolExecutor` | Load model once per worker |
| Streaming large files | `async for` + generator | Constant memory |

---

## 2. Observer / EventBus Pattern

The Observer pattern decouples event producers from consumers. The EventBus
variant allows many-to-many subscriptions without direct object references.

```mermaid
classDiagram
    class EventBus {
        -handlers: dict[str, list[ref]]
        +subscribe(event, handler)
        +unsubscribe(event, handler)
        +publish(event, data) int
    }

    class MetricsCollector {
        -counts: defaultdict[str, int]
        +on_event(event, data)
        +report() dict
    }

    class AuditLogger {
        -entries: list[dict]
        +on_event(event, data)
        +entries() list
    }

    class UserService {
        -bus: EventBus
        +login(user_id)
        +logout(user_id)
    }

    class OrderService {
        -bus: EventBus
        +place_order(order)
    }

    EventBus "1" o-- "0..*" MetricsCollector : notifies
    EventBus "1" o-- "0..*" AuditLogger : notifies
    UserService --> EventBus : publishes
    OrderService --> EventBus : publishes
```

**Sequence of a published event:**

```mermaid
sequenceDiagram
    participant Producer as UserService
    participant Bus as EventBus
    participant M as MetricsCollector
    participant A as AuditLogger

    Producer->>Bus: publish("user.login", {user_id: "u-001"})
    Bus->>M: on_event("user.login", data)
    M-->>Bus: ok
    Bus->>A: on_event("user.login", data)
    A-->>Bus: ok
    Bus-->>Producer: notified=2
```

**Key design choices:**
- Handlers are stored as `weakref.WeakMethod` so dead subscribers are
  automatically pruned (no memory leaks).
- Failed handlers log an error but do not prevent other handlers from running.
- The bus is synchronous; for async dispatch, wrap `publish` in `asyncio.create_task`.

---

## 3. Strategy Pattern Class Hierarchy

Strategy separates *what algorithm to use* from *how to invoke it*, making
algorithms interchangeable at runtime.

```mermaid
classDiagram
    class SamplingStrategy {
        <<abstract>>
        +sample(dataset, n) list
    }

    class RandomSampling {
        -rng: Random
        +sample(dataset, n) list
    }

    class StratifiedSampling {
        -label_key: str
        -rng: Random
        +sample(dataset, n) list
    }

    class WeightedSampling {
        -weight_key: str
        +sample(dataset, n) list
    }

    class DataSampler {
        -strategy: SamplingStrategy
        +set_strategy(strategy)
        +draw(dataset, n) list
    }

    SamplingStrategy <|-- RandomSampling
    SamplingStrategy <|-- StratifiedSampling
    SamplingStrategy <|-- WeightedSampling
    DataSampler o-- SamplingStrategy : uses
```

**When to use Strategy:**
- Multiple algorithms for the same task (sorting, sampling, pricing)
- Algorithm selection needs to change at runtime
- Avoiding large `if/elif` chains in business logic
- Testing individual algorithms in isolation

---

## 4. Retry Decorator – Decision Flow

```mermaid
flowchart TD
    Call([Call decorated function])
    Try[Execute function]
    Success{Succeeded?}
    Retryable{Exception is\nretryable?}
    MaxReached{Attempts ==\nmax_attempts?}
    CalcWait[Calculate wait:\nbackoff * 2^attempt]
    Jitter{jitter=True?}
    AddJitter[Add random 0..wait]
    Sleep[sleep wait seconds]
    Raise([Re-raise exception])
    Return([Return result])

    Call --> Try
    Try --> Success
    Success -->|Yes| Return
    Success -->|No| Retryable
    Retryable -->|No| Raise
    Retryable -->|Yes| MaxReached
    MaxReached -->|Yes| Raise
    MaxReached -->|No| CalcWait
    CalcWait --> Jitter
    Jitter -->|Yes| AddJitter --> Sleep
    Jitter -->|No| Sleep
    Sleep --> Try
```

**Why jitter?** Without jitter, all N clients that fail simultaneously will
retry at the same time (thundering herd). Uniform jitter `[0, wait]` spreads
retries across time, reducing peak load by ~50%.

---

## 5. Production Service Architecture

```mermaid
flowchart TB
    subgraph "startup"
        A[Load config.yaml] --> B[Setup logging\nRotatingFileHandler + QueueHandler]
        B --> C[Install signal handlers\nSIGTERM / SIGINT]
        C --> D[Register metrics\nCounter + Gauge + Histogram]
        D --> E[Service READY]
    end

    subgraph "main loop"
        E --> F{Shutdown\nrequested?}
        F -->|No| G[Fetch batch\nfrom source]
        G --> H[Validate records\nValidationError on failure]
        H --> I[Process batch\n_process_batch]
        I --> J[Record metrics\nlatency, errors, count]
        J --> K{Every N batches}
        K -->|Yes| L[gc.collect\nrelease memory]
        K -->|No| F
        L --> F
    end

    subgraph "shutdown"
        F -->|Yes| M[Finish current batch]
        M --> N[Flush log queue]
        N --> O[Log metrics report]
        O --> P[gc.collect final]
        P --> Q([Process exits 0])
    end

    subgraph "signals"
        R[SIGTERM / SIGINT] -->|sets shutdown flag| F
    end
```

**Key properties of the production service:**
1. **Graceful shutdown**: in-flight batch completes before exit.
2. **GC control**: `gc.set_threshold()` reduces mid-batch pauses;
   explicit `gc.collect()` at batch boundaries.
3. **Structured errors**: `ValidationError` is logged and counted, not fatal.
4. **Observable**: every batch emits latency to the histogram; percentiles
   available at any time via `MetricsRegistry.report()`.
5. **Config-driven**: all tuning knobs (`batch_size`, `gc_threshold`,
   `max_workers`) live in `config.yaml`.

---

## 6. Collections Module – When to Use Each

| Type | Use case | Key advantage |
|------|----------|---------------|
| `defaultdict(list)` | Grouping / adjacency lists | No `KeyError`; auto-initialise |
| `defaultdict(int)` | Frequency counting | Cleaner than `dict.get(k, 0) + 1` |
| `Counter` | Token counting, top-N | Arithmetic operators, `most_common()` |
| `deque(maxlen=N)` | Sliding window, bounded buffer | O(1) append/pop both ends |
| `namedtuple` | Immutable record (no methods) | Lightweight, tuple-compatible |
| `NamedTuple` | Typed record with defaults | Full type annotations |
| `ChainMap` | Layered config (env > file > defaults) | Read-through, live updates |
| `OrderedDict` | LRU cache, insertion-order dict | `move_to_end()`, `popitem(last=False)` |

---

## 7. Logging Architecture

```mermaid
flowchart LR
    App["Application code\nlogger.info(...)"] --> QH["QueueHandler\n(non-blocking put)"]
    QH --> Q["logging.Queue\n(thread-safe)"]
    Q --> QL["QueueListener\n(background thread)"]
    QL --> FH["RotatingFileHandler\nJSON lines"]
    QL --> CH["StreamHandler\ntext to stdout"]

    style Q fill:#f5f5f5,stroke:#999
    style QL fill:#dff0d8,stroke:#3c763d
```

**Why `QueueHandler`?** File I/O on the hot path (inside a request handler or
tight loop) adds latency. The `QueueHandler` puts the log record onto an
in-memory queue and returns immediately. The `QueueListener` drains the queue
in a dedicated daemon thread, so disk latency never affects application code.

---

## 8. Structural Patterns Summary

| Pattern | Problem solved | Example in codebase |
|---------|---------------|---------------------|
| **Adapter** | Two incompatible interfaces | `LegacyStoreAdapter` wraps `LegacyKeyValueStore` |
| **Decorator** | Add capability without subclassing | `ValidatingReader` + `TimingReader` wrap `DataReader` |
| **Facade** | Hide subsystem complexity | `MLPipelineFacade` hides extract/predict/format |
| **Proxy** | Lazy init + caching | `CachingProxy` over `ExpensiveService` |
| **Composite** | Tree of interchangeable components | `SequentialPipeline` contains `TransformStep` nodes |
