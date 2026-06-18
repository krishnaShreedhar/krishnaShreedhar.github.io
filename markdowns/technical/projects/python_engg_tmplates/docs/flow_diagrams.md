# Execution Flow Diagrams

---

## 1. Thread Pool Execution Flow

```mermaid
sequenceDiagram
    participant Main as Main Thread
    participant Exec as ThreadPoolExecutor
    participant W1 as Worker Thread 1
    participant W2 as Worker Thread 2
    participant W3 as Worker Thread 3
    participant Q as Internal Work Queue

    Main->>Exec: __enter__() – create pool (max_workers=N)
    Exec->>W1: spawn thread
    Exec->>W2: spawn thread
    Exec->>W3: spawn thread

    loop For each task
        Main->>Exec: submit(fn, arg_i)
        Exec->>Q: enqueue Future
    end

    Q->>W1: dequeue task-0
    Q->>W2: dequeue task-1
    Q->>W3: dequeue task-2
    Note over W1,W3: Threads execute concurrently

    W1-->>Main: future-0.set_result()
    W3-->>Main: future-2.set_result()
    W2-->>Main: future-1.set_result()
    Note over Main: as_completed() yields futures in\ncompletion order (not submission order)

    Main->>Exec: __exit__() – join all threads
```

**Key points:**
- `as_completed()` yields futures as they finish, allowing progressive result
  processing without waiting for the slowest task.
- Worker threads are reused across tasks (no thread creation overhead per task).
- `max_workers` controls the concurrency limit; set to `cpu_count()` for
  CPU-bound, higher for I/O-bound workloads.

---

## 2. Asyncio Event Loop Flow

```mermaid
flowchart TD
    subgraph "Event Loop (single OS thread)"
        EL([Event Loop starts])
        RQ[Ready Queue\ncoroutines to run]
        IO[I/O Selector\nepoll / kqueue]
        CB[Callback Queue\nI/O completions]

        EL --> RQ
        RQ -->|"pop next coroutine"| Run[Resume coroutine\nat await point]
        Run -->|"hits await asyncio.sleep\nor await network"| Suspend[Suspend coroutine\nregister with selector]
        Suspend --> IO
        IO -->|"I/O ready"| CB
        CB -->|"schedule resumption"| RQ
        Run -->|"coroutine returns"| Done[Result available\nfuture.set_result()]
    end

    App["asyncio.gather(\n  coro_1(), coro_2(), ...\n)"] --> EL
    Done --> App
```

**Cooperative multitasking:** A coroutine runs until it hits an `await`
expression. At that point it suspends and the event loop picks the next
ready coroutine. This is why `asyncio` can handle thousands of concurrent
I/O operations with a single OS thread – no context-switch overhead.

**asyncio.TaskGroup (Python 3.11+):**

```mermaid
sequenceDiagram
    participant Main
    participant TG as TaskGroup
    participant T1 as Task 1
    participant T2 as Task 2
    participant T3 as Task 3

    Main->>TG: async with TaskGroup() as tg:
    Main->>TG: tg.create_task(coro1)
    TG->>T1: schedule
    Main->>TG: tg.create_task(coro2)
    TG->>T2: schedule
    Main->>TG: tg.create_task(coro3)
    TG->>T3: schedule

    T1-->>TG: done
    T3-->>TG: done
    T2-->>TG: done

    Note over TG: All tasks done → exit __aexit__
    TG-->>Main: results via task.result()
```

If **any** task raises an exception, `TaskGroup` cancels the remaining tasks
and propagates the error as an `ExceptionGroup`.

---

## 3. Retry Decorator – Full Decision Flow

```mermaid
stateDiagram-v2
    [*] --> Attempt1 : Call decorated function

    Attempt1 --> Success : Returns value
    Success --> [*] : Return result

    Attempt1 --> Error1 : Raises exception
    Error1 --> Retryable1 : Is exception retryable?
    Retryable1 --> Raise1 : No → re-raise
    Raise1 --> [*]

    Retryable1 --> MaxCheck1 : Yes
    MaxCheck1 --> Raise2 : attempt == max_attempts
    Raise2 --> [*]

    MaxCheck1 --> Wait1 : attempt < max_attempts
    Wait1 --> Attempt2 : sleep(backoff * 2^0 [+ jitter])

    Attempt2 --> Success2 : Returns value
    Success2 --> [*] : Return result

    Attempt2 --> Error2 : Raises exception
    Error2 --> MaxCheck2 : Is retryable?
    MaxCheck2 --> Raise3 : attempt == max_attempts
    Raise3 --> [*]

    MaxCheck2 --> Wait2 : sleep(backoff * 2^1 [+ jitter])
    Wait2 --> AttemptN : ...

    AttemptN --> FinalRaise : Still failing after max_attempts
    FinalRaise --> [*]
```

**Backoff formula:**
```
wait = backoff_factor × 2^(attempt - 1)
if jitter:
    wait += uniform(0, wait)   # spreads retries ±100%
```

| attempt | backoff=2.0, no jitter | backoff=2.0, jitter range |
|---------|------------------------|---------------------------|
| 1       | 2.0 s                  | 2.0 – 4.0 s               |
| 2       | 4.0 s                  | 4.0 – 8.0 s               |
| 3       | 8.0 s                  | 8.0 – 16.0 s              |

---

## 4. Production Service Startup / Shutdown Flow

```mermaid
flowchart TB
    subgraph STARTUP ["Service Startup"]
        direction TB
        S1["Read config.yaml\nValidate required keys"] --> S2
        S2["Configure logging\nRotatingFileHandler + QueueHandler"] --> S3
        S3["Install signal handlers\nsignal.signal(SIGTERM, handler)\nsignal.signal(SIGINT, handler)"] --> S4
        S4["Initialise MetricsRegistry\nregister counters / gauges / histograms"] --> S5
        S5["Set GC thresholds\ngc.set_threshold(1000, 10, 10)"] --> S6
        S6["Log startup event\nService READY"]
    end

    subgraph RUNNING ["Main Processing Loop"]
        direction TB
        R1{shutdown_requested?} -->|No| R2
        R2["Pop next batch\n(batch_size from config)"] --> R3
        R3["_process_batch(batch)\nfor each record:"] --> R4
        R4{"_validate_record\npasses?"} -->|Yes| R5
        R4 -->|No| R6["log.warning + metrics.errors++"]
        R5["_process_record\ntransform / enrich"] --> R7
        R7["metrics.processed++\nmetrics.latency.observe(ms)"] --> R8
        R8{"Processed %\nbatch_size×10 == 0?"} -->|Yes| R9["gc.collect()"]
        R9 --> R1
        R8 -->|No| R1
        R6 --> R1
    end

    subgraph SHUTDOWN ["Graceful Shutdown"]
        direction TB
        G1["Complete current batch\n(no mid-batch abort)"] --> G2
        G2["QueueListener.stop()\nflush log queue"] --> G3
        G3["MetricsRegistry.log_report()\nfinal metrics snapshot"] --> G4
        G4["gc.collect()\nfinal sweep"] --> G5
        G5["log.info: Service stopped\nProcess exits 0"]
    end

    subgraph SIGNALS ["OS Signals"]
        SIG["SIGTERM / SIGINT"] --> SF["_handle_shutdown_signal\nshutdown_requested = True"]
    end

    STARTUP --> RUNNING
    RUNNING -->|"shutdown_requested=True"| SHUTDOWN
    SF -.->|"sets flag"| R1
```

**Graceful shutdown guarantees:**
1. The current batch completes – no partial writes.
2. Log records in the queue are flushed before the process exits.
3. Final metrics are logged for post-mortem analysis.
4. Exit code is `0` for clean shutdown, non-zero for unhandled exceptions.

---

## 5. Async Producer / Consumer Flow

```mermaid
sequenceDiagram
    participant P as Producer coroutine
    participant Q as asyncio.Queue
    participant C1 as Consumer 1
    participant C2 as Consumer 2

    P->>Q: await put("item-000")
    P->>Q: await put("item-001")
    C1->>Q: await get() → "item-000"
    C2->>Q: await get() → "item-001"
    P->>Q: await put("item-002")
    C1-->>Q: task_done()
    C1->>Q: await get() → "item-002"
    P->>Q: await put(None)  [sentinel]
    C2-->>Q: task_done()
    C2->>Q: await get() → None [sentinel]
    C2->>Q: await put(None)  [re-queue for C1]
    C2-->>C2: break loop
    C1->>Q: await get() → None
    C1-->>C1: break loop
    Note over P,C2: All done; gather() returns
```

**Sentinel re-queueing pattern:** When a consumer receives `None` (the
sentinel), it immediately puts `None` back before exiting. This ensures
every consumer eventually receives the termination signal, regardless of
how many consumers share the queue.

---

## 6. Shared Memory Flow (multiprocessing)

```mermaid
flowchart LR
    subgraph Main ["Main Process"]
        SM["SharedMemory.create()\nshm.name='psm_abc123'"]
        Write["Write array → shm.buf"]
    end

    subgraph Workers ["Worker Processes (spawn)"]
        W1["Process 1\nSharedMemory(name='psm_abc123')\nread slice [0:500]"]
        W2["Process 2\nSharedMemory(name='psm_abc123')\nread slice [500:1000]"]
    end

    subgraph Collect ["Result Collection"]
        Pool["Pool.map() collects\npartial sums"]
        Total["sum(partial_sums)"]
    end

    SM --> Write
    Write -->|"OS shared page"| W1
    Write -->|"OS shared page"| W2
    W1 -->|"partial_sum_1"| Pool
    W2 -->|"partial_sum_2"| Pool
    Pool --> Total
```

**Zero-copy:** Worker processes map the same physical memory pages. No
serialisation (pickle) occurs for the shared buffer – only the small result
values are pickled back to the main process.
