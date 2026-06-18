# Concurrency Patterns

Concurrency patterns provide reusable solutions to the challenges of writing correct, efficient multi-threaded and asynchronous programs. They address thread management, safe data sharing, producer-consumer coordination, and event handling in concurrent environments.

## Thread Pool Pattern

```mermaid
graph TD
    subgraph ThreadPool[Thread Pool]
        Queue[Task Queue\nbounded blocking queue]
        W1[Worker Thread 1\nidle / executing]
        W2[Worker Thread 2\nidle / executing]
        W3[Worker Thread 3\nidle / executing]
        WN[Worker Thread N\nidle / executing]

        Queue --> W1 & W2 & W3 & WN
    end

    Submitters[Task Submitters] -->|submit task| Queue

    subgraph Outcomes
        W1 -->|task complete| Result1[Result / Callback]
        W2 -->|task complete| Result2[Future / Promise]
    end

    Monitor[Pool Monitor\nmin/max threads\ncurrent load\nqueue depth]
    ThreadPool --- Monitor

    style Queue fill:#fef3c7,stroke:#d97706,stroke-width:2px
    style Monitor fill:#dbeafe,stroke:#2563eb
```

## Producer-Consumer Pattern

```mermaid
graph LR
    subgraph Producers
        P1[Producer 1\nData generator]
        P2[Producer 2\nAPI scraper]
        P3[Producer 3\nFile reader]
    end

    subgraph Buffer[Bounded Buffer / Channel]
        Q[Blocking Queue\nmax capacity: N]
        Note[Producers block when full\nConsumers block when empty]
    end

    subgraph Consumers
        C1[Consumer 1\nData processor]
        C2[Consumer 2\nDB writer]
        C3[Consumer 3\nNotifier]
    end

    P1 & P2 & P3 -->|put - blocks if full| Q
    Q -->|take - blocks if empty| C1 & C2 & C3

    style Q fill:#dcfce7,stroke:#16a34a,stroke-width:2px
```

## Read-Write Lock Pattern

```mermaid
stateDiagram-v2
    [*] --> Unlocked: Initial state

    Unlocked --> ReadLocked: reader acquires
    ReadLocked --> ReadLocked: additional reader acquires\n(multiple readers allowed)
    ReadLocked --> Unlocked: last reader releases

    Unlocked --> WriteLocked: writer acquires\n(exclusive)
    WriteLocked --> Unlocked: writer releases

    ReadLocked --> WriteLocked: NOT allowed\nwriter must wait for all readers
    WriteLocked --> ReadLocked: NOT allowed\nreaders must wait for writer
```

## Reactor Pattern

```mermaid
graph TD
    subgraph Reactor[Reactor / Event Loop]
        Selector[I/O Selector\nepoll / kqueue / IOCP]
        Dispatcher[Event Dispatcher]
        Selector -->|events ready| Dispatcher
    end

    subgraph Handlers[Event Handlers]
        ConnHandler[Connection Handler\non new connection]
        ReadHandler[Read Handler\non data available]
        WriteHandler[Write Handler\non write ready]
        TimerHandler[Timer Handler\non timeout]
    end

    Dispatcher --> ConnHandler & ReadHandler & WriteHandler & TimerHandler

    subgraph Clients[Many Concurrent Connections]
        C1[Client 1]
        C2[Client 2]
        CN[Client N]
    end

    C1 & C2 & CN -->|async I/O| Selector

    style Selector fill:#fef3c7,stroke:#d97706,stroke-width:2px
    style Dispatcher fill:#dbeafe,stroke:#2563eb
```

## Active Object Pattern

```mermaid
sequenceDiagram
    participant Client
    participant Proxy as Active Object Proxy
    participant Scheduler as Method Scheduler
    participant Servant as Servant Object
    participant Future as Future/Result

    Client->>Proxy: asyncMethod(args)
    Proxy->>Future: create Future
    Proxy->>Scheduler: enqueue MethodRequest
    Proxy-->>Client: return Future (immediately)

    Note over Client: Client continues other work

    Scheduler->>Scheduler: dequeue MethodRequest
    Scheduler->>Servant: invoke actual method
    Servant-->>Future: set result
    Client->>Future: get() - blocks until result ready
```

## Key Concepts

- **Thread Pool**: Maintains a pool of worker threads that can be reused to execute tasks, avoiding the overhead of creating and destroying threads for each task. Key parameters: core pool size (always-on threads), maximum pool size (burst capacity), queue capacity (buffer), and rejection policy (what to do when queue is full: reject, caller runs, discard oldest).

- **Producer-Consumer**: Decouples the production of data from its consumption using a bounded buffer. Producers add items to the buffer and block when full; consumers take items and block when empty. This back-pressure mechanism prevents producers from overwhelming consumers. Implemented with blocking queues (Java), channels (Go), or asyncio queues (Python).

- **Read-Write Lock**: Allows concurrent reads (multiple readers simultaneously) but exclusive writes (only one writer, no readers). Appropriate when reads far outnumber writes and the data structure is safe for concurrent reading. Trade-off: writer starvation if reads are continuous; write-preferring variants exist.

- **Reactor (Event Loop)**: A single-threaded event loop that demultiplexes I/O events from many connections and dispatches them to registered handlers. All handlers must be non-blocking — blocking a handler blocks the entire event loop. Foundation of Node.js, Netty, Nginx, and Python asyncio.

- **Proactor**: Like Reactor but initiates asynchronous I/O operations and receives completion notifications. The OS performs the I/O and notifies the application when complete, rather than the application polling for readiness (as in Reactor). Used by Windows IOCP.

- **Active Object**: Decouples method execution from method invocation for objects in their own thread of control. Method calls return immediately with a Future; the actual execution happens asynchronously. Provides a clean interface to asynchronous execution without callback hell.

- **Monitor Object**: Synchronizes concurrent execution of methods on an object and allows only one method to run within the object at a time. Methods acquire the monitor lock on entry and release on exit. Java synchronized methods implement this pattern.

## Trade-offs

| Pattern | Benefit | Cost |
|---------|---------|------|
| Thread Pool | Resource control, thread reuse | Tuning complexity, queue saturation |
| Producer-Consumer | Decoupled rates, back-pressure | Buffer sizing, deadlock risk |
| Read-Write Lock | High read throughput | Writer starvation, complexity |
| Reactor | High concurrency, low threads | No blocking I/O allowed |
| Active Object | Clean async API | Future management complexity |

## When to Use

- **Thread Pool**: Any server application handling concurrent requests — use a well-tuned thread pool instead of spawning threads per request
- **Producer-Consumer**: Data pipeline stages where processing rates differ between stages
- **Read-Write Lock**: Shared data structures with frequent reads and rare writes (caches, configuration)
- **Reactor**: High-concurrency network servers where the bottleneck is I/O concurrency, not CPU
- **Active Object**: When you need to provide a synchronous-looking interface to asynchronous execution
