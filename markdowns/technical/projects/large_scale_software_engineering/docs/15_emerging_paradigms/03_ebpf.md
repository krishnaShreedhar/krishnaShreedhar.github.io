# eBPF (Extended Berkeley Packet Filter)

eBPF is a revolutionary kernel technology that allows safely running sandboxed programs in the Linux kernel without modifying kernel source code or loading kernel modules. It enables powerful observability, networking, and security capabilities with minimal overhead.

## eBPF Architecture

```mermaid
graph TD
    subgraph UserSpace[User Space]
        BPFProg[eBPF Program\nwritten in C or Rust\ncompiled to BPF bytecode]
        BPFLib[BPF Library\nlibbpf, bpftool\nBCC Python frontend]
        Loader[Program Loader\nattaches to hooks]
    end

    subgraph Kernel[Linux Kernel]
        Verifier[eBPF Verifier\nSafety check - no infinite loops\nno invalid memory access\nmust terminate]
        JIT[JIT Compiler\nBytecode to native machine code]
        Runtime[eBPF Runtime\nexecutes at hook point]

        subgraph Hooks[Kernel Hook Points]
            Kprobes[kprobes\nkernel function entry/exit]
            Tracepoints[Tracepoints\nstatic instrumentation]
            XDP[XDP - eXpress Data Path\nnetwork packet processing]
            TC[Traffic Control\npacket filtering]
            LSM[LSM Hooks\nLinux Security Module]
            Syscalls[Syscall hooks\nenter exit]
        end
    end

    BPFProg --> Loader --> Verifier --> JIT --> Runtime
    Runtime --> Hooks

    subgraph BPFMaps[BPF Maps - Shared State]
        Maps[Hash tables, arrays, ring buffers\nShared between BPF program and user space\nFor metrics, events, configuration]
    end

    Runtime --> BPFMaps --> BPFLib

    style Verifier fill:#dcfce7,stroke:#16a34a,stroke-width:2px
    style Runtime fill:#dbeafe,stroke:#2563eb
```

## eBPF Use Cases

```mermaid
graph TD
    subgraph Observability[Observability without Agents]
        OBS1[CPU profiling\nFlame graphs from kernel\nno instrumentation needed]
        OBS2[Network tracing\nLatency between any two\nkernel network layers]
        OBS3[File I/O monitoring\nWhich process accessing what files]
        OBS4[Distributed tracing\nCorrelate kernel events\nwith application traces]
    end

    subgraph Security[Security Monitoring]
        SEC1[System call monitoring\nDetect unusual syscall patterns]
        SEC2[File access control\nPrevent unauthorized reads\neven from root]
        SEC3[Network policy enforcement\nBlock unauthorized connections\nat kernel level]
        SEC4[Container escape detection\nnon-standard capabilities\nor namespaces]
    end

    subgraph Networking[High-Performance Networking]
        NET1[XDP packet processing\nMillions of packets per second\nDDoS mitigation]
        NET2[Load balancing\nFacebook Katran uses eBPF\nfor L4 load balancing]
        NET3[Service mesh\nCilium replaces iptables\nwith eBPF for K8s networking]
        NET4[Traffic shaping\nBandwidth management\nQoS enforcement]
    end

    style Observability fill:#dbeafe,stroke:#2563eb
    style Security fill:#fee2e2,stroke:#dc2626
    style Networking fill:#dcfce7,stroke:#16a34a
```

## eBPF vs Traditional Approaches

```mermaid
graph LR
    subgraph Traditional[Traditional Observability]
        T1[Agent installed on every host]
        T2[Custom kernel modules - risky]
        T3[Ptrace or strace - very slow]
        T4[Application instrumentation required]
        T5[Data loss under high load]
        style T1 fill:#fee2e2,stroke:#dc2626
        style T2 fill:#fee2e2,stroke:#dc2626
    end

    subgraph EBPF[eBPF Approach]
        E1[No agents - kernel built-in]
        E2[Safe verified programs]
        E3[Production-safe 1-2 percent overhead]
        E4[No application modification]
        E5[Kernel-level performance]
        style E1 fill:#dcfce7,stroke:#16a34a
        style E2 fill:#dcfce7,stroke:#16a34a
    end
```

## Key Concepts

- **eBPF Verifier**: Before any eBPF program runs, the kernel's verifier checks it for safety: no unbounded loops (guaranteed termination), no invalid memory accesses, no accessing other processes' memory, and no kernel instability. This makes eBPF safe to run in production — unlike kernel modules, a buggy eBPF program cannot crash the kernel.

- **JIT Compilation**: After verification, the BPF bytecode is compiled to native machine code by the kernel's JIT compiler. eBPF programs run at near-native speed with minimal overhead (typically 1-2% CPU overhead for comprehensive observability).

- **XDP (eXpress Data Path)**: An eBPF hook point that processes network packets at the earliest possible point in the kernel network stack, before any memory allocation. Enables packet processing at millions of packets per second. Used by Facebook's Katran load balancer and DDoS mitigation systems.

- **Cilium**: A Kubernetes networking and security plugin (CNI) that replaces iptables with eBPF for all network policy enforcement and service networking. eBPF-based networking is significantly faster and more observable than iptables.

- **Tetragon**: A Kubernetes security observability and runtime enforcement tool by Isovalent/Cilium. Uses eBPF to detect and optionally block security events (privilege escalation, unauthorized network connections, sensitive file access) at the kernel level.

- **BPF Maps**: Key-value stores in the kernel that eBPF programs and user-space applications use to exchange data. Ring buffers (perf buffers) enable high-throughput event streaming from eBPF programs to user space. Hash maps store aggregated metrics.

- **Continuous Profiling**: Using eBPF-based profilers (Parca, Pyroscope, Grafana Phlare) to continuously profile production applications at low overhead. Generates always-on flame graphs showing where CPU time is spent, enabling performance regression detection without sampling bias.

## Trade-offs

| Approach | Performance | Safety | Portability | Kernel Version |
|---------|------------|--------|------------|----------------|
| Kernel module | Highest | Low (can crash) | Low | Any |
| eBPF | Very High | High (verified) | Medium (kernel 5.x+) | 5.x+ |
| ptrace/strace | Low | High | High | Any |
| User-space agent | Medium | High | High | Any |

## When to Apply

- **Production observability**: Replace sampling profilers with continuous eBPF profiling for always-on performance visibility
- **Kubernetes networking**: Adopt Cilium for better performance and native network policy visibility vs iptables
- **Security monitoring**: eBPF-based runtime security (Falco, Tetragon) provides kernel-level visibility that application agents cannot match
- **DDoS mitigation**: XDP-based packet processing at line rate before the kernel networking stack
- **Requires Linux 5.x+**: eBPF features improve rapidly with kernel versions; check capability requirements for your target kernel
