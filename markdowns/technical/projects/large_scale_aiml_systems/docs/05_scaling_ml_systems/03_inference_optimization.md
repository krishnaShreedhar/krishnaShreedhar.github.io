# Inference Optimization

Inference optimization reduces the latency and cost of running large neural networks at serving time. Training cost is paid once; inference cost is paid for every prediction in production — for large LLMs handling millions of requests, optimizing inference has 10-100x impact on total compute spend. Key techniques include quantization, KV-cache management, continuous batching, and speculative decoding.

## LLM Inference Bottlenecks

```mermaid
graph TD
    subgraph Bottlenecks[LLM Inference Performance Bottlenecks]
        subgraph Phases[Two Phases of LLM Inference]
            Prefill[Prefill Phase\nProcess entire input prompt\nParallel matrix multiplications\nCompute-bound\nGPU utilization high]

            Decode[Decode Phase - Token Generation\nGenerate one token at a time\nRead all KV cache from memory each step\nMemory bandwidth-bound\nGPU utilization low - 30-50%]
        end

        subgraph KVCacheProblem[KV Cache Memory Problem]
            KVC[KV Cache Size\nfor seq_len tokens\nnum_layers x 2 x num_heads x head_dim x seq_len x dtype_bytes\nA 13B model: 800 MB per request at 2048 tokens\nServing 100 concurrent requests = 80 GB\ncannot fit more requests]
        end
    end
```

## Quantization

```mermaid
graph TD
    subgraph Quantization[Quantization Techniques]
        subgraph W8A8[INT8 Quantization]
            Q8[Quantize weights to int8\nquantize activations to int8\ndequantize for output\n2x memory reduction from fp16\n15-20% speedup on A100]
        end

        subgraph W4[INT4 Weight Quantization]
            Q4[Weights only quantized to int4\nActivations remain fp16\nAWQ GPTQ bitsandbytes NF4\n4x memory reduction from fp16\nMinimal quality loss for LLMs 7B plus]
            style Q4 fill:#dcfce7,stroke:#16a34a,stroke-width:2px
        end

        subgraph Mixed[Mixed Precision Quantization]
            QM[Important layers kept at fp16\nSensitive attention layers at fp16\nFFN layers at int4\nMinimizes quality loss]
        end

        subgraph Calibration[Calibration Process]
            Cal[Collect representative inputs\nrun calibration forward pass\ncompute activation ranges\nfit quantization parameters\nvalidate quality on eval set]
        end
    end
```

## KV Cache Management with vLLM

```mermaid
graph TD
    subgraph PagedAttention[vLLM PagedAttention]
        subgraph NaiveKV[Naive KV Cache - Fragmented]
            NK1[Reserve max_seq_len memory\nfor every request upfront\nMost memory wasted\nfor short completions\nMax 10-20 concurrent requests]
            style NK1 fill:#fee2e2,stroke:#dc2626
        end

        subgraph Paged[PagedAttention - Efficient]
            PA1[KV cache divided into fixed-size blocks\nlike OS virtual memory pages\nBlocks allocated on demand\nas tokens are generated]
            PA2[Multiple requests share physical\nmemory blocks for common prefixes\nPrefix caching: system prompt\nshared across all requests\n3-10x more concurrent requests]
            PA3[GPU memory utilization: 90%+\nvs 20-40% naive reservation]
            style PA2 fill:#dcfce7,stroke:#16a34a,stroke-width:2px
        end
    end
```

## Continuous Batching

```mermaid
graph TD
    subgraph ContinuousBatching[Continuous Batching vs Static Batching]
        subgraph Static[Static Batching]
            SB[Wait for N requests\nprocess all together\nAll must finish before\nnext batch starts\nNew requests wait even if\nGPU has capacity]
            style SB fill:#fee2e2,stroke:#dc2626
        end

        subgraph Continuous[Continuous Batching - Iteration-Level Scheduling]
            CB[After each decode step\ncheck for completed requests\nimmediately insert new waiting requests\ninto freed batch slots\nGPU always processing maximum requests\n3-5x higher throughput]
            style CB fill:#dcfce7,stroke:#16a34a,stroke-width:2px
        end
    end
```

## Speculative Decoding

```mermaid
graph TD
    subgraph SpecDec[Speculative Decoding]
        DraftModel[Small Draft Model\n70M-7B parameters\ngenerates K draft tokens\nvery fast - K tokens in ~1 step]

        Verify[Target Model Verification\nVerify K draft tokens\nin parallel single forward pass\nfor all K positions simultaneously]

        Accept{Accept\ndraft token?}
        AcceptAll[All K tokens accepted\nK tokens generated\nfor cost of 1 target step plus K draft steps]
        PartAccept[Accept first M tokens\nreject remaining\nM + 1 tokens generated\nsample corrected token for M+1]

        Verify --> Accept
        Accept -->|Yes| AcceptAll
        Accept -->|No| PartAccept

        DraftModel --> Verify

        Speedup[Typical speedup: 2-3x\nfor tasks with predictable tokens\ncoding boilerplate, repetitive text]
        style Speedup fill:#dcfce7,stroke:#16a34a
    end
```

## Key Concepts

- **KV Cache**: During autoregressive token generation, each new token attends to all previous tokens. Rather than recomputing attention keys and values for previous tokens at each step, they are cached in GPU memory. This KV cache grows with sequence length and is the dominant memory consumer during LLM inference. A single 13B parameter model request at 4096 tokens requires ~1.6GB of KV cache storage.

- **Continuous Batching (Iteration-Level Batching)**: The core scheduling innovation in modern LLM serving. Traditional static batching processes all requests in a batch together, wasting capacity when short requests complete while long ones continue. Continuous batching evicts completed requests and inserts new ones at every decode step, maximizing GPU utilization. vLLM, TGI (Text Generation Inference), and TensorRT-LLM all implement continuous batching.

- **PagedAttention**: vLLM's innovation that applies virtual memory paging concepts to KV cache management. Instead of pre-allocating contiguous memory for the maximum sequence length, blocks are allocated dynamically as tokens are generated. This eliminates memory waste and enables prefix caching — multiple requests sharing a common system prompt can share the same cached KV blocks.

- **INT4 Quantization**: Reducing model weights from 16-bit (FP16/BF16) to 4-bit integers, achieving 4x memory reduction with minimal quality loss for models above ~7B parameters. Methods: GPTQ (post-training quantization by layer-wise optimization), AWQ (activation-aware weight quantization preserving outlier channels), bitsandbytes NF4 (used in QLoRA). Smaller models suffer more quality degradation from INT4 quantization.

- **Flash Attention**: A hardware-aware attention algorithm that computes attention without materializing the full N×N attention matrix in GPU memory. Instead, attention is computed in tiles that fit in GPU SRAM (fast memory), reducing memory bandwidth requirements from O(N²) to O(N). Flash Attention 2 and 3 achieve near-theoretical GPU bandwidth utilization and are the default implementation in all modern LLM frameworks.

- **Speculative Decoding**: Exploits the observation that the target model's time per token is the same regardless of batch size (it's memory-bandwidth-bound). A small draft model generates K candidate tokens cheaply; the target model verifies all K in one forward pass. When the draft is correct (common for predictable continuations), K tokens are produced for the cost of ~1 target forward pass. Rejection sampling ensures correctness — the output distribution matches the target model exactly.

- **TensorRT-LLM**: NVIDIA's optimized LLM inference library. Fuses operations (attention, layer norm, activation), uses TensorRT's kernel optimization, and implements continuous batching, quantization, and multi-GPU serving. Best-in-class latency and throughput for NVIDIA GPUs in production serving.

## Trade-offs

| Optimization | Latency Gain | Throughput Gain | Quality Loss | Implementation |
|-------------|-------------|----------------|-------------|---------------|
| INT8 quantization | 20-30% | 20-30% | Minimal | Low |
| INT4 quantization | 50-70% | 50-70% | Small (7B+) | Low |
| Flash Attention | 30-50% | 30-50% | None | Drop-in |
| Continuous batching | 0 | 3-5x | None | Framework |
| PagedAttention | 0 | 2-4x | None | vLLM |
| Speculative decoding | 2-3x | ~1x | None | Medium |

## When to Use

- **Flash Attention**: Always — it's a drop-in replacement with no quality loss and significant speedup, available in PyTorch 2.0+ natively
- **INT4 quantization**: When GPU memory is the constraint and serving models with 7B+ parameters — acceptable quality tradeoff for 4x memory reduction
- **vLLM**: Default serving framework for open-source LLMs — continuous batching and PagedAttention are pre-integrated and production-hardened
- **Speculative decoding**: High-throughput serving with predictable output patterns (coding assistants, structured generation) — 2-3x speedup with no quality loss
- **INT8 quantization**: When INT4 quality loss is unacceptable but memory savings are still needed — intermediate tradeoff
