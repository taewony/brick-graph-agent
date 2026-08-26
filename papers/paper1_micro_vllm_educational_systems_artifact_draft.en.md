# Windows-Native LLM Serving System Implementation and Performance Analysis Case Study

**Working title:** Windows-Native LLM Serving System Implementation and Performance Analysis Case Study: micro-vLLM, CUDA Python, Hybrid Prefill Dispatch, Prefix KV Cache, and Bounded CUDA Mechanism Analysis

**Author:** Taewon Kim

**Draft status:** Submission-candidate draft for Paper #1. This version frames the paper as a **Windows-native LLM serving system implementation and performance analysis case study**. The strongest results are fixed-context prefix KV-cache TTFT reduction and the eager-hybrid cuTile path reaching 557.66 tok/s after KV-store repair and hybrid prefill dispatch. CUDA Graph replay and Green Contexts are included as bounded or negative mechanism analyses, not as headline speedup claims.

## Abstract

High-performance LLM serving systems are typically developed around Linux-centric GPU software stacks, including FlashAttention, Triton, NCCL, vLLM-style schedulers, and production-oriented memory managers. These systems are effective, but their scale and dependency assumptions make it difficult for students and Windows-based local researchers to inspect how prefill, decode, KV cache management, scheduler decisions, CUDA contexts, and memory allocation interact inside an inference engine. This paper presents **micro-vLLM** as a **Windows-native LLM serving system implementation and performance analysis case study**. The goal is not to outperform mature Linux serving stacks, but to provide a compact, modifiable, and evidence-producing artifact for learning and analyzing LLM serving mechanisms on a Windows-native CUDA Python/cuTile path.

The artifact follows a staged path from tiled MatMul and fused multi-head attention to a minimal LLM loop and a nano-vLLM-style serving engine. We evaluate mechanisms that are useful for teaching complete serving-loop causality. First, WSL2 FlashAttention remains the optimized reference baseline, achieving 2138.55 tok/s on average versus 463.35 tok/s for the initial Windows-native cuTile path, a 4.62x throughput advantage. Second, after repairing the cuTile KV-cache store path and introducing hybrid prefill dispatch, the eager-hybrid cuTile path reaches 557.66 tok/s, a 20.4% gain over the initial cuTile mean. CUDA Graph decode replay is functionally repaired, but currently degrades throughput to roughly 393 tok/s, making it a useful negative mechanism result. Third, for fixed-context agent workloads that repeatedly reuse long static prefixes such as system prompts, database schemas, `SKILL.md` files, or course contexts, warm prefix KV-cache reuse reduces computed prefill tokens to 64 and reduces TTFT by 41.1%, 66.1%, and 79.7% for 1024-, 2048-, and 3072-token static prefixes, respectively. Fourth, CUDA-core Green Contexts successfully activate on an RTX 5070 target PC and moderately smooth protected decode latency under stress, but do not eliminate sequential prefill-induced pauses. Together, these results support a practical educational claim: inference-engine education should teach implementation, measurement, successful optimizations, bounded mechanisms, and negative results in the complete serving loop.

## Keywords

LLM serving, CUDA Python, cuTile, hybrid prefill dispatch, prefix KV cache, CUDA Graph, CUDA Green Contexts, nano-vLLM, fixed-context agents, Windows-native GPU computing, educational systems artifact

## 1. Introduction

Local LLM inference is increasingly relevant in classrooms, research laboratories, private data-analysis settings, and offline development environments. In these settings, learners do not only need to call an LLM API; they need to understand how prompts become prefill work, how KV cache blocks are allocated, why decode is latency sensitive, how scheduler decisions affect visible response time, and why a local optimization can fail when placed inside the complete serving loop.

Production-grade serving systems such as vLLM, TensorRT-LLM, FlashAttention-based stacks, and Triton kernels provide strong performance, but they are not ideal as first educational artifacts. They are large, Linux-oriented, and tightly coupled to optimized dependencies. WSL2 is a practical path for Windows users, but it adds another boundary between the learner and the native runtime. Windows-native GPU development remains important because many students and local researchers use consumer Windows machines with NVIDIA GPUs.

This paper therefore targets a different contribution: **a Windows-Native LLM serving implementation and performance-analysis case study** as an educational systems artifact. micro-vLLM is not positioned as a faster replacement for vLLM. It is designed to expose the serving mechanisms that students and researchers need to inspect: prefill, decode, paged KV cache, prefix reuse, CUDA context activation, scheduler behavior, allocation overhead, and tail-latency tradeoffs. The artifact is valuable when it turns serving behavior into reproducible evidence rather than hiding it behind a production stack.

A central workload in this paper is the **fixed-context agent workload**. Many agent systems repeatedly prepend stable context before a short user request: system prompts, database schemas, tool contracts, `SKILL.md` files, policy text, examples, rubrics, or course modules. Text-to-SQL agents repeatedly include database schema and rule context. CUDA tutoring agents repeatedly include course material and kernel templates. Educational knowledge agents repeatedly include selected knowledge-bundle excerpts. Such workloads are naturally suited to prefix KV-cache reuse, because the expensive static prefix can be reused across requests.

## 2. Research Questions

**RQ1.** Can a Windows-native CUDA Python/cuTile-based micro-vLLM artifact expose the main mechanisms of LLM serving in a form suitable for education and controlled experimentation?

**RQ2.** For fixed-context agent workloads, how much does prefix KV-cache reuse reduce computed prefill tokens and time-to-first-token in a Windows-native micro-vLLM implementation?

**RQ3.** How do hybrid prefill dispatch and cuTile KV-store repair affect Windows-native cuTile serving throughput?

**RQ4.** What do bounded and negative optimization results, including CUDA Graph replay, CUDA Green Contexts, and dynamic shape padding, teach about full serving-loop causality?

## 3. Contributions

This paper makes five contributions.

1. **A Windows-native educational migration artifact.** We organize micro-vLLM as a staged learning path from MatMul and fused attention to LLM-from-scratch and nano-vLLM-style serving. The artifact is compact enough to inspect and modify while still exercising real GPU behavior.

2. **A fixed-context agent benchmark.** We define and measure agent-like workloads where requests share long static prefixes followed by short dynamic user questions. This bridges inference-engine mechanisms with practical Text-to-SQL, tutoring, and knowledge-agent scenarios.

3. **Measured prefix KV-cache evidence.** Warm prefix cache reduces computed prefill tokens to 64 in all tested static-prefix settings and reduces TTFT by up to 79.7%. The changed-prefix negative control returns 0.0% cache hit and near no-cache TTFT, validating exact-prefix dependence.

4. **Hybrid prefill dispatch and cuTile serving repair.** After replacing the PyTorch advanced-indexing KV-cache update with a cuTile KV-store kernel and dispatching prefill by workload shape, the eager-hybrid cuTile path improves from the initial 463.35 tok/s mean to 557.66 tok/s.

5. **Serving-loop bottleneck analysis.** We report both bounded and negative systems results: WSL2 FlashAttention remains 4.62x faster than the initial Windows-native cuTile backend; CUDA Graph decode replay is functionally enabled but performance-negative at roughly 393 tok/s; dynamic shape padding reduces throughput by 67.04% due to allocator pressure; CUDA-core Green Contexts activate successfully and moderately smooth protected decode latency under stress, but do not eliminate sequential prefill-induced pauses.

## 4. Background

### 4.1 LLM Serving Loop

Autoregressive LLM serving consists of a **prefill** phase and a **decode** phase. Prefill processes the prompt and writes KV cache entries for all prompt tokens. Decode generates one token at a time while attending to previously cached keys and values. Prefill is sensitive to prompt length and parallel compute throughput. Decode is sensitive to per-token latency, memory bandwidth, scheduler overhead, KV-cache access, and kernel-launch behavior.

This distinction matters because an optimization can improve one phase while leaving the end-to-end service metric unchanged. For example, prefix cache can greatly reduce prefill work and TTFT, yet produce limited throughput benefit when the benchmark generates many output tokens and decode dominates total runtime.

### 4.2 Paged KV Cache and Prefix Reuse

Paged KV cache divides KV memory into blocks that can be allocated, mapped, reused, and evicted. Prefix caching extends this idea with token-identical prefix reuse. When a new request shares the same token prefix as an earlier request, the runtime can reuse the existing KV blocks and compute only the suffix.

The mechanism depends on exact token-prefix stability. Dynamic metadata such as timestamps, run IDs, randomized examples, or user-specific text placed early in the prompt can break reuse. Therefore, prompt layout is not only a modeling concern; it is also a runtime-efficiency concern. Static context should appear before dynamic user content when prefix reuse is desired.

### 4.3 CUDA Python, cuTile, and nano-vLLM

CUDA Python provides Python-level access to CUDA runtime and driver APIs, making it useful for teaching host-side GPU control. A cuTile-style workflow expresses tiled GPU kernels inside Python, lowering the barrier relative to C++ CUDA while exposing more hardware behavior than high-level PyTorch operators. nano-vLLM is a compact reference for vLLM-style serving. micro-vLLM adapts this pedagogical stance to a Windows-native CUDA Python experimentation path.

### 4.4 Green Contexts as Resource Partitioning

CUDA Green Contexts allow GPU resources, especially SMs, to be partitioned across contexts. In this paper they are treated as a **resource-isolation mechanism**, not as a general speedup mechanism. The relevant question is whether a latency-sensitive decode path can be protected under prefill interference. A Green Context result is only valid if activation metadata confirms that the requested resource-partitioning path was actually used.

## 5. System Design

### 5.1 Design Goals

micro-vLLM is designed around four goals.

- **Inspectability:** learners can trace how prompt tokens become prefill work, KV blocks, and decode steps.
- **Modifiability:** kernels, scheduler logic, and runtime instrumentation are small enough to change during experiments.
- **Windows-native execution:** the artifact runs in a Windows CUDA environment without treating WSL2 as the only viable path.
- **Agent-workload relevance:** benchmarks reflect repeated static context in educational agents, Text-to-SQL agents, and local knowledge agents.

### 5.2 Migration Stages

| Stage | Folder | Educational role |
| :--- | :--- | :--- |
| 0 | `KernelAgent/0-MatMul` | tiling, shared memory, swizzling, GEMM baseline learning |
| 1 | `KernelAgent/1-FMHA` | online softmax, causal masking, fused attention implementation |
| 2 | `KernelAgent/2-LLM-from-scratch` | minimal autoregressive loop, KV cache, CUDA Graph experiments |
| 3 | `KernelAgent/3-micro-vllm` | paged KV cache, prefix cache, continuous batching, Green Contexts |

This structure makes the inference engine a sequence of inspectable learning stages rather than a monolithic serving black box.

### 5.3 micro-vLLM Serving Architecture

micro-vLLM uses a lightweight Qwen-family serving loop. The scheduler manages waiting and running sequences. The block manager allocates KV-cache blocks and performs hash-based prefix reuse. Each sequence tracks prompt tokens, generated tokens, block tables, and cached-token counts. The model runner separates prefill and decode preparation, including input IDs, positions, slot mappings, sequence lengths, and block tables.

For prefix cache, the scheduler checks complete-block hashes during sequence allocation. If an identical prefix block exists, `seq.num_cached_tokens` increases and the prefill path sends only the uncached suffix to the model. The attention context still reflects the full key length, so the model can attend over reused prefix KV blocks and newly computed suffix KV blocks.

For Green Contexts, the runtime records whether the requested path actually activates. PyTorch GreenContext was importable in the target environment but rejected context creation. The working path used `cuda.core` with `cuda.bindings.driver`, `Device.set_current(ctx)`, and `Device.set_current()` for restoration. The benchmark JSON records `green_enabled`, `green_api_type`, `green_split_layout_width`, and `green_prefill_resource_source` to prevent false performance claims when activation falls back.

### 5.4 Fixed-Context Prompt Layout

The benchmark uses the following prompt layout.

```text
[static system prompt]
[static policy / tool contract]
[static DB schema or course context]
[static examples or rubric]
[dynamic user question]
```

The static prefix is intentionally identical across requests. The dynamic suffix changes per user question. This represents Text-to-SQL over a fixed SQLite schema, CUDA tutoring over a fixed module, nano-vLLM tutoring over a fixed source excerpt, and retrieval agents over stable course or knowledge-bundle context.

## 6. Experimental Methodology

### 6.1 Hardware and Software

Experiments were conducted on a Windows 11 target PC with an NVIDIA GeForce RTX 5070 GPU. The device reports 48 SMs, approximately 12GB VRAM, and 48MB L2 cache in the local artifact metadata. Recent Green Context preflight output reports PyTorch 2.13.0+cu130 and CUDA 13.0. micro-vLLM uses a Qwen-family local model; the stress benchmark logs identify `Qwen3-0.6B` on the target PC.

### 6.2 Baseline Throughput Benchmark

The baseline throughput benchmark compares the Windows-native cuTile path against the WSL2 FlashAttention reference path on the same dynamic multi-user benchmark. This comparison is not intended to show Windows cuTile superiority; it anchors the artifact against a mature optimized stack.

### 6.3 Prefix Cache Benchmark

The prefix-cache benchmark uses `KernelAgent/3-micro-vllm/bench_prefix_cache.py`. For each static-prefix length, the benchmark runs three conditions.

| Condition | Description |
| :--- | :--- |
| `no_cache` | Clear the persistent hash table before each request and perform full prefill. |
| `warm_cache` | Prime the cache and then reuse the same static prefix. |
| `prefix_changed` | Change the prefix so exact-prefix cache hits collapse. |

Static prefix lengths are 1024, 2048, and 3072 tokens. The dynamic suffix is 64 tokens and generation length is 64 tokens. The benchmark records cache-hit ratio, cached tokens, computed prefill tokens, TTFT, end-to-end latency, decode ITL, and throughput.

### 6.4 Green Context Stress Benchmark

The Green Context stress benchmark uses `KernelAgent/3-micro-vllm/bench_green_stress.py`. It keeps one protected decode request active and repeatedly injects large prefill requests. The main metric is **protected decode completion gap**, which includes prefill-induced pauses between visible decode tokens in the current sequential engine loop.

Default stress workload:

| Parameter | Value |
| :--- | ---: |
| Protected decode prompt | 32 tokens |
| Protected decode output | 256 tokens |
| Interfering prefill prompt | 3072 tokens |
| Interfering prefill output | 1 token |
| Prefill injections | 12 |
| Injection cadence | every 8 protected decode steps, starting after 4 decode steps |
| Green split | prefill 32 SMs, decode 16 SMs |

The Green-side subprocess must report `green_enabled=true` and `green_api_type="cuda_core"` to be accepted as a valid intervention.

## 7. Results

### 7.1 WSL2 FlashAttention Reference vs Windows cuTile

The dynamic multi-user benchmark processed 133,966 generated tokens.

| Backend | Runs | Mean time | Mean throughput | Relative throughput |
| :--- | ---: | ---: | ---: | ---: |
| Windows cuTile | 3 | 289.16 s | 463.35 tok/s | 1.00x |
| WSL2 FlashAttention | 4 | 62.65 s | 2138.55 tok/s | 4.62x |

The result shows that the initial Windows-native cuTile backend is slower than the mature WSL2 FlashAttention path. This is not a weakness of the paper framing; it is the reason the contribution is positioned as a Windows-native implementation and performance-analysis case study rather than a production serving replacement.


### 7.2 cuTile Hybrid Prefill Dispatch and CUDA Graph Replay

After the initial cuTile baseline, we repaired the decode graph-capture path by replacing the PyTorch advanced-indexing KV-cache update with a cuTile KV-store kernel. We also introduced **hybrid prefill dispatch**: multi-sequence non-prefix prefill uses the batched padded wrapper to reduce host/kernel launch count, while single-sequence and prefix-cache prefill use direct or paged wrappers to avoid unnecessary tensor materialization.

| cuTile mode | CUDA Graph | Prefill strategy | Total time | Throughput | Interpretation |
| :--- | :--- | :--- | ---: | ---: | :--- |
| Initial cuTile mean | off | eager baseline | 289.16 s | 463.35 tok/s | Initial Windows cuTile reference |
| Hybrid eager | off | hybrid | 240.23 s | 557.66 tok/s | Current paper-positive cuTile result |
| Hybrid graph | on | hybrid | 340.60 s | 393.32 tok/s | Capture succeeds but throughput regresses |
| Padded graph | on | padded | 341.02 s | 392.84 tok/s | Confirms graph replay is the main bottleneck |
| Direct graph | on | direct | stopped early | 95-101 tok/s progress | Not viable for the 256-request benchmark |

The eager-hybrid cuTile path improves throughput by 20.4% over the initial cuTile mean. In contrast, CUDA Graph replay is functionally repaired but currently performance-negative. This is an important systems lesson: a standard optimization mechanism must align with graph bucket replay behavior, kernel granularity, and paged decode memory-access patterns to improve end-to-end serving.

#### CUDA Graph Failure-Cause Analysis

The CUDA Graph result should be interpreted as two separate failures. The first was an **activation failure**. The initial cuTile decode path still updated KV cache entries through PyTorch boolean masking and advanced indexing operations, including `slot_mapping[mask]` and `k_cache_flat[valid_slots] = ...`. That path invoked host-runtime and dynamic-indexing behavior during capture, producing a `cudaErrorStreamCaptureInvalidated` failure. Replacing it with a fixed cuTile `store_kvcache_cutile_kernel` repaired graph capture.

The second failure was a **performance failure**. Even after capture succeeded, the current cuTile decode graph does not yet match the dynamic serving workload. Graph buckets may replay larger fixed shapes than the active token set requires, producing overcompute. The decode path contains many small kernels, so launch-elimination benefits are limited by kernel granularity. Paged KV-cache decode also preserves irregular memory-access behavior; graph replay alone does not remove that memory bottleneck. The direct graph-prefill mode additionally increases per-sequence launch pressure during multi-request prefill and therefore became non-viable in the 256-request benchmark. For this paper, CUDA Graph should therefore be reported as an optimization that has been made functionally available, but not yet validated as a speedup.

### 7.3 Prefix KV Cache for Fixed-Context Agent Workloads

| Static prefix | Prompt tokens | Warm cache hit | Prefill no-cache | Prefill warm | Prefill reduction | TTFT no-cache | TTFT warm | TTFT reduction | Prefix-changed TTFT | E2E delta warm vs no-cache |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 1088 | 94.1% | 1088 | 64 | 94.1% | 146.98 ms | 86.56 ms | 41.1% | 148.18 ms | +34.2% |
| 2048 | 2112 | 97.0% | 2112 | 64 | 97.0% | 255.58 ms | 86.72 ms | 66.1% | 256.93 ms | +4.9% |
| 3072 | 3136 | 98.0% | 3136 | 64 | 98.0% | 381.76 ms | 77.47 ms | 79.7% | 389.61 ms | +1.2% |

Warm prefix cache reduces computed prefill tokens to 64 in all static-prefix settings. This means the static prefix is reused as complete KV blocks and only the 64-token dynamic suffix is newly computed. TTFT reduction grows with prefix length: 41.1% for 1024 tokens, 66.1% for 2048 tokens, and 79.7% for 3072 tokens.

The prefix-changed negative control returns 0.0% cache hit and TTFT close to the no-cache condition. This validates the mechanism: the improvement comes from exact token-prefix reuse rather than measurement noise.

End-to-end throughput does not consistently improve in this 64-token generation benchmark. This is expected because the decode loop still dominates total runtime after prefill. The correct claim is therefore **prefill-work and TTFT reduction for fixed-context workloads**, not universal throughput improvement.

### 7.4 Dynamic Shape Padding as a Negative Result

Dynamic shape padding was intended to reduce JIT shape variation. In practice, it caused temporary tensor allocation and copy overhead at each decode step, increasing PyTorch CUDA caching allocator pressure. The no-padding eager path achieved 470.82 tok/s, while the padded path achieved 155.16 tok/s, a 67.04% throughput decrease.

This negative result is pedagogically important. It shows that local optimization hypotheses must be evaluated inside the full serving loop, including allocation behavior and host-side runtime overhead.

### 7.5 Green Context Activation and Non-Stress Result

Initial Green Context logs were not valid efficacy measurements because the requested Green path silently fell back. The runtime was then instrumented with activation metadata. A valid cuda-core run recorded `green_enabled=true` and `green_api_type="cuda_core"` for 20/20 Green-side runs, with a 32/16 SM split and `green_prefill_resource_source="device_sm_fallback"`.

Under the non-stress paired benchmark, the valid Green run showed no stable serving-level benefit after accounting for one baseline P99 outlier. Excluding that outlier, TTFT changed by +0.49% on average, P99 ITL by +0.32%, and throughput by +2.06%. This motivates the stress workload rather than a general speedup claim.

### 7.6 Green Context Stress Result

Under repeated 3072-token prefill interference, cuda-core Green Contexts show moderate protected-decode smoothing.

| Metric | Baseline mean | Green mean | Mean delta | Median delta | Improved runs |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Decode step P50 | 50.56 ms | 47.37 ms | -5.38% | -5.97% | 14/20 |
| Decode step P95 | 66.33 ms | 62.89 ms | -4.83% | -5.41% | 13/20 |
| Decode step P99 | 71.56 ms | 68.42 ms | -4.30% | -3.39% | 18/20 |
| Decode gap P50 | 51.03 ms | 47.89 ms | -5.28% | -5.39% | 14/20 |
| Decode gap P95 | 79.45 ms | 75.33 ms | -4.98% | -5.77% | 16/20 |
| Decode gap P99 | 432.31 ms | 426.35 ms | -1.29% | -1.15% | 13/20 |
| Decode gap max | 441.38 ms | 437.94 ms | -0.70% | -0.87% | 14/20 |
| Throughput | 2098.17 tok/s | 2188.79 tok/s | +4.75% | +5.19% | 14/20 |

The strongest Green Context result is not TTFT. It is decode-step tail smoothing under adversarial prefill injection: decode-step P99 improves in 18/20 paired runs, and decode-gap P95 improves in 16/20 paired runs. However, decode-gap P99 and maximum gap remain close to baseline. This indicates that the current sequential Python engine still inserts prefill pauses that SM partitioning cannot fully hide. Green Contexts are therefore a **bounded resource-isolation mechanism** in this artifact, not a complete solution for serving latency.

## 8. Discussion

### 8.1 Why Fixed-Context Agents Matter

Agent workloads often pay repeated prefill cost for context that is logically constant. A Text-to-SQL agent repeatedly includes schema and rule context. A CUDA tutor repeatedly includes course material, code snippets, and rubrics. A knowledge agent repeatedly includes selected knowledge-bundle content. Prefix KV cache makes this repeated structure visible to the runtime.

The prefix-cache result therefore connects prompt organization to serving efficiency. Stable context should be placed at the beginning of the prompt, and volatile metadata should be moved later or excluded from the cached prefix. For educational and data-query agents, runtime-aware prompt layout can reduce TTFT without changing the model.

### 8.2 What the Green Context Result Teaches

Green Contexts teach a different lesson. They are not a general accelerator; they are a resource-partitioning substrate. The artifact demonstrates three important engineering points.

First, activation validity must be measured. Earlier Green runs were invalid because the runtime silently fell back. Second, resource partitioning alone does not guarantee latency improvement when the serving loop is sequential. Third, under a stress workload with repeated prefill interference, Green Contexts moderately smooth protected decode latency but do not remove prefill-induced pauses.

This is a useful educational result because it prevents a common mistake: attributing latency variance to a GPU feature without proving that the feature activated or that the scheduler creates the right interference pattern.

### 8.3 Educational Value of Negative Results

The paper intentionally includes negative and bounded results. WSL2 FlashAttention is much faster than the initial Windows cuTile path. Hybrid eager improves the cuTile path, but CUDA Graph replay hurts throughput under the current graph bucket and paged decode conditions. Dynamic padding hurts throughput. Green Contexts require activation metadata and only show bounded benefit in the current engine. These results make the artifact more useful for education because they show how systems claims should be formed: mechanism, instrumentation, control condition, measurement, and conservative interpretation.

### 8.4 Submission Scope

This paper should not include the broader ActiveGraph or Tau Coding Agent framework as a technical contribution. Those topics fit a separate educational knowledge-agent paper. Paper #1 should stay focused on micro-vLLM as an inference-engine artifact and on measured serving-loop behavior.

## 9. Threats to Validity

1. The experiments use a single RTX 5070 target PC and may not generalize to datacenter GPUs or other CUDA versions.
2. Even after the eager-hybrid cuTile improvement to 557.66 tok/s, the Windows-native cuTile backend is slower than WSL2 FlashAttention, so the paper must not claim production serving superiority.
3. Prefix-cache gains depend on exact token-prefix stability. Early prompt variation can collapse cache hits.
4. The prefix-cache benchmark uses 64 generated tokens, so decode dominates end-to-end runtime and limits throughput gains.
5. The Green Context stress result uses `device_sm_fallback` as the prefill resource source because the target `cuda.core` split returns a one-resource layout for the decode carve-out. The artifact records this metadata, but future profiling should confirm the precise SM residency behavior.
6. The Green Context benchmark still uses a sequential Python engine loop. A future asynchronous prefill/decode loop is needed to test stronger resource-isolation claims.
7. CUDA Graph replay is repaired but performance-negative in the current cuTile decode path; future work must analyze graph bucket overcompute and kernel granularity before using it as a speedup claim.
8. Nsight-level counters such as L2 hit rate, achieved occupancy, and kernel overlap are not yet included in the reported evidence.

## 10. Conclusion

This paper presents micro-vLLM as a **Windows-native LLM serving system implementation and performance analysis case study**. The artifact does not outperform mature Linux serving stacks: WSL2 FlashAttention achieves 4.62x higher average throughput than the initial Windows cuTile backend. However, after cuTile KV-store repair and hybrid prefill dispatch, the eager-hybrid cuTile path improves to 557.66 tok/s, a 20.4% gain over the initial cuTile mean. micro-vLLM therefore contributes an inspectable migration and experimentation path where students and researchers can observe complete serving-loop causality.

The strongest measured result is prefix KV-cache reuse for fixed-context agent workloads. Warm cache reduces computed prefill tokens to 64 across all tested static-prefix lengths and reduces TTFT by 41.1%, 66.1%, and 79.7% for 1024-, 2048-, and 3072-token prefixes. The changed-prefix negative control validates exact-prefix dependence.

The Green Context experiments provide a bounded systems insight. cuda-core Green Contexts activate successfully on the RTX 5070 target PC and moderately smooth protected decode latency under repeated prefill interference. Decode-step P99 improves by 4.3% on average and improves in 18/20 paired stress runs. However, decode-gap P99 and maximum gap remain dominated by sequential prefill insertion, showing that SM partitioning alone is insufficient without an asynchronous serving loop.

Together, these results support the paper's educational thesis: effective inference-engine education should teach not only successful optimizations, but also implementation tradeoffs, activation validity, negative results, workload dependence, and the distinction between isolated kernel behavior and end-to-end serving behavior.

## References

1. W. Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention," SOSP 2023.
2. T. Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness," NeurIPS 2022.
3. P. Tillet, H. T. Kung, and D. Cox, "Triton: an intermediate language and compiler for tiled neural network computations," MAPL 2019.
4. NVIDIA Corporation, "CUDA Python Documentation," NVIDIA CUDA Documentation.
5. NVIDIA Corporation, "Green Contexts," CUDA Programming Guide.
6. GeeeekExplorer, "nano-vLLM," GitHub repository.
7. vLLM Project, "Automatic Prefix Caching," vLLM Documentation.
8. PyTorch Foundation, "torch.cuda.green_contexts: Granular Resource Partitioning for CUDA Kernels," PyTorch Documentation.


