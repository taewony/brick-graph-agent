# Windows 네이티브 LLM 서빙 시스템 구현 및 성능 분석 사례연구: 마이크로-vLLM, CUDA Python, 하이브리드 프리필 디스패치, 프리픽스 KV 캐시, CUDA 그린 컨텍스트

## Windows-Native LLM Serving System Implementation and Performance Analysis Case Study: micro-vLLM, CUDA Python, Hybrid Prefill Dispatch, Prefix KV Cache, and CUDA Green Contexts

### 국문 요약

고성능 LLM 서빙은 FlashAttention·Triton·NCCL 등 리눅스 중심 스택이라 학습자가 프리필·디코드·KV 캐시·스케줄러·메모리 할당 상호작용을 관찰하기 어렵다. 본 논문은 Windows 네이티브 CUDA Python/cuTile 경로의 소형 교육 아티팩트 마이크로-vLLM 구현·성능 분석 사례연구다. WSL2 FlashAttention 경로는 2138.55 tok/s로 초기 Windows cuTile 경로(463.35 tok/s) 대비 4.62배 빠르다. cuTile KV-스토어 수리와 하이브리드 프리필 디스패치로 eager-hybrid 경로는 557.66 tok/s(20.4% 향상)에 도달했고, CUDA 그래프 재생은 성능이 하락하는 부정적 결과를 보였다. 고정 컨텍스트 에이전트 워크로드에서 웜 프리픽스 KV 캐시 재사용은 연산 프리필 토큰을 64로 줄이고 TTFT를 41.1%·66.1%·79.7% 감소시켰다(프리픽스 변경 음성대조 0% 히트). CUDA 그린 컨텍스트는 활성화 검증 후 보호 디코드 지연을 완화(부호검정 p=0.0004)하나 순차 프리필 중단은 제거하지 못한다. 이는 구현·측정·긍정·부정 결과를 모두 가르치는 서빙 교육의 근거다.

### 국문 키워드

LLM 서빙, CUDA Python, cuTile, 하이브리드 프리필 디스패치, 프리픽스 KV 캐시, CUDA 그린 컨텍스트

### 영문 요약 (Abstract)

High-performance LLM serving relies on Linux-centric stacks (FlashAttention, Triton, NCCL, vLLM-style schedulers), making it hard for students and Windows-based local researchers to inspect how prefill, decode, KV-cache management, scheduling, and allocation interact inside an engine. We present micro-vLLM as a Windows-native CUDA Python/cuTile educational artifact. The WSL2 FlashAttention reference reaches 2138.55 tok/s versus 463.35 tok/s for the initial Windows cuTile path (4.62x). After cuTile KV-store repair and hybrid prefill dispatch, the eager-hybrid path reaches 557.66 tok/s (+20.4%); CUDA Graph replay is functionally repaired but performance-negative. For fixed-context agent workloads, warm prefix KV-cache reuse reduces computed prefill tokens to 64 and TTFT by 41.1%, 66.1%, and 79.7% for 1024/2048/3072-token prefixes, validated by a changed-prefix negative control (0% hit). CUDA Green Contexts, after activation validation, significantly smooth protected decode latency (sign test p=0.0004) but do not eliminate sequential prefill pauses. These results support teaching implementation, measurement, positive optimizations, and negative results in the full serving loop.

### 영문 키워드

LLM serving, CUDA Python, cuTile, hybrid prefill dispatch, prefix KV cache, CUDA Green Contexts

---

## 1. 서론 (Introduction)

Local LLM inference is increasingly relevant in classrooms, laboratories, and offline settings where learners must understand how prompts become prefill work, how KV-cache blocks are allocated, and why decode is latency-sensitive. Production systems such as vLLM, TensorRT-LLM, and FlashAttention-based stacks are effective but large, Linux-oriented, and tightly coupled to optimized dependencies, which limits their value as first educational artifacts.

We therefore target a different contribution: **a Windows-native LLM serving implementation and performance-analysis case study** as an educational artifact. micro-vLLM is not a faster replacement for vLLM; it exposes the serving mechanisms learners need to inspect — prefill, decode, paged KV cache, prefix reuse, CUDA context activation, scheduling, allocation overhead, and tail latency. Its value is turning serving behavior into reproducible evidence rather than hiding it behind a production stack.

A central workload is the **fixed-context agent workload**: Text-to-SQL agents repeatedly prepend database schemas and rule context; tutoring agents repeat course material; knowledge agents repeat bundle excerpts. Such workloads suit prefix KV-cache reuse because the expensive static prefix can be reused across requests.

**Code availability.** The micro-vLLM artifact is organized as a staged migration path (`0-MatMul`, `1-FMHA`, `2-LLM-from-scratch`, `3-micro-vllm`). The benchmark drivers are `bench_prefix_cache.py` and `bench_green_stress.py`. The repository is publicly available at: `[repository URL redacted for double-blind review — provided in the camera-ready version]`. Experiments ran on Windows 11, NVIDIA GeForce RTX 5070 (48 SMs, ≈12GB VRAM, 48MB L2), PyTorch 2.13.0+cu130, CUDA 13.0, Qwen3-0.6B.

## 2. 연구 질문 (Research Questions)

- **RQ1.** Can a Windows-native CUDA Python/cuTile micro-vLLM artifact expose the main LLM serving mechanisms in a form suitable for education and controlled experimentation?
- **RQ2.** For fixed-context agent workloads, how much does prefix KV-cache reuse reduce computed prefill tokens and time-to-first-token?
- **RQ3.** How do hybrid prefill dispatch and cuTile KV-store repair affect Windows-native cuTile serving throughput?
- **RQ4.** What do bounded and negative results (CUDA Graph, Green Contexts, dynamic shape padding) teach about full serving-loop causality?

## 3. 기여 (Contributions)

1. A Windows-native educational migration artifact (staged from tiled MatMul and fused attention to a nano-vLLM-style engine).
2. A fixed-context agent benchmark bridging inference mechanisms with Text-to-SQL, tutoring, and knowledge-agent scenarios.
3. Measured prefix KV-cache evidence: warm cache reduces computed prefill tokens to 64 and TTFT by up to 79.7%, with a changed-prefix negative control.
4. Hybrid prefill dispatch and cuTile KV-store repair improving the cuTile path from 463.35 to 557.66 tok/s.
5. A serving-loop causality analysis reporting bounded and negative results (WSL2 FlashAttention 4.62x faster; CUDA Graph performance-negative; dynamic padding −67.04%; Green Contexts bounded smoothing).

## 4. 시스템 설계 (System Design)

micro-vLLM follows four goals: **inspectability**, **modifiability**, **Windows-native execution**, and **agent-workload relevance**. The artifact is staged as a learning path:

| Stage | Folder | Educational role |
| :--- | :--- | :--- |
| 0 | `0-MatMul` | tiling, shared memory, swizzling, GEMM baseline |
| 1 | `1-FMHA` | online softmax, causal masking, fused attention |
| 2 | `2-LLM-from-scratch` | minimal autoregressive loop, KV cache, CUDA Graph |
| 3 | `3-micro-vllm` | paged KV cache, prefix cache, continuous batching, Green Contexts |

The serving loop uses a lightweight Qwen-family model. A scheduler manages waiting/running sequences; a block manager allocates KV-cache blocks and performs hash-based prefix reuse; the model runner separates prefill and decode preparation. For prefix cache, the scheduler checks complete-block hashes during allocation: identical prefixes raise `seq.num_cached_tokens` and send only the uncached suffix to the model, while attention still reflects the full key length. For Green Contexts, the runtime records activation metadata (`green_enabled`, `green_api_type`, `green_split_layout_width`, `green_prefill_resource_source`) to prevent false claims when activation falls back.

The benchmark prompt layout places static context before dynamic content:

```text
[static system prompt] [static policy / tool contract] [static DB schema or course context]
[static examples or rubric] [dynamic user question]
```

## 5. 실험 방법 (Experimental Methodology)

**Metrics.** Throughput (tok/s) is total generated tokens divided by wall-clock time. TTFT is the latency from request arrival to the first token; ITL is per-token decode latency; "computed prefill tokens" counts only tokens newly computed after prefix reuse.

Three benchmarks were used. (1) The **baseline throughput benchmark** compares the Windows cuTile path against the WSL2 FlashAttention reference on a dynamic multi-user benchmark. (2) The **prefix-cache benchmark** (`bench_prefix_cache.py`) tests `no_cache`, `warm_cache`, and `prefix_changed` conditions at 1024/2048/3072-token static prefixes, with a 64-token dynamic suffix and 64-token generation. (3) The **Green Context stress benchmark** (`bench_green_stress.py`) keeps one protected decode request active while injecting twelve 3072-token prefills; a valid run must report `green_enabled=true` and `green_api_type="cuda_core"`.

## 6. 결과 (Results)

### 6.1 Baseline: WSL2 FlashAttention vs Windows cuTile

The dynamic benchmark processed 133,966 generated tokens.

| Backend | Runs | Mean time | Mean throughput | Relative |
| :--- | ---: | ---: | ---: | ---: |
| Windows cuTile | 3 | 289.16 s | 463.35 tok/s | 1.00x |
| WSL2 FlashAttention | 4 | 62.65 s | 2138.55 tok/s | 4.62x |

The Windows-native path is slower; this anchors the contribution as an implementation and analysis case study rather than a production replacement.

### 6.2 Hybrid Prefill Dispatch and CUDA Graph Replay

Replacing the PyTorch advanced-indexing KV-cache update with a cuTile KV-store kernel repaired graph capture; hybrid prefill dispatch routes multi-sequence prefill to a batched padded wrapper and single-sequence/prefix-cache prefill to direct or paged wrappers.

| cuTile mode | Graph | Prefill | Total time | Throughput | Interpretation |
| :--- | :--- | :--- | ---: | ---: | :--- |
| Initial mean | off | eager | 289.16 s | 463.35 tok/s | baseline |
| Hybrid eager | off | hybrid | 240.23 s | 557.66 tok/s | paper-positive (+20.4%) |
| Hybrid graph | on | hybrid | 340.60 s | 393.32 tok/s | capture succeeds, throughput regresses |
| Padded graph | on | padded | 341.02 s | 392.84 tok/s | graph replay is the bottleneck |
| Direct graph | on | direct | stopped early | 95–101 tok/s | non-viable for 256 requests |

CUDA Graph failure decomposes into an **activation failure** (dynamic-indexing KV updates broke capture with `cudaErrorStreamCaptureInvalidated`, fixed by the cuTile store kernel) and a **performance failure** (graph buckets replay larger fixed shapes than the active token set; many small kernels limit launch-elimination benefit; paged decode keeps irregular memory access). It is reported as functionally available but not yet a speedup.

### 6.3 Prefix KV Cache for Fixed-Context Workloads

| Prefix | Prompt tokens | Warm hit | Prefill no-cache | Prefill warm | TTFT no-cache | TTFT warm | TTFT reduction | Changed-prefix TTFT |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 1088 | 94.1% | 1088 | 64 | 146.98 ms | 86.56 ms | 41.1% | 148.18 ms |
| 2048 | 2112 | 97.0% | 2112 | 64 | 255.58 ms | 86.72 ms | 66.1% | 256.93 ms |
| 3072 | 3136 | 98.0% | 3136 | 64 | 381.76 ms | 77.47 ms | 79.7% | 389.61 ms |

Warm cache reduces computed prefill tokens to 64 across all prefix lengths; the changed-prefix control returns 0% hit and no-cache-level TTFT, validating exact-prefix dependence. End-to-end throughput does not consistently improve because decode dominates the 64-token generation benchmark — the correct claim is prefill-work and TTFT reduction, not universal throughput gain.

### 6.4 Dynamic Shape Padding (Negative)

Padding increased PyTorch CUDA caching-allocator pressure via per-step temporary tensors, dropping throughput from 470.82 to 155.16 tok/s (−67.04%). Local hypotheses must be evaluated inside the full serving loop, including allocation behavior and host overhead.

### 6.5 Green Context Activation and Non-Stress Result

A valid `cuda_core` run recorded `green_enabled=true` and `green_api_type="cuda_core"` for 20/20 Green-side runs (32/16 SM split). Under the non-stress paired benchmark there was no stable serving-level benefit (+0.49% TTFT, +0.32% P99 ITL, +2.06% throughput after removing one P99 outlier), motivating the stress workload.

### 6.6 Green Context Stress Result

| Metric | Baseline | Green | Mean delta | Median delta | Improved runs |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Decode step P99 | 71.56 ms | 68.42 ms | −4.30% | −3.39% | 18/20 |
| Decode gap P95 | 79.45 ms | 75.33 ms | −4.98% | −5.77% | 16/20 |
| Decode step P50 | 50.56 ms | 47.37 ms | −5.38% | −5.97% | 14/20 |
| Throughput | 2098.17 tok/s | 2188.79 tok/s | +4.75% | +5.19% | 14/20 |

A two-sided binomial sign test over the 20 paired runs confirms the two headline smoothing effects (decode-step P99, 18/20, p=0.0004; decode-gap P95, 16/20, p=0.012), while throughput and decode-step P50/P95 differences are not significant at α=0.05 (14/20, 13/20). Decode-gap P99 and maximum gap remain near baseline: SM partitioning alone cannot hide pauses inserted by the sequential Python engine loop.

## 7. 논의 (Discussion)

**Why fixed-context agents matter.** Agent workloads repeatedly pay prefill cost for logically constant context. Prefix KV cache makes this structure visible, so runtime-aware prompt layout (static context first, volatile metadata later) can reduce TTFT without changing the model.

**What Green Contexts teach.** They are a resource-partitioning substrate, not a general accelerator. Three points follow: activation must be measured (earlier runs silently fell back); partitioning alone does not help a sequential loop; under stress they moderately smooth protected decode but do not remove prefill pauses.

**Educational value of negative results.** WSL2 FlashAttention is much faster; CUDA Graph regresses under current buckets; padding hurts; Green Contexts are bounded. These teach how systems claims should be formed — mechanism, instrumentation, control condition, measurement, and conservative interpretation.

**Submission scope.** This paper excludes broader agent-framework contributions; it remains focused on micro-vLLM as an inference-engine artifact and measured serving-loop behavior.

## 8. 타당성 위협 (Threats to Validity)

1. Single RTX 5070 target PC; results may not generalize to datacenter GPUs or other CUDA versions.
2. The Windows cuTile backend remains slower than WSL2 FlashAttention; no production-serving superiority is claimed.
3. Prefix-cache gains depend on exact token-prefix stability.
4. The prefix benchmark uses 64 generated tokens, so decode dominates end-to-end runtime and limits throughput gains.
5. Green stress uses `device_sm_fallback`; future profiling should confirm precise SM residency.
6. The Green benchmark uses a sequential Python engine loop; an asynchronous prefill/decode loop is needed for stronger isolation claims.
7. CUDA Graph is repaired but performance-negative; graph-bucket overcompute and kernel granularity require further analysis.
8. Nsight-level counters (L2 hit rate, occupancy, kernel overlap) are not yet included.
9. Throughput comparisons use 3–4 runs without reported within-backend variance; the +20.4% cuTile gain should be treated as a single-seed engineering observation until repeated measurements with standard deviations are available.

## 9. 결론 (Conclusion)

micro-vLLM is presented as a Windows-native LLM serving implementation and performance-analysis case study. WSL2 FlashAttention is 4.62x faster than the initial Windows cuTile path; after cuTile KV-store repair and hybrid prefill dispatch, the eager-hybrid path reaches 557.66 tok/s (+20.4%). The strongest result is prefix KV-cache reuse for fixed-context agents: computed prefill tokens drop to 64 and TTFT falls by 41.1%, 66.1%, and 79.7%, validated by a changed-prefix negative control. CUDA Green Contexts activate and significantly smooth protected decode latency (decode-step P99 sign test p=0.0004) but do not eliminate sequential prefill pauses. Effective inference-engine education should teach successful optimizations, implementation tradeoffs, activation validity, negative results, and the distinction between isolated kernel behavior and end-to-end serving behavior.

## 참고문헌 (References)

1. W. Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention," in Proceedings of ACM SOSP, 2023.
2. T. Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness," in Advances in Neural Information Processing Systems (NeurIPS), 2022.
3. P. Tillet, H. T. Kung, and D. Cox, "Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations," in Proceedings of MAPL, 2019.
4. NVIDIA Corporation, "CUDA Python Documentation," NVIDIA CUDA Documentation.
5. NVIDIA Corporation, "Green Contexts," CUDA Programming Guide.
6. vLLM Project, "Automatic Prefix Caching," vLLM Documentation.
7. PyTorch Foundation, "torch.cuda.green_contexts: Granular Resource Partitioning for CUDA Kernels," PyTorch Documentation.
8. GeeeekExplorer, "nano-VLLM," GitHub repository.
9. micro-vLLM artifact repository, "KernelAgent," GitHub repository — `[URL redacted for double-blind review]`.

## 부록 (Appendix)

### A.1 Prefix-Cache Benchmark Conditions

| Condition | Description |
| :--- | :--- |
| `no_cache` | Clear the persistent hash table before each request and perform full prefill. |
| `warm_cache` | Prime the cache, then reuse the same static prefix. |
| `prefix_changed` | Change the prefix so exact-prefix cache hits collapse. |

### A.2 Green Context Stress Workload

| Parameter | Value |
| :--- | ---: |
| Protected decode prompt | 32 tokens |
| Protected decode output | 256 tokens |
| Interfering prefill prompt | 3072 tokens |
| Interfering prefill output | 1 token |
| Prefill injections | 12 |
| Injection cadence | every 8 decode steps, after step 4 |
| Green split | prefill 32 SMs, decode 16 SMs |
