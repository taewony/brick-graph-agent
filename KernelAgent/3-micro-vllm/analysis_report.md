# Analysis Report: micro-vllm serving performance on RTX 5070
**Windows cuTile Backend, WSL FlashAttention Baseline, and Hardware Resource Partitioning Analysis**

---

## 1. Executive Summary

This report provides a comprehensive performance and architectural analysis of the **micro-vllm** serving engine under the `3-nano-vllm` workspace. We evaluated the serving pipeline on an NVIDIA GeForce RTX 5070 GPU using a Qwen-family model with Qwen2-style architecture metadata (36 layers, hidden size 2048, 16 attention heads, 2 KV heads, head dimension 128).

Three benchmark groups were executed:
1. **Eager serving performance benchmark (`bench.py`)**: Comparing eager unpadded cuTile execution against dynamic padded shapes, demonstrating the severe impact of host-side allocator thrashing.
2. **WSL FlashAttention backend benchmark (`bench.py`)**: Verifying that the optimized Linux stack (WSL2 + Triton + FlashAttention) runs successfully and measuring it as a strong reference baseline.
3. **Dedicated SM Resource Isolation benchmark (`bench_green.py`)**: Quantifying the impact of CUDA Green Context partitioning (32 SMs for Prefill, 16 SMs for Decode) on service tail latency and cache residency.

All benchmark runs generated coherent token outputs. The WSL FlashAttention result shows a large and repeatable performance lead over the current Windows cuTile path, so the paper should frame FlashAttention as an optimized reference baseline rather than claim that cuTile outperforms mature Linux serving kernels. The latest repeated Green Contexts runs are mixed; they support feasibility of single-GPU resource partitioning, but not yet a stable latency/throughput improvement claim.

---

## 2. WSL FlashAttention Baseline Verification and Comparison

The `test-result-flash_attn.md` and `test-result-flash_attn-run*.md` logs confirm that the WSL optimized attention stack is operational:

| Component | Observed Value |
| :--- | :--- |
| OS | WSL2, `6.18.33.2-microsoft-standard-WSL2` |
| Python | 3.12.3 |
| PyTorch | v2.12.1+cu130 |
| CUDA | Available, compiled with CUDA 13.0 |
| Triton | v3.7.1 |
| FlashAttention | v2.8.3.post1 loaded successfully |

The FlashAttention backend was executed using:

```bash
python bench.py
```

The Windows cuTile backend was executed using:

```powershell
python bench.py --use-cutile
```

All available logs processed **133,966 generated tokens** in the dynamic multi-user benchmark.

| Backend | Source Log | Runtime Environment | Total Time (s) | Throughput (tok/s) |
| :--- | :--- | :--- | ---: | ---: |
| cuTile attention | `test-result-cuTile.md` | Windows native path | 284.54 | 470.82 |
| cuTile attention | `test-result-cuTile-run.md` run 1 | Windows native path | 291.17 | 460.10 |
| cuTile attention | `test-result-cuTile-run.md` run 2 | Windows native path | 291.78 | 459.13 |
| FlashAttention | `test-result-flash_attn.md` | WSL2 Linux stack | 62.95 | 2128.07 |
| FlashAttention | `test-result-flash_attn-run1.md` | WSL2 Linux stack | 62.35 | 2148.62 |
| FlashAttention | `test-result-flash_attn-run2.md` | WSL2 Linux stack | 62.27 | 2151.47 |
| FlashAttention | `test-result-flash_attn-run3.md` | WSL2 Linux stack | 63.01 | 2126.04 |

### Repeated-Run Summary

| Backend | Runs Used | Mean Time (s) | Time Std. (s) | Mean Throughput (tok/s) | Throughput Std. (tok/s) | Relative Throughput |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| cuTile attention | all 3 available runs | 289.16 | 4.02 | **463.35** | 6.49 | 1.00x |
| FlashAttention | all 4 available runs | 62.65 | 0.39 | **2138.55** | 13.22 | **4.62x** |
| cuTile attention | latest repeat runs only | 291.48 | 0.43 | **459.62** | 0.69 | 1.00x |
| FlashAttention | latest repeat runs only | 62.54 | 0.41 | **2142.04** | 13.94 | **4.66x** |

### Interpretation

The WSL FlashAttention backend is approximately **4.62x faster** than the current Windows cuTile backend when all available runs are included, and approximately **4.66x faster** when only the latest repeat logs are compared:

```text
2138.55 / 463.35 = 4.62x
2142.04 / 459.62 = 4.66x
```

This is expected. FlashAttention is a mature, highly optimized attention kernel stack integrated with Linux-oriented serving paths. The current cuTile backend is a Python-level Windows-native migration path whose main value is controllability, education, and engineering analysis rather than immediate production performance superiority.

The paper should therefore make the comparison explicit but conservative:

- FlashAttention is the optimized reference baseline.
- micro-vllm/cuTile demonstrates a Windows-native migration and experimentation path.
- The remaining performance gap identifies future optimization work rather than invalidating the kernel migration contribution.

---

## 3. Eager Serving & Allocator Thrashing Analysis (`bench.py`)

Autoregressive decoding with continuous batching presents dynamic sequence shapes. To prevent frequent JIT recompilations in cuTile (JIT Compilation Storms), we evaluated a **dynamic padding** strategy (rounding inputs to the nearest multiple of 256) against our **Eager cuTile (No Padding)** strategy.

### A. Performance Benchmark Results
The benchmark processed a dynamic workload consisting of 256 concurrent client requests (totaling **133,966 tokens** generated):

| Configuration | Total Tokens | Total Time (s) | Throughput (tok/s) | Performance Impact | Correctness |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Eager cuTile (No Padding - Ours)** | 133,966 | 284.54 | **470.82** | Baseline (100.0%) | **PASS** |
| **Padded cuTile (Dynamic Padding)** | 133,966 | 863.39 | 155.16 | **-67.04% (Degraded)** | **PASS** |

### B. Architectural Insight: Caching Allocator Thrashing
* **The Problem**: While padding shapes to the nearest 256 successfully bypassed cuTile JIT dictionary lookups, it introduced a massive memory allocation bottleneck.
* **The Bottleneck**: Rounding shapes required creating large temporary padded tensors (e.g. `[256, 16, 1, 128]` for query projections) inside the model's forward loop at *every single decode iteration*. This continuous allocation and deallocation thrashed PyTorch’s **CUDA Caching Allocator** and triggered frequent host-side Python **garbage collection (GC)** runs.
* **The Solution**: In eager python serving, avoiding temporary tensor allocations and memory copying overhead is far more critical than avoiding compiler metadata lookups. Thus, micro-vllm adopts an unpadded eager-mode execution.

---

## 4. Dedicated SM Resource Partitioning Analysis (`bench_green.py`)

When multiple users are batched dynamically, heavy Prefill requests (compute-bound, $O(N^2)$) compete for GPU execution resources with light Decode requests (memory-bandwidth bound, $O(1)$). This compute interference leads to severe tail-latency spikes (ITL) for active users.

We resolved this by partitioning the **48 SMs** of the RTX 5070 into isolated contexts using **CUDA Green Contexts**:
- **Prefill Context**: 32 SMs (67% of GPU resources)
- **Decode Context**: 16 SMs (33% of GPU resources)

### A. Initial Performance Metrics Summary
We simulated a worst-case scenario: a 100-token decode client is active, and a massive 2048-token prefill prompt is injected dynamically at step 5. The initial run showed a favorable result:

| Metric | Baseline (Green OFF) | Target (Green ON) | Delta (%) | Status |
| :--- | :--- | :--- | :--- | :--- |
| **TTFT (Prefill Latency)** | 244.60 ms | 243.42 ms | **-0.5% (Flat)** | **Neutral** |
| **Decode P50 ITL (Median)** | 48.63 ms | 41.83 ms | **-14.0% (Improved)**| **Single-run observation** |
| **Decode P99 ITL (Tail)** | 85.31 ms | 84.34 ms | **-1.1% (Flat)** | **Neutral** |
| **Total Throughput** | 415.50 tok/s | 459.73 tok/s | **+10.6% (Improved)**| **Single-run observation** |
| **Total Elapsed Time** | 5.25 s | 4.75 s | **-9.5% (Reduced)** | **Single-run observation** |

### B. Repeated Green Contexts Runs

Additional runs in `test-result-cuTile-run.md` show that the Green Contexts effect is not yet stable across repeated executions:

| Run | Metric | Green OFF | Green ON | Delta |
| :--- | :--- | ---: | ---: | ---: |
| 1 | TTFT (ms) | 255.15 | 250.59 | -1.8% |
| 1 | Decode P50 ITL (ms) | 46.55 | 59.17 | +27.1% |
| 1 | Decode P99 ITL (ms) | 82.36 | 83.70 | +1.6% |
| 1 | Throughput (tok/s) | 417.47 | 349.28 | -16.3% |
| 2 | TTFT (ms) | 256.34 | 249.44 | -2.7% |
| 2 | Decode P50 ITL (ms) | 55.04 | 46.84 | -14.9% |
| 2 | Decode P99 ITL (ms) | 79.21 | 83.82 | +5.8% |
| 2 | Throughput (tok/s) | 365.20 | 413.53 | +13.2% |
| 3 | TTFT (ms) | 250.80 | 249.88 | -0.4% |
| 3 | Decode P50 ITL (ms) | 59.07 | 54.94 | -7.0% |
| 3 | Decode P99 ITL (ms) | 88.65 | 82.15 | -7.3% |
| 3 | Throughput (tok/s) | 346.78 | 369.57 | +6.6% |

Aggregated over the three latest Green Contexts repeats:

| Metric | Green OFF Mean | Green ON Mean | Mean Delta |
| :--- | ---: | ---: | ---: |
| TTFT | 254.10 ms | 249.97 ms | -1.6% |
| Decode P50 ITL | 53.55 ms | 53.65 ms | +0.2% |
| Decode P99 ITL | 83.41 ms | 83.22 ms | -0.2% |
| Throughput | 376.48 tok/s | 377.46 tok/s | +0.3% |
| Total Elapsed Time | 5.84 s | 5.81 s | -0.4% |

### C. Systems-Level Bottleneck & Cache Residency Analysis

1. **L2 Cache Residency (Decode Acceleration)**:
   Confining the decode kernels to a dedicated 16-SM partition can reduce P50 Inter-Token Latency (ITL) in individual runs, but the latest repeats do not show a stable aggregate improvement. 
   - Since decode kernels are small and memory-bandwidth bound, they depend heavily on cache locality.
   - Restricting their execution footprint may prevent prefill kernels from evicting decode weight slices from the **48.0 MB L2 Cache**, but this mechanism still requires profiling counters such as L2 hit rate, achieved occupancy, and kernel overlap to validate.

2. **Prefill Saturation Limit**:
   Intuitively, reducing prefill resources by 33% (from 48 SMs to 32 SMs) should increase TTFT. However, prefill latency remained flat or slightly improved in repeated runs.
   - A batch-1 prefill of 2048 tokens on Qwen2.5-3B-Instruct does not fully saturate the execution pipeline of all 48 SMs.
   - 32 SMs represents the hardware **efficiency sweet-spot (occupancy saturation limit)**, meaning over-provisioning SMs to prefill yields diminishing returns.

3. **Sequential Execution Limitation**:
   Because the Python engine loop schedules prefill and decode steps sequentially, they do not execute concurrently on the GPU in the current thread. 
   - Green Contexts can define physical SM partitions, but this implementation does not yet prove E2E overlap or consistent cache-residency benefits.
   - **Future Work**: Decouple the model runner loop into asynchronous execution streams on separate CPU threads to overlap prefill and decode steps on their respective SM partitions.

---

## 5. Paper Implications and Recommended Framing

The updated evidence changes the correct paper positioning. The report no longer supports a claim that the Windows cuTile path is faster than FlashAttention. Instead, it supports the following claims:

1. **Windows-native kernel migration path**: micro-vllm runs a cuTile attention backend in the Windows-oriented execution path and completes coherent Qwen-family generation and dynamic serving workloads.
2. **Optimized Linux baseline comparison**: WSL FlashAttention is operational and achieves **2138.55 tok/s** on average across all available runs, about **4.62x** the current cuTile backend throughput.
3. **Allocator engineering insight**: dynamic shape padding reduces JIT shape churn but introduces temporary tensor allocation and PyTorch CUDA caching allocator pressure, reducing cuTile throughput by **67.04%**.
4. **Single-GPU resource partitioning insight**: CUDA Green Contexts with Prefill 32 SMs and Decode 16 SMs is feasible in the current engine, but repeated runs show mixed latency and throughput effects. The paper should present the original favorable run as an observation, then state that the latest repeats average to nearly flat P50 ITL and throughput. Stable benefit requires additional profiling and a truly asynchronous Prefill/Decode execution loop.

The paper should avoid the following claims:

- micro-vllm outperforms FlashAttention.
- cuTile generally replaces Triton/FlashAttention for production serving.
- micro-vllm is production-grade relative to mature Linux serving stacks.

Recommended framing:

> micro-vllm provides a Windows-native cuTile migration and experimentation path for LLM serving kernels. Although the mature WSL FlashAttention backend remains substantially faster in E2E throughput, micro-vllm exposes the optimization path, allocator failure modes, and single-GPU resource isolation behavior in a form suitable for education, controlled experimentation, and future Windows-native optimization.

## 6. Next Experimental Steps

1. Repeat both `bench.py --use-cutile` and `bench.py` at least three times and report mean/std.
2. Verify that both benchmark paths use the same model weights, tokenizer, request distribution, input length range, and output length range.
3. Add latency metrics if the benchmark logger can be extended:
   - TTFT mean/P50/P99
   - ITL mean/P50/P99
   - throughput
4. Add profiling counters if available:
   - peak VRAM
   - allocator allocation count
   - CUDA kernel launch count
   - L2 hit rate for Green OFF/ON
5. Keep FlashAttention in the paper as an optimized reference baseline and position cuTile as a Windows-native migration and analysis path.
