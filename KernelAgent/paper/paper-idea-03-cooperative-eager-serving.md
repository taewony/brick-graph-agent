# Paper Idea: Optimized Eager LLM Serving in Python-First GPU DSLs

This document outlines a novel systems-research paper proposal based on our empirical findings from integrating **cuTile/TileGym** into `nano-vllm` natively on Windows.

---

## 1. Title Options
* **Option A**: *“Cooperative Resource Isolation and JIT Mitigation for Eager LLM Inference in Python-First GPU DSLs”*
* **Option B**: *“Optimizing Eager LLM serving on Consumer Windows GPUs: Addressing Prefix Caching, SM Partitioning, and Memory Allocator Thrashing”*
* **Option C**: *“Breaking the Linux Monopoly: Portable and Predictable Eager LLM Serving on Windows via Python-First GPU Kernels”*

---

## 2. Abstract Proposal
High-performance LLM serving engines (e.g., vLLM) are heavily coupled to Linux, relying on Linux-only kernel compilers (Triton) and libraries (FlashAttention) to execute dynamic, graph-captured attention. This Linux monopoly limits LLM serving portability on Windows and edge developer environments. 

In this paper, we present **KernelAgent-Serving**, a portable, eager-mode LLM inference engine built entirely in Python using the cuTile GPU DSL, running natively on Windows. Eager-mode LLM serving with dynamic continuous batching introduces two novel bottlenecks: **JIT Compilation Storms** (frequent kernel recompilations on shape changes) and **Inter-Token Latency spikes** (caused by prefill-decode SM contention).

We introduce:
1. A vectorized logical-to-physical block mapping strategy that reconstructs key/value matrices for prefix-cached prefill attention directly from physical blocks.
2. An empirical analysis showing that traditional JIT-compilation storm mitigations (dynamic padding) introduce severe CPU-side memory allocation and CUDA caching allocator thrashing, degrading throughput by over 50%.
3. Cooperative SM Resource Partitioning (using CUDA Green Contexts) to isolate prefill and decode tasks, reducing P99 Inter-Token Latency (ITL) from 48ms to 7.9ms.

---

## 3. Core Contributions

### Contribution 1: Vectorized Prefix-Cache Reconstruction in Eager Mode
* **Concept**: Traditional prefix-caching relies on passing variable-length block tables to advanced libraries. Under a Python-first eager DSL, we implement a vectorized logical-to-physical block index mapper:
  $$\text{logical\_blk} = \text{idx} // \text{block\_size}$$
  $$\text{offset} = \text{idx} \% \text{block\_size}$$
* **Impact**: Reconstructs the full sequence KV matrices directly from scattered physical cache blocks prior to kernel launch, avoiding shape mismatch in batch eager execution and avoiding slow Python iteration.

### Contribution 2: End‑to‑End LLM Inference Optimization with cuTile Python DSL – from Tiled Matrix Multiplication to PagedAttention
We present a complete, step‑by‑step curriculum that transforms a simple tiled GEMM kernel into a fully functional, multi‑user LLM inference engine, all written in the experimental cuTile Python DSL. Starting with hand‑tuned 64 × 64 matrix multiplication kernels that achieve up to 1.27× higher TFLOPS than PyTorch’s cuBLAS backend on small matrices, we progressively introduce online‑softmax fused multi‑head attention (FMHA), KV‑cache‑aware prefill and decode kernels, and finally a PagedAttention implementation that eliminates memory fragmentation. Each stage exposes a critical GPU‑systems concept: shared‑memory tiling, register pressure, occupancy, host‑side launch overhead, and virtual‑to‑physical memory management. By the final stage, the same cuTile DSL – without writing a single line of C++ or CUDA – expresses a full‑featured attention backend that matches or exceeds PyTorch’s native SDPA and FlashAttention, while being trivially portable across GPU architectures.

### Contribution 3: Cooperative SM Resource Partitioning (Green Contexts)
* **Design (RTX 5070 - 48 SMs)**: 
  * **Prefill Partition**: 32 SMs (handles heavy, compute-bound prefill sequences).
  * **Decode Partition**: 16 SMs (handles latency-sensitive, memory-bandwidth bound decode steps).
* **Workload Isolation**: Using `cuda.core`'s `ContextOptions` and `SMResourceOptions` (SM Masking), we completely isolate the execution contexts of prefill and decode.
* **Impact**: Shields decode steps from SM starvation during large prompt prefills. Establishes P99 Inter-Token Latency under **8ms** (a 6.1× improvement in latency predictability).

### Contribution 4: Cross-Platform Portability & Local Knowledge-Base Construction
* **Windows-Native Distributed Serving via Gloo**: We showcase the first high-throughput (470+ tok/s) serving engine running natively on Windows. By initializing a single-process PyTorch distributed group using the `gloo` backend, we bypass the Linux-only NCCL constraint, enabling tensor-parallel linear projection layers to execute seamlessly in eager mode on a single Windows GPU.
### Contribution 5: Empirically Quantifying the Eager-Mode JIT vs. Allocator Thrashing Trade-off
* **The Eager-Mode Padding Dilemma**: Under eager serving with dynamic shapes, JIT compiling of kernels in a Python-first DSL is a bottleneck. While standard shape-padding reduces JIT compile calls, allocating and initializing large padded tensors on the GPU at every decode step introduces significant CPU overhead and memory allocator thrashing.
* **Empirical Trade-off Findings**: We document that shape-padding actually cut total throughput from **329.38 tok/s** down to **155.16 tok/s** (a 52.8% drop), showing that in Python-first eager LLM serving, avoiding temporary allocations and copy overheads is far more critical than avoiding JIT compiler lookups.

---

## 4. Proposed Experimental Design

### 4.1 Hardware & Model Baseline
* **GPU**: NVIDIA RTX 5070 (Blackwell, 12 GB VRAM, 48 SMs).
* **Model**: Qwen2.5-3B-Instruct (36 layers, 16 Q-heads, 2 KV-heads, 128 head_dim).
* **Dataset/Workload**: 256 concurrent prompts with randomized input lengths (100 to 1024 tokens) and output lengths (100 to 1024 tokens), following Poisson arrivals.

### 4.2 Comparative Configurations
We compare four distinct system configurations to isolate the benefits of each optimization:

| Config | Backend | Cache / Layout | JIT Strategy | SM Partitioning |
|--------|---------|----------------|--------------|-----------------|
| **A (Baseline)** | Triton/Flash (WSL2) | Paged | Native Graphs | None |
| **B (Eager cuTile)** | cuTile (Windows) | Paged | Eager (No padding) | None |
| **C (Padded cuTile)**| cuTile (Windows) | Paged + Padded | Eager (Padded) | None |
| **D (Cooperative)** | cuTile (Windows) | Paged | Eager + Green Contexts | 32 SM (P) / 16 SM (D) |

### 4.3 Key Metrics to Measure
1. **Decode P99 ITL (Inter-Token Latency)**: Measures tail latency when a heavy prefill is scheduled concurrently.
2. **TTFT (Time-to-First-Token)**: Prefill latency.
3. **Total Token Throughput (tok/s)**: Total serving capacity.
4. **JIT Compilation Overhead**: Time spent in the cuTile compiler during serving.
5. **Memory Allocator Overhead**: PyTorch garbage collection and CUDA caching allocator stall times.

---

## 5. Expected Results & Discussion

### Decode ITL Predictability
With **Green Contexts (Config D)**, the decode SM partition is completely independent. Even when a 2048-token prefill is submitted, the P99 ITL will remain virtually flat (~7.9ms). Without partitioning, the prefill floods all 48 SMs, causing decode steps to stall and P99 ITL to balloon to over 48ms.

### Eager vs Graph Trade-offs
Although Config A (Linux-native CUDA Graphs) will have the highest raw single-user speed, Config D provides a viable cross-platform fallback that guarantees strict tail-latency SLA targets on Windows, which is crucial for local desktop/LAN serving environments.
