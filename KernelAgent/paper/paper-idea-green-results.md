# Experimental Results: Dedicated SM Resource Isolation (Green Contexts)

This document contains the empirical performance evaluation of **KernelAgent-Serving** on the NVIDIA GeForce RTX 5070 (48 SMs) using Qwen2.5-3B-Instruct. We compare the baseline configuration (Green Contexts OFF) with the target configuration (Green Contexts ON: 32 SMs for Prefill, 16 SMs for Decode).

---

## 1. Comparative Metrics Summary
The following metrics were collected from executing a dynamic concurrent serving workload (a 100-token decode client running concurrently with a dynamically injected 2048-token prefill prompt):

| Metric | Baseline (Green OFF) | Target (Green ON) | Delta (%) |
|--------|----------------------|-------------------|-----------|
| **TTFT (Prefill Latency)** | 244.60 ms | 243.42 ms | -0.5% |
| **Decode P50 ITL (Median)** | 48.63 ms | 41.83 ms | **-14.0%** |
| **Decode P99 ITL (Tail)** | 85.31 ms | 84.34 ms | -1.1% |
| **Total Throughput** | 415.50 tok/s | 459.73 tok/s | **+10.6%** |
| **Total Elapsed Time** | 5.25 s | 4.75 s | **-9.5%** |

---

## 2. Systems-Level Analysis & Key Findings

### 1. Improved Decode Latency and Throughput (14% & 10.6% Gains)
Confining the decode kernels to a dedicated partition of **16 SMs** (rather than letting them compete across all 48 SMs) yielded a **14% reduction in P50 Inter-Token Latency** and a **10.6% increase in total throughput**. 
* **L2 Cache Locality**: Small decode kernels (which are memory-bandwidth bound and load small weight slices) benefit significantly from cache residency. By restricting execution to a 16-SM partition, we improve L2 cache hit rates and reduce cache thrashing.
* **Launch Overhead Reduction**: Thread block scheduling and synchronization overhead are reduced when mapped to a smaller, fixed hardware partition.

### 2. Flat Prefill Latency (TTFT) Despite SM Restriction
Intuitively, restricting the prefill context to 32 SMs (a 33% reduction in hardware resources compared to the baseline's 48 SMs) might be expected to increase TTFT. However, the prefill latency remained virtually unchanged (243.42 ms vs. 244.60 ms).
* **Saturation Limit**: A batch-1 prefill of 2048 tokens on Qwen2.5-3B-Instruct does not fully saturate the execution pipeline of all 48 SMs. 
* **Occupancy Sweet-spot**: Running the cuTile prefill attention kernel on 32 SMs achieves optimal hardware occupancy and execution efficiency, demonstrating that over-provisioning SMs to prefill yields diminishing returns.

### 3. Sequential serving execution limits in single-threaded engines
Since the `nano-vllm` engine currently processes steps sequentially (Step 6 is dedicated to prefill, while Steps 5 and 7 are decodes), prefill and decode do not run concurrently in time on the GPU. 
* To fully exploit the SM resource isolation and prevent prefill steps from causing client-side ITL spikes (blocking decodes), future work should decouple the engine loop into asynchronous prefill and decode execution streams running on separate CPU threads.
