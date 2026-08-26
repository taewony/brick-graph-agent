## KernelAgent: cuTile-Based GPU Kernel Optimization for LLM Inference

**Project Lead:** Tech Lead (Kernel Engineering & Performance Analysis)  
**Target Hardware:** NVIDIA GeForce RTX 5070 (Blackwell Architecture, SM 100/101)  
**Objective:** Establish a foundational research platform demonstrating the performance superiority of NVIDIA cuTile DSL + CUDA Python 1.0 over conventional PyTorch/Triton implementations. We focus on matrix multiplication, fused multi-head attention (FMHA), and end‑to‑end LLM inference with advanced KV‑cache management, CUDA Graphs, and PagedAttention-based multi-user serving.

---

## 1. Project Overview

Modern LLM inference demands extreme GPU efficiency, yet existing frameworks often suffer from host-side Python launch overhead, suboptimal memory access patterns, and rigid kernel dispatch boundaries. **KernelAgent** systematically replaces critical GPU kernels (MatMul, FMHA, full decoder blocks) with hand‑tuned cuTile implementations and compares them in a strictly apple‑to‑apple manner against PyTorch/Triton/FlashAttention baselines.

The ultimate goal is to **write a peer‑reviewed paper** based on the measured data, proving that a cuTile‑optimized stack can outperform current production‑grade solutions, particularly in latency-sensitive serving scenarios.

---

## 2. Repository Structure

The actual structure of the repository and its experimental modules is organized as follows:

```
KernelAgent/
├── README.md                          # <--- You are here
├── 0-cuTile/                          # Matrix Multiplication Experiments
│   ├── cutile_matmul.py               # cuTile Matrix Multiplication kernel
│   ├── cutile_matmul_perf.py          # Benchmark: cuTile MatMul vs. PyTorch matmul
│   └── test.py                        # Correctness checks for cuTile MatMul
├── 1-FMHA/                            # Fused Multi-Head Attention Experiments
│   ├── AttentionFMHA.py               # FMHA correctness and tuning wrapper
│   └── AttentionFMHA_v3.py            # Latency-hiding FMHA implementation
├── 2-LLM-from-scratch/                # End-to-End LLM Inference Experiments
│   ├── compare_infer.py               # E2E Benchmark: PyTorch vs. cuTile Attention (with KV-Cache)
│   ├── compare_infer_cuda_graph.py    # E2E Benchmark: cuTile Raw vs. cuTile + CUDA Graphs
│   ├── src/                           # Model source codes
│   │   ├── model.py                   # Baseline GPT-2 Model (PyTorch)
│   │   ├── model_cutile.py            # cuTile Attention GPT-2 Model
│   │   ├── cutile_kernel.py           # cuTile FMHA kernel wrapper
│   │   ├── generate.py                # Generation helper (PyTorch)
│   │   └── generate_cutile.py         # Generation helper (cuTile)
│   └── study/                         # Design specifications and design documents
│       └── compare_infer.md           # Engineering guide for the E2E benchmark
└── micro-vllm/                       # Multi-User Serving & PagedAttention (Completed)
    ├── nanovllm/                      # Custom lightweight vLLM inference engine
    │   ├── layers/
    │   │   ├── cutile_attention.py    # Custom cuTile prefill & paged decode kernels (GQA, causal padding)
    │   │   ├── attention.py           # Conditional routing & optional Triton/Flash imports
    │   │   ├── layernorm.py           # Optional JIT compilation support on Windows
    │   │   ├── rotary_embedding.py    # Optional JIT compilation support on Windows
    │   │   └── activation.py          # Optional JIT compilation support on Windows
    │   └── engine/
    │       └── model_runner.py        # Windows-compatible Gloo fallback and eager routing
    ├── src/
    │   ├── tests/
    │   │   ├── test_migration.py      # 5-step TDD verification test suite (CPU/GPU)
    │   │   └── test_layers.py         # GPU-targeted unit tests (Linear & RoPE)
    │   ├── cpu_sim/
    │   │   └── run_cpu_sim.py         # Mock-up scheduler simulation on CPU
    │   └── download_model.py          # Model downloader with Windows path expansion
    ├── bench.py                       # Multi-user serving benchmark (Supports --use-cutile)
    ├── example.py                     # Eager-mode text generation example
    ├── test_asynchronous.py           # Dynamic asynchronous user arrival simulation
    └── asynchronous_testing.md        # Guide to asynchronous simulation and dynamic hooking
```

---

## 3. Experiments at a Glance

| Experiment | Key Kernel(s) | Baseline | cuTile Advantage |
|------------|---------------|----------|------------------|
| **0 - MatMul** | Tiled GEMM with vectorized loads, unrolled loops | `torch.matmul` (FP16) | Explicit shared memory tiling and register blocking optimized for Tensor Cores. |
| **1 - FMHA** | Online softmax Fused MHA (64×64 tile) | `F.scaled_dot_product_attention` | Reduces memory bandwidth pressure by fusing QK multiplication, Softmax, and PV reduction. |
| **2 - LLM-from-scratch** | FMHA Prefill + Decode (KV-Cache & CUDA Graph enabled) | GPT-2 Block with PyTorch attention | Bypasses Python launch latency during token-by-token decoding via static graphs. |
| **micro-vllm** | Fused Prefill & Paged Decode (GQA, Causal Padding) | PyTorch SDPA & Triton `store_kvcache` | **Completed.** Bypasses Linux-only Triton/FlashAttention dependencies, allowing native Windows GPU serving. Achieves 1,900+ tok/s throughput. |

All measurements use **warm-up (5–20×) and repeated runs (20–100×)** with `torch.cuda.synchronize()` to guarantee accurate GPU timings. Numerical correctness is verified using `torch.allclose` or exact output token matching.

---

## 4. Co-work Workflow (Tech Lead & Experiment Runner)

Since the experiments run on the **RTX 5070** target machine, we will collaborate as follows:

1. **Tech Lead (AI)**:
   - Authors and refactors cuTile kernels and wrapper scripts.
   - Designs benchmark suites to capture execution metrics.
   - Evaluates execution logs, profiles kernel performance, and drafts academic paper segments.
2. **Experiment Runner (User)**:
   - Runs the benchmark scripts locally on the RTX 5070.
   - Passes execution output, raw timing tables, and console logs back to the Tech Lead.
   - Provides feedback on performance bottlenecks and system configurations.

### RTX 5070 Optimization Directives:
- **Compute Capability**: Blackwell (SM 100/101).
- **Precision**: Tensors are cast to `float16` to leverage NVIDIA Tensor Cores.
- **Tiling Bounds**: Prefill uses $64 \times 64$ FMHA tiling (proved to yield optimal occupancy and SM utilization).
- **Graph Compilation**: Autoregressive decode steps are captured in CUDA Graphs using static cache buffers to eliminate Python host-side dispatch overhead.
- **Windows Portability**: Uses conditional JIT compilation (via `@optional_compile`) and the `gloo` distributed backend to run natively on Windows GPUs.

---

## 5. How to Run the Comparisons

### Step 0: Prerequisites & Environment
Ensure PyTorch (with CUDA support) and the `cuda-python` / `cuda.tile` libraries are installed in your Python environment.
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install tilegym[tileiras]
```

### Step 1: Matrix Multiplication Benchmark (`0-MatMul`)
To compare raw GEMM performance:
```bash
python 0-MatMul/compare_matmul.py
```

### Step 2: Fused Multi-Head Attention Benchmark (`1-FMHA`)
To run correctness verification and performance analysis of the raw attention kernel:
```bash
python 1-FMHA/AttentionFMHA.py --correctness-check
```

### Step 3: End-to-End LLM Inference (KV-Cache Aware)
To compare PyTorch and cuTile attention with KV-Cache during text generation:
```bash
python 2-LLM-from-scratch/compare_infer.py
```

### Step 4: CUDA Graph Acceleration Benchmark
To run the full comparison showing the speedup obtained by bypassing Python launch overhead with CUDA Graphs:
```bash
python 2-LLM-from-scratch/compare_infer_cuda_graph.py
```

### Step 5-1: Native Windows Multi-User Serving (`micro-vllm`)
출력 결과: micro-vllm/test-result-cutile.md

1. **Download & Rename Model**:
   ```powershell
   pip install huggingface_hub
   python micro-vllm/src/download_model.py --repo Qwen/Qwen2.5-3B-Instruct --dest ~/huggingface
   Move-Item -Path ~/huggingface/Qwen2.5-3B-Instruct -Destination ~/huggingface/Qwen3-0.6B
   ```
2. **Run TDD Verification Test Suite**:
   Runs 5 logical and GPU-compiled kernel correctness checks against Golden PyTorch references:
   ```powershell
   python micro-vllm/src/tests/test_migration.py
   ```
3. **Run E2E Eager-mode Text Generation**:
   ```powershell
   $env:NANO_VLLM_USE_CUTILE="1"
   python micro-vllm/example.py
   ```
4. **Run serving Benchmark**:
   ```powershell
   python micro-vllm/bench.py --use-cutile
   ```
5. **Run Asynchronous Dynamic User Simulation**:
   ```powershell
   python micro-vllm/test_asynchronous.py
   ```

---

### Step 5-2: Linux(WSL) Multi-User Serving (`nano-vllm`)
출력 결과: micro-vllm/test-result-flash_attn.md

1. **Download & Rename Model**:
   ```bash
   cd micro-vllm
   pip install huggingface_hub
   python3 src/download_model.py --repo Qwen/Qwen2.5-3B-Instruct --dest ~/huggingface
   mv ~/huggingface/Qwen2.5-3B-Instruct ~/huggingface/Qwen3-0.6B
   ```
2. **Check Downloaded & Renamed qwen Model**:
   ```bash
   pip install transformers
   python src/inspect_model.py /home/linux/huggingface/Qwen3-0.6B >> test-result-flash_attn.md
   ```
3. **Run E2E flash_attn mode Text Generation**:
   ```bash
   python example.py
   ```
4. **Run serving Benchmark in flash_attn mode**:
   ```bash
   python bench.py
   ```
5. **Run Asynchronous Dynamic User Simulation**:
   ```bash
   python test_asynchronous.py
   ```

---

## 6. Target Performance Goals

On the target RTX 5070 GPU, we aim for the following outcomes:

- **MatMul**: 1.2–1.5× higher TFLOPS than PyTorch for large matrix dimensions ($N \ge 2048$) through optimal L1 caching and register utilization.
- **FMHA**: Stable prefill latencies and lower memory bandwidth bottlenecking.
- **LLM-from-scratch (Raw)**: Consistent prefill performance with `T >= 64` sequences.
- **LLM-from-scratch (CUDA Graphs)**: **>2.0× speedup** in token-per-second decoding throughput over raw cuTile execution by resolving host-side dispatch overhead.
- **micro-vllm (cuTile)**: Multi-user batched serving performance reaching **1,900+ tok/s** overall throughput natively on Windows GPUs.

