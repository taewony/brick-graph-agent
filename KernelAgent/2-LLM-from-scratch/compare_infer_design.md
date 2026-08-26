# Design & Architecture: End-to-End LLM Inference Optimization

This document describes the design, system architecture, and GPU optimization strategies for our custom cuTile-based GPT-2 inference pipeline compared to PyTorch's native baseline. It highlights the design of our **Static KV Cache** and **CUDA Graph Capture** mechanisms, which target the latency and dispatch overhead bottlenecks of autoregressive decoding on the NVIDIA RTX 5070 GPU.

---

## 1. Inference Phases & Performance Bottlenecks

LLM autoregressive inference consists of two distinct computational phases, each presenting unique engineering bottlenecks:

```
                            ┌─────────────────────────┐
                            │      Input Prompt       │
                            └────────────┬────────────┘
                                         │
                                         ▼
                            ┌─────────────────────────┐
                            │    Prefill (Prompt)     │  ◄── Compute-Bound (GEMM)
                            │   Computes KV Cache     │      Optimized via cuTile FMHA
                            └────────────┬────────────┘
                                         │
                                         ▼
                              ┌─────────────────────┐
                              │  Sample Next Token  │
                              └──────────┬──────────┘
                                         │
                                         ▼
                            ┌─────────────────────────┐
    ┌───────────────────────┤    Decode (Autoregr.)   │  ◄── Memory-Bound (GEMV)
    │                       │  Generates 1 token/step │      Optimized via Static Cache
    │                       └────────────┬────────────┘      & CUDA Graph Replays
    │                                    │
    └────────────────────────────────────┘
```

### A. Prefill Phase (Prompt Processing)
- **Operation**: The model processes the entire prompt sequence at once to generate the initial logits and build the key-value (KV) cache.
- **Bottleneck**: This is compute-bound. The self-attention operation scales quadratically ($O(S^2)$) with the prompt length.
- **Optimization**: We launch our optimized `cutile_fmha` kernel (utilizing $64 \times 64$ tile sizes, L2 cache swizzling, and prefetch pipelining) to accelerate the GEMM-based attention calculations.

### B. Decode Phase (Autoregressive Generation)
- **Operation**: The model generates text token-by-token. In each step, the query sequence length is exactly 1 ($T=1$).
- **Bottleneck**: This phase is heavily memory-bandwidth bound and launch-overhead bound. The computation consists of Vector-Matrix multiplications (GEMV), which run extremely fast on Tensor Cores. The execution time is dominated by:
  1. **Python Interpreter Overhead**: Python dispatching layer-by-layer.
  2. **PyTorch Dispatch Overhead**: PyTorch checking tensor shapes, strides, and memory layouts in every forward step.
  3. **Memory Allocation Latency**: Dynamic resizing of the KV Cache (using `torch.cat`) causing memory fragmentation and reallocation.

---

## 2. Static KV Cache Architecture

To eliminate memory reallocation and fragmentation during the decode loop, we implement a **Static KV Cache** within the `StaticGPTRunner` wrapper:

1. **Pre-Allocation**:
   At initialization, we allocate fixed-size contiguous tensors for the Key and Value cache for each layer:
   $$\text{Cache Shape} = [\text{Batch}, \text{Heads}, \text{MaxSeqLen}, \text{HeadDim}]$$
   Using a fixed size (e.g., `MaxSeqLen = 1024`) prevents any dynamic memory allocation during decoding.
2. **In-Place Updates**:
   During each decode step $t$, the new key and value projection vectors (of length 1) are written directly into their respective slices of the pre-allocated buffers using the graph-compatible `scatter_` operator with a pre-allocated index tensor:
   ```python
   self.static_k[i].scatter_(2, self.static_scatter_index, k)
   self.static_v[i].scatter_(2, self.static_scatter_index, v)
   ```
3. **Softmax Masking of Unpopulated Slots**:
   Since the cache tensor has a static shape, future slots ($t > \text{current\_step}$) contain garbage data. To prevent this data from affecting attention, we initialize the static key cache (`static_k`) with a large negative value (`-10000.0`).
   - When `cutile_fmha` performs the softmax update, these unpopulated slots result in $e^{-10000.0} \approx 0$, effectively masking them out.

---

## 3. CUDA Graph Optimization

Even with a static KV cache, launching dozens of small PyTorch operators in every layer at every step generates significant host-to-device launch latency (approx. 2–3 ms per token). To bypass this, we capture the entire decode forward step into a **CUDA Graph**:

```
                       [CUDA Graph Capture Flow]

1. Set inputs in-place   ==>  runner.static_input.copy_(next_token)
                              runner.static_step.copy_(current_step)

2. Replay GPU Stream     ==>  graph.replay()  <-- Bypasses Python/PyTorch dispatch

3. Sample next token     ==>  next_token = static_logits.argmax()
```

### How Graph Capture Works:
1. **Warmup**: Run a single forward step of the model outside the graph to trigger JIT compilation of custom cuTile kernels and allocate internal CUDA resources.
2. **Capture**: Using `torch.cuda.graph(g)`, we record all kernel dispatches onto the CUDA stream during a single forward pass:
   ```python
   with torch.cuda.graph(g):
       runner.static_logits = runner.forward_step(runner.static_input, runner.static_step)
   ```
   This records the exact execution sequence of GPU kernels (including our custom `cutile_fmha` kernel) and their memory addresses.
3. **Execution (Replay)**:
   In the decode loop, we copy the new input token and step index into the pre-allocated static tensors, then replay the graph:
   ```python
   g.replay()
   ```
   This launches the entire layer stack on the GPU in a single operation, reducing host-overhead latency to near-zero.

---

## 4. Benchmark Configurations

We evaluate three execution modes:
1. **Baseline PyTorch**: Native PyTorch GPT-2 inference (uses `torch.nn.functional.scaled_dot_product_attention` and dynamic `torch.cat` KV caches).
2. **cuTile Attention (Raw)**: cuTile attention kernel integration, maintaining dynamic KV caches (subject to Python launcher overheads).
3. **cuTile + CUDA Graphs**: Fully optimized static KV cache runner replayed via CUDA Graphs.

### Evaluation Metrics
- **TTFT (Time To First Token)**: Prefill time in milliseconds.
- **Decoding Speed (tokens/sec)**: Throughput of the decode phase, computed as:
  $$\text{Throughput} = \frac{\text{Generated Tokens} - 1}{\text{Total Time} - \text{TTFT}}$$
- **Parity Check**: Strict token-by-token comparison to guarantee that our optimizations preserve numerical correctness and model output.

---

## 5. Execution Instructions (For Experiment Runner)

Since the development machine does not have a GPU, the benchmark scripts must be run on the target RTX 5070 machine.

1. **Autoregressive Inference & KV Cache Benchmark**:
   ```bash
   python 2-LLM-from-scratch/compare_infer.py
   ```
2. **CUDA Graph Optimized Inference Benchmark**:
   ```bash
   python 2-LLM-from-scratch/compare_infer_cuda_graph.py
   ```
   *Note: Ensure `checkpoint_final.pt` exists in the root directory or configure the path accordingly using the `--checkpoint` argument. The script will print an error message and exit if the checkpoint file is not found.*
