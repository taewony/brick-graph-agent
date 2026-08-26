# Design & Architecture: nano-vllm Attention Migration to cuTile / TileGym

This document describes the design, system architecture, and kernel mapping strategy for migrating the attention mechanism in `nano-vllm` from Triton/FlashAttention to optimized cuTile/TileGym-based custom kernels. It details how PagedAttention, block tables, and CUDA Graphs operate in `nano-vllm` and outlines the technical changes required for the migration.

---

## 1. Initial Setup and Execution Guide (Original nano-vllm)

Setting up the original `nano-vllm` repository on a target machine equipped with an NVIDIA RTX 5070 requires installing the necessary GPU execution dependencies and downloading the target model.

> [!IMPORTANT]
> **Windows OS Compatibility Note**:
> OpenAI Triton and FlashAttention do **not** natively support Windows. Trying to install them directly in Windows PowerShell will result in compilation and installation errors.
> - **To run the full GPU pipeline**: You **must** run these steps inside **WSL 2 (Windows Subsystem for Linux)** with Ubuntu.
> - **To run natively on Windows/PowerShell**: You must use the **CPU Simulation mode** (`python src/cpu_sim/run_cpu_sim.py`), which bypasses the GPU and avoids Triton/FlashAttention dependencies.

### A. Environment Preparation (WSL 2 / Ubuntu Bash)
1. **Virtual Environment Setup**:
   Create and activate a clean Python 3.10 virtual environment:
   ```bash
   conda create -n nanovllm python=3.10 -y
   conda activate nanovllm
   ```
2. **Install PyTorch (CUDA-compatible)**:
   RTX 5070 (Blackwell architecture) performs best with PyTorch compiled with CUDA 12.1 or newer:
   ```bash
   pip3 install torch --index-url https://download.pytorch.org/whl/cu121
   ```
3. **Install Core Dependencies**:
   Install Triton, Hugging Face Hub, and transformers:
   ```bash
   pip install triton transformers huggingface_hub accelerate
   ```
4. **Install original FlashAttention**:
   Since the original `nano-vllm` imports attention interfaces directly from `flash_attn`, build and install the library:
   ```bash
   pip install flash-attn --no-build-isolation
   ```

### B. Model Download and Setup
1. **Download Default Qwen Model**:
   `nano-vllm` expects a Qwen-architecture model. We use `Qwen/Qwen2.5-0.5B-Instruct` as it has a small memory footprint (~1.2 GB in FP16), making it fast to download and leaving maximum VRAM available for the KV cache pool on the RTX 5070:
   ```bash
   # Create download directory
   mkdir -p ~/huggingface
   # Download the model weights and config files (uses local_dir_use_symlinks=False for safe Windows copy)
   python src/download_model.py --repo Qwen/Qwen2.5-0.5B-Instruct --dest ~/huggingface
   ```
2. **Rename Model Directory**:
   `bench.py` and `example.py` are configured to load the model from the local path `~/huggingface/Qwen3-0.6B/`. Rename the downloaded directory to match this expectation:
   ```bash
   mv ~/huggingface/Qwen2.5-0.5B-Instruct ~/huggingface/Qwen3-0.6B
   ```

### C. Execution and Verification
1. **Model Architecture Inspection**:
   Run the metadata inspector script to analyze GQA heads, Layer sizes, and estimated KV Cache memory:
   ```bash
   python src/inspect_model.py ~/huggingface/Qwen3-0.6B
   ```
2. **Run Inference Example**:
   Execute the simple generation wrapper to verify text completions:
   ```bash
   python example.py
   ```
3. **Run Throughput Benchmark**:
   Measure decoding throughput speed (tokens/second) using the benchmark suite:
   ```bash
   python bench.py
   ```

---

## 2. nano-vllm Attention Architecture Overview

`nano-vllm` implements a lightweight version of the vLLM serving engine. It manages memory allocation and request execution using **PagedAttention** to eliminate memory fragmentation.

```
                    [Logical KV Cache (Sequence)]
                    ┌───────┬───────┬───────┬───────┐
                    │Blk 0  │Blk 1  │Blk 2  │Blk 3  │
                    └───┬───┴───┬───┴───┬───┴───┬───┘
                        │       │       │       │
                        ▼       ▼       ▼       ▼  (Mapped by BlockManager)
                    ┌───────┬───────┬───────┬───────┐
                    │Blk 12 │Blk 45 │Blk 7  │Blk 89 │
                    └───────┴───────┴───────┴───────┘
                    [Physical KV Cache (Paged Buffers)]
```

### A. KV Cache Layout
The global physical KV cache is allocated as a single contiguous tensor of shape:
$$\text{KV Cache Shape} = [2, \text{layers}, \text{num\_blocks}, \text{block\_size}, \text{kv\_heads}, \text{head\_dim}]$$
- Dimension 0 represents keys (0) and values (1).
- Memory is split into fixed-size physical blocks (`block_size` tokens, typically 16 or 32).
- The `BlockManager` maps each sequence's logical token positions to physical blocks using a `block_table` tensor: `[batch_size, max_num_blocks_per_seq]`.

### B. Triton Store Kernel
In every forward pass, new keys and values are generated eagerly for the new prompt/generated tokens. A Triton kernel (`store_kvcache_kernel`) writes these keys/values directly into the physical cache slots mapped by a `slot_mapping` tensor:
```python
# slot_mapping points to the flat physical address of each token inside self.kv_cache
slot = tl.load(slot_mapping_ptr + idx)
# Store key/value at cache offsets corresponding to the slot
tl.store(k_cache_ptr + slot * D + offset, key)
```

---

## 3. FlashAttention vs. cuTile/TileGym Kernel Mapping

Currently, `nano-vllm` uses `flash-attention` library calls in [nanovllm/layers/attention.py](file:///D:/Capstone/metaprogramming/KernelAgent/3-nano-vllm/nanovllm/layers/attention.py):

```python
if context.is_prefill:
    o = flash_attn_varlen_func(q, k, v, ...)
else:  # Decode
    o = flash_attn_with_kvcache(q, k_cache, v_cache, ...)
```

The migration to cuTile/TileGym requires replacing these calls with equivalent custom kernels:

| nano-vllm Path | Eager API | Kernel Mechanism | cuTile / TileGym Migration Strategy |
| :--- | :--- | :--- | :--- |
| **Prefill Phase** | `flash_attn_varlen_func` | Computes attention over variable-length sequence prompts using cumulative offsets (`cu_seqlens_q`, `cu_seqlens_k`) without padding. | **cuTile FMHA Causal Kernel**: Launch our optimized `cutile_fmha` kernel per sequence. To handle variable-length prompts without padding, we can iterate over sequences in a loop, launching `cutile_fmha` on each sequence's slices separately. |
| **Decode Phase** | `flash_attn_with_kvcache` | Performs PagedAttention decoding by fetching keys/values directly from physical blocks using the `block_table` mapping. | **cuTile Paged Decode Kernel**: Since cuTile's standard decode kernel assumes a contiguous KV cache, we must implement a custom `fmha_paged_decode_kernel` where the physical key/value address calculation is performed dynamically inside the kernel using the `block_table`. |

---

## 4. cuTile Paged Decoding Attention Kernel Design

To support PagedAttention natively within cuTile without copying blocks into a contiguous buffer on the host (which degrades performance), we must modify the tile loading logic inside our custom cuTile kernel.

### A. Coordinate Lookup for Paged KV Cache
In standard `cutile_fmha`, key loading is done using a contiguous offset:
```python
k = ct.load(K, index=(batch_idx, off_kv_h, 0, j), shape=(1, 1, TILE_D, TILE_N))
```
In a Paged KV Cache, the columns along the sequence dimension are scattered across physical blocks. For key column index `col_idx` (corresponding to slot position `col_idx` in history):
1. **Block ID Lookup**: $\text{block\_id} = \text{block\_table}[\text{batch\_idx}, \text{col\_idx} // \text{block\_size}]$
2. **Block Slot Offset**: $\text{slot\_offset} = \text{col\_idx} \% \text{block\_size}$
3. **Physical Address**: The target key/value is loaded from `K[block_id, slot_offset, head_idx, dim]`.

### B. cuTile Paged Tile Load Kernel Implementation
We can define a custom cuTile kernel that resolves block table offsets dynamically:
```python
@ct.kernel(occupancy=2)
def fmha_paged_decode_kernel(Q, K_cache, V_cache, Out, block_table, block_size, ...):
    bid_x = ct.bid(0)
    bid_y = ct.bid(1)
    batch_idx = bid_y // H
    head_idx = bid_y % H

    # Load query tile (T=1, contiguous)
    q = ct.load(Q, index=(batch_idx, head_idx, bid_x, 0), shape=(1, 1, TILE_M, TILE_D))
    
    # Loop over key/value sequence blocks
    for j in range(0, Tc):
        # Resolve physical block ID from block_table
        # Since block_table resides in GPU global memory, load it dynamically:
        block_id_tile = ct.load(block_table, index=(batch_idx, j), shape=(1,))
        block_id = ct.astype(block_id_tile, np.int32).reshape(())
        
        # Load key tile from physical block cache
        k = ct.load(K_cache, index=(block_id, 0, head_idx, 0), shape=(1, block_size, 1, TILE_D))
        # Perform MMA, softmax update, and load V similarly...
```

---

## 5. CUDA Graph Orchestration in ModelRunner

`nano-vllm` uses bucketing/padding and CUDA Graphs to optimize decode latency in `ModelRunner` (` capture_cudagraph` and `run_model`).

```
                    [ModelRunner CUDA Graph Capture]
   Capture individual graphs for bucketing batch sizes:
   Batch Sizes: [1, 2, 4, 8, 16, 32, 48, ..., max_bs]
   
   Replay matching graph for batch size bs:
   graph = graphs[next_bs >= bs]
   Copy inputs to static graph vars -> graph.replay() -> argmax logits
```

### A. CUDA Graph Capture Flow
1. During initialization, `capture_cudagraph()` creates individual CUDA Graphs for a set of bucketed batch sizes: `[1, 2, 4, 8, 16, 32, ...]` up to `max_bs`.
2. Static placeholders (`input_ids`, `positions`, `slot_mapping`, `context_lens`, `block_tables`) are pre-allocated at their maximum sizes.
3. For each bucket size, a warmup pass is run, and the execution trace of the attention layers is captured into a `CUDAGraph` object.

### B. Integrating cuTile inside CUDA Graphs
- Because our cuTile custom attention kernels compile to standard GPU kernel launches on the active CUDA stream, they are automatically recorded during the graph capture phase.
- Passing `static_step` or other dynamic indices as 1-element GPU tensors ensures that no CPU-GPU synchronization is compiled into the graph stream, preserving 100% graph capture capability.

---

## 6. Migration Roadmap

1. **Triton KV Cache Store Check**: Retain the Triton `store_kvcache` kernel, as it efficiently handles writing newly projected K/V tensors into scattered slots.
2. **Implement cuTile Paged Attention Kernel**: Create the custom cuTile decode attention kernel (`cutile_fmha_paged_decode`) that accepts the `block_table` and `block_size` inputs and performs block address calculations natively on the GPU.
3. **Integrate into layers/attention.py**:
   - For **Prefill**: Loop over prompts in the batch and launch `cutile_fmha` (applying padding to 64 if the prompt length is shorter than 64 to avoid SDPA fallbacks).
   - For **Decode**: Call `cutile_fmha_paged_decode` using the pre-allocated cache buffers and block table.
4. **Graph Capture Warmup & Benchmarking**: Run `bench.py` to compare throughput (tokens/second) against the FlashAttention baseline.
