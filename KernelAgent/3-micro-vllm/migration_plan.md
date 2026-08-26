# Migration Plan: nano-vllm Attention to cuTile/TileGym

This document outlines the step-by-step migration plan to integrate **cuTile / TileGym** custom attention kernels into `nano-vllm`. By replacing the Linux-only dependencies (**FlashAttention** and **Triton**) with cuTile and pure PyTorch operations, we enable **GPU-accelerated native execution directly on Windows OS** (via PowerShell) while maintaining the original PyTorch/FlashAttention codebase for reference and comparison.

---

## 1. Architectural Highlights & Experience Leveraged

This plan builds directly on the optimization findings from our previous tasks:
- **Phase 1 (FMHA)**: Proven best tile configurations ($64 \times 64$ and $128 \times 64$), memory coalescing, and prefetch pipelining.
- **Phase 2 (LLM-from-Scratch)**:
  - **Causal Padding**: If sequence length $T < 64$, pad to 64 using `F.pad` before running the custom cuTile kernel, and slice back to $T$ to avoid falling back to PyTorch SDPA.
  - **No Cache Clearing (`fill_`)**: Remove unnecessary `.fill_(0.0)` cache resets since causal masking (`offs_m >= offs_n`) automatically ignores future slots.
  - **Direct Prefill Writing**: Write projections directly into pre-allocated cache buffers on the GPU during the prefill forward pass rather than performing post-hoc copies.
  - **CUDA Graph Harmony**: Bypassing CPU-GPU synchronization by loading dynamic inputs (like step indices) from 1-element GPU tensors.

---

## 2. Phase-by-Phase Migration Plan

### 🛠️ Phase 1: Triton to Pure PyTorch KV Store Migration
Currently, `nano-vllm` uses a Triton kernel (`store_kvcache_kernel`) to copy projected K/V tensors into the physical cache blocks. Triton is Linux-only, preventing native execution on Windows.

- **Action**: Replace the Triton-based `store_kvcache` function in `nanovllm/layers/attention.py` with a pure PyTorch flat index-write operation:
```python
def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    D = key.shape[1] * key.shape[2]  # num_heads * head_dim
    N = key.shape[0]
    
    # Flat views of KV cache and projections
    k_cache_flat = k_cache.view(-1, D)
    v_cache_flat = v_cache.view(-1, D)
    key_flat = key.view(-1, D)
    value_flat = value.view(-1, D)
    
    # Write only active slots (ignoring slots that map to -1)
    mask = slot_mapping != -1
    valid_slots = slot_mapping[mask].long()
    
    k_cache_flat[valid_slots] = key_flat[mask]
    v_cache_flat[valid_slots] = value_flat[mask]
```
- **Validation**: Test natively on Windows/PowerShell CPU simulation or GPU to ensure values are copied into identical memory locations.

---

### 🚀 Phase 2: cuTile Custom Kernels for Prefill and Paged Decode
We will write two custom cuTile kernels to replace `flash-attn` calls:

1. **Prefill Attention (`cutile_fmha`)**:
   - Computes causal self-attention for variable-length prompts.
   - **Padding Mechanism**: If the prompt length $T < 64$, we pad $Q, K, V$ to 64 tokens, launch our optimized cuTile kernel, and slice the output back to $T$ to avoid any SDPA fallback.
2. **Paged Decode Attention (`cutile_fmha_paged_decode`)**:
   - Performs PagedAttention decoding using `block_table`.
   - **Block Resolution**: Compute physical cache addresses directly on the GPU within the cuTile load instructions:
     $$\text{block\_id} = \text{block\_table}[\text{batch\_idx}, \text{col\_idx} // \text{block\_size}]$$
     $$\text{slot\_offset} = \text{col\_idx} \% \text{block\_size}$$

---

### 🎨 Phase 3: Integration into `nanovllm/layers/attention.py`
To preserve the original PyTorch/FlashAttention source code, we will introduce a `use_cutile` flag in the config/context. The `Attention` layer's `forward` pass will conditionally route execution:

```python
class Attention(nn.Module):
    # ...
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        
        # 1. Update Cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
            
        # 2. Select Backend Routing
        if context.use_cutile:
            if context.is_prefill:
                # Launch custom cuTile prefill attention (with causal padding if T < 64)
                o = cutile_fmha_prefill(q, k, v, causal=True, ...)
            else:
                # Launch custom cuTile PagedAttention decode
                o = cutile_fmha_paged_decode(q, k_cache, v_cache, context.block_tables, ...)
        else:
            # ORIGINAL PyTorch / FlashAttention code paths
            if context.is_prefill:
                o = flash_attn_varlen_func(q, k, v, ...)
            else:
                o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache, ...)
        return o
```

---

### 📈 Phase 4: CUDA Graph Capture Harmony
- Ensure the custom cuTile PagedAttention kernels do not trigger any CPU-GPU synchronization (e.g. by passing the current step/token indices as 1-element GPU tensors).
- The JIT-compiled cuTile kernels will be automatically recorded during `ModelRunner.capture_cudagraph()` for each bucketed batch size.

---

## 3. Step-by-Step Windows Native Setup & Execution (with cuTile)

Once migrated, the original Linux dependencies are eliminated. You can run the GPU-enabled server directly in PowerShell:

1. **Install CUDA-compatible PyTorch & cuTile**:
   ```powershell
   pip3 install torch --index-url https://download.pytorch.org/whl/cu121
   pip install tilegym[tileiras]
   ```
2. **Download Model**:
   ```powershell
   python src/download_model.py --repo Qwen/Qwen2.5-0.5B-Instruct --dest ~/huggingface
   mv ~/huggingface/Qwen2.5-0.5B-Instruct ~/huggingface/Qwen3-0.6B
   ```
3. **Execute Benchmark**:
   ```powershell
   python bench.py --use-cutile
   ```
---

  ### Summary of Modifications

  1. Windows Native Path Expansion: Updated download_model.py using  os.path.expanduser  so that the directory paths
  containing  ~  resolve correctly on Windows PowerShell.
  2. Syntax Error Fixes: Fixed broken multiline print syntax errors in test_layers.py and run_cpu_sim.py.
  3. No Process Group Requirement for Unit Tests: Mocked  torch.distributed  in test_layers.py when running outside
  of rank-based contexts, allowing simple single-process unit testing.
  4. cuTile Kernels & Wrapper: Created cutile_attention.py containing:
      • A variable-sequence-length causal FMHA prefill kernel using padded 4D tensors (avoiding SDPA fallbacks by
      padding to a multiple of 64).
      • A dynamic PagedAttention decoding kernel resolving block tables internally on the GPU.
      • Robust  try-except  imports for  cuda.tile  to prevent startup crashes on CPU-only machines.
  5. Conditional Routing in Main Attention: Updated attention.py to:
      • Lazily and conditionally import  triton  and  flash_attn .
      • Fall back to the pure PyTorch flat-view index write ( store_kvcache_pytorch ) on Windows or when cuTile is
      used.
      • Route to  cutile_fmha_prefill  or  cutile_fmha_paged_decode  if  use_cutile=True  or the
      NANO_VLLM_USE_CUTILE=1  environment variable is present.
  6. Command-Line Integration in Benchmark: Updated bench.py to parse  --use-cutile  using  argparse  and set
  the environment variable.
  ──────
  ### Step-by-Step TDD Verification Guide for Your RTX 5070 Machine

  Run the following commands in order in your PowerShell terminal:

  #### Step 1: Run CPU Simulation & Existing Unit Tests

  These check that existing linear/rotary layers and the vLLM scheduler logic work perfectly on CPU without requiring
  any GPU or CUDA packages:

    # Run the core scheduler and sequence model simulation
    python src/cpu_sim/run_cpu_sim.py

    # Run layer-specific unit tests (Linear & RoPE)
    python src/tests/test_layers.py

  #### Step 2: Run the CPU/GPU Verification Suite

  Verify the logic of our migrated KV-store, causal padding, and paged cache lookup:

    python src/tests/test_migration.py

  #### Step 3: Set Up the GPU Environment (RTX 5070)

  Installs Windows-native CUDA-compatible PyTorch and the cuTile compiler package:

    # Install GPU PyTorch (ensure your CUDA toolkit matches the wheel version)
    pip install torch --index-url https://download.pytorch.org/whl/cu121

    # Install TileGym/cuTile compiler library
    pip install tilegym[tileiras]

  #### Step 4: Download and Prepare the Target Model

  Download the model to the local cache directory:

    # Download Qwen2.5-0.5B-Instruct to your user profile directory
    python src/download_model.py --repo Qwen/Qwen2.5-0.5B-Instruct --dest ~/huggingface

    # Rename folder to match path expected by example and benchmark
    Move-Item -Path ~/huggingface/Qwen2.5-0.5B-Instruct -Destination ~/huggingface/Qwen3-0.6B

  #### Step 5: Execute and Validate Output

  Run the example text generation using cuTile:

    # Run using the environment variable flag to route to cuTile
    $env:NANO_VLLM_USE_CUTILE="1"
    python example.py

  #### Step 6: Benchmark Throughput

  Measure tokens/sec with and without the cuTile backend:

    # 1. Run baseline FlashAttention/original path (will raise exception on Windows if flash-attn is missing)
    python bench.py

    # 2. Run cuTile accelerated path
    python bench.py --use-cutile
    
---

  I have aligned the implementation of our custom cuTile kernels in cutile_attention.py with the reference designs from
  NVIDIA/TileGym/src/tilegym/ops .

  Specifically, we leveraged:

  1. Online Softmax Scaling: Computes and normalizes scaling using intermediate maximum log-sum-exp variables ( m_i ,
  l_i ) in float32 for maximum numerical precision, matching the reference FMHA tiling strategy.
  2. Dynamic Block Transposition: Loaded keys transposed to  (head_dim, block_size)  directly at the hardware load
  level via custom transposition orders ( order=(0, 3, 2, 1) ), avoiding high-overhead global memory transpose
  operations.
  3. Internal Block Table Lookup: Resolved physical memory addresses within the CUDA execution loop:

    block\_id = block\_table [batch\_idx,j]

  and fetched from  k_cache[physical_block_id, offset, head, :]  dynamically.
  
  ---
  
    The model ran successfully on your Windows GPU (RTX 5070) with cuTile/TileGym acceleration!

  ### Key Accomplishments & Metrics

  • TDD Validation: The 5-step test suite ( test_migration.py ) successfully executed, logically verifying the flat-
  indexing store, causal padding offsets, and block table mappings on CPU, and then compiling and verifying the JIT
  cuTile kernels on your GPU.
  • Zero Linux Dependency: By conditionally importing  triton  /  flash_attn  and making  torch.compile  optional
  (fallback to eager mode in layers like  RMSNorm ,  SwiGLU , and  Sampler ), we eliminated all Linux-only build
  blocks.
  • cuTile Acceleration Performance:
      • Prefill Speed: 882 tok/s (variable-sequence causal padding up to a multiple of 64 worked flawlessly).
      • Decode Speed: 37 tok/s (physical block lookup inside the cuTile load instructions using the block table
      worked dynamically on the GPU).


  ### Completed Directory Architecture

  1. migration_plan.md: Verified as 100% OK for native Windows PowerShell execution (with updated directory resolution
  guidelines).
  2. cutile_attention.py: Houses the optimized prefill and paged decode kernels, handling head-group mapping, causal
  padding, and dynamic address calculations on the GPU.
  3. attention.py: Integrates the conditional routing to cuTile under the  use_cutile=True  flag (or
  NANO_VLLM_USE_CUTILE=1  environment variable).
  4. test_migration.py: The complete 5-stage verification test suite.
  5. model_runner.py: Updated to use the GLOO backend for distributed initialization on Windows/single-GPU
  configurations.
  6. layernorm.py, activation.py, sampler.py, and rotary_embedding.py: Bypasses compiler errors on Windows.
  
  ---
  
  The  nano-vllm  engine is designed to be fully dynamic. During startup, it reads the model's dimensions (hidden
  size, head dimension, number of layers, attention heads, GQA ratios) directly from the downloaded model's
  config.json .
  ──────
  ### How to Download and Run a Different Model

Qwen2.5-3B  is an excellent candidate for your RTX 5070 GPU! It offers a great balance of model intelligence (much
  stronger than the 0.5B model) and memory usage.

  #### Memory & Hardware Specs for Qwen2.5-3B

  • Weight Memory: A 3B model at FP16 precision takes  3 * 2 bytes = ~6.2 GB  of VRAM.
  • VRAM Margin: On your RTX 5070 (which has 12 GB of VRAM), this leaves about  ~5.8 GB  of free VRAM. This is more
  than enough for the GPU memory workspace and a very large vLLM Paged KV Cache (allowing high concurrency).
  • GQA / Attention: Unlike some larger models,  Qwen2.5-3B  uses MHA (16 Query heads and 16 KV heads). This is fully
  supported and will run with high occupancy.

  #### How to Run it

  Simply download the model and rename it to the benchmark directory:

    # 1. Download the 3B Instruct model
    python src/download_model.py --repo Qwen/Qwen2.5-3B-Instruct --dest ~/huggingface

    # 2. Rename the old folder to back it up
    Rename-Item -Path ~/huggingface/Qwen3-0.6B -NewName Qwen3-0.6B-old

    # 3. Rename the 3B model to Qwen3-0.6B
    Move-Item -Path ~/huggingface/Qwen2.5-3B-Instruct -Destination ~/huggingface/Qwen3-0.6B

  Then run the generation or benchmark as usual:

    # Run generation
    python example.py

    # Run benchmark
    python bench.py --use-cutile

──────

  ### Compatibility Check list

  1. Grouped Query Attention (GQA): Larger Qwen models (like 7B or 8B) use GQA (where the number of Query heads is
  larger than the number of Key/Value heads). Our cuTile attention kernels are fully compatible with GQA because they
  dynamically calculate  query_group_size = H // num_kv_heads  at runtime.
  2. VRAM Size Warning:
      • Qwen2.5-1.5B or 3B: These will run extremely fast and fit easily inside your GPU VRAM.
      • Qwen2.5-7B/8B: A 7B model at FP16 requires  ~14 GB  of VRAM just to store the weights. On a GPU with 12 GB or
      16 GB VRAM, this might run close to the limit once the vLLM KV Cache is allocated, so you may need to reduce
      the  gpu_memory_utilization  flag in  Config  (e.g. to  0.7  or  0.8 ) to prevent Out-Of-Memory (OOM) errors.