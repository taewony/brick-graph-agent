# Guide: Asynchronous User Simulation & Hacking vLLM Operations

This document describes how to simulate real-world, asynchronous (dynamic) user request arrivals in `nano-vllm`, and how developers can utilize this framework to intercept, profile, or hack vLLM core operations (such as scheduling, KV Cache allocation, and attention routing).

---

## 1. Simulating Asynchronous User Arrivals

In production, users submit prompts at random timestamps. While the standard vLLM entrypoint wraps this in asynchronous async-io loops, we can simulate this deterministically using an iterative stepping loop.

Below is the implementation for `test_asynchronous.py`. It adds a secondary request dynamically after a specific number of decode steps have already run for the first request.

### `test_asynchronous.py`
Create this file in the `3-nano-vllm` folder:

```python
import os
import time
import torch
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer

def main():
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    
    # Initialize LLM in eager mode for native Windows execution
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)
    sampling_params = SamplingParams(temperature=0.6, max_tokens=100)

    # 1. User 1 submits a prompt immediately
    llm.add_request("Explain the theory of relativity in one sentence.", sampling_params)
    print("📥 User 1 request added.")

    step_count = 0
    
    # 2. Run the generation iteratively
    finished_outputs = {}
    while not llm.is_finished():
        step_count += 1
        
        # Simulate User 2 submitting a request dynamically at Step 5
        if step_count == 5:
            llm.add_request("What is the capital of France?", sampling_params)
            print("\n📥 User 2 request added dynamically while User 1 is decoding!")

        # Execute a single engine step
        outputs, num_tokens = llm.step()
        for seq_id, token_ids in outputs:
            finished_outputs[seq_id] = token_ids
        
        # Print active generations at each step
        active_ids = [seq.seq_id for seq in llm.scheduler.running]
        print(f"Step {step_count}: Active User IDs={active_ids}, Tokens processed={num_tokens}")

    # 3. Print final outputs
    print("\n" + "="*60)
    for seq_id in sorted(finished_outputs.keys()):
        token_ids = finished_outputs[seq_id]
        text = tokenizer.decode(token_ids)
        print(f"User {seq_id} Completion: {text.strip()}\n" + "-"*60)

if __name__ == "__main__":
    main()
```

---

## 2. Hacking vLLM Core Operations

Simulating the engine step-by-step gives you hookable access to the internal states. You can monkey-patch or inject diagnostic logic at runtime.

### A. Hijacking the Scheduler (Preemption & Stealing)
By hooking into the scheduling loop, you can inspect the waiting queues or forcibly preempt a low-priority user's KV cache blocks to allocate them to a high-priority user.

Add this code before the loop to intercept scheduling:
```python
# Save original schedule method
original_schedule = llm.scheduler.schedule

def hacked_schedule():
    # Inspect wait queue
    if llm.scheduler.waiting:
        print(f"🕵️ HACK: Detected {len(llm.scheduler.waiting)} requests waiting in queue.")
        # Modify priority or force schedule
    
    # Inspect block utilization
    free_blocks = len(llm.scheduler.block_manager.free_block_ids)
    print(f"🕵️ HACK: Free physical blocks remaining: {free_blocks}")
    
    return original_schedule()

# Hijack scheduler
llm.scheduler.schedule = hacked_schedule
```

---

### B. Dynamic Layer Monkey-Patching (Attention Interception)
You can dynamically intercept the attention layer's forward pass at runtime to inspect outputs, log scaling factors, or dynamically swap backends.

```python
for name, module in llm.model_runner.model.named_modules():
    if module.__class__.__name__ == "Attention":
        original_forward = module.forward
        
        def hacked_forward(q, k, v, self=module):
            print(f"🕵️ HACK: Intercepted Attention Forward!")
            print(f"    - Query Shape: {q.shape}")
            print(f"    - Key Shape: {k.shape}")
            
            # You can inject noise, modify inputs, or change outputs here!
            return original_forward(q, k, v)
            
        module.forward = hacked_forward
```

---

### C. Directly Reading / Writing the physical KV Cache (VRAM hacking)
The physical KV Cache is allocated as a static tensor inside the `ModelRunner`. You can directly query or corrupt the values of historical tokens in VRAM during execution:

```python
# The physical cache buffer is shape (2, num_layers, num_blocks, block_size, num_kv_heads, head_dim)
kv_cache = llm.model_runner.kv_cache

# Inspect the Key Cache of layer 0, block 5
layer_idx = 0
block_id = 5
key_cache_slice = kv_cache[0, layer_idx, block_id] 
print(f"🕵️ HACK: Cached values of block {block_id}: {key_cache_slice.mean().item()}")

# Force corruption (e.g. inject adversarial perturbation or wipe memory)
# kv_cache[0, layer_idx, block_id].fill_(0.0)
```

---

### D. Profiling Latency (TTFT & Decode Speed)
You can measure the exact **Time to First Token (TTFT)** for User 2 by timing the step where `num_tokens > 0` occurs for their request.

```python
user_2_prefill_time = None
t_user_2_start = None

# In the loop:
if step_count == 5:
    t_user_2_start = time.perf_counter()

if t_user_2_start is not None and user_2_prefill_time is None:
    # Check if User 2 is running prefill
    running_ids = [seq.seq_id for seq in llm.scheduler.running]
    if 1 in running_ids: # User 2 ID is 1 (0-indexed)
        # Prefill runs in a step where num_tokens > 0
        user_2_prefill_time = time.perf_counter() - t_user_2_start
        print(f"⏱️ TTFT for User 2: {user_2_prefill_time * 1000:.2f} ms")
```
