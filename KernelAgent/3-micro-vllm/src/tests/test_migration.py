import os
import sys
import torch
import torch.nn as nn
import math

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.tests.utils import fix_seed, compare_outputs

# =========================================================================
# 1. Migrated PyTorch KV Cache Store implementation
# =========================================================================
def store_kvcache_pytorch(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    """
    Pure PyTorch implementation of KV cache storing (eliminates Triton).
    """
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


def test_store_kvcache_cpu():
    """
    [TDD Step 1] Verify the pure PyTorch store_kvcache implementation on CPU.
    """
    print("\n[TDD Step 1] Verification of store_kvcache_pytorch (CPU)")
    fix_seed(42)
    
    # Dimensions: 4 tokens, 6 heads, 64 head_dim
    N, num_heads, head_dim = 4, 6, 64
    D = num_heads * head_dim
    num_blocks, block_size = 8, 16
    
    # Allocate cache buffers
    k_cache_ref = torch.zeros((num_blocks, block_size, num_heads, head_dim))
    v_cache_ref = torch.zeros((num_blocks, block_size, num_heads, head_dim))
    k_cache_target = torch.zeros((num_blocks, block_size, num_heads, head_dim))
    v_cache_target = torch.zeros((num_blocks, block_size, num_heads, head_dim))
    
    # New keys and values
    key = torch.randn(N, num_heads, head_dim)
    value = torch.randn(N, num_heads, head_dim)
    
    # Define slots to write to (including a -1 for padding/inactive token)
    slot_mapping = torch.tensor([5, 12, -1, 38], dtype=torch.int32)
    
    # 1. Reference: Eager loop implementation
    k_cache_ref_flat = k_cache_ref.view(-1, D)
    v_cache_ref_flat = v_cache_ref.view(-1, D)
    for idx in range(N):
        slot = slot_mapping[idx].item()
        if slot != -1:
            k_cache_ref_flat[slot] = key[idx].view(D)
            v_cache_ref_flat[slot] = value[idx].view(D)
            
    # 2. Target: PyTorch optimized implementation
    store_kvcache_pytorch(key, value, k_cache_target, v_cache_target, slot_mapping)
    
    # 3. Compare Results
    k_match = compare_outputs(k_cache_ref, k_cache_target)
    v_match = compare_outputs(v_cache_ref, v_cache_target)
    if k_match and v_match:
        print("✅ store_kvcache_pytorch Verification passed!")
    else:
        print("❌ store_kvcache_pytorch Verification failed!")


# =========================================================================
# 2. Prefill Causal Padding verification
# =========================================================================
def test_prefill_causal_padding_cpu():
    """
    [TDD Step 2] Verify the mathematical equivalence of causal padding.
    """
    print("\n[TDD Step 2] Verification of Causal Padding (CPU)")
    fix_seed(42)
    
    # Sequence length T = 32 (less than block size 64)
    B, H, T, D = 1, 6, 32, 64
    pad_len = 64 - T
    
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)
    
    # 1. Reference: PyTorch Native Causal Attention on unpadded inputs
    q_ref = q
    k_ref = k
    v_ref = v
    o_ref = torch.nn.functional.scaled_dot_product_attention(q_ref, k_ref, v_ref, is_causal=True)
    
    # 2. Target: Causal attention on inputs padded to 64, then sliced back
    q_padded = torch.nn.functional.pad(q, (0, 0, 0, pad_len))
    k_padded = torch.nn.functional.pad(k, (0, 0, 0, pad_len))
    v_padded = torch.nn.functional.pad(v, (0, 0, 0, pad_len))
    
    o_padded = torch.nn.functional.scaled_dot_product_attention(q_padded, k_padded, v_padded, is_causal=True)
    o_target = o_padded[:, :, :T, :]
    
    # 3. Compare
    match = compare_outputs(o_ref, o_target)
    if match:
        print("✅ Causal Sequence Padding Verification passed!")
    else:
        print("❌ Causal Sequence Padding Verification failed!")


# =========================================================================
# 3. Paged KV Cache vs Contiguous KV Cache Attention verification
# =========================================================================
def test_paged_attention_lookup_cpu():
    """
    [TDD Step 3] Verify that retrieving from Paged KV Cache block table
    produces identical attention inputs to contiguous cache blocks.
    """
    print("\n[TDD Step 3] Verification of Paged KV Cache lookup mapping (CPU)")
    fix_seed(42)
    
    B, H, T_q, D = 1, 6, 1, 64  # Decode step: 1 token
    block_size = 16
    context_len = 35 # We have generated 35 tokens so far
    num_blocks = math.ceil(context_len / block_size)  # 3 blocks
    
    # Allocate contiguous KV Cache for comparison
    k_contiguous = torch.randn(B, H, context_len, D)
    
    # Allocate Paged KV Cache: [total_physical_blocks, block_size, H, D]
    total_physical_blocks = 10
    k_paged = torch.zeros(total_physical_blocks, block_size, H, D)
    
    # Create block table mapping logical blocks to physical block IDs
    # Let's map logical blocks 0, 1, 2 to physical blocks 5, 2, 7
    block_table = [5, 2, 7]
    
    # Populate Paged cache using block table mapping
    for i in range(context_len):
        logical_block_idx = i // block_size
        block_offset = i % block_size
        physical_block_id = block_table[logical_block_idx]
        
        # Copy contiguous element to paged location
        k_paged[physical_block_id, block_offset, :, :] = k_contiguous[0, :, i, :]
        
    # Reconstruct contiguous cache back from paged cache using block table to verify mapping correctness
    k_reconstructed = torch.zeros(B, H, context_len, D)
    for i in range(context_len):
        logical_block_idx = i // block_size
        block_offset = i % block_size
        physical_block_id = block_table[logical_block_idx]
        
        k_reconstructed[0, :, i, :] = k_paged[physical_block_id, block_offset, :, :]
        
    match = compare_outputs(k_contiguous, k_reconstructed)
    if match:
        print("✅ Paged KV Cache block mapping Verification passed!")
    else:
        print("❌ Paged KV Cache block mapping Verification failed!")


# =========================================================================
# 4. cuTile FMHA Prefill attention verification on GPU
# =========================================================================
def test_cutile_prefill_attention_gpu():
    """
    [TDD Step 4] Verify cuTile FMHA Prefill attention on GPU vs Golden PyTorch.
    """
    print("\n[TDD Step 4] Verification of cutile_fmha_prefill (GPU)")
    if not torch.cuda.is_available():
        print("⏭️ Skip: CUDA is not available.")
        return
    try:
        from nanovllm.layers.cutile_attention import HAS_CUTILE, cutile_fmha_prefill
    except ImportError:
        print("⏭️ Skip: Could not import cutile_attention module.")
        return

    if not HAS_CUTILE:
        print("⏭️ Skip: cuTile is not installed/available.")
        return
        
    fix_seed(42)
    B, H, D = 2, 8, 64
    seq_lens = [32, 45] # Variable lengths
    cu_seqlens_q = torch.tensor([0, 32, 77], dtype=torch.int32, device="cuda")
    cu_seqlens_k = torch.tensor([0, 32, 77], dtype=torch.int32, device="cuda")
    total_tokens = 77
    
    # Generate random Q, K, V on GPU (float16)
    q = torch.randn(total_tokens, H, D, dtype=torch.float16, device="cuda")
    k = torch.randn(total_tokens, H, D, dtype=torch.float16, device="cuda")
    v = torch.randn(total_tokens, H, D, dtype=torch.float16, device="cuda")
    
    # Golden reference using PyTorch SDPA
    o_refs = []
    for b in range(B):
        start = cu_seqlens_q[b].item()
        end = cu_seqlens_q[b+1].item()
        qb = q[start:end].transpose(0, 1).unsqueeze(0) # (1, H, seqlen, D)
        kb = k[start:end].transpose(0, 1).unsqueeze(0)
        vb = v[start:end].transpose(0, 1).unsqueeze(0)
        ob = torch.nn.functional.scaled_dot_product_attention(qb, kb, vb, is_causal=True)
        o_refs.append(ob.squeeze(0).transpose(0, 1)) # (seqlen, H, D)
    o_ref = torch.cat(o_refs, dim=0)
    
    # Target: cuTile FMHA prefill (Standard)
    match_standard = False
    try:
        o_target = cutile_fmha_prefill(
            q, k, v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=45,
            max_seqlen_k=45,
            scale=1.0 / math.sqrt(D),
            causal=True
        )
        match_standard = compare_outputs(o_ref, o_target, rtol=1e-2, atol=1e-2)
        if match_standard:
            print("✅ cuTile FMHA Prefill (Standard) GPU Verification passed!")
        else:
            print("❌ cuTile FMHA Prefill (Standard) GPU Verification failed!")
    except Exception as e:
        print(f"❌ cuTile FMHA Prefill (Standard) compilation or execution failed: {e}")

    # Target: cuTile FMHA prefill (Prefix Cache)
    match_prefix = False
    try:
        cu_seqlens_q_pc = torch.tensor([0, 16, 24], dtype=torch.int32, device="cuda")
        cu_seqlens_k_pc = torch.tensor([0, 48, 56], dtype=torch.int32, device="cuda")
        block_size = 16
        num_blocks = 10
        
        k_cache_pc = torch.randn(num_blocks, block_size, H, D, dtype=torch.float16, device="cuda")
        v_cache_pc = torch.randn(num_blocks, block_size, H, D, dtype=torch.float16, device="cuda")
        block_table_pc = torch.tensor([[1, 3, 5], [2, 4, -1]], dtype=torch.int32, device="cuda")
        
        q_pc = torch.randn(24, H, D, dtype=torch.float16, device="cuda")
        k_dummy = torch.randn(24, H, D, dtype=torch.float16, device="cuda")
        v_dummy = torch.randn(24, H, D, dtype=torch.float16, device="cuda")
        
        o_refs_pc = []
        for b in range(B):
            start_q = cu_seqlens_q_pc[b].item()
            end_q = cu_seqlens_q_pc[b+1].item()
            start_k = cu_seqlens_k_pc[b].item()
            end_k = cu_seqlens_k_pc[b+1].item()
            seqlen_q = end_q - start_q
            seqlen_k = end_k - start_k
            
            qb = q_pc[start_q:end_q].transpose(0, 1).unsqueeze(0) # (1, H, seqlen_q, D)
            kb = torch.zeros(1, H, seqlen_k, D, dtype=torch.float16, device="cuda")
            vb = torch.zeros(1, H, seqlen_k, D, dtype=torch.float16, device="cuda")
            for i in range(seqlen_k):
                logical_blk = i // block_size
                offset = i % block_size
                physical_blk = block_table_pc[b, logical_blk].item()
                kb[0, :, i, :] = k_cache_pc[physical_blk, offset, :, :]
                vb[0, :, i, :] = v_cache_pc[physical_blk, offset, :, :]
                
            # 프리픽스 캐시는 bottom-right 인과 마스크(q_i attends k_j iff j <= i + (seqlen_k - seqlen_q)).
            # PyTorch is_causal=True는 q_len != k_len일 때 top-left(j <= i)라서 프리픽스 오프셋을 무시하므로,
            # 명시적 float 마스크(0 = attend, -inf = masked)로 대체한다.
            q_idx = torch.arange(seqlen_q, device="cuda")[:, None].float()
            k_idx = torch.arange(seqlen_k, device="cuda")[None, :].float()
            attn_mask = torch.where(k_idx <= q_idx + (seqlen_k - seqlen_q), 0.0, float("-inf"))
            # SDPA의 flash backend는 float 마스크(attn bias) dtype이 query dtype과 일치해야 함.
            attn_mask = attn_mask.to(qb.dtype)
            ob = torch.nn.functional.scaled_dot_product_attention(
                qb, kb, vb, attn_mask=attn_mask, is_causal=False
            )
            o_refs_pc.append(ob.squeeze(0).transpose(0, 1))
        o_ref_pc = torch.cat(o_refs_pc, dim=0)
        
        o_target_pc = cutile_fmha_prefill(
            q_pc, k_dummy, v_dummy,
            cu_seqlens_q=cu_seqlens_q_pc,
            cu_seqlens_k=cu_seqlens_k_pc,
            max_seqlen_q=16,
            max_seqlen_k=48,
            scale=1.0 / math.sqrt(D),
            causal=True,
            k_cache=k_cache_pc,
            v_cache=v_cache_pc,
            block_table=block_table_pc
        )
        match_prefix = compare_outputs(o_ref_pc, o_target_pc, rtol=1e-2, atol=1e-2)
        if match_prefix:
            print("✅ cuTile FMHA Prefill (Prefix Cache) GPU Verification passed!")
        else:
            print("❌ cuTile FMHA Prefill (Prefix Cache) GPU Verification failed!")
    except Exception as e:
        print(f"❌ cuTile FMHA Prefill (Prefix Cache) compilation or execution failed: {e}")


# =========================================================================
# 5. cuTile Paged Decode attention verification on GPU
# =========================================================================
def test_cutile_paged_decode_attention_gpu():
    """
    [TDD Step 5] Verify cuTile Paged Decode attention on GPU vs Golden PyTorch.
    """
    print("\n[TDD Step 5] Verification of cutile_fmha_paged_decode (GPU)")
    if not torch.cuda.is_available():
        print("⏭️ Skip: CUDA is not available.")
        return
    try:
        from nanovllm.layers.cutile_attention import HAS_CUTILE, cutile_fmha_paged_decode
    except ImportError:
        print("⏭️ Skip: Could not import cutile_attention module.")
        return

    if not HAS_CUTILE:
        print("⏭️ Skip: cuTile is not installed/available.")
        return
        
    fix_seed(42)
    B, H, D = 2, 8, 64
    block_size = 16
    context_lens = torch.tensor([35, 12], dtype=torch.int32, device="cuda")
    
    # Query: 1 token per batch item (B, H, D)
    q = torch.randn(B, H, D, dtype=torch.float16, device="cuda")
    
    # KV cache allocation
    num_blocks = 8
    k_cache = torch.randn(num_blocks, block_size, H, D, dtype=torch.float16, device="cuda")
    v_cache = torch.randn(num_blocks, block_size, H, D, dtype=torch.float16, device="cuda")
    
    # Block table mapping
    block_table = torch.tensor([[1, 3, 5], [2, -1, -1]], dtype=torch.int32, device="cuda")
    
    # Golden reference using PyTorch SDPA
    o_refs = []
    for b in range(B):
        cur_len = context_lens[b].item()
        kb = torch.zeros(1, H, cur_len, D, dtype=torch.float16, device="cuda")
        vb = torch.zeros(1, H, cur_len, D, dtype=torch.float16, device="cuda")
        for i in range(cur_len):
            logical_blk = i // block_size
            offset = i % block_size
            physical_blk = block_table[b, logical_blk].item()
            kb[0, :, i, :] = k_cache[physical_blk, offset, :, :]
            vb[0, :, i, :] = v_cache[physical_blk, offset, :, :]
        
        qb = q[b].unsqueeze(0).unsqueeze(2) # (1, H, 1, D)
        ob = torch.nn.functional.scaled_dot_product_attention(qb, kb, vb, is_causal=False)
        o_refs.append(ob.squeeze(2)) # (1, H, D)
    o_ref = torch.cat(o_refs, dim=0) # (B, H, D)
    
    # Target: cuTile paged decode attention
    try:
        o_target_raw = cutile_fmha_paged_decode(
            q, k_cache, v_cache,
            block_table=block_table,
            context_lens=context_lens,
            scale=1.0 / math.sqrt(D),
            block_size=block_size
        )
        o_target = o_target_raw.squeeze(1)
        
        # Compare
        match = compare_outputs(o_ref, o_target, rtol=1e-2, atol=1e-2)
        if match:
            print("✅ cuTile Paged Decode GPU Verification passed!")
        else:
            print("❌ cuTile Paged Decode GPU Verification failed!")
    except Exception as e:
        print(f"❌ cuTile Paged Decode compilation or execution failed: {e}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Running nano-vllm migration TDD suite on device: {device}")
    
    test_store_kvcache_cpu()
    test_prefill_causal_padding_cpu()
    test_paged_attention_lookup_cpu()
    
    test_cutile_prefill_attention_gpu()
    test_cutile_paged_decode_attention_gpu()

