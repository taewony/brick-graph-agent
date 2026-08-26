import os
import torch
from torch import nn
from nanovllm.utils.context import get_context
from nanovllm.layers.cutile_attention import cutile_fmha_prefill, cutile_fmha_paged_decode, HAS_CUTILE

HAS_TRITON = False
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    pass

HAS_FLASH = False
try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
    HAS_FLASH = True
except ImportError:
    pass

if HAS_CUTILE:
    import cuda.tile as ct
    ConstInt = ct.Constant[int]

    @ct.kernel(occupancy=2)
    def store_kvcache_cutile_kernel(
        key_flat,
        value_flat,
        k_cache_flat,
        v_cache_flat,
        slot_mapping,
        D: ConstInt,
    ):
        idx = ct.bid(0)
        slot = ct.load(slot_mapping, index=(idx,), shape=(1,)).reshape(())
        if slot != -1:
            key = ct.load(key_flat, index=(idx, 0), shape=(1, D))
            value = ct.load(value_flat, index=(idx, 0), shape=(1, D))
            ct.store(k_cache_flat, index=(slot, 0), tile=key)
            ct.store(v_cache_flat, index=(slot, 0), tile=value)

    def store_kvcache_cutile(
        key: torch.Tensor,
        value: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ):
        N, num_heads, head_dim = key.shape
        D = num_heads * head_dim
        assert key.stride(-1) == 1 and value.stride(-1) == 1
        assert key.stride(1) == head_dim and value.stride(1) == head_dim
        assert k_cache.stride(1) == D and v_cache.stride(1) == D
        assert slot_mapping.numel() == N
        key_flat = key.view(-1, D)
        value_flat = value.view(-1, D)
        k_cache_flat = k_cache.view(-1, D)
        v_cache_flat = v_cache.view(-1, D)
        ct.launch(
            torch.cuda.current_stream(),
            (N, 1, 1),
            store_kvcache_cutile_kernel,
            (key_flat, value_flat, k_cache_flat, v_cache_flat, slot_mapping, D),
        )


if HAS_TRITON:
    @triton.jit
    def store_kvcache_kernel(
        key_ptr,
        key_stride,
        value_ptr,
        value_stride,
        k_cache_ptr,
        v_cache_ptr,
        slot_mapping_ptr,
        D: tl.constexpr,
    ):
        idx = tl.program_id(0)
        slot = tl.load(slot_mapping_ptr + idx)
        if slot == -1: return
        key_offsets = idx * key_stride + tl.arange(0, D)
        value_offsets = idx * value_stride + tl.arange(0, D)
        key = tl.load(key_ptr + key_offsets)
        value = tl.load(value_ptr + value_offsets)
        cache_offsets = slot * D + tl.arange(0, D)
        tl.store(k_cache_ptr + cache_offsets, key)
        tl.store(v_cache_ptr + cache_offsets, value)

    def store_kvcache_triton(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
        N, num_heads, head_dim = key.shape
        D = num_heads * head_dim
        assert key.stride(-1) == 1 and value.stride(-1) == 1
        assert key.stride(1) == head_dim and value.stride(1) == head_dim
        assert k_cache.stride(1) == D and v_cache.stride(1) == D
        assert slot_mapping.numel() == N
        store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


def store_kvcache_pytorch(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    D = key.shape[1] * key.shape[2]
    k_cache_flat = k_cache.view(-1, D)
    v_cache_flat = v_cache.view(-1, D)
    key_flat = key.view(-1, D)
    value_flat = value.view(-1, D)

    mask = slot_mapping != -1
    valid_slots = slot_mapping[mask].long()

    k_cache_flat[valid_slots] = key_flat[mask]
    v_cache_flat[valid_slots] = value_flat[mask]


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    context = get_context()
    use_cutile = context.use_cutile or (os.environ.get("NANO_VLLM_USE_CUTILE", "0") == "1")

    if use_cutile and HAS_CUTILE:
        store_kvcache_cutile(key, value, k_cache, v_cache, slot_mapping)
    elif HAS_TRITON and not use_cutile:
        store_kvcache_triton(key, value, k_cache, v_cache, slot_mapping)
    else:
        store_kvcache_pytorch(key, value, k_cache, v_cache, slot_mapping)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        use_cutile = context.use_cutile or (os.environ.get("NANO_VLLM_USE_CUTILE", "0") == "1")

        if use_cutile:
            if context.is_prefill:
                o = cutile_fmha_prefill(
                    q, k, v,
                    cu_seqlens_q=context.cu_seqlens_q,
                    cu_seqlens_k=context.cu_seqlens_k,
                    max_seqlen_q=context.max_seqlen_q,
                    max_seqlen_k=context.max_seqlen_k,
                    scale=self.scale,
                    causal=True,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    block_table=context.block_tables,
                )
            else:
                block_size = k_cache.shape[1]
                o = cutile_fmha_paged_decode(
                    q, k_cache, v_cache,
                    block_table=context.block_tables,
                    context_lens=context.context_lens,
                    scale=self.scale,
                    block_size=block_size,
                )
        else:
            if not HAS_FLASH:
                raise RuntimeError("FlashAttention is not installed/available, and use_cutile is False/disabled.")
            if context.is_prefill:
                if context.block_tables is not None:
                    k, v = k_cache, v_cache
                o = flash_attn_varlen_func(q, k, v,
                                           max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                           max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                           softmax_scale=self.scale, causal=True, block_table=context.block_tables)
            else:
                o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                            cache_seqlens=context.context_lens, block_table=context.block_tables,
                                            softmax_scale=self.scale, causal=True)
        return o
