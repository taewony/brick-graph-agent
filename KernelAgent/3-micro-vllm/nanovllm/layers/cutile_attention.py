import math
import torch
import numpy as np

HAS_CUTILE = False
try:
    import cuda.tile as ct
    from cuda.tile import RoundingMode as RMd
    HAS_CUTILE = True
except ImportError:
    pass

if HAS_CUTILE:
    INV_LOG_2 = 1.0 / math.log(2)
    ConstInt = ct.Constant[int]
    ConstBool = ct.Constant[bool]

    @ct.kernel(occupancy=2)
    def fmha_prefill_kernel(Q, K, V, Out,
                            qk_scale: float,
                            TILE_D: ConstInt,
                            H: ConstInt,
                            TILE_M: ConstInt,
                            TILE_N: ConstInt,
                            QUERY_GROUP_SIZE: ConstInt,
                            CAUSAL: ConstBool,
                            Q_START_IN_K: int):
        bid_x = ct.bid(0)
        bid_y = ct.bid(1)
        batch_idx = bid_y // H
        head_idx = bid_y % H
        off_kv_h = head_idx // QUERY_GROUP_SIZE

        qk_scale = qk_scale * INV_LOG_2

        offs_m = bid_x * TILE_M + ct.arange(TILE_M, dtype=np.int32)
        offs_m = offs_m[:, None]

        offs_n_tile = ct.arange(TILE_N, dtype=np.int32)
        offs_n_tile = offs_n_tile[None, :]

        m_i = ct.full((TILE_M, 1), -np.inf, dtype=np.float32)
        l_i = ct.full((TILE_M, 1), 0.0, dtype=np.float32)
        acc = ct.full((TILE_M, TILE_D), 0.0, dtype=np.float32)

        q = ct.load(
            Q, index=(batch_idx, head_idx, bid_x, 0), shape=(1, 1, TILE_M, TILE_D),
            padding_mode=ct.PaddingMode.ZERO,
            latency=2,
        ).reshape((TILE_M, TILE_D))

        q_seqlen = Q.shape[2]
        k_seqlen = K.shape[2]
        m_end = (bid_x + 1) * TILE_M
        if CAUSAL:
            mask_start = (Q_START_IN_K + bid_x * TILE_M) // TILE_N
            mask_start = min(mask_start, k_seqlen // TILE_N)
            Tc = ct.cdiv(min(Q_START_IN_K + m_end, k_seqlen), TILE_N)
        else:
            Tc = ct.cdiv(k_seqlen, TILE_N)
            mask_start = k_seqlen // TILE_N

        for j in range(0, Tc):
            k = ct.load(
                K, index=(batch_idx, off_kv_h, 0, j), shape=(1, 1, TILE_D, TILE_N),
                order=(0, 1, 3, 2),
                padding_mode=ct.PaddingMode.ZERO,
                latency=2,
            ).reshape((TILE_D, TILE_N))

            qk = ct.full((TILE_M, TILE_N), 0., dtype=np.float32)
            qk = ct.mma(q, k, qk)

            offs_n = j * TILE_N + offs_n_tile
            qk += ct.where(offs_n < k_seqlen, 0.0, -np.inf)
            if CAUSAL and j >= mask_start:
                mask = ct.where(Q_START_IN_K + offs_m >= offs_n, 0.0, -np.inf)
                qk += mask

            m_ij = ct.maximum(m_i, ct.max(qk, axis=-1, keepdims=True) * qk_scale)
            qk = qk * qk_scale - m_ij

            p = ct.exp2(qk, flush_to_zero=True)
            l_ij = ct.sum(p, axis=-1, keepdims=True)
            alpha = ct.exp2(m_i - m_ij, flush_to_zero=True)

            l_i = l_i * alpha + l_ij
            acc = acc * alpha

            v = ct.load(
                V, index=(batch_idx, off_kv_h, j, 0), shape=(1, 1, TILE_N, TILE_D),
                padding_mode=ct.PaddingMode.ZERO,
                latency=4,
            ).reshape((TILE_N, TILE_D))

            p = p.astype(Q.dtype)
            acc = ct.mma(p, v, acc)
            m_i = m_ij

        acc = ct.truediv(acc, l_i, flush_to_zero=True, rounding_mode=RMd.APPROX)
        acc = ct.where(offs_m < q_seqlen, acc, 0.0)
        acc = acc.reshape((1, 1, TILE_M, TILE_D)).astype(Out.dtype)
        ct.store(Out, index=(batch_idx, head_idx, bid_x, 0), tile=acc)

    @ct.kernel(occupancy=2)
    def fmha_prefill_paged_kernel(
        Q, K_cache, V_cache, block_table, Out,
        qk_scale: float,
        K_SEQLEN: int,
        Q_START_IN_K: int,
        TILE_D: ConstInt,
        H: ConstInt,
        TILE_M: ConstInt,
        TILE_N: ConstInt,
        QUERY_GROUP_SIZE: ConstInt,
        BLOCK_SIZE: ConstInt,
        CAUSAL: ConstBool,
    ):
        bid_x = ct.bid(0)
        bid_y = ct.bid(1)
        batch_idx = bid_y // H
        head_idx = bid_y % H
        off_kv_h = head_idx // QUERY_GROUP_SIZE

        qk_scale = qk_scale * INV_LOG_2
        q_seqlen = Q.shape[2]
        k_seqlen = K_SEQLEN

        offs_m = bid_x * TILE_M + ct.arange(TILE_M, dtype=np.int32)
        offs_m = offs_m[:, None]
        offs_n_tile = ct.arange(TILE_N, dtype=np.int32)
        offs_n_tile = offs_n_tile[None, :]

        m_i = ct.full((TILE_M, 1), -np.inf, dtype=np.float32)
        l_i = ct.full((TILE_M, 1), 0.0, dtype=np.float32)
        acc = ct.full((TILE_M, TILE_D), 0.0, dtype=np.float32)

        q = ct.load(
            Q, index=(batch_idx, head_idx, bid_x, 0), shape=(1, 1, TILE_M, TILE_D),
            padding_mode=ct.PaddingMode.ZERO,
            latency=2,
        ).reshape((TILE_M, TILE_D))

        m_end = (bid_x + 1) * TILE_M
        if CAUSAL:
            mask_start = (Q_START_IN_K + bid_x * TILE_M) // TILE_N
            mask_start = min(mask_start, k_seqlen // TILE_N)
            Tc = ct.cdiv(min(Q_START_IN_K + m_end, k_seqlen), TILE_N)
        else:
            Tc = ct.cdiv(k_seqlen, TILE_N)
            mask_start = k_seqlen // TILE_N

        for j in range(0, Tc):
            token_start = j * TILE_N
            logical_block = token_start // BLOCK_SIZE
            block_tile = token_start % BLOCK_SIZE
            physical_block_id = ct.load(block_table, index=(batch_idx, logical_block), shape=(1, 1)).reshape(())

            k = ct.load(
                K_cache, index=(physical_block_id, block_tile, off_kv_h, 0),
                shape=(1, TILE_D, 1, TILE_N),
                order=(0, 3, 2, 1),
                padding_mode=ct.PaddingMode.ZERO,
                latency=2,
            ).reshape((TILE_D, TILE_N))

            qk = ct.full((TILE_M, TILE_N), 0., dtype=np.float32)
            qk = ct.mma(q, k, qk)

            offs_n = token_start + offs_n_tile
            qk += ct.where(offs_n < k_seqlen, 0.0, -np.inf)
            if CAUSAL and j >= mask_start:
                mask = ct.where(Q_START_IN_K + offs_m >= offs_n, 0.0, -np.inf)
                qk += mask

            m_ij = ct.maximum(m_i, ct.max(qk, axis=-1, keepdims=True) * qk_scale)
            qk = qk * qk_scale - m_ij

            p = ct.exp2(qk, flush_to_zero=True)
            l_ij = ct.sum(p, axis=-1, keepdims=True)
            alpha = ct.exp2(m_i - m_ij, flush_to_zero=True)

            l_i = l_i * alpha + l_ij
            acc = acc * alpha

            v = ct.load(
                V_cache, index=(physical_block_id, block_tile, off_kv_h, 0),
                shape=(1, TILE_N, 1, TILE_D),
                padding_mode=ct.PaddingMode.ZERO,
                latency=4,
            ).reshape((TILE_N, TILE_D))

            p = p.astype(Q.dtype)
            acc = ct.mma(p, v, acc)
            m_i = m_ij

        acc = ct.truediv(acc, l_i, flush_to_zero=True, rounding_mode=RMd.APPROX)
        acc = ct.where(offs_m < q_seqlen, acc, 0.0)
        acc = acc.reshape((1, 1, TILE_M, TILE_D)).astype(Out.dtype)
        ct.store(Out, index=(batch_idx, head_idx, bid_x, 0), tile=acc)

    @ct.kernel(occupancy=2)
    def paged_decode_kernel(
        Q, K_cache, V_cache, block_table, context_lens, Out,
        qk_scale: float,
        TILE_D: ConstInt,
        H: ConstInt,
        QUERY_GROUP_SIZE: ConstInt,
        BLOCK_SIZE: ConstInt,
    ):
        bid_y = ct.bid(0)
        batch_idx = bid_y // H
        head_idx = bid_y % H
        off_kv_h = head_idx // QUERY_GROUP_SIZE

        cur_len = ct.load(context_lens, index=(batch_idx,), shape=(1,)).reshape(())
        qk_scale = qk_scale * INV_LOG_2

        q = ct.load(
            Q, index=(batch_idx, head_idx, 0, 0), shape=(1, 1, 1, TILE_D),
            padding_mode=ct.PaddingMode.ZERO,
        ).reshape((1, TILE_D))

        m_i = ct.full((1, 1), -np.inf, dtype=np.float32)
        l_i = ct.full((1, 1), 0.0, dtype=np.float32)
        acc = ct.full((1, TILE_D), 0.0, dtype=np.float32)

        num_seq_blocks = ct.cdiv(cur_len, BLOCK_SIZE)

        for j in range(0, num_seq_blocks):
            physical_block_id = ct.load(block_table, index=(batch_idx, j), shape=(1, 1)).reshape(())

            k = ct.load(
                K_cache, index=(physical_block_id, 0, off_kv_h, 0), shape=(1, TILE_D, 1, BLOCK_SIZE),
                order=(0, 3, 2, 1),
                padding_mode=ct.PaddingMode.ZERO,
            ).reshape((TILE_D, BLOCK_SIZE))

            qk = ct.full((1, BLOCK_SIZE), 0., dtype=np.float32)
            qk = ct.mma(q, k, qk)

            if j == num_seq_blocks - 1:
                offs_n = j * BLOCK_SIZE + ct.arange(BLOCK_SIZE, dtype=np.int32)[None, :]
                mask = ct.where(offs_n < cur_len, 0.0, -np.inf)
                qk += mask

            m_ij = ct.maximum(m_i, ct.max(qk, axis=-1, keepdims=True) * qk_scale)
            qk = qk * qk_scale - m_ij

            p = ct.exp2(qk, flush_to_zero=True)
            l_ij = ct.sum(p, axis=-1, keepdims=True)
            alpha = ct.exp2(m_i - m_ij, flush_to_zero=True)

            l_i = l_i * alpha + l_ij
            acc = acc * alpha

            v = ct.load(
                V_cache, index=(physical_block_id, 0, off_kv_h, 0), shape=(1, BLOCK_SIZE, 1, TILE_D),
                padding_mode=ct.PaddingMode.ZERO,
            ).reshape((BLOCK_SIZE, TILE_D))

            p = p.astype(Q.dtype)
            acc = ct.mma(p, v, acc)
            m_i = m_ij

        acc = ct.truediv(acc, l_i, flush_to_zero=True, rounding_mode=RMd.APPROX)
        acc = acc.reshape((1, 1, 1, TILE_D)).astype(Out.dtype)
        ct.store(Out, index=(batch_idx, head_idx, 0, 0), tile=acc)


def _bhtd_view(tokens_h_d: torch.Tensor, start: int, length: int) -> torch.Tensor:
    return tokens_h_d.narrow(0, start, length).permute(1, 0, 2).unsqueeze(0)


def _launch_grid(seqlen_q: int, num_heads: int, tile_m: int) -> tuple[int, int, int]:
    return (max(1, math.ceil(seqlen_q / tile_m)), num_heads, 1)


def _cutile_fmha_prefill_padded(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    scale: float,
    causal: bool,
    k_cache: torch.Tensor = None,
    v_cache: torch.Tensor = None,
    block_table: torch.Tensor = None,
) -> torch.Tensor:
    tile_m, tile_n = 64, 64
    pad_len_q = max(64, math.ceil(max_seqlen_q / tile_m) * tile_m)
    pad_len_k = max(64, math.ceil(max_seqlen_k / tile_n) * tile_n)

    B = cu_seqlens_q.numel() - 1
    total_tokens_q = q.shape[0]
    num_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    head_dim = q.shape[2]

    q_padded = torch.empty((B, num_heads, pad_len_q, head_dim), dtype=q.dtype, device=q.device)
    k_padded = torch.empty((B, num_kv_heads, pad_len_k, head_dim), dtype=k.dtype, device=k.device)
    v_padded = torch.empty((B, num_kv_heads, pad_len_k, head_dim), dtype=v.dtype, device=v.device)

    cu_q = cu_seqlens_q.tolist()
    cu_k = cu_seqlens_k.tolist()
    for b in range(B):
        start_q = cu_q[b]
        end_q = cu_q[b + 1]
        seqlen_q = end_q - start_q
        q_padded[b, :, :seqlen_q, :] = q[start_q:end_q].transpose(0, 1)

        start_k = cu_k[b]
        end_k = cu_k[b + 1]
        seqlen_k = end_k - start_k
        if block_table is not None and k_cache is not None and k_cache.numel() > 0:
            block_size = k_cache.shape[1]
            idx_seq = torch.arange(seqlen_k, device=block_table.device)
            logical_blk = torch.div(idx_seq, block_size, rounding_mode="floor")
            offset = idx_seq % block_size
            physical_blk = block_table[b, logical_blk].long()
            k_padded[b, :, :seqlen_k, :] = k_cache[physical_blk, offset].transpose(0, 1)
            v_padded[b, :, :seqlen_k, :] = v_cache[physical_blk, offset].transpose(0, 1)
        else:
            k_padded[b, :, :seqlen_k, :] = k[start_k:end_k].transpose(0, 1)
            v_padded[b, :, :seqlen_k, :] = v[start_k:end_k].transpose(0, 1)

    padded_out = torch.empty((B, num_heads, pad_len_q, head_dim), dtype=q.dtype, device=q.device)
    grid = (pad_len_q // tile_m, B * num_heads, 1)
    query_group_size = num_heads // num_kv_heads

    ct.launch(torch.cuda.current_stream(), grid, fmha_prefill_kernel, (
        q_padded, k_padded, v_padded, padded_out,
        scale, head_dim, num_heads,
        tile_m, tile_n, query_group_size, causal, 0,
    ))

    out = torch.empty((total_tokens_q, num_heads, head_dim), dtype=q.dtype, device=q.device)
    for b in range(B):
        start_q = cu_q[b]
        end_q = cu_q[b + 1]
        seqlen_q = end_q - start_q
        out[start_q:end_q] = padded_out[b, :, :seqlen_q, :].transpose(0, 1)

    return out

def cutile_fmha_prefill(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    cu_seqlens_q: torch.Tensor, cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int, max_seqlen_k: int, scale: float, causal: bool,
    k_cache: torch.Tensor = None, v_cache: torch.Tensor = None,
    block_table: torch.Tensor = None,
) -> torch.Tensor:
    if not HAS_CUTILE:
        raise RuntimeError("cuTile is not installed or import failed.")

    tile_m, tile_n = 64, 64
    total_tokens_q = q.shape[0]
    num_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    head_dim = q.shape[2]
    query_group_size = num_heads // num_kv_heads
    out = torch.empty((total_tokens_q, num_heads, head_dim), dtype=q.dtype, device=q.device)
    stream = torch.cuda.current_stream()
    batch_size = cu_seqlens_q.numel() - 1

    import os
    strategy = os.environ.get("NANO_VLLM_CUTILE_PREFILL_STRATEGY", "hybrid").lower()
    if strategy not in {"hybrid", "direct", "padded"}:
        strategy = "hybrid"

    use_paged_prefill = block_table is not None and k_cache is not None and k_cache.numel() > 0
    if not use_paged_prefill and (strategy == "padded" or (strategy == "hybrid" and batch_size > 1)):
        return _cutile_fmha_prefill_padded(
            q, k, v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            scale=scale,
            causal=causal,
            k_cache=k_cache,
            v_cache=v_cache,
            block_table=block_table,
        )

    if use_paged_prefill:
        block_size = k_cache.shape[1]
        if block_size % tile_n != 0:
            tile_n = block_size

    cu_q = cu_seqlens_q.tolist()
    cu_k = cu_seqlens_k.tolist()
    for b in range(batch_size):
        start_q = cu_q[b]
        end_q = cu_q[b + 1]
        start_k = cu_k[b]
        end_k = cu_k[b + 1]
        seqlen_q = end_q - start_q
        seqlen_k = end_k - start_k
        if seqlen_q <= 0:
            continue

        q_view = _bhtd_view(q, start_q, seqlen_q)
        out_view = _bhtd_view(out, start_q, seqlen_q)
        q_start_in_k = max(0, seqlen_k - seqlen_q)
        grid = _launch_grid(seqlen_q, num_heads, tile_m)

        if use_paged_prefill:
            # Paged prefill: read K/V directly from the block table without
            # materializing a contiguous cache. The kernel's bottom-right causal
            # mask (Q_START_IN_K + offs_m >= offs_n) is correct; the earlier
            # "bug" was the test golden reference using top-left is_causal=True.
            block_table_view = block_table.narrow(0, b, 1)
            ct.launch(stream, grid, fmha_prefill_paged_kernel, (
                q_view, k_cache, v_cache, block_table_view, out_view,
                scale, seqlen_k, q_start_in_k,
                head_dim, num_heads, tile_m, tile_n, query_group_size, k_cache.shape[1], causal,
            ))
        else:
            k_view = _bhtd_view(k, start_k, seqlen_k)
            v_view = _bhtd_view(v, start_k, seqlen_k)
            ct.launch(stream, grid, fmha_prefill_kernel, (
                q_view, k_view, v_view, out_view,
                scale, head_dim, num_heads,
                tile_m, tile_n, query_group_size, causal, q_start_in_k,
            ))

    return out


def cutile_fmha_paged_decode(
    q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
    block_table: torch.Tensor, context_lens: torch.Tensor, scale: float,
    block_size: int,
) -> torch.Tensor:
    if not HAS_CUTILE:
        raise RuntimeError("cuTile is not installed or import failed.")

    B, H, D = q.shape
    q_4d = q.unsqueeze(2)
    Out = torch.empty((B, H, 1, D), dtype=q.dtype, device=q.device)

    grid = (B * H, 1, 1)
    query_group_size = H // k_cache.shape[2]

    ct.launch(torch.cuda.current_stream(), grid, paged_decode_kernel, (
        q_4d, k_cache, v_cache, block_table, context_lens, Out, scale,
        D, H, query_group_size, block_size,
    ))

    return Out.transpose(1, 2)




