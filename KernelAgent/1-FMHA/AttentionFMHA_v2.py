# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import cuda.tile as ct
try:
    import cuda.tile_experimental as ct_experimental
except ImportError:
    ct_experimental = None
import torch
import math
import sys

from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import sdpa_kernel, SDPBackend
from types import SimpleNamespace

import numpy as np
from cuda.tile import RoundingMode as RMd

INV_LOG_2 = 1.0 / math.log(2)
ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]

@ct.kernel(occupancy=2)
def fmha_kernel(Q, K, V, Out,
                qk_scale: float,
                input_pos: int,
                TILE_D: ConstInt,  # TILE_D = hidden_size
                H: ConstInt,
                TILE_M: ConstInt,
                TILE_N: ConstInt,
                QUERY_GROUP_SIZE: ConstInt,
                CAUSAL: ConstBool,
                EVEN_K: ConstBool):
    # Map block IDs to batch and head indices
    bid_x = ct.bid(0)
    bid_y = ct.bid(1)
    batch_idx = bid_y // H
    head_idx = bid_y % H
    off_kv_h = head_idx // QUERY_GROUP_SIZE

    # Adjust qk_scale for exp2
    qk_scale = qk_scale * INV_LOG_2

    # Initialize offsets for current query tile (M-dimension)
    offs_m = bid_x * TILE_M + ct.arange(TILE_M, dtype=np.int32)
    offs_m += input_pos
    offs_m = offs_m[:, None]

    # Initialize local offsets for key/value tile (N-dimension)
    offs_n_tile = ct.arange(TILE_N, dtype=np.int32)
    offs_n_tile = offs_n_tile[None, :]

    # Allocate Shared Memory (SRAM Staging to reduce Register Pressure)
    smem_q = ct.shared_memory((TILE_M, TILE_D), dtype=Q.dtype)
    smem_k = ct.shared_memory((TILE_D, TILE_N), dtype=K.dtype)
    smem_v = ct.shared_memory((TILE_N, TILE_D), dtype=V.dtype)

    # Initialize online softmax accumulators in float32 for stability
    m_i = ct.full((TILE_M, 1), -np.inf, dtype=np.float32)
    l_i = ct.full((TILE_M, 1), 0.0, dtype=np.float32)
    acc = ct.full((TILE_M, TILE_D), 0.0, dtype=np.float32)

    # Load query tile from Global Memory to Shared Memory
    q_global = ct.load(Q, index=(batch_idx, head_idx, bid_x, 0), shape=(1, 1, TILE_M, TILE_D))
    smem_q[...] = q_global.reshape((TILE_M, TILE_D))
    ct.syncthreads()

    q_tile = smem_q[...] 
    
    m_end = input_pos + (bid_x + 1) * TILE_M
    k_seqlen = K.shape[2]
    if CAUSAL:
        mask_start = (input_pos + bid_x * TILE_M) // TILE_N
        mask_start = min(mask_start, k_seqlen // TILE_N)
        Tc = ct.cdiv(min(m_end, k_seqlen), TILE_N)
    else:
        Tc = ct.cdiv(k_seqlen, TILE_N)
        mask_start = k_seqlen // TILE_N

    # Loop over K, V blocks (N-dimension chunks)
    for j in range(0, Tc):
        # --- Compute QK product ---
        # Load K from Global Memory to Shared Memory
        k_global = ct.load(
            K, index=(batch_idx, off_kv_h, 0, j), shape=(1, 1, TILE_D, TILE_N),
            order=(0, 1, 3, 2)
        )
        smem_k[...] = k_global.reshape((TILE_D, TILE_N))
        ct.syncthreads()
        
        k_tile = smem_k[...]

        qk = ct.full((TILE_M, TILE_N), 0., dtype=np.float32)
        qk = ct.mma(q_tile, k_tile, qk)

        # --- Apply Causal Masking ---
        if (CAUSAL or not EVEN_K) and j >= mask_start:
            offs_n = j * TILE_N + offs_n_tile
            mask = ct.full((TILE_M, TILE_N), True, dtype=np.bool)
            if not EVEN_K:
                mask = mask & (offs_n < k_seqlen)
            if CAUSAL:
                mask = mask & (offs_m >= offs_n)
            mask = ct.where(mask, 0.0, -np.inf)
            qk += mask

        # --- Online Softmax Update ---
        m_ij = max(m_i, ct.max(qk, axis=-1, keepdims=True) * qk_scale)
        qk = qk * qk_scale - m_ij

        p = ct.exp2(qk, flush_to_zero=True)
        l_ij = ct.sum(p, axis=-1, keepdims=True)
        alpha = ct.exp2(m_i - m_ij, flush_to_zero=True)
        
        l_i = l_i * alpha + l_ij
        acc = acc * alpha

        # --- Compute PV product ---
        # Load V from Global Memory to Shared Memory
        v_global = ct.load(
            V, index=(batch_idx, off_kv_h, j, 0), shape=(1, 1, TILE_N, TILE_D)
        )
        smem_v[...] = v_global.reshape((TILE_N, TILE_D))
        ct.syncthreads()
        
        v_tile = smem_v[...]

        p = p.astype(Q.dtype)
        acc = ct.mma(p, v_tile, acc)
        m_i = m_ij

    # --- Final Normalization and Store ---
    acc = ct.truediv(acc, l_i, flush_to_zero=True, rounding_mode=RMd.APPROX)
    acc = acc.reshape((1, 1, TILE_M, TILE_D)).astype(Out.dtype)
    ct.store(Out, index=(batch_idx, head_idx, bid_x, 0), tile=acc)


# --- Wrapper function to launch the FMHA kernel ---
def cutile_fmha(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                qk_scale: float | None = None,
                input_pos: int = 0,
                tile_m: int = 128,
                tile_n: int = 128,
                query_group_size: int = 1,
                causal: bool = False) -> torch.Tensor:
    
    Batch, Heads, SeqLen_Q, D_k = Q.shape
    _, KV_Heads, SeqLen_KV, D_v = V.shape
    even_k = (SeqLen_KV % tile_n) == 0

    if qk_scale is None:
        qk_scale = 1.0 / math.sqrt(D_k)

    Out = torch.empty((Batch, Heads, SeqLen_Q, D_v), dtype=Q.dtype, device=Q.device)

    grid_x = math.ceil(SeqLen_Q / tile_m)
    grid_y = Batch * Heads
    grid = (grid_x, grid_y, 1)

    ct.launch(torch.cuda.current_stream(), grid, fmha_kernel, (
        Q, K, V, Out,
        qk_scale, input_pos, D_k, Heads,
        tile_m, tile_n, query_group_size, causal, even_k
    ))

    return Out


def torch_fmha(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
               is_causal: bool, enable_gqa: bool) -> torch.Tensor:
    backend = SDPBackend.CUDNN_ATTENTION \
            if (Q.shape[2] == K.shape[2]) \
            else SDPBackend.FLASH_ATTENTION
    with sdpa_kernel(backend):
        ret = scaled_dot_product_attention(Q, K, V,
                                           is_causal=is_causal,
                                           enable_gqa=enable_gqa)
    return ret

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--correctness-check", action="store_true", help="Check the correctness of the results")
    parser.add_argument("--tile-m", type=int, default=128, help="Tile size M")
    parser.add_argument("--tile-n", type=int, default=128, help="Tile size N")
    parser.add_argument("--num-ctas", type=int, default=1, help="Number of CTAs")
    parser.add_argument("--occupancy", type=int, default=2, help="Occupancy")
    parser.add_argument("--skip-small-tests", action="store_true", help="Skip small tests")
    args = parser.parse_args()
    
    print("--- Running cuTile Fused Multi-Head Attention (FMHA) v2 (SMEM Staging) ---")

    DTYPE = torch.float16
    QUERY_GROUP_SIZE = 1

    if not args.skip_small_tests:
        BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, SEQ_LEN_KV, D_K, D_V = 2, 8, 128, 128, 64, 64
        Q_input = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, D_K, dtype=DTYPE, device='cuda')
        K_input = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN_KV, D_K, dtype=DTYPE, device='cuda')
        V_input = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN_KV, D_V, dtype=DTYPE, device='cuda')

        print("\n--- Test 1: Non-Causal Attention ---")
        out = cutile_fmha(Q=Q_input, K=K_input, V=V_input, tile_m=args.tile_m, tile_n=args.tile_n, causal=False)
        if args.correctness_check:
            ref = torch_fmha(Q_input, K_input, V_input, is_causal=False, enable_gqa=False)
            torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)
            print("Correctness check passed")

        print("\n--- Test 2: Causal Attention ---")
        out = cutile_fmha(Q=Q_input, K=K_input, V=V_input, tile_m=args.tile_m, tile_n=args.tile_n, causal=True)
        if args.correctness_check:
            ref = torch_fmha(Q_input, K_input, V_input, is_causal=True, enable_gqa=False)
            torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)
            print("Correctness check passed")

    print("\n--- Main Profiling Benchmark (Causal) ---")
    BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, SEQ_LEN_KV, D_K, D_V = 8, 16, 1024, 1024, 64, 64
    Q_input = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, D_K, dtype=DTYPE, device='cuda')
    K_input = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN_KV, D_K, dtype=DTYPE, device='cuda')
    V_input = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN_KV, D_V, dtype=DTYPE, device='cuda')

    # Warmup
    for _ in range(3):
        cutile_fmha(Q=Q_input, K=K_input, V=V_input, tile_m=args.tile_m, tile_n=args.tile_n, causal=True)
    
    torch.cuda.synchronize()
    
    # Run targeted execution for NCU
    out = cutile_fmha(Q=Q_input, K=K_input, V=V_input, tile_m=args.tile_m, tile_n=args.tile_n, causal=True)
    torch.cuda.synchronize()
    
    if args.correctness_check:
        ref = torch_fmha(Q_input, K_input, V_input, is_causal=True, enable_gqa=False)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=5e-2)
        print("Main Benchmark Correctness check passed")