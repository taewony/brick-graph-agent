# compare_FMHA.py
# cuTile Fused Multi-Head Attention (FMHA) vs PyTorch SDPA performance comparison
# Optimized tile size 64x64, tailored for NVIDIA RTX 5070 GPU (SM 120 / Blackwell)

import argparse
import math
import time
import torch
import numpy as np
import cuda.tile as ct
from cuda.tile import RoundingMode as RMd

INV_LOG_2 = 1.0 / math.log(2)
ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]

# ============================================================
# 1. cuTile FMHA Kernel Definition
# ============================================================

@ct.kernel(occupancy=2)
def fmha_kernel(Q, K, V, Out,
                qk_scale: float,
                input_pos: int,
                TILE_D: ConstInt,  # head_dim
                H: ConstInt,       # number of heads
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

    # Scale query-key scaling factor for exp2
    qk_scale = qk_scale * INV_LOG_2

    # Initialize offsets for current query tile (M-dimension)
    offs_m = bid_x * TILE_M + ct.arange(TILE_M, dtype=np.int32)
    offs_m += input_pos
    offs_m = offs_m[:, None]

    # Initialize local offsets for key/value tile (N-dimension)
    offs_n_tile = ct.arange(TILE_N, dtype=np.int32)
    offs_n_tile = offs_n_tile[None, :]

    # Initialize online softmax accumulators in float32 for numerical stability
    m_i = ct.full((TILE_M, 1), -np.inf, dtype=np.float32)
    l_i = ct.full((TILE_M, 1), 0.0, dtype=np.float32)
    acc = ct.full((TILE_M, TILE_D), 0.0, dtype=np.float32)

    # Load query tile directly (hint latency=2 to prefetch)
    q = ct.load(
        Q, index=(batch_idx, head_idx, bid_x, 0), shape=(1, 1, TILE_M, TILE_D),
        padding_mode=ct.PaddingMode.ZERO,
        latency=2
    ).reshape((TILE_M, TILE_D))
    
    m_end = input_pos + (bid_x + 1) * TILE_M
    k_seqlen = K.shape[2]
    if CAUSAL:
        mask_start = (input_pos + bid_x * TILE_M) // TILE_N
        mask_start = min(mask_start, k_seqlen // TILE_N)
        Tc = ct.cdiv(min(m_end, k_seqlen), TILE_N)
    else:
        Tc = ct.cdiv(k_seqlen, TILE_N)
        mask_start = k_seqlen // TILE_N

    # Loop over K, V blocks along the N-dimension (sequence length)
    for j in range(0, Tc):
        # Load K (hint latency=2 for pipelined prefetch)
        k = ct.load(
            K, index=(batch_idx, off_kv_h, 0, j), shape=(1, 1, TILE_D, TILE_N),
            order=(0, 1, 3, 2),
            padding_mode=ct.PaddingMode.ZERO,
            latency=2
        ).reshape((TILE_D, TILE_N))
        
        # Compute QK product
        qk = ct.full((TILE_M, TILE_N), 0., dtype=np.float32)
        qk = ct.mma(q, k, qk)

        # Apply Causal Masking & Boundary Padding Masking
        if (CAUSAL or not EVEN_K) and j >= mask_start:
            offs_n = j * TILE_N + offs_n_tile
            mask = ct.full((TILE_M, TILE_N), True, dtype=np.bool_)
            if not EVEN_K:
                mask = mask & (offs_n < k_seqlen)
            if CAUSAL:
                mask = mask & (offs_m >= offs_n)
            mask = ct.where(mask, 0.0, -np.inf)
            qk += mask

        # Online Softmax Update step
        m_ij = max(m_i, ct.max(qk, axis=-1, keepdims=True) * qk_scale)
        qk = qk * qk_scale - m_ij

        p = ct.exp2(qk, flush_to_zero=True)
        l_ij = ct.sum(p, axis=-1, keepdims=True)
        alpha = ct.exp2(m_i - m_ij, flush_to_zero=True)
        
        l_i = l_i * alpha + l_ij
        acc = acc * alpha

        # Load V (hint latency=4 to hide high DRAM latency during math)
        v = ct.load(
            V, index=(batch_idx, off_kv_h, j, 0), shape=(1, 1, TILE_N, TILE_D),
            padding_mode=ct.PaddingMode.ZERO,
            latency=4
        ).reshape((TILE_N, TILE_D))

        p = p.astype(Q.dtype)
        acc = ct.mma(p, v, acc)
        m_i = m_ij

    # Final Normalization and Store
    acc = ct.truediv(acc, l_i, flush_to_zero=True, rounding_mode=RMd.APPROX)
    acc = acc.reshape((1, 1, TILE_M, TILE_D)).astype(Out.dtype)
    ct.store(Out, index=(batch_idx, head_idx, bid_x, 0), tile=acc)


# ---- Helper Function to Launch Kernel ----
def cutile_fmha(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                qk_scale: float | None = None,
                input_pos: int = 0,
                tile_m: int = 64,
                tile_n: int = 64,
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


# ============================================================
# 2. Benchmarking Utilities
# ============================================================

def compute_tflops(B, H, S_Q, S_KV, D, time_sec):
    # FLOPs formula for Fused Attention: 4 * Batch * Heads * SeqLen_Q * SeqLen_KV * Dim
    ops = 4 * B * H * S_Q * S_KV * D
    return ops / (time_sec * 1e12)


def benchmark_pytorch(Q, K, V, causal=False, warmup=20, repeats=100):
    for _ in range(warmup):
        _ = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=causal)
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(repeats):
        out = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=causal)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / repeats
    return out, elapsed


def benchmark_cutile(Q, K, V, causal=False, warmup=20, repeats=100):
    for _ in range(warmup):
        _ = cutile_fmha(Q, K, V, tile_m=64, tile_n=64, causal=causal)
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(repeats):
        out = cutile_fmha(Q, K, V, tile_m=64, tile_n=64, causal=causal)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / repeats
    return out, elapsed


def verify(label, out_pt, out_cutile):
    try:
        # Use slightly relaxed tolerance to prevent false negatives due to FP16 accumulation differences
        torch.testing.assert_close(out_pt, out_cutile, atol=1e-2, rtol=5e-2)
        print(f"  {label}: ✅ Correctness Check Passed")
        return True
    except Exception as e:
        diff = (out_pt.float() - out_cutile.float()).abs().max().item()
        print(f"  {label}: ❌ Correctness Check Failed! Max diff = {diff:.4f}")
        return False


# ============================================================
# 3. Main Benchmark
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark PyTorch SDPA vs. cuTile FMHA")
    parser.add_argument("--warmup", type=int, default=20, help="Number of warmup steps")
    parser.add_argument("--repeats", type=int, default=100, help="Number of repetitions")
    args = parser.parse_args()

    print("="*80)
    print("cuTile Fused Multi-Head Attention (FMHA) Benchmark - Optimized Tile Size 64x64")
    print("Target GPU: RTX 5070 (Blackwell SM120) | Data Type: float16")
    print("="*80)

    # Benchmark Configuration matching typical LLM workloads
    B = 8
    H = 12
    D = 64
    sizes = [512, 1024, 2048, 4096]
    
    results = []

    for causal in [False, True]:
        mode_str = "Causal" if causal else "Non-Causal"
        print(f"\n⚡ Running Benchmarks for [{mode_str} Attention] (Batch={B}, Heads={H}, Dim={D})")
        print("-" * 80)
        
        for S in sizes:
            print(f"\n[SeqLen {S}x{S}]")
            
            Q = torch.randn((B, H, S, D), dtype=torch.float16, device='cuda')
            K = torch.randn((B, H, S, D), dtype=torch.float16, device='cuda')
            V = torch.randn((B, H, S, D), dtype=torch.float16, device='cuda')

            # 1. PyTorch SDPA Baseline
            out_pt, t_pt = benchmark_pytorch(Q, K, V, causal=causal, warmup=args.warmup, repeats=args.repeats)
            tflops_pt = compute_tflops(B, H, S, S, D, t_pt)

            # 2. cuTile FMHA (64x64 tiles)
            try:
                out_cu, t_cu = benchmark_cutile(Q, K, V, causal=causal, warmup=args.warmup, repeats=args.repeats)
                tflops_cu = compute_tflops(B, H, S, S, D, t_cu)
                ok = verify("cuTile FMHA", out_pt, out_cu)
            except Exception as e:
                print(f"  cuTile FMHA Execution Error: {e}")
                ok = False
                t_cu = float('inf')
                tflops_cu = 0.0

            results.append((mode_str, S, t_pt, tflops_pt, t_cu, tflops_cu, ok))

            # Print stats
            print(f"  PyTorch:       {t_pt*1000:7.3f} ms ({tflops_pt:6.2f} TFLOPS)")
            if ok:
                speedup = t_pt / t_cu
                print(f"  cuTile FMHA:   {t_cu*1000:7.3f} ms ({tflops_cu:6.2f} TFLOPS) | Speedup: {speedup:.2f}x")

    # Summary Table
    print("\n" + "="*80)
    print("                    PERFORMANCE SUMMARY TABLE")
    print("="*80)
    print(f"{'Mode':>12} | {'Size':>6} | {'PyTorch Time':>18} | {'cuTile FMHA Time':>18} | {'Speedup':>9}")
    print("-"*80)
    for (mode, sz, pt, ptf, cu, cuf, ok) in results:
        if ok:
            speedup_str = f"{pt/cu:7.2f}x"
            cu_str = f"{cu*1000:6.3f} ms ({cuf:5.1f}T)"
        else:
            speedup_str = "FAILED"
            cu_str = "FAILED"
        pt_str = f"{pt*1000:6.3f} ms ({ptf:5.1f}T)"
        print(f"{mode:>12} | {sz:>6} | {pt_str:>18} | {cu_str:>18} | {speedup_str:>9}")
    print("-"*80)
