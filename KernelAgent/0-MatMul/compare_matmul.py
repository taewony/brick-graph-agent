# compare_matmul.py
# cuTile matrix multiplication vs PyTorch matmul performance comparison
# Fixed tile size at 64, optimized for NVIDIA RTX 5070 GPU (SM 100/101)

import torch
import time
import numpy as np
import cuda.tile as ct

# ============================================================
# 1. cuTile Kernels
# ============================================================

TILE_SIZE = 64

# ---- Version 1: Standard Tiled MatMul ----
@ct.kernel
def matmul_sample(A, B, C, NUM_BID_M: int, NUM_BID_N: int, NUM_K_TILES: int):
    # Standard grid mapping using bid(0) for m and bid(1) for n
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)

    acc = ct.zeros((64, 64), dtype=ct.float32)

    for k in range(NUM_K_TILES):
        tile_A = ct.load(A, index=(bid_m, k), shape=(64, 64))
        tile_B = ct.load(B, index=(k, bid_n), shape=(64, 64))

        tile_A = ct.astype(tile_A, ct.float16)
        tile_B = ct.astype(tile_B, ct.float16)

        acc = ct.mma(tile_A, tile_B, acc=acc)

    ct.store(C, index=(bid_m, bid_n), tile=ct.astype(acc, C.dtype))


# ---- Version 2: Optimized TileGym Grouped Persistent MatMul ----
@ct.kernel
def matmul_tilegym(A, B, C, NUM_BID_M: int, NUM_BID_N: int, NUM_K_TILES: int, GROUP_SIZE_M: int):
    start_tile_id = ct.bid(0)
    num_programs = ct.num_blocks(0)
    num_groups = (NUM_BID_M + GROUP_SIZE_M - 1) // GROUP_SIZE_M
    total_tiles = num_groups * GROUP_SIZE_M * NUM_BID_N

    for tile_id in range(start_tile_id, total_tiles, num_programs):
        tiles_per_group_strip = GROUP_SIZE_M * NUM_BID_N
        group_id = tile_id // tiles_per_group_strip
        group_offset = tile_id % tiles_per_group_strip
        
        bid_n_inner = group_offset // GROUP_SIZE_M
        bid_m_inner = group_offset % GROUP_SIZE_M
        
        bid_m = group_id * GROUP_SIZE_M + bid_m_inner
        bid_n = bid_n_inner

        if bid_m >= NUM_BID_M or bid_n >= NUM_BID_N:
            continue

        acc = ct.zeros((64, 64), dtype=ct.float32)

        for k in range(NUM_K_TILES):
            tile_A = ct.load(A, index=(bid_m, k), shape=(64, 64))
            tile_B = ct.load(B, index=(k, bid_n), shape=(64, 64))

            tile_A = ct.astype(tile_A, ct.float16)
            tile_B = ct.astype(tile_B, ct.float16)

            acc = ct.mma(tile_A, tile_B, acc=acc)

        ct.store(C, index=(bid_m, bid_n), tile=ct.astype(acc, C.dtype))


# ============================================================
# 2. Benchmarking Utilities
# ============================================================

def compute_tflops(M, N, K, time_sec):
    ops = 2 * M * N * K
    return ops / (time_sec * 1e12)


def benchmark_pytorch(A, B, warmup=20, repeats=100):
    C = torch.empty(A.shape[0], B.shape[1], dtype=torch.float16, device='cuda')
    for _ in range(warmup):
        torch.matmul(A, B, out=C)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        torch.matmul(A, B, out=C)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / repeats
    return C, elapsed


def benchmark_cutile(kernel, A, B, M, N, K, warmup=20, repeats=100):
    C = torch.empty(M, N, dtype=torch.float16, device='cuda')
    stream = torch.cuda.current_stream()
    
    num_bid_m = M // TILE_SIZE
    num_bid_n = N // TILE_SIZE
    num_k_tiles = K // TILE_SIZE
    
    if kernel == matmul_sample:
        grid_dim = (num_bid_m, num_bid_n, 1)
        args = (A, B, C, num_bid_m, num_bid_n, num_k_tiles)
    else:
        # matmul_tilegym uses persistent threads
        # Optimized for RTX 5070 (SM occupancy & L2 cache swizzling)
        num_ctas = 192    # 4 blocks per SM (across 48 SMs) to hide memory latency
        grid_dim = (num_ctas, 1, 1)
        group_size_m = 8  # Group size 8 to exploit the massive 48MB L2 cache
        args = (A, B, C, num_bid_m, num_bid_n, num_k_tiles, group_size_m)

    for _ in range(warmup):
        ct.launch(stream, grid_dim, kernel, args)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        ct.launch(stream, grid_dim, kernel, args)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / repeats
    return C, elapsed


def verify(label, C_torch, C_cutile):
    if torch.allclose(C_torch, C_cutile, atol=2e-1, rtol=2e-2):
        print(f"  {label}: ✅ Correctness Check Passed")
        return True
    else:
        diff = (C_torch.float() - C_cutile.float()).abs().max().item()
        print(f"  {label}: ❌ Correctness Check Failed! Max diff = {diff:.4f}")
        return False


# ============================================================
# 3. Main Benchmark
# ============================================================

if __name__ == "__main__":
    print("="*75)
    print("cuTile Matrix Multiplication Benchmark - Tile Size 64 Fixed")
    print("Target GPU: RTX 5070 (Blackwell SM100/101) | Data Type: float16")
    print("="*75)

    sizes = [256, 512, 1024, 2048, 4096]
    results = []

    for size in sizes:
        M = N = K = size
        print(f"\n[Matrix Dimension {M}x{K} x {K}x{N}]")

        A = torch.randn(M, K, dtype=torch.float16, device='cuda')
        B = torch.randn(K, N, dtype=torch.float16, device='cuda')

        # 1. PyTorch Benchmark
        C_pt, t_pt = benchmark_pytorch(A, B)
        tflops_pt = compute_tflops(M, N, K, t_pt)

        # 2. cuTile Sample
        try:
            C_sample, t_sample = benchmark_cutile(matmul_sample, A, B, M, N, K)
            tflops_sample = compute_tflops(M, N, K, t_sample)
            ok_sample = verify("cuTile Sample", C_pt, C_sample)
        except Exception as e:
            print(f"  cuTile Sample Execution Error: {e}")
            ok_sample = False
            t_sample = float('inf')
            tflops_sample = 0.0

        # 3. cuTile TileGym (Optimized)
        try:
            C_tilegym, t_tilegym = benchmark_cutile(matmul_tilegym, A, B, M, N, K)
            tflops_tilegym = compute_tflops(M, N, K, t_tilegym)
            ok_tilegym = verify("cuTile TileGym", C_pt, C_tilegym)
        except Exception as e:
            print(f"  cuTile TileGym Execution Error: {e}")
            ok_tilegym = False
            t_tilegym = float('inf')
            tflops_tilegym = 0.0

        results.append((size, t_pt, tflops_pt, t_sample, tflops_sample, t_tilegym, tflops_tilegym))

        # Printing intermediate performance figures
        print(f"  PyTorch:           {t_pt*1000:.3f} ms ({tflops_pt:.2f} TFLOPS)")
        if ok_sample:
            speedup = t_pt / t_sample
            print(f"  cuTile Sample:     {t_sample*1000:.3f} ms ({tflops_sample:.2f} TFLOPS) | Speedup: {speedup:.2f}x")
        if ok_tilegym:
            speedup = t_pt / t_tilegym
            print(f"  cuTile TileGym:    {t_tilegym*1000:.3f} ms ({tflops_tilegym:.2f} TFLOPS) | Speedup: {speedup:.2f}x")

    # Summary Table
    print("\n" + "="*75)
    print("                    PERFORMANCE SUMMARY TABLE")
    print("="*75)
    print(f"{'Size':>8} | {'PyTorch':>18} | {'cuTile Sample':>18} | {'cuTile TileGym':>18}")
    print("-"*75)
    for (sz, pt, ptf, sp, spf, gy, gyf) in results:
        print(f"{sz:>8} | {pt*1000:>7.2f} ms ({ptf:>5.1f}T) | {sp*1000:>7.2f} ms ({spf:>5.1f}T) | {gy*1000:>7.2f} ms ({gyf:>5.1f}T)")
    print("-"*75)
