import cuda.tile as ct
import cupy as cp
import numpy as np
import torch

# ============================================================
# 1. Configuration
# ============================================================
M_SIZE = N_SIZE = K_SIZE = 4096
TILE_SIZE = 128

NUM_BID_M = M_SIZE // TILE_SIZE   # 16
NUM_BID_N = N_SIZE // TILE_SIZE   # 16
NUM_K_TILES = K_SIZE // TILE_SIZE # 16

# Benchmarking params
N_WARMUP = 10
N_ITER = 100
NUM_CTAS = 16  # Persistent Threads count

# ============================================================
# 2. Coordinate Calculation
# ============================================================
def compute_tile_coords(tile_id, num_bid_m, num_bid_n, group_size_m):
    """
    If group_size_m == 0: Returns Standard Raster Order (Row-Major).
    If group_size_m > 0 : Returns Swizzled Order for L2 Locality.
    """
    
    # [Case 0] No Swizzling (Standard Raster Order)
    # bid_n changes fast (0,1,2...), bid_m changes slow
    if group_size_m == 0:
        bid_m = tile_id // num_bid_n
        bid_n = tile_id % num_bid_n
        return bid_m, bid_n

    # [Case 1] Swizzled Order
    tiles_per_group_strip = group_size_m * num_bid_n
    
    group_id = tile_id // tiles_per_group_strip
    group_offset = tile_id % tiles_per_group_strip

    # Iterate bid_m fast within the group to reuse Tile B
    bid_n_inner = group_offset // group_size_m
    bid_m_inner = group_offset % group_size_m
    
    bid_m = group_id * group_size_m + bid_m_inner
    bid_n = bid_n_inner

    return bid_m, bid_n


# ============================================================
# 3. cuTile Kernel
# ============================================================
@ct.kernel
def matmul_kernel_universal(A, B, C, group_size_m):
    # Persistent Kernel Pattern
    start_tile_id = ct.bid(0)
    num_programs = ct.num_blocks(0)
    total_tiles = NUM_BID_M * NUM_BID_N

    for tile_id in range(start_tile_id, total_tiles, num_programs):
        
        # Calculate coords based on group_size_m
        bid_m, bid_n = compute_tile_coords(
            tile_id,
            NUM_BID_M,
            NUM_BID_N,
            group_size_m
        )

        # Boundary Check
        if bid_m >= NUM_BID_M or bid_n >= NUM_BID_N:
            continue

        acc = ct.zeros((TILE_SIZE, TILE_SIZE), dtype=ct.float32)

        for k in range(NUM_K_TILES):
            tile_A = ct.load(A, index=(bid_m, k), shape=(TILE_SIZE, TILE_SIZE))
            tile_B = ct.load(B, index=(k, bid_n), shape=(TILE_SIZE, TILE_SIZE))
            
            tile_A = ct.astype(tile_A, ct.float16)
            tile_B = ct.astype(tile_B, ct.float16)
            
            acc = ct.mma(tile_A, tile_B, acc=acc)

        ct.store(C, index=(bid_m, bid_n), tile=ct.astype(acc, C.dtype))


# ============================================================
# 4. Benchmarking Helper
# ============================================================
def benchmark_cutile(label, d_A, d_B, d_C, group_size):
    stream = cp.cuda.get_current_stream()
    
    # 1. Warmup
    for _ in range(N_WARMUP):
        ct.launch(
            stream, (NUM_CTAS, 1, 1), 
            matmul_kernel_universal, 
            (d_A, d_B, d_C, group_size)
        )
    stream.synchronize()
    
    # 2. Measure
    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()
    
    start_event.record()
    for _ in range(N_ITER):
        ct.launch(
            stream, (NUM_CTAS, 1, 1), 
            matmul_kernel_universal, 
            (d_A, d_B, d_C, group_size)
        )
    end_event.record()
    end_event.synchronize()
    
    avg_ms = cp.cuda.get_elapsed_time(start_event, end_event) / N_ITER
    print(f"{label:<25} : {avg_ms:.3f} ms")
    return avg_ms

def benchmark_pytorch(A_t, B_t):
    # 1. Warmup
    for _ in range(N_WARMUP):
        _ = torch.matmul(A_t, B_t)
    torch.cuda.synchronize()
    
    # 2. Measure
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(N_ITER):
        _ = torch.matmul(A_t, B_t)
    end_event.record()
    torch.cuda.synchronize()
    
    avg_ms = start_event.elapsed_time(end_event) / N_ITER
    print(f"{'PyTorch (cuBLAS)':<25} : {avg_ms:.3f} ms")
    return avg_ms

# ============================================================
# 5. Main
# ============================================================
def main():
    print(f"=== MatMul Benchmark ({M_SIZE}x{N_SIZE}x{K_SIZE}) ===")
    print(f"Iterations: {N_ITER}, CTAs: {NUM_CTAS}")
    print("-" * 50)

    # Prepare Data
    np.random.seed(0)
    h_A = np.random.randn(M_SIZE, K_SIZE).astype(np.float16)
    h_B = np.random.randn(K_SIZE, N_SIZE).astype(np.float16)
    h_C = np.zeros((M_SIZE, N_SIZE), dtype=np.float16)

    d_A = cp.asarray(h_A)
    d_B = cp.asarray(h_B)
    d_C = cp.asarray(h_C)
    
    A_t = torch.from_numpy(h_A).cuda()
    B_t = torch.from_numpy(h_B).cuda()

    # --------------------------------------------------------
    # Benchmarking
    # --------------------------------------------------------
    t_torch = benchmark_pytorch(A_t, B_t)
    t_g0    = benchmark_cutile("cuTile (No Swizzle)", d_A, d_B, d_C, group_size=0)
    t_g2    = benchmark_cutile("cuTile (Group=2)",    d_A, d_B, d_C, group_size=2)
    t_g4    = benchmark_cutile("cuTile (Group=4)",    d_A, d_B, d_C, group_size=4)

    # --------------------------------------------------------
    # Verification (Check correctness with Group=4 result)
    # --------------------------------------------------------
    print("-" * 50)
    print("Verifying correctness (based on Group=4 run)...")
    
    C_ref = torch.matmul(A_t, B_t).cpu().numpy()
    C_cutile = cp.asnumpy(d_C)

    if np.allclose(C_cutile, C_ref, atol=2e-1, rtol=2e-2):
        print("✅ Verification: Success!")
    else:
        diff = np.abs(C_cutile - C_ref)
        print(f"❌ Verification: Failed! (Max error: {diff.max()})")

    # --------------------------------------------------------
    # Summary Table
    # --------------------------------------------------------
    print("=" * 60)
    print(f"{'Method':<25} | {'Time (ms)':<10} | {'Rel. Speed':<10}")
    print("-" * 60)
    print(f"{'PyTorch (cuBLAS)':<25} | {t_torch:.3f} ms  | 1.00x")
    print(f"{'cuTile (No Swizzle)':<25} | {t_g0:.3f} ms  | {t_torch/t_g0:.2f}x")
    print(f"{'cuTile (Group=2)':<25} | {t_g2:.3f} ms  | {t_torch/t_g2:.2f}x")
    print(f"{'cuTile (Group=4)':<25} | {t_g4:.3f} ms  | {t_torch/t_g4:.2f}x")
    print("=" * 60)
    
    # Analysis
    best_cutile = min(t_g0, t_g2, t_g4)
    worst_cutile = max(t_g0, t_g2, t_g4)
    
    if worst_cutile > 0:
        gap = ((worst_cutile - best_cutile) / worst_cutile) * 100
        print(f">> Gap between Best/Worst cuTile: {gap:.2f}%")
        
    if t_g0 < t_g4:
        print(">> Note: Swizzling overhead might be outweighing benefits at this size.")
    else:
        print(">> Note: Swizzling is providing performance gain.")

if __name__ == "__main__":
    main()
