import cuda.tile as ct
import cupy as cp
import numpy as np
import torch

# ============================================================
# 1. System & Problem Configuration
# ============================================================
# [Optimization 1] Get GPU Specs dynamically to maximize Occupancy
dev_id = cp.cuda.get_device_id()
props = cp.cuda.runtime.getDeviceProperties(dev_id)
NUM_SMS = props['multiProcessorCount']

# Launch configuration: 
# We launch enough blocks (CTAs) to saturate all SMs.
# 2 CTAs per SM is a good heuristic to hide latency.
NUM_CTAS = NUM_SMS * 2 

print(f">> Detected GPU with {NUM_SMS} SMs. Launching {NUM_CTAS} CTAs.")

# Problem Size (Large enough to test bandwidth)
M_SIZE = N_SIZE = K_SIZE = 4096 
TILE_SIZE = 128

NUM_BID_M = M_SIZE // TILE_SIZE
NUM_BID_N = N_SIZE // TILE_SIZE
NUM_K_TILES = K_SIZE // TILE_SIZE

# Benchmarking params
N_WARMUP = 5
N_ITER = 20

# ============================================================
# 2. Coordinate Calculation (Swizzling Logic)
# ============================================================
def compute_tile_coords(tile_id, num_bid_m, num_bid_n, group_size_m):
    """
    Map linear tile_id to (bid_m, bid_n)
    group_size_m=0 : Raster Order (Standard)
    group_size_m>0 : Swizzled Order (L2 Locality Optimized)
    """
    
    # [Case 0] Standard Raster Order
    if group_size_m == 0:
        bid_m = tile_id // num_bid_n
        bid_n = tile_id % num_bid_n
        return bid_m, bid_n

    # [Case 1] Swizzled Order
    # 1. Calculate grouping dimensions
    tiles_per_group_strip = group_size_m * num_bid_n
    
    group_id = tile_id // tiles_per_group_strip
    group_offset = tile_id % tiles_per_group_strip

    # 2. Swizzle: Iterate 'm' fast, 'n' slow WITHIN the group
    # This keeps 'n' constant for several 'm' steps, reusing B-tile in L2 cache.
    bid_n_inner = group_offset // group_size_m
    bid_m_inner = group_offset % group_size_m
    
    bid_m = group_id * group_size_m + bid_m_inner
    bid_n = bid_n_inner

    return bid_m, bid_n


# ============================================================
# 3. cuTile Kernel (Universal)
# ============================================================
@ct.kernel
def matmul_kernel_universal(A, B, C, group_size_m):
    # Persistent Thread Loop
    # Now 'num_programs' will be large (e.g., 200+), so this loop runs fewer times per block
    # but runs on many more blocks in parallel.
    start_tile_id = ct.bid(0)
    num_programs = ct.num_blocks(0)
    total_tiles = NUM_BID_M * NUM_BID_N

    for tile_id in range(start_tile_id, total_tiles, num_programs):
        
        bid_m, bid_n = compute_tile_coords(
            tile_id,
            NUM_BID_M,
            NUM_BID_N,
            group_size_m
        )

        if bid_m >= NUM_BID_M or bid_n >= NUM_BID_N:
            continue

        # Accumulator (FP32)
        acc = ct.zeros((TILE_SIZE, TILE_SIZE), dtype=ct.float32)

        for k in range(NUM_K_TILES):
            tile_A = ct.load(A, index=(bid_m, k), shape=(TILE_SIZE, TILE_SIZE))
            tile_B = ct.load(B, index=(k, bid_n), shape=(TILE_SIZE, TILE_SIZE))
            
            # Use TF32 (TensorFloat-32) or FP16 for Tensor Cores
            # Assuming input is FP16, we cast to appropriate type for mma
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
    # Warmup
    for _ in range(N_WARMUP): _ = torch.matmul(A_t, B_t)
    torch.cuda.synchronize()
    
    # Measure
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(N_ITER): _ = torch.matmul(A_t, B_t)
    end.record()
    torch.cuda.synchronize()
    
    avg_ms = start.elapsed_time(end) / N_ITER
    print(f"{'PyTorch (cuBLAS)':<25} : {avg_ms:.3f} ms")
    return avg_ms

# ============================================================
# 5. Main Execution
# ============================================================
def main():
    print("=" * 60)
    print(f"=== MatMul Benchmark Optimized ({M_SIZE}x{N_SIZE}x{K_SIZE}) ===")
    print(f"SM Count: {NUM_SMS}, Active CTAs: {NUM_CTAS}")
    print("-" * 60)

    # Data Prep
    np.random.seed(0)
    # Using float16 for Tensor Core performance
    h_A = np.random.randn(M_SIZE, K_SIZE).astype(np.float16)
    h_B = np.random.randn(K_SIZE, N_SIZE).astype(np.float16)
    h_C = np.zeros((M_SIZE, N_SIZE), dtype=np.float16)

    d_A = cp.asarray(h_A)
    d_B = cp.asarray(h_B)
    d_C = cp.asarray(h_C)
    
    A_t = torch.from_numpy(h_A).cuda()
    B_t = torch.from_numpy(h_B).cuda()

    # --- Run Benchmarks ---
    t_torch = benchmark_pytorch(A_t, B_t)
    t_g0    = benchmark_cutile("cuTile (No Swizzle)", d_A, d_B, d_C, group_size=0)
    t_g2    = benchmark_cutile("cuTile (Group=2)",    d_A, d_B, d_C, group_size=2)
    t_g4    = benchmark_cutile("cuTile (Group=4)",    d_A, d_B, d_C, group_size=4)

    # --- Verification ---
    print("-" * 60)
    print("Verifying correctness (based on Group=4)...")
    C_ref = torch.matmul(A_t, B_t).cpu().numpy()
    C_cutile = cp.asnumpy(d_C)
    
    # Relaxed tolerance for large FP16 accumulation
    if np.allclose(C_cutile, C_ref, atol=2e-1, rtol=2e-2):
        print("✅ Verification: Success!")
    else:
        diff = np.abs(C_cutile - C_ref)
        print(f"❌ Verification: Failed! Max error: {diff.max()}")

    # --- Final Report ---
    print("=" * 60)
    print(f"{'Method':<25} | {'Time (ms)':<10} | {'Rel. Speed':<10}")
    print("-" * 60)
    print(f"{'PyTorch (cuBLAS)':<25} | {t_torch:.3f} ms  | 1.00x")
    print(f"{'cuTile (No Swizzle)':<25} | {t_g0:.3f} ms  | {t_torch/t_g0:.2f}x")
    print(f"{'cuTile (Group=2)':<25} | {t_g2:.3f} ms  | {t_torch/t_g2:.2f}x")
    print(f"{'cuTile (Group=4)':<25} | {t_g4:.3f} ms  | {t_torch/t_g4:.2f}x")
    print("=" * 60)

    # Analysis Comment
    speedup = (t_g0 - t_g4) / t_g0 * 100
    if speedup > 2.0:
        print(f">> ✅ SUCCESS: Swizzling provided {speedup:.1f}% speedup due to better L2 hit rate.")
    else:
        print(f">> ℹ️ NOTE: Performance is similar. Bottleneck might still be compute-bound or launch overhead.")

if __name__ == "__main__":
    main()
