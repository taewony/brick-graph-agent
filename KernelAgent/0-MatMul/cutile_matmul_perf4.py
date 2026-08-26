import cuda.tile as ct
import cupy as cp
import numpy as np

# ============================================================
# 1. System Configuration
# ============================================================
dev_id = cp.cuda.get_device_id()
props = cp.cuda.runtime.getDeviceProperties(dev_id)
NUM_SMS = props['multiProcessorCount']

# Launch configuration:
# Using enough CTAs to saturate the GPU.
NUM_CTAS = NUM_SMS * 2

print(f">> GPU Detected: {props['name'].decode()} with {NUM_SMS} SMs.")
print(f">> Using {NUM_CTAS} CTAs for persistent kernels (Swizzling Enabled).\n")

# Matrix sizes to test
MATRIX_SIZES = [1024, 2048, 4096, 8192, 16384, 32768]

# Tile configurations to test (Tile_M, Tile_N)
TILE_CONFIGS = [
    (128, 128), # Baseline
    (128, 64),  # Rectangular M-major
    (64, 128),  # Rectangular N-major (Swizzling helps here!)
    (64, 64)    # Smallest
]

# ============================================================
# 2. Benchmark Function (with Swizzling)
# ============================================================
def run_benchmark(M, N, K, tile_m, tile_n):
    """
    Runs a benchmark with Block Swizzling (L2 Cache Optimization).
    """
    
    # 1. Basic Grid setup
    if M % tile_m != 0 or N % tile_n != 0:
        return -1.0 

    grid_m = M // tile_m
    grid_n = N // tile_n
    k_step = 64 
    grid_k = K // k_step

    # [Optimization Parameter]
    # Process 8 rows of A for every column of B to maximize B's reuse in L2 Cache.
    GROUP_SIZE_M = 8

    # --------------------------------------------------------
    # Kernel Definition
    # --------------------------------------------------------
    @ct.kernel
    def matmul_kernel_swizzled(A, B, C):
        # Persistent Thread Block ID
        bid = ct.bid(0)
        num_programs = ct.num_blocks(0)
        total_tiles = grid_m * grid_n

        # Grid-stride loop (Persistent Threads)
        for tile_id in range(bid, total_tiles, num_programs):
            
            # ========================================================
            # [KEY LOGIC] Block Swizzling / Rasterization Re-mapping
            # ========================================================
            # Goal: Visit tiles in column-major order within a group of rows.
            # This keeps a tile of B in L2 cache while we iterate through A-rows.

            # 1. Determine the "Group" (A horizontal strip of GROUP_SIZE_M rows)
            #    One group contains (GROUP_SIZE_M * grid_n) tiles.
            num_pid_in_group = GROUP_SIZE_M * grid_n
            group_id = tile_id // num_pid_in_group

            # 2. Identify the starting row index of this group
            first_pid_m = group_id * GROUP_SIZE_M
            
            # 3. Handle boundary: The last group might have fewer than 8 rows
            group_size_m = min(grid_m - first_pid_m, GROUP_SIZE_M)
            
            # 4. Identify local ID within the current group
            pid_in_group = tile_id % num_pid_in_group

            # 5. Map linear ID to (M, N) coordinates
            #    - pid_in_group % group_size_m  -> Changes FAST (Row index moves down)
            #    - pid_in_group // group_size_m -> Changes SLOW (Col index moves right)
            #    Result: (0,0)->(1,0)->...->(7,0) then (0,1)->(1,1)...
            bid_m = first_pid_m + (pid_in_group % group_size_m)
            bid_n = pid_in_group // group_size_m
            # ========================================================

            # Initialize Accumulator
            acc = ct.zeros((tile_m, tile_n), dtype=ct.float32)

            # Main Compute Loop (K-dimension)
            for k in range(grid_k):
                # Load Tile A: (tile_m, k_step)
                tile_A = ct.load(A, index=(bid_m, k), shape=(tile_m, k_step))
                
                # Load Tile B: (k_step, tile_n)
                # Thanks to Swizzling, this tile_B is likely already in L2 Cache!
                tile_B = ct.load(B, index=(k, bid_n), shape=(k_step, tile_n))
                
                # MMA
                tile_A = ct.astype(tile_A, ct.float16)
                tile_B = ct.astype(tile_B, ct.float16)
                acc = ct.mma(tile_A, tile_B, acc=acc)

            # Store Result
            ct.store(C, index=(bid_m, bid_n), tile=ct.astype(acc, C.dtype))

    # --------------------------------------------------------
    # Data Preparation & Execution
    # --------------------------------------------------------
    try:
        # Generate random data
        d_A = cp.random.randn(M, K, dtype=cp.float32).astype(cp.float16)
        d_B = cp.random.randn(K, N, dtype=cp.float32).astype(cp.float16)
        d_C = cp.zeros((M, N), dtype=cp.float16)
    except cp.cuda.memory.OutOfMemoryError:
        print(f"   [Error] OOM for size {M}x{N}. Skipping...")
        return -1.0

    stream = cp.cuda.get_current_stream()
    
    n_warmup = 3
    n_iter = 10 if M <= 8192 else 3

    # Warmup
    for _ in range(n_warmup):
        ct.launch(stream, (NUM_CTAS, 1, 1), matmul_kernel_swizzled, (d_A, d_B, d_C))
    stream.synchronize()

    # Measure
    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()
    
    start_event.record()
    for _ in range(n_iter):
        ct.launch(stream, (NUM_CTAS, 1, 1), matmul_kernel_swizzled, (d_A, d_B, d_C))
    end_event.record()
    end_event.synchronize()
    
    avg_ms = cp.cuda.get_elapsed_time(start_event, end_event) / n_iter
    
    # Cleanup
    del d_A, d_B, d_C
    cp.get_default_memory_pool().free_all_blocks()
    
    return avg_ms

# ============================================================
# 3. Main Loop
# ============================================================
def main():
    print("=" * 70)
    print(f"{'Size (MxNxK)':<15} | {'Tile':<10} | {'Time (ms)':<12} | {'TFLOPS':<10}")
    print("=" * 70)

    for size in MATRIX_SIZES:
        M = N = K = size
        
        best_time = float('inf')
        best_tile = None

        print(f"Testing {size}x{size}x{size}...")

        for t_m, t_n in TILE_CONFIGS:
            avg_ms = run_benchmark(M, N, K, t_m, t_n)
            
            if avg_ms < 0:
                print(f"{size:<15} | {t_m}x{t_n:<7} | {'N/A':<12} | {'-':<10}")
                continue

            # TFLOPS Calculation
            ops = 2.0 * M * N * K
            tflops = (ops / (avg_ms * 1e-3)) / 1e12

            print(f"{'':<15} | {t_m}x{t_n:<7} | {avg_ms:<12.3f} | {tflops:<10.2f}")

            if avg_ms < best_time:
                best_time = avg_ms
                best_tile = (t_m, t_n)

        print("-" * 70)
        print(f">> Best for {size}: Tile {best_tile} ({best_time:.3f} ms)")
        print("=" * 70)

if __name__ == "__main__":
    main()
