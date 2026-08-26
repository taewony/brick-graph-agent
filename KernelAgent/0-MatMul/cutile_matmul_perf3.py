import cuda.tile as ct
import cupy as cp
import numpy as np
from functools import partial

# ============================================================
# 1. System Configuration
# ============================================================
dev_id = cp.cuda.get_device_id()
props = cp.cuda.runtime.getDeviceProperties(dev_id)
NUM_SMS = props['multiProcessorCount']

# Launch configuration:
# Launching enough blocks (CTAs) to saturate all SMs.
# 2 CTAs per SM is a good heuristic to hide memory latency on RTX 5070.
NUM_CTAS = NUM_SMS * 2

print(f">> GPU Detected: {props['name'].decode()} with {NUM_SMS} SMs.")
print(f">> Using {NUM_CTAS} CTAs for persistent kernels.\n")

# Matrix sizes to test
MATRIX_SIZES = [1024, 2048, 4096, 8192, 16384, 32768]

# Tile configurations to test (Tile_M, Tile_N)
TILE_CONFIGS = [
    (128, 128), # Baseline: High register pressure
    (128, 64),  # Rectangular: Reduced register pressure for M
    (64, 128),  # Rectangular: Reduced register pressure for N
    (64, 64)    # Small: Maximize Occupancy
]

# ============================================================
# 2. Benchmark Function
# ============================================================
def run_benchmark(M, N, K, tile_m, tile_n):
    """
    Runs a benchmark for specific Matrix size and Tile size.
    Generates a JIT-compiled kernel via closure to optimize for constants.
    """
    
    # --------------------------------------------------------
    # Grid & Tiling Logic
    # --------------------------------------------------------
    # Ensure matrix dimensions are divisible by tile size
    # (For simplicity in this benchmark, we assume they are. 
    # In production, boundary checks are needed.)
    if M % tile_m != 0 or N % tile_n != 0:
        return -1.0 # Skip invalid configurations if any

    grid_m = M // tile_m
    grid_n = N // tile_n
    
    # Standard K-step for Tensor Core operations (usually 32 or 64)
    k_step = 64 
    grid_k = K // k_step

    # --------------------------------------------------------
    # Kernel Definition (Closure captures tile_m, tile_n, etc.)
    # --------------------------------------------------------
    @ct.kernel
    def matmul_kernel_dynamic(A, B, C):
        # Persistent Thread Block ID
        bid = ct.bid(0)
        num_programs = ct.num_blocks(0)
        total_tiles = grid_m * grid_n

        # Grid-stride loop (Persistent Threads)
        for tile_id in range(bid, total_tiles, num_programs):
            # 1. Map linear tile_id to 2D coordinates
            # (Using simple Raster Order for consistent comparison)
            bid_m = tile_id // grid_n
            bid_n = tile_id % grid_n
            
            # 2. Initialize Accumulator (FP32 for precision)
            # The size is fixed at compile-time via closure capture
            acc = ct.zeros((tile_m, tile_n), dtype=ct.float32)

            # 3. K-dimension Loop (Main Compute)
            for k in range(grid_k):
                # Load Tile A: (tile_m, k_step)
                tile_A = ct.load(A, index=(bid_m, k), shape=(tile_m, k_step))
                
                # Load Tile B: (k_step, tile_n)
                tile_B = ct.load(B, index=(k, bid_n), shape=(k_step, tile_n))
                
                # Cast to FP16 for Tensor Core MMA
                tile_A = ct.astype(tile_A, ct.float16)
                tile_B = ct.astype(tile_B, ct.float16)
                
                # Matrix Multiply-Accumulate
                acc = ct.mma(tile_A, tile_B, acc=acc)

            # 4. Store Result
            ct.store(C, index=(bid_m, bid_n), tile=ct.astype(acc, C.dtype))

    # --------------------------------------------------------
    # Data Preparation
    # --------------------------------------------------------
    # Generate random data directly on GPU to save Host RAM and PCIe time.
    # Note: 32768^2 * 2 bytes = ~2 GB. 3 matrices = ~6 GB. 
    # This fits within RTX 5070 12GB VRAM.
    try:
        # [FIX] generate in float32, then cast to float16
        # CuPy random generator does not support float16 directly.
        d_A = cp.random.randn(M, K, dtype=cp.float32).astype(cp.float16)
        d_B = cp.random.randn(K, N, dtype=cp.float32).astype(cp.float16)
        d_C = cp.zeros((M, N), dtype=cp.float16)
    except cp.cuda.memory.OutOfMemoryError:
        print(f"   [Error] OOM for size {M}x{N}. Skipping...")
        return -1.0

    stream = cp.cuda.get_current_stream()

    # --------------------------------------------------------
    # Execution & Measurement
    # --------------------------------------------------------
    # Adjust iterations based on problem size to save time
    n_warmup = 3
    n_iter = 10 if M <= 8192 else 3  # Run fewer iterations for very large matrices

    # Warmup
    for _ in range(n_warmup):
        ct.launch(stream, (NUM_CTAS, 1, 1), matmul_kernel_dynamic, (d_A, d_B, d_C))
    stream.synchronize()

    # Measure
    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()
    
    start_event.record()
    for _ in range(n_iter):
        ct.launch(stream, (NUM_CTAS, 1, 1), matmul_kernel_dynamic, (d_A, d_B, d_C))
    end_event.record()
    end_event.synchronize()
    
    avg_ms = cp.cuda.get_elapsed_time(start_event, end_event) / n_iter
    
    # Clean up memory explicitly
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
        
        # Track best time for this matrix size
        best_time = float('inf')
        best_tile = None

        print(f"Testing {size}x{size}x{size}...")

        for t_m, t_n in TILE_CONFIGS:
            # Run Benchmark
            avg_ms = run_benchmark(M, N, K, t_m, t_n)
            
            if avg_ms < 0:
                print(f"{size:<15} | {t_m}x{t_n:<7} | {'N/A':<12} | {'-':<10}")
                continue

            # Calculate TFLOPS (2 * M * N * K / time / 1e12)
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

