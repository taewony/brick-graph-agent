import cuda.tile as ct
import cupy as cp
import numpy as np
import torch

# ============================================================
# 1. System Configuration
# ============================================================
dev_id = cp.cuda.get_device_id()
props = cp.cuda.runtime.getDeviceProperties(dev_id)
NUM_SMS = props['multiProcessorCount']

NUM_CTAS = NUM_SMS * 2   # Saturate GPU
print(f">> GPU SMs: {NUM_SMS}, CTAs launched: {NUM_CTAS}")

# Problem Size (fixed)
M_SIZE = N_SIZE = K_SIZE = 4096

# Sweep candidates
TILE_SIZES = [32, 64, 128, 256]
GROUP_SIZES = [0, 2, 4]

N_WARMUP = 3
N_ITER = 10

# ============================================================
# 2. Swizzle Mapping
# ============================================================
def compute_tile_coords(tile_id, num_m, num_n, group_size_m):
    if group_size_m == 0:
        return tile_id // num_n, tile_id % num_n

    tiles_per_group = group_size_m * num_n
    group_id = tile_id // tiles_per_group
    offset = tile_id % tiles_per_group

    bid_n = offset // group_size_m
    bid_m = group_id * group_size_m + (offset % group_size_m)
    return bid_m, bid_n

# ============================================================
# 3. Universal Kernel (Tile Size is dynamic)
# ============================================================
@ct.kernel
def matmul_kernel(A, B, C,
                  TILE_SIZE,
                  NUM_BID_M, NUM_BID_N, NUM_K_TILES,
                  group_size_m):

    start = ct.bid(0)
    stride = ct.num_blocks(0)
    total_tiles = NUM_BID_M * NUM_BID_N

    for tile_id in range(start, total_tiles, stride):
        bid_m, bid_n = compute_tile_coords(
            tile_id, NUM_BID_M, NUM_BID_N, group_size_m
        )

        if bid_m >= NUM_BID_M or bid_n >= NUM_BID_N:
            continue

        acc = ct.zeros((TILE_SIZE, TILE_SIZE), dtype=ct.float32)

        for k in range(NUM_K_TILES):
            A_tile = ct.load(A, (bid_m, k), (TILE_SIZE, TILE_SIZE))
            B_tile = ct.load(B, (k, bid_n), (TILE_SIZE, TILE_SIZE))

            acc = ct.mma(
                ct.astype(A_tile, ct.float16),
                ct.astype(B_tile, ct.float16),
                acc
            )

        ct.store(C, (bid_m, bid_n), ct.astype(acc, C.dtype))

# ============================================================
# 4. Benchmark Helper
# ============================================================
def benchmark(d_A, d_B, d_C,
              TILE_SIZE, group_size):

    NUM_BID_M = M_SIZE // TILE_SIZE
    NUM_BID_N = N_SIZE // TILE_SIZE
    NUM_K_TILES = K_SIZE // TILE_SIZE

    stream = cp.cuda.get_current_stream()

    for _ in range(N_WARMUP):
        ct.launch(
            stream, (NUM_CTAS, 1, 1),
            matmul_kernel,
            (d_A, d_B, d_C,
             TILE_SIZE,
             NUM_BID_M, NUM_BID_N, NUM_K_TILES,
             group_size)
        )
    stream.synchronize()

    start = cp.cuda.Event()
    end = cp.cuda.Event()

    start.record()
    for _ in range(N_ITER):
        ct.launch(
            stream, (NUM_CTAS, 1, 1),
            matmul_kernel,
            (d_A, d_B, d_C,
             TILE_SIZE,
             NUM_BID_M, NUM_BID_N, NUM_K_TILES,
             group_size)
        )
    end.record()
    end.synchronize()

    return cp.cuda.get_elapsed_time(start, end) / N_ITER

# ============================================================
# 5. Main Sweep
# ============================================================
def main():
    np.random.seed(0)

    h_A = np.random.randn(M_SIZE, K_SIZE).astype(np.float16)
    h_B = np.random.randn(K_SIZE, N_SIZE).astype(np.float16)
    h_C = np.zeros((M_SIZE, N_SIZE), dtype=np.float16)

    d_A = cp.asarray(h_A)
    d_B = cp.asarray(h_B)
    d_C = cp.asarray(h_C)

    A_t = torch.from_numpy(h_A).cuda()
    B_t = torch.from_numpy(h_B).cuda()

    # PyTorch baseline
    for _ in range(N_WARMUP):
        _ = torch.matmul(A_t, B_t)
    torch.cuda.synchronize()

    start = torch.cuda.Event(True)
    end = torch.cuda.Event(True)

    start.record()
    for _ in range(N_ITER):
        _ = torch.matmul(A_t, B_t)
    end.record()
    torch.cuda.synchronize()

    t_torch = start.elapsed_time(end) / N_ITER
    print(f"\nPyTorch (cuBLAS): {t_torch:.3f} ms\n")

    print("Tile | Group | Time(ms) | Rel.Speed")
    print("-" * 40)

    for TILE in TILE_SIZES:
        for G in GROUP_SIZES:
            t = benchmark(d_A, d_B, d_C, TILE, G)
            print(f"{TILE:4d} | {G:5d} | {t:8.3f} | {t_torch/t:6.2f}x")

if __name__ == "__main__":
    main()

