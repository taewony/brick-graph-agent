import cuda.tile as ct
import cupy as cp
import numpy as np
import torch

# ============================================================
# 1. Problem Configuration
# ============================================================
M_SIZE = N_SIZE = K_SIZE = 2048
TILE_SIZE = 128

NUM_BID_M = M_SIZE // TILE_SIZE   # 16
NUM_BID_N = N_SIZE // TILE_SIZE   # 16
NUM_K_TILES = K_SIZE // TILE_SIZE # 16

# [NOTE] Key parameter for L2 Cache Locality (Swizzling)
GROUP_SIZE_M = 4


# ============================================================
# 2. tile_id -> (bid_m, bid_n) Mapping Function
# ============================================================
def compute_tile_coords(tile_id, num_bid_m, num_bid_n, group_size_m):
    """
    Maps a linear tile_id to (bid_m, bid_n) coordinates with Grouped Swizzling.
    To maximize L2 cache hit rate for Matrix B, we iterate bid_m internally
    while keeping bid_n constant within the group.
    """
    
    # Total tiles handled in one horizontal strip of groups
    # We process 'group_size_m' rows across all 'num_bid_n' columns
    tiles_per_group_strip = group_size_m * num_bid_n

    # 1. Identify which horizontal strip (group) we are in
    group_id = tile_id // tiles_per_group_strip

    # 2. Offset within the current strip
    group_offset = tile_id % tiles_per_group_strip

    # 3. Calculate Coordinates
    # [FIX] Logic changed to iterate 'm' fast and 'n' slow within the group
    # to reuse the loaded tile_B (which depends on n).
    
    # bid_n changes slowly: 0, 0, 1, 1 ... (for group_size_m=2)
    bid_n_inner = group_offset // group_size_m
    
    # bid_m changes fast: 0, 1, 0, 1 ... (cycles within the group size)
    bid_m_inner = group_offset % group_size_m
    
    # Final bid_m combines the strip index (group_id) and inner index
    bid_m = group_id * group_size_m + bid_m_inner
    bid_n = bid_n_inner

    return bid_m, bid_n


# ============================================================
# 3. cuTile Kernel with Grouped Scheduling
# ============================================================
@ct.kernel
def matmul_tile_grouped_kernel(A, B, C):
    # --------------------------------------------------------
    # Persistent Kernel Pattern
    # --------------------------------------------------------
    start_tile_id = ct.bid(0)          # Starting tile ID for this CTA
    num_programs = ct.num_blocks(0)    # Total number of CTAs

    # Total number of output tiles in C
    total_tiles = NUM_BID_M * NUM_BID_N

    # --------------------------------------------------------
    # Persistent Scheduling Loop
    # --------------------------------------------------------
    for tile_id in range(start_tile_id, total_tiles, num_programs):

        # ----------------------------------------------------
        # Calculate Tile Coordinates (Swizzled)
        # ----------------------------------------------------
        bid_m, bid_n = compute_tile_coords(
            tile_id,
            NUM_BID_M,
            NUM_BID_N,
            GROUP_SIZE_M
        )

        # ----------------------------------------------------
        # Boundary Check
        # (Prevents out-of-bounds access if grid is not perfectly divisible)
        # ----------------------------------------------------
        if bid_m >= NUM_BID_M or bid_n >= NUM_BID_N:
            continue # Use continue instead of return to process remaining tiles

        # ----------------------------------------------------
        # Initialize Accumulator (FP32)
        # ----------------------------------------------------
        acc = ct.zeros((TILE_SIZE, TILE_SIZE), dtype=ct.float32)

        # ----------------------------------------------------
        # K-dimension Loop
        # ----------------------------------------------------
        for k in range(NUM_K_TILES):

            # Load tile from A
            tile_A = ct.load(
                A,
                index=(bid_m, k),
                shape=(TILE_SIZE, TILE_SIZE)
            )

            # Load tile from B
            # [NOTE] Thanks to Grouped Swizzling, multiple iterations (bid_m)
            # will share the same (k, bid_n) index, reusing this tile_B from L2 cache.
            tile_B = ct.load(
                B,
                index=(k, bid_n),
                shape=(TILE_SIZE, TILE_SIZE)
            )

            # Cast to FP16 for Tensor Core MMA
            # (Assuming standard Tensor Core usage for FP16 input)
            tile_A = ct.astype(tile_A, ct.float16)
            tile_B = ct.astype(tile_B, ct.float16)

            # Perform Matrix Multiply-Accumulate
            acc = ct.mma(tile_A, tile_B, acc=acc)

        # ----------------------------------------------------
        # Store Result
        # ----------------------------------------------------
        ct.store(
            C,
            index=(bid_m, bid_n),
            tile=ct.astype(acc, C.dtype)
        )


# ============================================================
# 4. Host Code
# ============================================================
def main():
    print(f"=== Grouped Tile MatMul (GROUP_SIZE_M = {GROUP_SIZE_M}) ===")

    # --------------------------------------------------------
    # Data Initialization
    # --------------------------------------------------------
    np.random.seed(0)
    h_A = np.random.randn(M_SIZE, K_SIZE).astype(np.float16)
    h_B = np.random.randn(K_SIZE, N_SIZE).astype(np.float16)
    h_C = np.zeros((M_SIZE, N_SIZE), dtype=np.float16)

    d_A = cp.asarray(h_A)
    d_B = cp.asarray(h_B)
    d_C = cp.asarray(h_C)

    # --------------------------------------------------------
    # Kernel Launch configuration
    # (Fewer CTAs than tiles -> Persistent Threads)
    # --------------------------------------------------------
    num_ctas = 8  # Can be tuned based on device SM count

    stream = cp.cuda.get_current_stream()
    ct.launch(
        stream,
        (num_ctas, 1, 1),
        matmul_tile_grouped_kernel,
        (d_A, d_B, d_C)
    )

    stream.synchronize()

    # --------------------------------------------------------
    # PyTorch Verification
    # --------------------------------------------------------
    print("Verifying results...")
    A_t = torch.from_numpy(h_A).cuda()
    B_t = torch.from_numpy(h_B).cuda()
    
    # Reference calculation (accumulates in FP32 usually, output FP16)
    C_ref = torch.matmul(A_t, B_t).cpu().numpy()
    C_cutile = cp.asnumpy(d_C)

    diff = np.abs(C_cutile - C_ref)
    print(f"Max error : {diff.max()}")
    print(f"Mean error: {diff.mean()}")

    # [FIX] Relaxed tolerance for FP16 precision limits (approx 1-2 bits error allowed)
    if np.allclose(C_cutile, C_ref, atol=2e-1, rtol=2e-2):
        print("✅ Verification: Success! (Within FP16 limits)")
    else:
        print("❌ Verification: Failed!")


if __name__ == "__main__":
    main()
