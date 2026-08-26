import cuda.tile as ct
import cupy as cp
import numpy as np
import torch

# ============================================================
# 1. Problem size
# ============================================================
M_SIZE = 2048
N_SIZE = 2048
K_SIZE = 2048

TILE_SIZE = 128   # tile size

GRID_M = M_SIZE // TILE_SIZE   # 16
GRID_N = N_SIZE // TILE_SIZE   # 16
GRID_K = K_SIZE // TILE_SIZE   # 16


# ============================================================
# 2. cuTile kernel
# ============================================================
@ct.kernel
def matmul_tile_kernel(A, B, C):
    # Block (tile) indices
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)

    # FP32 accumulator
    acc = ct.zeros((TILE_SIZE, TILE_SIZE), dtype=ct.float32)

    for k in range(GRID_K):
        # Load A tile
        tile_A = ct.load(
            A,
            index=(bid_m, k),
            shape=(TILE_SIZE, TILE_SIZE)
        )

        # Load B tile
        tile_B = ct.load(
            B,
            index=(k, bid_n),
            shape=(TILE_SIZE, TILE_SIZE)
        )

        # Convert for compute
        tile_A = ct.astype(tile_A, ct.float16)
        tile_B = ct.astype(tile_B, ct.float16)

        # Compute (lowered internally to MMA-safe form)
        acc += tile_A @ tile_B

    # Store result
    ct.store(
        C,
        index=(bid_m, bid_n),
        tile=ct.astype(acc, C.dtype)
    )


# ============================================================
# 3. Main
# ============================================================
def main():
    print("=== cuTile 2048x2048 MatMul (WORKING) ===")

    # --------------------------------------------------------
    # Host data
    # --------------------------------------------------------
    np.random.seed(0)
    h_A = np.random.randn(M_SIZE, K_SIZE).astype(np.float16)
    h_B = np.random.randn(K_SIZE, N_SIZE).astype(np.float16)
    h_C = np.zeros((M_SIZE, N_SIZE), dtype=np.float16)

    # --------------------------------------------------------
    # Device copy
    # --------------------------------------------------------
    d_A = cp.asarray(h_A)
    d_B = cp.asarray(h_B)
    d_C = cp.asarray(h_C)

    # --------------------------------------------------------
    # Launch kernel
    # --------------------------------------------------------
    stream = cp.cuda.get_current_stream()
    ct.launch(
        stream,
        (GRID_M, GRID_N, 1),
        matmul_tile_kernel,
        (d_A, d_B, d_C)
    )
    stream.synchronize()

    result_cutile = cp.asnumpy(d_C)

    # ========================================================
    # Verification with PyTorch's result
    # ========================================================
    print("Verifying with PyTorch...")

    A_torch = torch.from_numpy(h_A).cuda()
    B_torch = torch.from_numpy(h_B).cuda()

    with torch.no_grad():
        C_ref = torch.matmul(A_torch, B_torch)

    C_ref_np = C_ref.cpu().numpy()

    abs_diff = np.abs(result_cutile - C_ref_np)
    print("Max error :", abs_diff.max())
    print("Mean error:", abs_diff.mean())

    # Relax absolute tolerance accounting for FP16 precision limits 
    # A deviation of 0.125 corresponds to the quantization error of FP16.
    # We set atol to 0.2 to accommodate these inherent 1-2 bit differences.
    if np.allclose(result_cutile, C_ref_np, atol=2e-1, rtol=2e-2):
        print("O Verification: Success! (Within FP16 precision limits)")
    else:
        print("X Verification: Failed!")


if __name__ == "__main__":
    main()

