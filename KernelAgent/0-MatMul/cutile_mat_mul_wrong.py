"""
A minimal example demonstrating matrix multiplication:
- cuTile kernel definition
- tile load / compute / store
- correct execution and verification
"""

import cuda.tile as ct
import cupy as cp
import numpy as np


# ------------------------------------------------------------
# Tile size (entire matrix fits in one tile)
# ------------------------------------------------------------
TILE_M = 4
TILE_N = 4


# ------------------------------------------------------------
# cuTile kernel
# ------------------------------------------------------------
@ct.kernel
def hello_cutile_kernel(A, B, C):
    """
    A, B, C are 2D tensors in global memory.
    This kernel loads one tile, adds them, and stores the result.
    """

    # Load one tile from A and B
    # index=(0,0) means "the first tile"
    a_tile = ct.load(A, index=(0, 0), shape=(TILE_M, TILE_N))
    b_tile = ct.load(B, index=(0, 0), shape=(TILE_M, TILE_N))

    # Simple elementwise computation
    c_tile = a_tile @ b_tile

    # Store the result tile
    ct.store(C, index=(0, 0), tile=c_tile)


# ------------------------------------------------------------
# Host-side test code
# ------------------------------------------------------------
def main():
    # Create small input matrices
    h_A = np.array(
        [[ 1,  2,  3,  4],
         [ 5,  6,  7,  8],
         [ 9, 10, 11, 12],
         [13, 14, 15, 16]],
        dtype=np.float32
    )

    h_B = np.ones((TILE_M, TILE_N), dtype=np.float32)

    # Allocate output
    h_C = np.zeros_like(h_A)

    # Copy to device
    d_A = cp.asarray(h_A)
    d_B = cp.asarray(h_B)
    d_C = cp.asarray(h_C)

    # Launch kernel
    # Only one tile → grid = (1, 1, 1)
    ct.launch(
        cp.cuda.get_current_stream(),
        (1, 1, 1),
        hello_cutile_kernel,
        (d_A, d_B, d_C)
    )

    # Copy result back
    result = cp.asnumpy(d_C)

    # Print results
    print("Matrix A:")
    print(h_A)

    print("\nMatrix B:")
    print(h_B)

    print("\nResult C = A + B:")
    print(result)

    # Verification
    expected = h_A @ h_B
    if np.allclose(result, expected):
        print("\n✅ Hello cuTile! Verification passed.")
    else:
        print("\n❌ Verification failed.")


if __name__ == "__main__":
    main()

