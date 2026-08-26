import cuda.tile as ct
import cupy as cp
import numpy as np

# 1. Define tile dimensions (equal to the full matrix size)
# By making B a 2x4 matrix, L=4, so all dimensions are even / powers of two.
M_SIZE = 4
N_SIZE = 2
L_SIZE = 4 

# 2. Kernel definition (processes a single tile)
@ct.kernel
def matmul_simple_kernel(A, B, C):
    # A, B, C are pointers (tensors) in global memory
    
    # [Load] Load the entire matrices as single tiles
    # A: shape (4 x 2)
    tile_A = ct.load(A, index=(0, 0), shape=(M_SIZE, N_SIZE))
    
    # B: shape (2 x 4)
    tile_B = ct.load(B, index=(0, 0), shape=(N_SIZE, L_SIZE))
    
    # [Compute] Matrix multiplication (MMA: Matrix Multiply-Accumulate)
    # (4 x 2) @ (2 x 4) -> (4 x 4)
    tile_C = tile_A @ tile_B
    
    # [Store] Store the result tile
    ct.store(C, index=(0, 0), tile=tile_C)

# 3. Main execution code
def main():
    # Prepare input data
    h_A = np.array([
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8]
    ], dtype=np.float32)  # 4x2

    h_B = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8]
    ], dtype=np.float32)  # 2x4

    # Allocate space for the result (4x4)
    h_C = np.zeros((M_SIZE, L_SIZE), dtype=np.float32)

    # Copy data to device (using CuPy)
    d_A = cp.asarray(h_A)
    d_B = cp.asarray(h_B)
    d_C = cp.asarray(h_C)

    # Launch the kernel
    # Since the entire computation fits in a single tile,
    # we use a grid of (1, 1, 1)
    stream = cp.cuda.get_current_stream()
    ct.launch(stream, (1, 1, 1), matmul_simple_kernel, (d_A, d_B, d_C))
    
    # Synchronize and copy result back to host
    result_C = cp.asnumpy(d_C)

    # Print results
    print("Matrix A (4x2):\n", h_A)
    print("\nMatrix B (2x4):\n", h_B)
    print("\nResult Matrix C (4x4):\n", result_C)

    # Verification using NumPy
    expected = np.dot(h_A, h_B)
    if np.allclose(result_C, expected):
        print("\nVerification: Success!")

if __name__ == "__main__":
    main()

