"""
Example demonstrating simple vector addition.
Shows how to perform elementwise operations on vectors.
"""
import cupy as cp
import numpy as np
import cuda.tile as ct

# Define the cuTile kernel using a decorator
@ct.kernel
def vector_add_kernel(a, b, result, tile_size: ct.Constant[int]):
    # Get the 1D block ID (processing unit identifier)
    pid = ct.bid(0)

    # Load input data into tiles from global memory
    a_tile = ct.load(a, index=(pid,), shape=(tile_size,))
    b_tile = ct.load(b, index=(pid,), shape=(tile_size,))

    # Perform element-wise addition on the tiles (this runs on the GPU)
    result_tile = a_tile + b_tile

    # Store the result tile back to the output array in global memory
    ct.store(result, index=(pid,), tile=result_tile)

def run_test():
    # Define vector and tile sizes (must be powers of two for tiles)
    vector_size = 4096 * 256 # no of blocks = 4098
    tile_size = 256
    
    # Calculate the grid size needed to cover the entire vector
    grid = (ct.cdiv(vector_size, tile_size), 1, 1) # 1D grid for a 1D problem

    # Create input data using CuPy arrays (these reside in GPU memory)
    a = cp.random.uniform(-1, 1, vector_size, dtype=cp.float32)
    b = cp.random.uniform(-1, 1, vector_size, dtype=cp.float32)
    c = cp.zeros_like(a)

    # Launch the cuTile kernel
    ct.launch(cp.cuda.get_current_stream(), grid, vector_add_kernel, (a, b, c, tile_size))

    # --- Verification (optional, typically done on the host) ---
    # Copy results from GPU back to host (NumPy arrays in RAM)
    a_np = cp.asnumpy(a)
    b_np = cp.asnumpy(b)
    c_np = cp.asnumpy(c)

   # Verify results
    expected = a_np + b_np
    np.testing.assert_array_almost_equal(c_np, expected)

    print("✓ vector_add_example passed!")


if __name__ == "__main__":
    run_test()

