import math
import torch
import cuda.tile as ct

# Define constant type
ConstInt = ct.Constant[int]

def next_power_of_2(n):
    """Returns the next power of 2 greater than or equal to n."""
    return 1 << (n - 1).bit_length()

# -------------------------------------------------------------------------
# 1. GPU Kernel Definition (Simplified using load/store)
# -------------------------------------------------------------------------
@ct.kernel(occupancy=4)
def simple_softmax_kernel(
    output,              # Output tensor pointer
    input,               # Input tensor pointer
    n_rows: ConstInt,    # Number of rows
    n_cols: ConstInt,    # Number of columns (actual data width)
    TILE_SIZE: ConstInt, # Tile width (power of 2)
):
    # [Persistent Threading]
    # Get the program ID (block ID) and total number of programs
    pid = ct.bid(0)
    num_programs = ct.num_blocks(0)

    # Each block processes multiple rows in a loop
    for row_idx in range(pid, n_rows, num_programs):
        
        # (1) Load Data
        # Load a 1D tile from global memory. 
        # padding_mode=NEG_INF ensures padded values don't affect max/exp calculation.
        row = ct.load(
            input, 
            index=(row_idx, 0), 
            shape=(1, TILE_SIZE), 
            padding_mode=ct.PaddingMode.NEG_INF
        )
        
        # (2) Type Conversion
        # Convert to float32 for high-precision calculation
        row = ct.astype(row, ct.float32)

        # (3) Numerical Stability
        # Find max value in the row and subtract it to prevent exp() overflow
        row_max = ct.max(row, 1, keepdims=True)
        row_minus_max = ct.sub(row, row_max)

        # (4) Softmax Calculation
        # Compute exponential
        numerator = ct.exp(row_minus_max)
        
        # Compute sum of exponentials
        denominator = ct.sum(numerator, 1, keepdims=True)
        
        # Divide to get probabilities
        softmax_output = ct.truediv(numerator, denominator)

        # (5) Store Result
        # Convert back to original data type and store to global memory
        softmax_output = ct.astype(softmax_output, input.dtype)
        
        ct.store(
            output, 
            index=(row_idx, 0), 
            tile=softmax_output
        )

# -------------------------------------------------------------------------
# 2. Host Launcher
# -------------------------------------------------------------------------
def run_tile_softmax(input_tensor):
    n_rows, n_cols = input_tensor.shape
    
    # Create output tensor
    output_tensor = torch.empty_like(input_tensor)
    
    # Ensure memory is contiguous
    input_tensor = input_tensor.contiguous()
    output_tensor = output_tensor.contiguous()

    # Determine tile size (must be power of 2 to cover n_cols)
    tile_size = next_power_of_2(n_cols)

    # Calculate grid size based on GPU SM count
    num_sm = torch.cuda.get_device_properties(input_tensor.device).multi_processor_count
    occupancy = 4
    num_programs = min(num_sm * occupancy, n_rows)
    
    grid = (num_programs, 1, 1)

    # Launch kernel
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        simple_softmax_kernel,
        (
            output_tensor,
            input_tensor,
            n_rows,
            n_cols,
            tile_size
        ),
    )
    
    return output_tensor

# -------------------------------------------------------------------------
# 3. Main: Execution & Benchmarking
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # Configuration
    ROWS = 4096
    COLS = 1024
    dtype = torch.float16 # Use fp16 for speed
    device = "cuda"

    print(f"Configuration -> Size: ({ROWS}, {COLS}), Dtype: {dtype}")

    # Generate random data
    x = torch.randn(ROWS, COLS, device=device, dtype=dtype)

    # 1. Verification (Compare with PyTorch)
    print("Verifying correctness...")
    y_ref = torch.softmax(x.float(), dim=1).to(dtype) # PyTorch Reference
    y_tile = run_tile_softmax(x)                      # TileGym Kernel

    if torch.allclose(y_ref, y_tile, atol=1e-2, rtol=1e-2):
        print("✅ Correctness Pass!")
    else:
        print("❌ Correctness Fail!")
        # Debug: check max difference
        diff = (y_ref - y_tile).abs().max().item()
        print(f"   Max Difference: {diff}")

    # 2. Benchmarking
    print("\nBenchmarking...")
    
    # Warmup
    for _ in range(10):
        run_tile_softmax(x)
    torch.cuda.synchronize()

    # Measure execution time
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    iters = 100
    start_event.record()
    for _ in range(iters):
        run_tile_softmax(x)
    end_event.record()
    
    torch.cuda.synchronize()
    
    elapsed_time_ms = start_event.elapsed_time(end_event) / iters
    print(f"🚀 Average Execution Time: {elapsed_time_ms:.4f} ms")

