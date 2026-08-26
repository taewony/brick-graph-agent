import cuda.tile as ct
import cupy as cp
import torch
from dataclasses import dataclass
from contextlib import contextmanager

# ============================================================
# 1. Configuration & Constants
# ============================================================
@dataclass
class Config:
    tile_m: int
    tile_n: int
    tile_k: int
    occupancy: int  # 1, 2, 4
    
    @property
    def num_ctas(self):
        # Note: NUM_SMS will be set in main()
        return NUM_SMS * self.occupancy

@contextmanager
def compiler_timeout(timeout_sec: int):
    try:
        from cuda.tile._cext import default_tile_context
        old = default_tile_context.config.compiler_timeout_sec
        default_tile_context.config.compiler_timeout_sec = timeout_sec
        yield
        default_tile_context.config.compiler_timeout_sec = old
    except (ImportError, AttributeError):
        yield

# ============================================================
# 2. High-Performance Kernel Factory (Fixed)
# ============================================================
def make_kernel_source(M, N, K, cfg: Config):
    """
    Returns a Python function (Closure) with constants baked in.
    FIX: Unpack 'cfg' attributes to local variables so JIT can capture them as constants.
    """
    
    # [FIX] Unpack attributes here! The JIT cannot read 'cfg.tile_m' inside the kernel.
    tile_m = cfg.tile_m
    tile_n = cfg.tile_n
    tile_k = cfg.tile_k
    
    # Pre-calculate Grid constants
    grid_m = (M + tile_m - 1) // tile_m
    grid_n = (N + tile_n - 1) // tile_n
    total_tiles = grid_m * grid_n
    num_k_tiles = (K + tile_k - 1) // tile_k
    
    # Swizzling Constant
    GROUP_SIZE_M = 8

    # Note: Do NOT add @ct.kernel here. We apply it in run_benchmark.
    def _kernel(A, B, C):
        bid = ct.bid(0)
        num_programs = ct.num_blocks(0)
        
        # [Persistent Loop]
        for tile_id in range(bid, total_tiles, num_programs):
            
            # [Swizzling Logic]
            num_bid_in_group = GROUP_SIZE_M * grid_n
            group_id = tile_id // num_bid_in_group
            first_bid_m = group_id * GROUP_SIZE_M
            
            group_size_m = min(grid_m - first_bid_m, GROUP_SIZE_M)
            
            bid_m = first_bid_m + (tile_id % group_size_m)
            bid_n = (tile_id % num_bid_in_group) // group_size_m
            
            # [FIX] Use local variables (tile_m, tile_n) instead of cfg.tile_m
            acc = ct.zeros((tile_m, tile_n), dtype=ct.float32)
            
            # [Main Compute Loop]
            for k_idx in range(num_k_tiles):
                # Load A
                a = ct.load(A, index=(bid_m, k_idx), 
                            shape=(tile_m, tile_k), 
                            padding_mode=ct.PaddingMode.ZERO)
                
                # Load B
                b = ct.load(B, index=(k_idx, bid_n), 
                            shape=(tile_k, tile_n), 
                            padding_mode=ct.PaddingMode.ZERO)
                
                # MMA
                acc = ct.mma(ct.astype(a, ct.float16), ct.astype(b, ct.float16), acc)
                
            # Store Result
            ct.store(C, index=(bid_m, bid_n), tile=ct.astype(acc, C.dtype))
            
    return _kernel

# ============================================================
# 3. Benchmarking Logic
# ============================================================
def run_benchmark(M, N, K, configs, stream):
    results = []
    
    # -------------------------------------------------------
    # A. Data Preparation
    # -------------------------------------------------------
    try:
        d_A = cp.random.randn(M, K, dtype=cp.float32).astype(cp.float16)
        d_B = cp.random.randn(K, N, dtype=cp.float32).astype(cp.float16)
        d_C = cp.zeros((M, N), dtype=cp.float16)
        
        t_A = torch.as_tensor(d_A, device='cuda')
        t_B = torch.as_tensor(d_B, device='cuda')
    except cp.cuda.memory.OutOfMemoryError:
        return "OOM", []

    # -------------------------------------------------------
    # B. PyTorch Baseline
    # -------------------------------------------------------
    for _ in range(5): torch.matmul(t_A, t_B)
    torch.cuda.synchronize()
    
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    
    start_evt.record()
    for _ in range(10): torch.matmul(t_A, t_B)
    end_evt.record()
    torch.cuda.synchronize()
    
    torch_ms = start_evt.elapsed_time(end_evt) / 10.0
    torch_tflops = (2.0 * M * N * K) / (torch_ms * 1e-3) / 1e12

    # -------------------------------------------------------
    # C. CuTile Configs
    # -------------------------------------------------------
    best_cutile_tflops = 0
    best_cfg_str = ""

    for cfg in configs:
        # 1. Get Source (with constants captured)
        kernel_source = make_kernel_source(M, N, K, cfg)
        
        try:
            # 2. Compile
            with compiler_timeout(4):
                kernel = ct.kernel(kernel_source, occupancy=cfg.occupancy)
        except Exception as e:
            # print(f"Compile skipped for {cfg}: {e}")
            continue

        # 3. Launch Setup
        grid = (cfg.num_ctas, 1, 1)
        args = (d_A, d_B, d_C)
        
        def run_cutile():
            ct.launch(stream, grid, kernel, args)
            
        stream.synchronize()
        for _ in range(3): run_cutile()
        stream.synchronize()
        
        start_evt.record()
        for _ in range(5): run_cutile()
        end_evt.record()
        torch.cuda.synchronize()
        
        cutile_ms = start_evt.elapsed_time(end_evt) / 5.0
        cutile_tflops = (2.0 * M * N * K) / (cutile_ms * 1e-3) / 1e12
        
        results.append((cfg, cutile_ms, cutile_tflops))
        
        if cutile_tflops > best_cutile_tflops:
            best_cutile_tflops = cutile_tflops
            best_cfg_str = f"{cfg.tile_m}x{cfg.tile_n}x{cfg.tile_k} (Occ={cfg.occupancy})"

    del d_A, d_B, d_C, t_A, t_B
    cp.get_default_memory_pool().free_all_blocks()
    
    return (torch_tflops, best_cutile_tflops, best_cfg_str, results)

# ============================================================
# 4. Main Execution
# ============================================================
def main():
    global NUM_SMS
    dev_id = cp.cuda.get_device_id()
    props = cp.cuda.runtime.getDeviceProperties(dev_id)
    NUM_SMS = props['multiProcessorCount']
    
    print(f">> GPU: {props['name'].decode()} | SMs: {NUM_SMS}")
    print(">> Strategy: Fixed Tile_K=64, Sweeping Occupancy [1, 2, 4], Swizzling Enabled")
    print("=" * 100)
    print(f"{'Size':<10} | {'PyTorch':<8} | {'CuTile':<8} | {'Speedup':<8} | {'Best Config':<30}")
    print(f"{'(MxNxK)':<10} | {'(TFLOPS)':<8} | {'(TFLOPS)':<8} | {'(%)':<8}     | {'(Tile_M x Tile_N x Tile_K)'}")
    print("=" * 100)

    configs = []
    # 1. High Occupancy (Latency Hiding)
    configs.append(Config(64, 64, 64, occupancy=4))
    configs.append(Config(64, 64, 64, occupancy=2))
    
    # 2. Balanced
    configs.append(Config(128, 64, 64, occupancy=2))
    configs.append(Config(128, 64, 64, occupancy=1))
    
    # 3. Large Tiles (Throughput)
    configs.append(Config(128, 128, 64, occupancy=1))

    sizes = [1024, 2048, 4096, 8192, 16384]
    stream = cp.cuda.get_current_stream()

    for size in sizes:
        M = N = K = size
        
        ret = run_benchmark(M, N, K, configs, stream)
        
        if ret == "OOM":
            print(f"{size:<10} | {'OOM':<30}")
            continue
            
        torch_tf, cutile_tf, best_cfg, all_res = ret
        
        speedup = (cutile_tf - torch_tf) / torch_tf * 100
        sign = "+" if speedup > 0 else ""
        
        print(f"{size:<10} | {torch_tf:<8.2f} | {cutile_tf:<8.2f} | {sign}{speedup:<7.1f}% | {best_cfg}")
        
    print("=" * 100)

if __name__ == "__main__":
    main()
