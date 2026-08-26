import cuda.tile as ct
import cupy as cp
import torch
from dataclasses import dataclass
from typing import Any
from contextlib import contextmanager

# ============================================================
# 1. Autotuner Infrastructure
# ============================================================

@dataclass
class Config:
    tile_m: int
    tile_n: int
    tile_k: int
    num_ctas: int
    occupancy: int
    opt_level: int = 3

@dataclass
class TunedResult:
    config: Config
    best_time: float

def _time_ms(run_once, args, stream, warmup=2, rep=5):
    stream.synchronize()
    for _ in range(warmup): run_once(args)
    stream.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record(stream=torch.cuda.ExternalStream(stream.ptr))
    for _ in range(rep): run_once(args)
    end.record(stream=torch.cuda.ExternalStream(stream.ptr))
    end.synchronize()
    
    return start.elapsed_time(end) / max(1, rep)

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
# 2. Kernel Factory (The Robust Way)
# ============================================================
def make_matmul_kernel_source(M, N, K, cfg: Config):
    """
    Returns a RAW Python function (Closure) with constants baked in.
    Note: NO @ct.kernel decorator here! We apply it later in the tuner.
    """
    
    # Pre-calculate constants in Python to avoid kernel-side logic errors
    grid_m = M // cfg.tile_m
    grid_n = N // cfg.tile_n
    total_tiles = grid_m * grid_n
    num_k_tiles = K // cfg.tile_k
    
    # [FIX] Do NOT use @ct.kernel here. Return the raw function.
    def _kernel(A, B, C):
        bid = ct.bid(0)
        num_programs = ct.num_blocks(0)
        
        # Using captured 'total_tiles' (Hardcoded constant for this instance)
        for tile_id in range(bid, total_tiles, num_programs):
            
            # Simple Raster Order
            bid_m = tile_id // grid_n
            bid_n = tile_id % grid_n
            
            acc = ct.zeros((cfg.tile_m, cfg.tile_n), dtype=ct.float32)
            
            for k_idx in range(num_k_tiles):
                a = ct.load(A, index=(bid_m, k_idx), shape=(cfg.tile_m, cfg.tile_k))
                b = ct.load(B, index=(k_idx, bid_n), shape=(cfg.tile_k, cfg.tile_n))
                
                acc = ct.mma(ct.astype(a, ct.float16), ct.astype(b, ct.float16), acc)
                
            ct.store(C, index=(bid_m, bid_n), tile=ct.astype(acc, C.dtype))
            
    return _kernel

# ============================================================
# 3. Main Logic
# ============================================================
class Autotuner:
    def __init__(self, configs):
        self.configs = configs

    def tune(self, M, N, K, d_A, d_B, d_C, stream):
        best_time = float('inf')
        best_res = None
        
        print(f"   [Tuning] {M}x{N}x{K} ...")
        
        for cfg in self.configs:
            # Skip invalid tiles (must divide exactly)
            if M % cfg.tile_m != 0 or N % cfg.tile_n != 0:
                continue

            # 1. Get the RAW function (Source)
            kernel_func_raw = make_matmul_kernel_source(M, N, K, cfg)
            
            # 2. Compile it HERE (Apply decorator programmatically)
            # Now 'kernel_func_raw' is a real Python function, so this works.
            tuned_kernel = ct.kernel(
                kernel_func_raw,
                occupancy=cfg.occupancy,
                opt_level=cfg.opt_level
            )
            
            # 3. Define Run Closure
            # Note: num_ctas is used for GRID SIZE, not passed to kernel options
            grid = (cfg.num_ctas, 1, 1)
            args = (d_A, d_B, d_C)
            
            def run_once(rt_args):
                ct.launch(stream, grid, tuned_kernel, rt_args)
            
            try:
                # 4. Measure
                with compiler_timeout(5):
                    time = _time_ms(run_once, args, stream)
                
                # Debug Output (Optional)
                # tflops = (2.0*M*N*K) / (time*1e-3) / 1e12
                # print(f"     -> {cfg.tile_m}x{cfg.tile_n} (CTAs={cfg.num_ctas}): {time:.3f} ms ({tflops:.2f} TF)")
                
                if time < best_time:
                    best_time = time
                    best_res = TunedResult(cfg, best_time)
                    
            except Exception as e:
                # print(f"     [Skip] Error: {e}")
                continue
                
        if not best_res:
            return None
            
        return best_res

def main():
    cp.random.seed(0)
    dev = cp.cuda.Device(0)
    dev.use()
    
    props = cp.cuda.runtime.getDeviceProperties(0)
    NUM_SMS = props['multiProcessorCount']
    print(f">> Detected GPU: {props['name'].decode()} (SMs: {NUM_SMS})")
    
    # Define Search Space
    configs = []
    for tm, tn in [(128, 128), (128, 64), (64, 128), (64, 64)]:
        for occ in [1, 2]:
            configs.append(Config(
                tile_m=tm, tile_n=tn, tile_k=32,
                num_ctas=NUM_SMS * occ,
                occupancy=occ
            ))
            
    tuner = Autotuner(configs)
    
    sizes = [1024, 2048, 4096, 8192, 16384]
    
    print("=" * 80)
    print(f"{'Size':<10} | {'Best Tile':<15} | {'CTAs':<6} | {'Time (ms)':<10} | {'TFLOPS':<8}")
    print("=" * 80)
    
    stream = cp.cuda.get_current_stream()
    
    for size in sizes:
        M = N = K = size
        
        try:
            d_A = cp.random.randn(M, K, dtype=cp.float32).astype(cp.float16)
            d_B = cp.random.randn(K, N, dtype=cp.float32).astype(cp.float16)
            d_C = cp.zeros((M, N), dtype=cp.float16)
        except cp.cuda.memory.OutOfMemoryError:
            print(f"{size:<10} | OOM")
            continue
            
        res = tuner.tune(M, N, K, d_A, d_B, d_C, stream)
        
        if res:
            ops = 2.0 * M * N * K
            tflops = (ops / (res.best_time * 1e-3)) / 1e12
            tile_str = f"{res.config.tile_m}x{res.config.tile_n}"
            print(f"{size:<10} | {tile_str:<15} | {res.config.num_ctas:<6} | {res.best_time:<10.3f} | {tflops:<8.2f}")
        else:
            print(f"{size:<10} | Tuning Failed")
            
        del d_A, d_B, d_C
        cp.get_default_memory_pool().free_all_blocks()
        
    print("=" * 80)

if __name__ == "__main__":
    main()
