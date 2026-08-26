import torch
import cupy as cp

def print_gpu_details():
    print(f"{'='*60}")
    print(f" GPU Architecture Analysis for MatMul Optimization")
    print(f"{'='*60}")

    if not torch.cuda.is_available():
        print("No CUDA device found.")
        return

    # PyTorch Info
    device_id = torch.cuda.current_device()
    props_torch = torch.cuda.get_device_properties(device_id)
    
    # CuPy Low-level Info (cudaDeviceProp struct)
    dev_id_cupy = cp.cuda.get_device_id()
    props_cupy = cp.cuda.runtime.getDeviceProperties(dev_id_cupy)

    # 1. Basic Info
    print(f"Model Name           : {props_torch.name}")
    print(f"Architecture         : sm_{props_torch.major}{props_torch.minor}")
    print(f"Total VRAM (Global)  : {props_torch.total_memory / 1024**3:.2f} GB")
    print(f"SM Count (Cores)     : {props_torch.multi_processor_count} SMs")
    
    # 2. Memory Hierarchy (Crucial for Tiling & Swizzling)
    print(f"\n[Memory Hierarchy - For Tiling Strategy]")
    
    # L2 Cache: Swizzling 효율과 직결됩니다. (행렬 B가 캐시에 얼마나 들어가는지)
    l2_size = props_cupy['l2CacheSize']
    print(f"  L2 Cache Size      : {l2_size / 1024 / 1024:.2f} MB")
    
    # Shared Memory (L1): TILE_SIZE 결정에 결정적입니다. (예: 128x128 타일이 들어가는지)
    # sharedMemPerBlock: 블록당 최대 사용량
    # sharedMemPerMultiprocessor: SM당 물리적 한계 (Occupancy 계산용)
    sm_per_block = props_cupy['sharedMemPerBlock']
    sm_per_sm = props_cupy['sharedMemPerMultiprocessor']
    
    print(f"  Shared Mem / Block : {sm_per_block / 1024:.2f} KB (Max Tile Size constraint)")
    print(f"  Shared Mem / SM    : {sm_per_sm / 1024:.2f} KB (Occupancy constraint)")

    # 3. Register File (For Kernel Occupancy)
    regs_per_block = props_cupy['regsPerBlock']
    regs_per_sm = props_cupy['regsPerMultiprocessor'] if 'regsPerMultiprocessor' in props_cupy else "N/A"
    
    print(f"\n[Register File - For Occupancy]")
    print(f"  Registers / Block  : {regs_per_block} (32-bit)")
    print(f"  Registers / SM     : {regs_per_sm}")
    
    # 4. Bandwidth & Bus
    mem_bus_width = props_cupy['memoryBusWidth']
    mem_clock = props_cupy['memoryClockRate'] # kHz
    
    # Theoretical Bandwidth (GB/s) = (Clock * BusWidth * 2(DDR) / 8) / 1e6
    # Note: GDDR6/6X/7 effective clock calculation varies, rough estimate here.
    print(f"\n[Bus Interface]")
    print(f"  Memory Bus Width   : {mem_bus_width}-bit")
    print(f"  Memory Clock Rate  : {mem_clock / 1000:.0f} MHz")

    print(f"{'='*60}")
    
    # ----------------------------------------------------
    # Analysis for Optimization
    # ----------------------------------------------------
    print(f"\n💡 [Optimization Tips based on Specs]")
    
    # Tip 1: Tile Size
    # FP16 128x128 tile needs: 128*128*2 bytes (A) + 128*128*2 bytes (B) = 64KB (Double buffering -> 128KB)
    print(f"  1. Max Tile Size:")
    if sm_per_block >= 64 * 1024:
         print(f"     -> 128x128 FP16 Tile (Requires ~64KB+ SM) is SAFE.")
    else:
         print(f"     -> 128x128 might be too large. Consider 64x64.")

    # Tip 2: Swizzling (L2 Reuse)
    # How many 128x128 tiles fit in L2? (Assuming 4096 K dim)
    # Tile B (128x4096) size = 1MB. 
    l2_mb = l2_size / (1024**2)
    print(f"  2. Swizzling Strategy:")
    print(f"     -> With {l2_mb:.1f} MB L2 Cache, you can cache a significant portion of Matrix B.")
    print(f"     -> Large L2 allows larger GROUP_SIZE_M (e.g., 4 or 8) effectively.")

if __name__ == "__main__":
    print_gpu_details()


