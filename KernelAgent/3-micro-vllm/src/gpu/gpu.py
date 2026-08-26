# gpu.py
import torch

if not torch.cuda.is_available():
    print("CUDA is NOT available.")
    exit()

device_count = torch.cuda.device_count()
print(f"Detected {device_count} CUDA device(s).\n")

for i in range(device_count):
    props = torch.cuda.get_device_properties(i)

    print(f"===== Device {i} =====")
    print(f"Name: {props.name}")
    print(f"Compute Capability: {props.major}.{props.minor}")
    print(f"Total Global Memory: {props.total_memory / (1024 ** 3):.2f} GB")
    print(f"Shared Memory per Block: {props.shared_memory_per_block // 1024} KB")
    print(f"Registers per Block: {props.regs_per_block}")
    print(f"Warp Size: {props.warp_size}")
    print(f"Max Threads per Block: {props.max_threads_per_block}")
    print(f"Max Threads Dim: {tuple(props.max_threads_dim)}")
    print(f"Max Grid Size: {tuple(props.max_grid_size)}")
    print(f"Clock Rate: {props.clock_rate / 1000:.2f} MHz")
    print(f"MultiProcessor Count: {props.multi_processor_count}")
    print("=====================\n")