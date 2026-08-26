PS D:\code\metaprogramming\KernelAgent> python --version
Python 3.13.14
PS D:\code\metaprogramming\KernelAgent>   python -c "import torch; print('torch', torch.__version__); print('cuda', torch.version.cuda); print('gpu',
>>   torch.cuda.get_device_name(0))"
torch 2.12.0+cu130
cuda 13.0
gpu NVIDIA GeForce RTX 5070
PS D:\code\metaprogramming\KernelAgent> nvidia-smi
Thu Jul  2 10:26:25 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 610.47                 KMD Version: 610.47        CUDA UMD Version: 13.3     |
+-----------------------------------------+------------------------+----------------------+


PS D:\code\metaprogramming\KernelAgent> python micro-vllm/bench.py --use-cutile
🚀 Using cuTile attention backend
`torch_dtype` is deprecated! Use `dtype` instead!
Generating: 100%|█████████████████████████████████████████████████████████████████████████████████| 1/1 [00:03<00:00,  3.63s/it, Prefill=21tok/s, Decode=27tok/s]
🚀 Starting benchmark generation loop...
⏱️ [Progress] Elapsed: 30.1s | Finished: 13/256 (5.1%) | Active: 61 running, 182 waiting | Generated: 16318 tokens | Decode Throughput: 542.3 tok/s
⏱️ [Progress] Elapsed: 60.1s | Finished: 40/256 (15.6%) | Active: 51 running, 165 waiting | Generated: 32043 tokens | Decode Throughput: 532.9 tok/s
⏱️ [Progress] Elapsed: 90.2s | Finished: 68/256 (26.6%) | Active: 49 running, 139 waiting | Generated: 46079 tokens | Decode Throughput: 510.9 tok/s
⏱️ [Progress] Elapsed: 120.2s | Finished: 99/256 (38.7%) | Active: 58 running, 99 waiting | Generated: 60951 tokens | Decode Throughput: 506.9 tok/s
⏱️ [Progress] Elapsed: 150.3s | Finished: 130/256 (50.8%) | Active: 53 running, 73 waiting | Generated: 75940 tokens | Decode Throughput: 505.2 tok/s
⏱️ [Progress] Elapsed: 180.4s | Finished: 154/256 (60.2%) | Active: 52 running, 50 waiting | Generated: 91004 tokens | Decode Throughput: 504.5 tok/s
⏱️ [Progress] Elapsed: 210.4s | Finished: 183/256 (71.5%) | Active: 59 running, 14 waiting | Generated: 106085 tokens | Decode Throughput: 504.1 tok/s
⏱️ [Progress] Elapsed: 240.5s | Finished: 213/256 (83.2%) | Active: 43 running, 0 waiting | Generated: 121984 tokens | Decode Throughput: 507.2 tok/s
⏱️ [Progress] Elapsed: 270.5s | Finished: 249/256 (97.3%) | Active: 7 running, 0 waiting | Generated: 132440 tokens | Decode Throughput: 489.5 tok/s
Total: 133966tok, Time: 291.17s, Throughput: 460.10tok/s
PS D:\code\metaprogramming\KernelAgent> python micro-vllm/bench.py --use-cutile
🚀 Using cuTile attention backend
`torch_dtype` is deprecated! Use `dtype` instead!
Generating: 100%|█████████████████████████████████████████████████████████████████████████████████| 1/1 [00:03<00:00,  3.13s/it, Prefill=23tok/s, Decode=25tok/s]
🚀 Starting benchmark generation loop...
⏱️ [Progress] Elapsed: 30.0s | Finished: 13/256 (5.1%) | Active: 61 running, 182 waiting | Generated: 16623 tokens | Decode Throughput: 553.5 tok/s
⏱️ [Progress] Elapsed: 60.1s | Finished: 40/256 (15.6%) | Active: 52 running, 164 waiting | Generated: 32095 tokens | Decode Throughput: 534.1 tok/s
⏱️ [Progress] Elapsed: 90.2s | Finished: 70/256 (27.3%) | Active: 52 running, 134 waiting | Generated: 46531 tokens | Decode Throughput: 516.1 tok/s
⏱️ [Progress] Elapsed: 120.2s | Finished: 101/256 (39.5%) | Active: 58 running, 97 waiting | Generated: 61356 tokens | Decode Throughput: 510.5 tok/s
⏱️ [Progress] Elapsed: 150.2s | Finished: 131/256 (51.2%) | Active: 54 running, 71 waiting | Generated: 76584 tokens | Decode Throughput: 509.7 tok/s
⏱️ [Progress] Elapsed: 180.3s | Finished: 154/256 (60.2%) | Active: 52 running, 50 waiting | Generated: 91316 tokens | Decode Throughput: 506.5 tok/s
⏱️ [Progress] Elapsed: 210.4s | Finished: 183/256 (71.5%) | Active: 58 running, 15 waiting | Generated: 106614 tokens | Decode Throughput: 506.8 tok/s
⏱️ [Progress] Elapsed: 240.4s | Finished: 213/256 (83.2%) | Active: 43 running, 0 waiting | Generated: 122285 tokens | Decode Throughput: 508.7 tok/s
⏱️ [Progress] Elapsed: 270.4s | Finished: 250/256 (97.7%) | Active: 6 running, 0 waiting | Generated: 132538 tokens | Decode Throughput: 490.1 tok/s
Total: 133966tok, Time: 291.78s, Throughput: 459.13tok/s
PS D:\code\metaprogramming\KernelAgent> python micro-vllm\bench_green.py
=============================================================
🚀 Running Dedicated SM Resource Isolation (Green Contexts) Benchmark
=============================================================

⏱️ Running Baseline Configuration (Green Contexts OFF)...

🟢 Running Target Configuration (Green Contexts ON)...

======================================================================
📊 BENCHMARK COMPARISON REPORT: BASELINE VS GREEN CONTEXTS
======================================================================
Model: Qwen2.5-3B-Instruct (Eager cuTile Backend)
Workload: Concurrent 2048-token Prefill + 100-token Decode Client
----------------------------------------------------------------------
Metric                         | Baseline (Green OFF) | Target (Green ON)  | Delta
----------------------------------------------------------------------
TTFT (Prefill Latency)         |           255.15 ms |         250.59 ms |    -1.8%
Decode P50 ITL (Median)        |            46.55 ms |          59.17 ms |   +27.1%
Decode P99 ITL (Tail)          |            82.36 ms |          83.70 ms |    +1.6%
Total Throughput               |        417.47 tok/s |      349.28 tok/s |   -16.3%
----------------------------------------------------------------------
Total Tokens Processed: Baseline = 2183 tok, Green = 2183 tok
Total Elapsed Time:     Baseline = 5.23 s, Green = 6.25 s
======================================================================

💡 Key Insights:
  * No significant P99 decode tail latency reduction observed in this run.
PS D:\code\metaprogramming\KernelAgent> python micro-vllm\bench_green.py
=============================================================
🚀 Running Dedicated SM Resource Isolation (Green Contexts) Benchmark
=============================================================

⏱️ Running Baseline Configuration (Green Contexts OFF)...

🟢 Running Target Configuration (Green Contexts ON)...

======================================================================
📊 BENCHMARK COMPARISON REPORT: BASELINE VS GREEN CONTEXTS
======================================================================
Model: Qwen2.5-3B-Instruct (Eager cuTile Backend)
Workload: Concurrent 2048-token Prefill + 100-token Decode Client
----------------------------------------------------------------------
Metric                         | Baseline (Green OFF) | Target (Green ON)  | Delta
----------------------------------------------------------------------
TTFT (Prefill Latency)         |           256.34 ms |         249.44 ms |    -2.7%
Decode P50 ITL (Median)        |            55.04 ms |          46.84 ms |   -14.9%
Decode P99 ITL (Tail)          |            79.21 ms |          83.82 ms |    +5.8%
Total Throughput               |        365.20 tok/s |      413.53 tok/s |   +13.2%
----------------------------------------------------------------------
Total Tokens Processed: Baseline = 2183 tok, Green = 2183 tok
Total Elapsed Time:     Baseline = 5.98 s, Green = 5.28 s
======================================================================

💡 Key Insights:
  * No significant P99 decode tail latency reduction observed in this run.
PS D:\code\metaprogramming\KernelAgent> python micro-vllm\bench_green.py
=============================================================
🚀 Running Dedicated SM Resource Isolation (Green Contexts) Benchmark
=============================================================

⏱️ Running Baseline Configuration (Green Contexts OFF)...

🟢 Running Target Configuration (Green Contexts ON)...

======================================================================
📊 BENCHMARK COMPARISON REPORT: BASELINE VS GREEN CONTEXTS
======================================================================
Model: Qwen2.5-3B-Instruct (Eager cuTile Backend)
Workload: Concurrent 2048-token Prefill + 100-token Decode Client
----------------------------------------------------------------------
Metric                         | Baseline (Green OFF) | Target (Green ON)  | Delta
----------------------------------------------------------------------
TTFT (Prefill Latency)         |           250.80 ms |         249.88 ms |    -0.4%
Decode P50 ITL (Median)        |            59.07 ms |          54.94 ms |    -7.0%
Decode P99 ITL (Tail)          |            88.65 ms |          82.15 ms |    -7.3%
Total Throughput               |        346.78 tok/s |      369.57 tok/s |    +6.6%
----------------------------------------------------------------------
Total Tokens Processed: Baseline = 2183 tok, Green = 2183 tok
Total Elapsed Time:     Baseline = 6.30 s, Green = 5.91 s
======================================================================

💡 Key Insights:
  * Decode P99 tail latency was reduced by 7.3% under Green Contexts!
PS D:\code\metaprogramming\KernelAgent>
PS D:\code\metaprogramming\KernelAgent> python micro-vllm\bench_green.py
=============================================================
🚀 Running Dedicated SM Resource Isolation (Green Contexts) Benchmark
=============================================================

⏱️ Running Baseline Configuration (Green Contexts OFF)...

🟢 Running Target Configuration (Green Contexts ON)...

======================================================================
📊 BENCHMARK COMPARISON REPORT: BASELINE VS GREEN CONTEXTS
======================================================================
Model: Qwen2.5-3B-Instruct (Eager cuTile Backend)
Workload: Concurrent 2048-token Prefill + 100-token Decode Client
----------------------------------------------------------------------
Metric                         | Baseline (Green OFF) | Target (Green ON)  | Delta
----------------------------------------------------------------------
TTFT (Prefill Latency)         |           250.49 ms |         247.57 ms |    -1.2%
Decode P50 ITL (Median)        |            45.12 ms |          59.73 ms |   +32.4%
Decode P99 ITL (Tail)          |            98.03 ms |          94.03 ms |    -4.1%
Total Throughput               |        419.97 tok/s |      345.19 tok/s |   -17.8%
----------------------------------------------------------------------
Total Tokens Processed: Baseline = 2183 tok, Green = 2183 tok
Total Elapsed Time:     Baseline = 5.20 s, Green = 6.32 s
======================================================================

💡 Key Insights:
  * Decode P99 tail latency was reduced by 4.1% under Green Contexts!
PS D:\code\metaprogramming\KernelAgent> python micro-vllm\bench_green.py
=============================================================
🚀 Running Dedicated SM Resource Isolation (Green Contexts) Benchmark
=============================================================

⏱️ Running Baseline Configuration (Green Contexts OFF)...

🟢 Running Target Configuration (Green Contexts ON)...

======================================================================
📊 BENCHMARK COMPARISON REPORT: BASELINE VS GREEN CONTEXTS
======================================================================
Model: Qwen2.5-3B-Instruct (Eager cuTile Backend)
Workload: Concurrent 2048-token Prefill + 100-token Decode Client
----------------------------------------------------------------------
Metric                         | Baseline (Green OFF) | Target (Green ON)  | Delta
----------------------------------------------------------------------
TTFT (Prefill Latency)         |           246.97 ms |         251.09 ms |    +1.7%
Decode P50 ITL (Median)        |            57.69 ms |          43.41 ms |   -24.8%
Decode P99 ITL (Tail)          |            90.43 ms |          93.24 ms |    +3.1%
Total Throughput               |        350.59 tok/s |      445.39 tok/s |   +27.0%
----------------------------------------------------------------------
Total Tokens Processed: Baseline = 2183 tok, Green = 2183 tok
Total Elapsed Time:     Baseline = 6.23 s, Green = 4.90 s
======================================================================

💡 Key Insights:
  * No significant P99 decode tail latency reduction observed in this run.
  * Prefill TTFT increased by 1.7% as expected (since prefill SM resources were limited to 32 SMs instead of 48 SMs).
PS D:\code\metaprogramming\KernelAgent> python micro-vllm\bench_green.py
=============================================================
🚀 Running Dedicated SM Resource Isolation (Green Contexts) Benchmark
=============================================================

⏱️ Running Baseline Configuration (Green Contexts OFF)...

🟢 Running Target Configuration (Green Contexts ON)...

======================================================================
📊 BENCHMARK COMPARISON REPORT: BASELINE VS GREEN CONTEXTS
======================================================================
Model: Qwen2.5-3B-Instruct (Eager cuTile Backend)
Workload: Concurrent 2048-token Prefill + 100-token Decode Client
----------------------------------------------------------------------
Metric                         | Baseline (Green OFF) | Target (Green ON)  | Delta
----------------------------------------------------------------------
TTFT (Prefill Latency)         |           255.28 ms |         250.41 ms |    -1.9%
Decode P50 ITL (Median)        |            44.38 ms |          61.58 ms |   +38.7%
Decode P99 ITL (Tail)          |            91.08 ms |          84.95 ms |    -6.7%
Total Throughput               |        432.16 tok/s |      334.73 tok/s |   -22.5%
----------------------------------------------------------------------
Total Tokens Processed: Baseline = 2183 tok, Green = 2183 tok
Total Elapsed Time:     Baseline = 5.05 s, Green = 6.52 s
======================================================================

💡 Key Insights:
  * Decode P99 tail latency was reduced by 6.7% under Green Contexts!
PS D:\code\metaprogramming\KernelAgent> python micro-vllm\bench_green.py
=============================================================
🚀 Running Dedicated SM Resource Isolation (Green Contexts) Benchmark
=============================================================

⏱️ Running Baseline Configuration (Green Contexts OFF)...

🟢 Running Target Configuration (Green Contexts ON)...

======================================================================
📊 BENCHMARK COMPARISON REPORT: BASELINE VS GREEN CONTEXTS
======================================================================
Model: Qwen2.5-3B-Instruct (Eager cuTile Backend)
Workload: Concurrent 2048-token Prefill + 100-token Decode Client
----------------------------------------------------------------------
Metric                         | Baseline (Green OFF) | Target (Green ON)  | Delta
----------------------------------------------------------------------
TTFT (Prefill Latency)         |           250.16 ms |         252.87 ms |    +1.1%
Decode P50 ITL (Median)        |            46.17 ms |          43.25 ms |    -6.3%
Decode P99 ITL (Tail)          |            93.45 ms |          92.83 ms |    -0.7%
Total Throughput               |        416.98 tok/s |      440.23 tok/s |    +5.6%
----------------------------------------------------------------------
Total Tokens Processed: Baseline = 2183 tok, Green = 2183 tok
Total Elapsed Time:     Baseline = 5.24 s, Green = 4.96 s
======================================================================

💡 Key Insights:
  * Decode P99 tail latency was reduced by 0.7% under Green Contexts!
  * Prefill TTFT increased by 1.1% as expected (since prefill SM resources were limited to 32 SMs instead of 48 SMs).
PS D:\code\metaprogramming\KernelAgent> python micro-vllm\bench_green.py
=============================================================
🚀 Running Dedicated SM Resource Isolation (Green Contexts) Benchmark
=============================================================

⏱️ Running Baseline Configuration (Green Contexts OFF)...

🟢 Running Target Configuration (Green Contexts ON)...

======================================================================
📊 BENCHMARK COMPARISON REPORT: BASELINE VS GREEN CONTEXTS
======================================================================
Model: Qwen2.5-3B-Instruct (Eager cuTile Backend)
Workload: Concurrent 2048-token Prefill + 100-token Decode Client
----------------------------------------------------------------------
Metric                         | Baseline (Green OFF) | Target (Green ON)  | Delta
----------------------------------------------------------------------
TTFT (Prefill Latency)         |           249.15 ms |         250.56 ms |    +0.6%
Decode P50 ITL (Median)        |            44.03 ms |          49.70 ms |   +12.9%
Decode P99 ITL (Tail)          |           105.88 ms |          86.07 ms |   -18.7%
Total Throughput               |        427.42 tok/s |      397.42 tok/s |    -7.0%
----------------------------------------------------------------------
Total Tokens Processed: Baseline = 2183 tok, Green = 2183 tok
Total Elapsed Time:     Baseline = 5.11 s, Green = 5.49 s
======================================================================

💡 Key Insights:
  * Decode P99 tail latency was reduced by 18.7% under Green Contexts!
  * Prefill TTFT increased by 0.6% as expected (since prefill SM resources were limited to 32 SMs instead of 48 SMs).
PS D:\code\metaprogramming\KernelAgent>