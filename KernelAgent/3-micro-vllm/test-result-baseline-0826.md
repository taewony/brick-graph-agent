PS D:\code\brick-graph-agent\KernelAgent\3-micro-vllm> python src/inspect_model.py $env:USERPROFILE\huggingface\Qwen3-0.6B

🔍 [vLLM Metadata Inspector] 모델 분석 시작: C:\Users\실습실1\huggingface\Qwen3-0.6B
------------------------------------------------------------
🏗️  Architecture: Qwen2ForCausalLM
📏 Hidden Size: 2048
🧠 Layers: 36
🧩 Attention Heads: 16
💾 KV Heads (GQA): 2 (Ratio: 8:1)
📐 Head Dimension: 128
📜 Max Position: 32768
📖 Vocab Size: 151936
------------------------------------------------------------
🚀 [vLLM Engine Simulation]
🔹 Token당 KV Cache 크기: 36.00 KB
🔹 한 블록(16 tokens) 메모리: 0.56 MB
✅ Tensor Parallel (2 GPUs) 가능
   - GPU당 Attention Heads: 8
   - GPU당 KV Heads: 1
------------------------------------------------------------
✅ 분석 완료. vLLM은 이 정보들을 바탕으로 'BlockManager'를 설정합니다.
PS D:\code\brick-graph-agent\KernelAgent\3-micro-vllm> $env:NANO_VLLM_USE_CUTILE="1"
PS D:\code\brick-graph-agent\KernelAgent\3-micro-vllm> $env:NANO_VLLM_CUTILE_PREFILL_STRATEGY="hybrid"
PS D:\code\brick-graph-agent\KernelAgent\3-micro-vllm> python bench.py --use-cutile --cutile-prefill-strategy hybrid
🚀 Using cuTile attention backend
cuTile prefill strategy: hybrid
cuTile CUDA Graph decode: disabled by default
`torch_dtype` is deprecated! Use `dtype` instead!
Generating: 100%|████████████████████████████████████████| 1/1 [00:04<00:00,  5.00s/it, Prefill=3tok/s, Decode=25tok/s]
🚀 Starting benchmark generation loop...
⏱️ [Progress] Elapsed: 30.0s | Finished: 4/256 (1.6%) | Active: 70 running, 182 waiting | Generated: 10458 tokens | Decode Throughput: 348.5 tok/s
⏱️ [Progress] Elapsed: 60.1s | Finished: 24/256 (9.4%) | Active: 54 running, 178 waiting | Generated: 23603 tokens | Decode Throughput: 392.8 tok/s
⏱️ [Progress] Elapsed: 90.2s | Finished: 49/256 (19.1%) | Active: 51 running, 156 waiting | Generated: 35314 tokens | Decode Throughput: 391.7 tok/s
⏱️ [Progress] Elapsed: 120.2s | Finished: 70/256 (27.3%) | Active: 52 running, 134 waiting | Generated: 46531 tokens | Decode Throughput: 387.0 tok/s
⏱️ [Progress] Elapsed: 151.0s | Finished: 94/256 (36.7%) | Active: 59 running, 103 waiting | Generated: 57922 tokens | Decode Throughput: 383.7 tok/s
⏱️ [Progress] Elapsed: 181.1s | Finished: 115/256 (44.9%) | Active: 52 running, 89 waiting | Generated: 69431 tokens | Decode Throughput: 383.5 tok/s
⏱️ [Progress] Elapsed: 211.1s | Finished: 138/256 (53.9%) | Active: 53 running, 65 waiting | Generated: 81195 tokens | Decode Throughput: 384.7 tok/s
⏱️ [Progress] Elapsed: 241.2s | Finished: 155/256 (60.5%) | Active: 51 running, 50 waiting | Generated: 92911 tokens | Decode Throughput: 385.3 tok/s
⏱️ [Progress] Elapsed: 271.3s | Finished: 181/256 (70.7%) | Active: 58 running, 17 waiting | Generated: 104908 tokens | Decode Throughput: 386.7 tok/s
⏱️ [Progress] Elapsed: 301.3s | Finished: 204/256 (79.7%) | Active: 52 running, 0 waiting | Generated: 117363 tokens | Decode Throughput: 389.5 tok/s
⏱️ [Progress] Elapsed: 331.3s | Finished: 229/256 (89.5%) | Active: 27 running, 0 waiting | Generated: 128533 tokens | Decode Throughput: 388.0 tok/s
⏱️ [Progress] Elapsed: 361.4s | Finished: 254/256 (99.2%) | Active: 2 running, 0 waiting | Generated: 133257 tokens | Decode Throughput: 368.8 tok/s
Total: 133966tok, Time: 372.44s, Throughput: 359.70tok/s
PS D:\code\brick-graph-agent\KernelAgent\3-micro-vllm>