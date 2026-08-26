PS D:\Capstone\metaprogramming\KernelAgent\3-nano-vllm> python .\bench_green.py
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
TTFT (Prefill Latency)         |           244.60 ms |         243.42 ms |    -0.5%
Decode P50 ITL (Median)        |            48.63 ms |          41.83 ms |   -14.0%
Decode P99 ITL (Tail)          |            85.31 ms |          84.34 ms |    -1.1%
Total Throughput               |        415.50 tok/s |      459.73 tok/s |   +10.6%
----------------------------------------------------------------------
Total Tokens Processed: Baseline = 2183 tok, Green = 2183 tok
Total Elapsed Time:     Baseline = 5.25 s, Green = 4.75 s
======================================================================

PS D:\Capstone\metaprogramming\KernelAgent\3-nano-vllm> python bench.py --use-cutile
🚀 Using cuTile attention backend
`torch_dtype` is deprecated! Use `dtype` instead!
Generating: 100%|███████████████████████████████████████| 1/1 [00:03<00:00,  3.15s/it, Prefill=23tok/s, Decode=23tok/s]
🚀 Starting benchmark generation loop...
⏱️ [Progress] Elapsed: 30.1s | Finished: 14/256 (5.5%) | Active: 62 running, 180 waiting | Generated: 16807 tokens | Decode Throughput: 558.8 tok/s
⏱️ [Progress] Elapsed: 60.1s | Finished: 42/256 (16.4%) | Active: 51 running, 163 waiting | Generated: 32666 tokens | Decode Throughput: 543.3 tok/s
⏱️ [Progress] Elapsed: 90.2s | Finished: 71/256 (27.7%) | Active: 51 running, 134 waiting | Generated: 47041 tokens | Decode Throughput: 521.7 tok/s
⏱️ [Progress] Elapsed: 120.2s | Finished: 104/256 (40.6%) | Active: 58 running, 94 waiting | Generated: 62401 tokens | Decode Throughput: 519.3 tok/s
⏱️ [Progress] Elapsed: 150.2s | Finished: 133/256 (52.0%) | Active: 52 running, 71 waiting | Generated: 77855 tokens | Decode Throughput: 518.3 tok/s
⏱️ [Progress] Elapsed: 180.3s | Finished: 155/256 (60.5%) | Active: 51 running, 50 waiting | Generated: 93013 tokens | Decode Throughput: 515.9 tok/s
⏱️ [Progress] Elapsed: 210.4s | Finished: 186/256 (72.7%) | Active: 56 running, 14 waiting | Generated: 108626 tokens | Decode Throughput: 516.3 tok/s
⏱️ [Progress] Elapsed: 240.4s | Finished: 218/256 (85.2%) | Active: 38 running, 0 waiting | Generated: 124440 tokens | Decode Throughput: 517.6 tok/s
⏱️ [Progress] Elapsed: 270.4s | Finished: 253/256 (98.8%) | Active: 3 running, 0 waiting | Generated: 133043 tokens | Decode Throughput: 492.0 tok/s
Total: 133966tok, Time: 284.54s, Throughput: 470.82tok/s

---

PS D:\Capstone\metaprogramming\KernelAgent\3-nano-vllm> python src/inspect_model.py C:\Users\실습실1/huggingface\Qwen2.5-3B-Instruct

🔍 [vLLM Metadata Inspector] 모델 분석 시작: C:\Users\실습실1/huggingface\Qwen2.5-3B-Instruct
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

---

Generating: 100%|█████████████████████████████████████| 4/4 [00:14<00:00,  3.57s/it, Prefill=1282tok/s, Decode=27tok/s]

============================================================
Prompt: '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nintroduce yourself<|im_end|>\n<|im_start|>assistant\n'
------------------------------------------------------------
Completion: Hello! I'm Qwen, an artificial intelligence model created by Alibaba Cloud. I'm designed to assist with a wide range of tasks, from generating text and answering questions to helping with language translation, summarizing information, and more. I'm constantly learning and improving, so my responses are based on the latest data and algorithms available to me. How can I assist you today?<|im_end|>
Generated Tokens: 77

============================================================
Prompt: '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nlist all prime numbers within 100<|im_end|>\n<|im_start|>assistant\n'
------------------------------------------------------------
Completion: Certainly! Here is the list of all prime numbers within 100:

2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97

These are all the prime numbers less than or equal to 100.<|im_end|>
Generated Tokens: 128

============================================================
Prompt: '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nWrite a short poem about coding and artificial intelligence.<|im_end|>\n<|im_start|>assistant\n'
------------------------------------------------------------
Completion: In the digital dawn, under neon light,
Coders weave the threads of AI's might.
Bits and bytes dance in the silicon sky,
A symphony of logic so divine.

From zeros and ones, a mind takes shape,
A spark of thought, in circuits awake.
Neurons hum, connections form,
In the vast ocean of data, it's found.

Artificial, yet with spirit so true,
Empowering humans, in tasks anew.
In the realm where code and thought collide,
A future blossoms, with promise on high.<|im_end|>
Generated Tokens: 110

============================================================
Prompt: '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nExplain the difference between TCP and UDP in simple terms.<|im_end|>\n<|im_start|>assistant\n'
------------------------------------------------------------
Completion: Sure! Let's break down the differences between TCP (Transmission Control Protocol) and UDP (User Datagram Protocol) in simple terms:

### TCP (Transmission Control Protocol):
- **Purpose**: TCP is designed for applications that require reliable, connection-oriented communication. It ensures that data is delivered correctly and in order.
- **Features**:
  - **Connection-Oriented**: Before any data can be sent, a connection must be established between the sender and receiver.
  - **Error Checking and Recovery**: TCP checks the integrity of data during transmission and can request retransmission if errors are detected.
  - **Flow Control**: It manages the flow of data to prevent overloading the receiver.
  - **Acknowledgments**: The receiver sends acknowledgments back to the sender to confirm that the data has been received correctly.
- **Use Cases**: Web browsing, email, file transfers (like FTP), and most other internet services that require reliability.

### UDP (User Datagram Protocol):
- **Purpose**: UDP is simpler and faster than TCP. It's ideal for applications where low latency and bandwidth are more important than reliability.
- **Features**:
  - **Connectionless**: UDP does not establish a connection before sending data. It simply sends packets as fast as possible.
  - **
Generated Tokens: 256

---

PS D:\Capstone\metaprogramming\KernelAgent\3-nano-vllm> python .\test_asynchronous.py
`torch_dtype` is deprecated! Use `dtype` instead!
📥 User 1 request added.
Step 1: Active User IDs=[4], Tokens processed=12
Step 2: Active User IDs=[4], Tokens processed=-1
Step 3: Active User IDs=[4], Tokens processed=-1
Step 4: Active User IDs=[4], Tokens processed=-1

📥 User 2 request added dynamically while User 1 is decoding!
Step 5: Active User IDs=[4, 5], Tokens processed=8
Step 6: Active User IDs=[4, 5], Tokens processed=-2
Step 7: Active User IDs=[4, 5], Tokens processed=-2
Step 8: Active User IDs=[4, 5], Tokens processed=-2
Step 9: Active User IDs=[4, 5], Tokens processed=-2
Step 10: Active User IDs=[4, 5], Tokens processed=-2
Step 11: Active User IDs=[4, 5], Tokens processed=-2
Step 12: Active User IDs=[4, 5], Tokens processed=-2
Step 13: Active User IDs=[4, 5], Tokens processed=-2
Step 14: Active User IDs=[4, 5], Tokens processed=-2
Step 15: Active User IDs=[4, 5], Tokens processed=-2
Step 16: Active User IDs=[4, 5], Tokens processed=-2
Step 17: Active User IDs=[4, 5], Tokens processed=-2
Step 18: Active User IDs=[4, 5], Tokens processed=-2
Step 19: Active User IDs=[4, 5], Tokens processed=-2
Step 20: Active User IDs=[4, 5], Tokens processed=-2
Step 21: Active User IDs=[4, 5], Tokens processed=-2
Step 22: Active User IDs=[4, 5], Tokens processed=-2
Step 23: Active User IDs=[4, 5], Tokens processed=-2
Step 24: Active User IDs=[4, 5], Tokens processed=-2
Step 25: Active User IDs=[4, 5], Tokens processed=-2
Step 26: Active User IDs=[4, 5], Tokens processed=-2
Step 27: Active User IDs=[4, 5], Tokens processed=-2
Step 28: Active User IDs=[4, 5], Tokens processed=-2
Step 29: Active User IDs=[4, 5], Tokens processed=-2
Step 30: Active User IDs=[4, 5], Tokens processed=-2
Step 31: Active User IDs=[4, 5], Tokens processed=-2
Step 32: Active User IDs=[4, 5], Tokens processed=-2
Step 33: Active User IDs=[4, 5], Tokens processed=-2
Step 34: Active User IDs=[4, 5], Tokens processed=-2
Step 35: Active User IDs=[4, 5], Tokens processed=-2
Step 36: Active User IDs=[4, 5], Tokens processed=-2
Step 37: Active User IDs=[4, 5], Tokens processed=-2
Step 38: Active User IDs=[4, 5], Tokens processed=-2
Step 39: Active User IDs=[4, 5], Tokens processed=-2
Step 40: Active User IDs=[4, 5], Tokens processed=-2
Step 41: Active User IDs=[4, 5], Tokens processed=-2
Step 42: Active User IDs=[4, 5], Tokens processed=-2
Step 43: Active User IDs=[4, 5], Tokens processed=-2
Step 44: Active User IDs=[4, 5], Tokens processed=-2
Step 45: Active User IDs=[4, 5], Tokens processed=-2
Step 46: Active User IDs=[4, 5], Tokens processed=-2
Step 47: Active User IDs=[4, 5], Tokens processed=-2
Step 48: Active User IDs=[4, 5], Tokens processed=-2
Step 49: Active User IDs=[4, 5], Tokens processed=-2
Step 50: Active User IDs=[4, 5], Tokens processed=-2
Step 51: Active User IDs=[4, 5], Tokens processed=-2
Step 52: Active User IDs=[4, 5], Tokens processed=-2
Step 53: Active User IDs=[4, 5], Tokens processed=-2
Step 54: Active User IDs=[4, 5], Tokens processed=-2
Step 55: Active User IDs=[4, 5], Tokens processed=-2
Step 56: Active User IDs=[4, 5], Tokens processed=-2
Step 57: Active User IDs=[4, 5], Tokens processed=-2
Step 58: Active User IDs=[4, 5], Tokens processed=-2
Step 59: Active User IDs=[4, 5], Tokens processed=-2
Step 60: Active User IDs=[4, 5], Tokens processed=-2
Step 61: Active User IDs=[4, 5], Tokens processed=-2
Step 62: Active User IDs=[4, 5], Tokens processed=-2
Step 63: Active User IDs=[4, 5], Tokens processed=-2
Step 64: Active User IDs=[4, 5], Tokens processed=-2
Step 65: Active User IDs=[4, 5], Tokens processed=-2
Step 66: Active User IDs=[4, 5], Tokens processed=-2
Step 67: Active User IDs=[4, 5], Tokens processed=-2
Step 68: Active User IDs=[4, 5], Tokens processed=-2
Step 69: Active User IDs=[4, 5], Tokens processed=-2
Step 70: Active User IDs=[4, 5], Tokens processed=-2
Step 71: Active User IDs=[4, 5], Tokens processed=-2
Step 72: Active User IDs=[4, 5], Tokens processed=-2
Step 73: Active User IDs=[4, 5], Tokens processed=-2
Step 74: Active User IDs=[4, 5], Tokens processed=-2
Step 75: Active User IDs=[4, 5], Tokens processed=-2
Step 76: Active User IDs=[4, 5], Tokens processed=-2
Step 77: Active User IDs=[4, 5], Tokens processed=-2
Step 78: Active User IDs=[4, 5], Tokens processed=-2
Step 79: Active User IDs=[4, 5], Tokens processed=-2
Step 80: Active User IDs=[4, 5], Tokens processed=-2
Step 81: Active User IDs=[4, 5], Tokens processed=-2
Step 82: Active User IDs=[4, 5], Tokens processed=-2
Step 83: Active User IDs=[4, 5], Tokens processed=-2
Step 84: Active User IDs=[4, 5], Tokens processed=-2
Step 85: Active User IDs=[4, 5], Tokens processed=-2
Step 86: Active User IDs=[4, 5], Tokens processed=-2
Step 87: Active User IDs=[4, 5], Tokens processed=-2
Step 88: Active User IDs=[4, 5], Tokens processed=-2
Step 89: Active User IDs=[4, 5], Tokens processed=-2
Step 90: Active User IDs=[4, 5], Tokens processed=-2
Step 91: Active User IDs=[4, 5], Tokens processed=-2
Step 92: Active User IDs=[4, 5], Tokens processed=-2
Step 93: Active User IDs=[4, 5], Tokens processed=-2
Step 94: Active User IDs=[4, 5], Tokens processed=-2
Step 95: Active User IDs=[4, 5], Tokens processed=-2
Step 96: Active User IDs=[4, 5], Tokens processed=-2
Step 97: Active User IDs=[4, 5], Tokens processed=-2
Step 98: Active User IDs=[4, 5], Tokens processed=-2
Step 99: Active User IDs=[4, 5], Tokens processed=-2
Step 100: Active User IDs=[4, 5], Tokens processed=-2
Step 101: Active User IDs=[5], Tokens processed=-2
Step 102: Active User IDs=[5], Tokens processed=-1
Step 103: Active User IDs=[5], Tokens processed=-1
Step 104: Active User IDs=[], Tokens processed=-1

============================================================
User 4 Completion: The theory of relativity, proposed by Albert Einstein, describes the relationship between space and time, and the fact that the laws of physics are the same for all non-accelerating observers, and that the speed of light in a vacuum is constant, independent of the motion of the light source or observer. It comprises two parts: special relativity, which applies to all physical phenomena in the absence of gravity, and general relativity, which describes gravity as a geometric property of space and time.
------------------------------------------------------------
User 5 Completion: The capital of France is Paris. Paris is a vibrant city located in the northern part of the country and is known for its rich history, culture, and beauty. It is home to iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum, which houses the world-famous Mona Lisa painting. Paris is also famous for its cuisine, fashion, and art scene, making it a popular destination for tourists from all over the world. The city is the political
------------------------------------------------------------