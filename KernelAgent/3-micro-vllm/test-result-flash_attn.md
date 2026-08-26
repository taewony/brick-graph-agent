(env) linux@localhost:~/code/metaprogramming/KernelAgent/micro-vllm$ python --version
Python 3.12.3

==================================================
▶ OS Version       :  6.18.33.2-microsoft-standard-WSL2
▶ PyTorch Version  : v2.12.1+cu130
▶ CUDA Available   : True (Compiled with CUDA 13.0)
▶ Triton Version   : v3.7.1
--------------------------------------------------
Flash Attention v2.8.3.post1 로드 성공
==================================================
linux@localhost:/mnt/d/code$ nvidia-smi
Wed Jul  1 17:25:04 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 610.43.02              KMD Version: 610.47        CUDA UMD Version: 13.3     |

(env) linux@localhost:~/code/metaprogramming/KernelAgent/micro-vllm$ python src/inspect_model.py /home/linux/huggingface/Qwen3-0.6B

🔍 [vLLM Metadata Inspector] 모델 분석 시작: /home/linux/huggingface/Qwen3-0.6B
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

(env) linux@localhost:~/code/metaprogramming/KernelAgent/micro-vllm$ python example.py
[transformers] `torch_dtype` is deprecated! Use `dtype` instead!

🚀 Starting generation...
Generating: 100%|███████████████████████████████████████| 4/4 [00:11<00:00,  2.98s/it, Prefill=82tok/s, Decode=57tok/s]

============================================================
Prompt: '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nintroduce yourself<|im_end|>\n<|im_start|>assistant\n'
------------------------------------------------------------
Completion: Hello! I'm Qwen, the latest model created by Alibaba Cloud. My primary function is to assist with a variety of tasks, answer questions, and provide information across multiple domains. I can help with tasks such as summarizing text, translating between languages, generating text, answering questions, and much more. I'm designed to be helpful, harmless, and honest in my interactions. How can I assist you today?<|im_end|>
Generated Tokens: 85

============================================================
Prompt: '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nlist all prime numbers within 100<|im_end|>\n<|im_start|>assistant\n'
------------------------------------------------------------
Completion: Sure, here is the list of all prime numbers within 100:

2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97<|im_end|>
Generated Tokens: 111

============================================================
Prompt: '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nWrite a short poem about coding and artificial intelligence.<|im_end|>\n<|im_start|>assistant\n'
------------------------------------------------------------
Completion: In the symphony of bits and bytes,
Where algorithms dance in the night,
Artificial intelligence sings,
A symphony of human might.

Through the code, the unseen hand,
Shapes the world anew and clear,
With every line, a thought is made,
A dream become a digital heir.

In the realm of logic and thought,
Where bytes and bytes do meet,
We craft the future's face,
With coding's power, we create, we beat.

So let the AI sing its song,
Of human ingenuity's flight,
For in the code, the future's born,
And AI's the voice in the night.<|im_end|>
Generated Tokens: 127

============================================================
Prompt: '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nExplain the difference between TCP and UDP in simple terms.<|im_end|>\n<|im_start|>assistant\n'
------------------------------------------------------------
Completion: Sure! TCP (Transmission Control Protocol) and UDP (User Datagram Protocol) are two of the most common types of protocols used in networking for transferring data between devices.

### TCP (Transmission Control Protocol):

- **Purpose**: TCP is used for reliable data transfer. It ensures that data is delivered correctly, in order, and without errors.
- **Process**: TCP provides a connection-oriented service. This means that before any data can be sent, a connection must be established. After the connection is set up, data can be sent back and forth. TCP also handles retransmission of lost or misplaced data to ensure all the data arrives correctly.
- **Example**: When you download a file from the internet, the file you get is usually sent via TCP. Websites, emails, and most other data transfers over the internet use TCP.
- **Use Case**: Suitable for applications where data integrity and reliability are crucial, such as web browsing, file transfers, and email.

### UDP (User Datagram Protocol):

- **Purpose**: UDP is used for faster data transfer. It does not guarantee delivery or order of data. If a packet is lost or arrives out of order, the receiver must handle the situation.
- **Process**: UDP is a connectionless protocol, meaning that before any data
Generated Tokens: 256

============================================================
Total time: 11.91s
Total tokens: 579tok
Average throughput: 48.62 tok/s
============================================================

(env) linux@localhost:~/code/metaprogramming/KernelAgent/micro-vllm$ python bench.py
⚡ Using default (FlashAttention) backend
[transformers] `torch_dtype` is deprecated! Use `dtype` instead!
Generating: 100%|████████████████████████████████████████| 1/1 [00:01<00:00,  1.12s/it, Prefill=9tok/s, Decode=87tok/s]
🚀 Starting benchmark generation loop...
⏱️ [Progress] Elapsed: 30.0s | Finished: 108/256 (42.2%) | Active: 71 running, 77 waiting | Generated: 66210 tokens | Decode Throughput: 2206.9 tok/s
⏱️ [Progress] Elapsed: 60.2s | Finished: 251/256 (98.0%) | Active: 5 running, 0 waiting | Generated: 133081 tokens | Decode Throughput: 2212.3 tok/s
Total: 133966tok, Time: 62.95s, Throughput: 2128.07tok/s