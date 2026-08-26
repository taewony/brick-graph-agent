import os
import argparse
from transformers import AutoConfig

def inspect_vllm_metadata(model_path):
    """
    vLLM이 모델을 로드할 때 가장 먼저 확인하는 
    핵심 메타데이터와 메모리 요구사항을 분석합니다.
    """
    if not os.path.exists(model_path):
        print(f"❌ Error: {model_path} 경로를 찾을 수 없습니다.")
        return

    print(f"\n🔍 [vLLM Metadata Inspector] 모델 분석 시작: {model_path}")
    print("-" * 60)

    # 1. HuggingFace Config 로드
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    
    # 2. 핵심 아키텍처 정보 추출
    arch = getattr(config, "architectures", ["Unknown"])[0]
    hidden_size = getattr(config, "hidden_size", 0)
    num_heads = getattr(config, "num_attention_heads", 0)
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads) # GQA 확인
    num_layers = getattr(config, "num_hidden_layers", 0)
    head_dim = hidden_size // num_heads
    max_pos = getattr(config, "max_position_embeddings", 0)
    vocab_size = getattr(config, "vocab_size", 0)
    
    print(f"🏗️  Architecture: {arch}")
    print(f"📏 Hidden Size: {hidden_size}")
    print(f"🧠 Layers: {num_layers}")
    print(f"🧩 Attention Heads: {num_heads}")
    print(f"💾 KV Heads (GQA): {num_kv_heads} (Ratio: {num_heads // num_kv_heads}:1)")
    print(f"📐 Head Dimension: {head_dim}")
    print(f"📜 Max Position: {max_pos}")
    print(f"📖 Vocab Size: {vocab_size}")

    print("-" * 60)
    
    # 3. vLLM 추론 엔진의 시뮬레이션 (GPU 1개 기준)
    print("🚀 [vLLM Engine Simulation]")
    
    # KV Cache 한 블록(예: 16개 단어) 당 필요한 메모리 계산 (FP16 기준)
    block_size = 16 
    # Key(2 bytes) + Value(2 bytes) * Layer * KV_Heads * Head_Dim * Block_Size
    bytes_per_token = 2 * 2 * num_layers * num_kv_heads * head_dim
    memory_per_block_mb = (bytes_per_token * block_size) / (1024 * 1024)
    
    print(f"🔹 Token당 KV Cache 크기: {bytes_per_token / 1024:.2f} KB")
    print(f"🔹 한 블록({block_size} tokens) 메모리: {memory_per_block_mb:.2f} MB")
    
    # 텐서 병렬화(TP) 시뮬레이션
    tp_size = 2
    if num_heads % tp_size == 0:
        print(f"✅ Tensor Parallel ({tp_size} GPUs) 가능")
        print(f"   - GPU당 Attention Heads: {num_heads // tp_size}")
        print(f"   - GPU당 KV Heads: {num_kv_heads // tp_size}")
    else:
        print(f"⚠️  Tensor Parallel ({tp_size} GPUs) 주의: Head 개수가 GPU 개수로 나누어지지 않습니다.")

    print("-" * 60)
    print("✅ 분석 완료. vLLM은 이 정보들을 바탕으로 'BlockManager'를 설정합니다.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str, help="HuggingFace 모델 경로 (로컬 또는 ID)")
    args = parser.parse_args()
    
    inspect_vllm_metadata(args.model_path)
