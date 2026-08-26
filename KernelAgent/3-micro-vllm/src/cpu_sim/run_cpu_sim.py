import os
import sys

# 프로젝트 루트를 경로에 추가 (nanovllm 패키지 로드를 위해)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from src.cpu_sim.mock_model_runner import MockModelRunner

def run_simulation():
    print("\n🚀 [vLLM CPU Simulation Mode] 가동 시작!")
    print("-" * 60)

    # 1. 설정 (가짜 모델 설정)
    config = Config(
        model="fake-model-path",
        max_num_seqs=4,
        max_num_batched_tokens=1024,
        num_kvcache_blocks=100, # 가상 GPU 메모리 블록 개수
        kvcache_block_size=16   # 블록당 16개 단어 저장
    )
    # config의 __post_init__에서 파일 경로 체크를 우회하기 위해 hf_config 직접 설정
    config.hf_config = type('obj', (object,), {'max_position_embeddings': 4096})

    # 2. 부품 준비 (진짜 두뇌 + 가짜 근육)
    scheduler = Scheduler(config)
    model_runner = MockModelRunner(vocab_size=32000)

    # 3. 사용자 요청 추가 (2개의 질문)
    prompts = [
        [1, 2, 3, 4, 5, 6, 7, 8], # 질문 1 (8개 토큰)
        [10, 11, 12, 13, 14]      # 질문 2 (5개 토큰)
    ]
    sp = SamplingParams(max_tokens=32)
    
    for p in prompts:
        seq = Sequence(p, sp)
        scheduler.add(seq)
        print(f"📥 요청 추가: ID={seq.seq_id}, Prompt Length={len(p)}")

    # 4. 추론 루프 (Simulation)
    step_count = 0
    while not scheduler.is_finished():
        step_count += 1
        print(f"\n--- [Step {step_count}] ---")

        # A. 스케줄링 (진짜 로직!)
        seqs, is_prefill = scheduler.schedule()
        mode = "Prefill" if is_prefill else "Decode"
        
        # B. 가짜 모델 연산
        token_ids = model_runner.call("run", seqs, is_prefill)
        
        # C. 후처리 및 로그 출력
        scheduler.postprocess(seqs, token_ids)
        
        for seq in seqs:
            print(f"[{mode}] Seq {seq.seq_id}: "
                  f"Total Tokens={len(seq)}, "
                  f"Blocks={len(seq.block_table)}, "
                  f"Status={seq.status.name}")
        
        print(f"📦 Free Blocks: {len(scheduler.block_manager.free_block_ids)}")

    print("-" * 60)
    print("✅ 시뮬레이션 완료! 모든 요청이 성공적으로 처리되었습니다.")

if __name__ == "__main__":
    run_simulation()
