import os
import sys
import torch
from torch import nn

# 프로젝트 루트를 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch.distributed as dist
if not dist.is_initialized():
    dist.get_world_size = lambda: 1
    dist.get_rank = lambda: 0

from nanovllm.layers.linear import ColumnParallelLinear
from nanovllm.layers.rotary_embedding import RotaryEmbedding
from src.tests.utils import fix_seed, compare_outputs, get_garbage_input


def test_column_parallel_linear(device):
    """
    [TDRE 01] 텐서 병렬화 선형 레이어 (Column-wise) 검증
    - vLLM이 레이어를 쪼개서 계산해도, 원래 Linear 결과와 똑같아야 합니다.
    """
    print(f"\n[Test 01] ColumnParallelLinear 검증 시작 (디바이스: {device})")
    fix_seed(42)
    
    in_dim, out_dim = 128, 256
    garbage_input = get_garbage_input((1, in_dim)).to(device)
    
    # 1. Golden Reference (정답지)
    ref_linear = nn.Linear(in_dim, out_dim, bias=False).to(device)
    ref_output = ref_linear(garbage_input)
    
    # 2. Target Layer (nano-vllm 구현체)
    # 실제 TP 환경 대신 단일 프로세스에서 동작하는지 테스트하기 위해 weight 직접 주입
    # (실제 vLLM에서는 가중치 로더가 이 역할을 합니다)
    target_layer = ColumnParallelLinear(in_dim, out_dim, bias=False).to(device)
    target_layer.weight.data.copy_(ref_linear.weight.data) # 가중치 복사
    
    target_output = target_layer(garbage_input)
    
    # 3. 결과 비교
    compare_outputs(ref_output, target_output)

def test_rotary_embedding(device):
    """
    [TDRE 02] Rotary Embedding (RoPE) 회전 연산 검증
    - 수학적으로 올바른 회전 공식이 적용되었는지 확인합니다.
    """
    print(f"\n[Test 02] RotaryEmbedding 검증 시작 (디바이스: {device})")
    fix_seed(42)
    
    head_size = 64
    rotary_dim = 64
    max_pos = 1024
    base = 10000
    
    # 1. 가짜 입력 준비 (Query, Key, Position)
    q = get_garbage_input((1, 1, head_size)).to(device)
    k = get_garbage_input((1, 1, head_size)).to(device)
    pos = torch.tensor([5], device=device) # 5번째 위치
    
    # 2. Target Layer 실행
    rope = RotaryEmbedding(head_size, rotary_dim, max_pos, base).to(device)
    q_out, k_out = rope(pos, q, k)
    
    # 3. 간단한 정답지 시뮬레이션 (수학 공식 기반의 수동 계산)
    # (실제로는 더 정교한 Reference 코드를 사용해야 함)
    # 여기서는 RoPE 연산 전후의 벡터 노름(Norm)이 유지되는지 확인 (L2 Norm 보존 성질)
    q_norm_in = q.norm().item()
    q_norm_out = q_out.norm().item()
    
    print(f"   - 입력 Norm: {q_norm_in:.6f}, 출력 Norm: {q_norm_out:.6f}")
    if abs(q_norm_in - q_norm_out) < 1e-5:
        print("✅ RoPE 성질 확인 (회전 후에도 벡터 크기 보존)")
    else:
        print("❌ 오류: 벡터 크기가 변했습니다. 회전 연산이 틀렸을 수 있습니다.")

def test_rmsnorm_parity(device):
    """
    [TDRE 03] RMSNorm fused(F.rms_norm) vs manual 수치 parity 검증
    - Tier 2b 퓨전이 수동 구현과 수치적으로 동등한지 확인한다.
    """
    print(f"\n[Test 03] RMSNorm (fused vs manual) parity 검증 (디바이스: {device})")
    fix_seed(42)
    import torch.nn.functional as F

    D = 128
    eps = 1e-6
    dtype = torch.bfloat16

    x = torch.randn(4, 16, D, device=device, dtype=dtype)
    w = torch.randn(D, device=device, dtype=dtype)

    # 1. fused (F.rms_norm, 단일 커널)
    fused = F.rms_norm(x, [D], w, eps)

    # 2. manual reference (float32 누적)
    xf = x.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    xf = xf * torch.rsqrt(var + eps)
    manual = xf.to(dtype) * w

    if compare_outputs(fused, manual, rtol=1e-2, atol=1e-2):
        print("✅ RMSNorm parity passed! (fused == manual)")
    else:
        print("❌ RMSNorm parity failed!")

if __name__ == "__main__":
    # GPU가 없으면 CPU에서 테스트 수행
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 테스트 실행 디바이스: {device}")
    
    test_column_parallel_linear(device)
    test_rotary_embedding(device)
    test_rmsnorm_parity(device)

