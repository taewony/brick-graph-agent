import torch
import random
import numpy as np

def fix_seed(seed=42):
    """
    모든 랜덤 시드를 고정하여 '똑같은 가비지(Same Garbage)' 데이터가 생성되게 합니다.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compare_outputs(output_ref, output_target, rtol=1e-3, atol=1e-5):
    """
    두 결과값(텐서)을 비교하고, 오차 범위를 넘어가면 상세 정보를 출력합니다.
    """
    if not torch.allclose(output_ref, output_target, rtol=rtol, atol=atol):
        diff = (output_ref - output_target).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        print(f"❌ 수치적 불일치 발견!")
        print(f"   - 최대 오차: {max_diff:.8f}")
        print(f"   - 평균 오차: {mean_diff:.8f}")
        return False
    else:
        print("✅ 수치적 일치! (Golden Reference와 동일)")
        return True

def get_garbage_input(shape, dtype=torch.float32):
    """
    고정된 형태의 랜덤 텐서를 생성합니다.
    """
    return torch.randn(shape, dtype=dtype)
