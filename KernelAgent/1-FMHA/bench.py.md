`cuTile` 기반의 FMHA(Fused Multi-Head Attention)와 PyTorch 네이티브(SDPA/FlashAttention) 간의 성능을 정밀하게 비교하고, 특히 **타일 크기(Tile Size) 튜닝** 및 **커널 융합(Fused Ops)의 이점**을 검증하기 위한 마이크로 벤치마크 스크립트(`bench_fmha.py`)를 제안합니다.

이 벤치마크는 앞서 우리가 논의했던 **Prefill(긴 시퀀스) 환경**과 **Decode(짧은 시퀀스, T=1) 환경**을 모두 테스트하여 아키텍처적 병목을 눈으로 확인할 수 있도록 설계되었습니다.

### 📊 `bench_fmha.py` (FMHA 마이크로 벤치마크 스크립트)

```python
import torch
import itertools
from cutile_kernel import cutile_fmha, torch_fmha

def do_bench(fn, *args, warmup=10, rep=50, **kwargs):
    """PyTorch CUDA 이벤트를 이용한 정밀 측정 함수"""
    # Warmup
    for _ in range(warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(rep):
        fn(*args, **kwargs)
    end_event.record()
    torch.cuda.synchronize()
    
    # 평균 실행 시간(ms) 반환
    return start_event.elapsed_time(end_event) / rep

def run_benchmark():
    print("=" * 80)
    print(" 🚀 FMHA Micro-Benchmark: PyTorch SDPA vs cuTile ")
    print("=" * 80)
    
    # 고정 파라미터 (RTX 5070 등 타겟 장비 기준)
    BATCH = 8
    HEADS = 16
    D_K = 64
    D_V = 64
    DTYPE = torch.float16
    DEVICE = 'cuda'

    # 테스트할 시퀀스 길이 (1: Decode, 256~4096: Prefill)
    seq_lens = [1, 64, 256, 1024, 2048]
    
    # 탐색할 타일 크기 (Tile_M, Tile_N) 
    # SM Starvation(점유율) 및 레지스터 압박(Spilling) 비교용
    tile_configs = [(64, 64), (128, 64), (128, 128)]

    print(f"{'Seq_Len':<10} | {'Backend / Tile Config':<25} | {'Latency (ms)':<15} | {'Speedup':<10}")
    print("-" * 65)

    for seq in seq_lens:
        # 데이터 준비
        Q = torch.randn(BATCH, HEADS, seq, D_K, dtype=DTYPE, device=DEVICE)
        K = torch.randn(BATCH, HEADS, seq, D_K, dtype=DTYPE, device=DEVICE)
        V = torch.randn(BATCH, HEADS, seq, D_V, dtype=DTYPE, device=DEVICE)

        # 1. Baseline: PyTorch SDPA (Fused 커널 자동 선택 - FlashAttention/xFormers)
        baseline_ms = do_bench(torch_fmha, Q, K, V, is_causal=True, enable_gqa=False)
        print(f"{seq:<10} | {'PyTorch Native (SDPA)':<25} | {baseline_ms:>10.4f} ms  | {'1.00x':>7}")

        # 2. cuTile FMHA: 타일 크기 튜닝
        for tile_m, tile_n in tile_configs:
            # 디코딩(T=1)의 경우 큰 타일을 쓰면 Out-of-bounds나 낭비가 심함
            try:
                cutile_ms = do_bench(cutile_fmha, Q, K, V, tile_m=tile_m, tile_n=tile_n, causal=True)
                speedup = baseline_ms / cutile_ms
                
                # 최고 성능일 경우 강조
                mark = "🔥" if speedup > 1.05 else ""
                
                print(f"{'':<10} | cuTile ({tile_m:3d} x {tile_n:3d})     | {cutile_ms:>10.4f} ms  | {speedup:>6.2f}x {mark}")
            except Exception as e:
                print(f"{'':<10} | cuTile ({tile_m:3d} x {tile_n:3d})     | {'Failed':>10}      | {'-':>7}")
                
        print("-" * 65)

if __name__ == "__main__":
    run_benchmark()

```

### 💡 벤치마크 결과 분석 가이드 (관전 포인트)

위 코드를 실행한 뒤 다음 세 가지 현상을 중점적으로 관찰하여 최적화 방향을 결정하십시오.

1. **디코딩 단계 (Seq_Len = 1)**
* **예상 결과:** `PyTorch Native`가 `cuTile`보다 압도적으로 빠를 것입니다 (Speedup < 1.0).
* **원인:** 데이터 연산량은 극도로 적은데, `cutile.launch`를 호출하면서 발생하는 **파이썬 바운더리 오버헤드**가 전체 지연 시간을 지배하기 때문입니다. 이는 커널 자체의 문제가 아니라 런처의 문제임을 벤치마크가 증명해 줄 것입니다.


2. **단기 프리필 단계 (Seq_Len = 64 ~ 256) & 타일 튜닝**
* **예상 결과:** `cuTile (64 x 64)`가 `cuTile (128 x 128)`보다 더 나은 성능(혹은 비슷함)을 보여줄 것입니다.
* **원인:** 시퀀스 길이가 짧을 때 `128x128` 타일을 사용하면 그리드 크기(스레드 블록 수)가 너무 작게 쪼개져 GPU의 SM이 굶는 현상(Starvation)이 발생합니다. `64x64`로 쪼개어 스레드 블록 수를 늘려주는 것이 유리합니다.


3. **장기 프리필 단계 (Seq_Len = 1024 이상) & 커널 융합 효과**
* **예상 결과:** `cuTile` (특히 128x64 또는 128x128)이 PyTorch SDPA를 앞서거나 근접하는 성능(🔥)을 낼 수 있습니다.
* **원인:** 시퀀스 길이가 길어지면 커널 런치 오버헤드는 숨겨지고, 메모리 병목이 지배하게 됩니다. 이때 `cuTile` 내부의 온라인 소프트맥스와 레지스터 퓨전(Fusion) 연산의 효율성이 PyTorch 백엔드(FlashAttention)의 효율성과 진검승부를 벌이게 됩니다.