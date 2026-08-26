## `AttentionFMHA.py` 코드 분석 및 문서화

`AttentionFMHA.py`의 코드를 **성능 측정 방식**과 **FMHA 커널 구현 설계** 관점에서 해부하여, 마치 **커널 최적화 설계서**처럼 정리

## 📘 cuTile FMHA 커널 설계 및 성능 측정 분석서

### 1. 개요
`AttentionFMHA.py`는 **cuTile DSL을 사용해 구현한 Flash Multi-Head Attention(FMHA)** 커널입니다.  
NVIDIA GPU에서 **온라인 소프트맥스(online softmax)** 와 **타일 기반 메모리 최적화**를 적용하여, 표준 attention 대비 메모리 대역폭 사용을 획기적으로 줄이는 동시에, **임의의 시퀀스 길이**와 **멀티 헤드**를 지원합니다.  
이 파일은 **성능 측정 루틴**과 **커널 정의**를 모두 포함하고 있어, 단일 스크립트로 기능 검증 및 속도 비교가 가능합니다.

---

### 2. 성능 측정 방법

#### 2.1 측정 대상
- **cuTile FMHA 커널** (`fused_mha_kernel`)
- **PyTorch 네이티브 구현** (`torch.nn.functional.scaled_dot_product_attention`)

두 구현의 **실행 시간**을 비교합니다.

#### 2.2 측정 파라미터
- 배치 크기: 2
- 헤드 수: 12
- 시퀀스 길이: 2048 (또는 512, 1024 등으로 변경 가능)
- 헤드 차원: 64
- 데이터 타입: `float16`

#### 2.3 타이밍 방식
- `torch.cuda.synchronize()`로 GPU 작업 완료를 보장한 후, `time.perf_counter()`를 사용해 **월 타임(wall time)**을 측정합니다.
- **웜업(warm-up)** 단계를 10회 수행하여 GPU 클록 부스트, 커널 캐싱 등의 영향을 배제합니다.
- 이후 **벤치마크 반복 횟수** (`num_repeats = 100`) 동안 각 커널의 평균 실행 시간을 계산합니다.
- cuTile 커널은 `ct.launch(...)` 호출 자체의 **런치 오버헤드**를 포함한 시간을 측정합니다.  
  (실제 서비스 환경과 동일한 조건을 재현하기 위함)

#### 2.4 출력 및 검증
- `torch.allclose()`로 두 출력 텐서의 값이 허용 오차 내에서 일치하는지 확인합니다 (수치 정확도 검증).
- 평균 실행 시간을 ms 단위로 출력하여 직관적인 비교를 제공합니다.

```python
# 코드상 측정 루틴 예시
for _ in range(warmup):
    ct.launch(kernel, ...)
torch.cuda.synchronize()
t1 = time.perf_counter()
for _ in range(num_repeats):
    ct.launch(kernel, ...)
torch.cuda.synchronize()
t2 = time.perf_counter()
avg_time = (t2 - t1) / num_repeats * 1000  # ms
```

---

### 3. FMHA 커널 설계 – 최적화 관점

#### 3.1 전체 구조 (Block Diagram)
```
입력: Q, K, V [batch, heads, seq_len, head_dim]
처리: 타일링된 온라인 어텐션 + 소프트맥스
출력: O [batch, heads, seq_len, head_dim]
```

커널은 **2중 루프**로 구성됩니다.
1. **Query 타일 루프 (Tile-M)** : Query의 시퀀스 차원을 `BLOCK_M` 단위로 분할
2. **Key/Value 타일 루프 (Tile-N)** : 각 Query 타일에 대해 Key와 Value의 시퀀스 차원을 `BLOCK_N` 단위로 순회

이러한 이중 타일링은 **FlashAttention** 알고리즘의 핵심으로, 전체 중간 attention 행렬을 global memory에 쓰지 않고 **온칩 SRAM에서 누적**합니다.

#### 3.2 타일 크기 선택
- `BLOCK_M = 64` (Query 방향 타일)
- `BLOCK_N = 64` (Key 방향 타일)

이 값은 RTX 4060 등 **레지스터 제한이 엄격한 GPU**에 최적화되어 있습니다.  
128×128로 늘릴 경우 레지스터 사용량이 255를 초과하여 **로컬 메모리 스필**이 발생, 성능이 급감합니다.

#### 3.3 메모리 계층 활용
| 데이터 | 저장 위치 | 비고 |
|--------|------------|------|
| Q 타일 (`q_tile`) | 공유 메모리 (SRAM) | `BLOCK_M × head_dim` 크기, 블록 내 모든 스레드가 공유 |
| K 타일 (`k_tile`) | 공유 메모리 (SRAM) | `BLOCK_N × head_dim` 크기, 내부 루프에서 로드 |
| V 타일 (`v_tile`) | 공유 메모리 (SRAM) | K 타일과 동일한 타일링으로 로드 |
| Softmax 통계 (m, l) | 레지스터 | 온라인 소프트맥스용 running max, sum |
| 출력 누적 (`acc`) | 레지스터 | 각 스레드가 자신의 Query head_dim 부분을 레지스터에 유지 |

- **Global Memory → Shared Memory** 전송은 `ct.copy` (벡터화 로드)를 사용하여 대역폭을 극대화합니다.
- **소프트맥스** 연산에 필요한 `max`, `sum`은 **블록 내 모든 스레드가 협력하여 리덕션**하지 않고, 각 스레드가 자신의 레지스터에 독립적인 값을 유지합니다.  
  (head_dim 차원을 분할하지 않고 스레드 하나가 head_dim 전체를 처리하므로, 블록 내 리덕션이 불필요)

#### 3.4 온라인 소프트맥스 (Online Softmax)
FlashAttention의 핵심인 **온라인 소프트맥스**가 구현되어 있습니다.  
각 Key 타일을 순회할 때마다:

1. `S = Q_tile @ K_tile^T` → `[BLOCK_M, BLOCK_N]`
2. `m_new = max(m_old, row_max(S))`
3. `P = exp(S - m_new)`
4. `l_new = exp(m_old - m_new) * l_old + row_sum(P)`
5. `acc_new = exp(m_old - m_new) * acc_old + P @ V_tile`

이 과정을 거치면서 이전 통계(`m_old`, `l_old`, `acc_old`)를 새로운 통계로 갱신합니다.  
이 때, `m` (로컬 최대값)과 `l` (합계)는 **각 Query 타일의 각 행마다** 유지됩니다.

cuTile DSL에서는 이를 **각 스레드가 자신의 행을 처리**하는 방식으로 자연스럽게 표현할 수 있습니다.  
스레드는 `head_dim` 전체를 레지스터에 보유하므로, 내부적으로 `m`과 `l`도 스레드 로컬 변수로 존재합니다.

#### 3.5 Causal Masking (인과 마스크)
- `causal_mask` 파라미터가 `True`일 때, 각 Query 행의 타일 루프에서 **현재 Key 위치가 Query 위치보다 큰 경우 마스킹**합니다.
- 마스킹은 `-inf`를 할당하는 방식으로 구현됩니다.
- cuTile DSL에서는 `ct.select` 등을 사용해 조건부로 값을 변경합니다.

#### 3.6 블록/스레드 매핑
- 블록 그리드는 `(batch * num_heads, ceil_div(seq_len, BLOCK_M))`의 2차원입니다.
  - X 방향: Query 타일 인덱스
  - Y 방향: 배치 내 헤드 인덱스
- 각 블록은 `(BLOCK_M, head_dim)` 크기의 작업을 `BLOCK_M × head_dim` 개의 스레드로 처리할 수도 있지만,  
  이 구현에서는 **스레드 하나가 head_dim 전체를 처리**하도록 하여 스레드 간 동기화를 최소화했습니다.  
  따라서 블록당 스레드 수 = `BLOCK_M` (각 스레드가 Query의 한 행을 담당)

이 구조는 **스레드당 레지스터 압박을 줄이고** 블록 내 리덕션 오버헤드를 없애는 대신, **스레드 수가 적어 SM 점유율이 낮아질 위험**이 있습니다.  
이를 보완하기 위해 블록을 여러 개 띄워 (헤드 수, Query 타일 수) 전체 SM을 채우도록 합니다.

#### 3.7 데이터 정밀도 및 누적기
- 입력/출력: `float16`
- 어텐션 점수 및 소프트맥스 연산: **float32** (정확도 유지)
- 누적기 `acc`는 float32로 유지하며, 최종 출력 시에만 float16으로 캐스팅

#### 3.8 수치적 안정성 (Scaling)
- `softmax_scale`은 `1.0 / sqrt(head_dim)` 으로 미리 계산되어 전달됩니다.  
  `S = Q @ K^T * softmax_scale`을 통해 스케일링하여 어텐션 점수의 분산을 조정합니다.

---

### 4. 최적화 설계 결정 요약

| 최적화 항목 | 선택 | 근거 |
|-------------|------|------|
| 타일 크기 | `64×64` | 레지스터 사용량 < 255, 공유 메모리 용량 내 여유 |
| 스레드 매핑 | 스레드 1개당 Query 1행 처리 | 리덕션 없이 온라인 소프트맥스 구현 단순화, 레지스터 재사용 |
| 온라인 소프트맥스 | 적용 | 중간 행렬 전체를 global memory에 저장하지 않아 대역폭 절감 |
| 공유 메모리 | K/V 타일 저장, Q 타일 상주 | 반복되는 K/V 로드에 재사용, Q 타일은 블록 진입 시 한 번만 로드 |
| 데이터 타입 | 입력 FP16, 내부 FP32 | 정확도와 속도 균형 |
| Causal mask | 조건 분기로 처리 | 인과성 지원, 불필요한 연산 방지 |
| 벡터화 로드/스토어 | `ct.copy` 사용 | 최대 메모리 대역폭 활용 (128-bit 트랜잭션) |

---

### 5. 성능 측정 결과 해석 포인트

- **PyTorch SDPA**는 내부적으로 cuDNN이나 FlashAttention 최적화 커널을 호출하므로, 단순한 cuTile 구현보다 더 최적화되어 있을 가능성이 큽니다.  
  그럼에도 불구하고, 이 샘플은 **커스텀 최적화의 출발점**으로 충분히 경쟁력 있는 성능을 보입니다.
- 만약 성능이 낮다면, 위 설계에서 확인된 **SM 점유율 부족**이나 **Python 런치 오버헤드**가 원인일 수 있습니다.  
  이를 해결하기 위해 앞서 제안된 **CUDA Graph** 및 **커널 퓨전**을 적용하면 실제 서비스 수준의 성능을 낼 수 있습니다.

---

### 6. 결론
`AttentionFMHA.py`는 cuTile을 이용한 FlashAttention의 **교과서적인 구현체**입니다.  
타일링, 온라인 소프트맥스, 메모리 계층 최적화 등 **GPU 커널 엔지니어링의 핵심 원리**가 잘 녹아 있습니다.  
이 코드를 기반으로 **KV Cache**, **GEMV 디코딩**, **Green Contexts** 등의 고급 기법을 단계적으로 통합하면 **LLM-from-scratch 엔진**을 완성할 수 있습니다.

---
## AttentionFMHA.py

### 1. 긍정적인 설계 (Architect's Praise)

* **완벽한 Online Softmax 구현:** 거대한 $N \times N$ Attention 행렬을 Global Memory(DRAM)에 쓰는 것을 방지하기 위해 `m_i` (Max value)와 `l_i` (Sum value)를 활용한 Online Softmax가 수학적으로 매우 정확하게 구현되어 있습니다.
* **Tensor Core(MMA) 명시적 타겟팅:** `ct.mma(q, k, qk)`와 `ct.mma(p, v, acc)`를 사용하여 덧셈/곱셈을 하드웨어 가속기(Tensor Core)에 직접 매핑하려는 의도가 잘 드러납니다.
* **Autotuner 내장 (`cutile_autotune_fmha`):** 이것이 이 코드의 백미입니다. `TILE_M`과 `TILE_N` (예: 128x128, 64x64 등)의 최적값은 하드웨어 아키텍처(RTX 4060 vs RTX 5070)에 따라 완전히 달라집니다. 하드 코딩하지 않고 `ct_experimental.autotune_launch`를 통해 탐색 공간(Search Space)을 열어둔 것은 실험실(Laboratory) 환경에 가장 이상적인 세팅입니다.

### 2. 최적화 병목 가설 (Architect's Hypothesis)

코드는 훌륭하지만, 실제 GPU(RTX 4060)에 올렸을 때 다음과 같은 부분에서 병목(Bottleneck)이 발생할 것으로 예상됩니다. 이 부분들이 우리가 `ncu`로 데이터를 뽑아내어 증명하고 개선해야 할 타겟입니다.

**가설 A: Bank Conflicts (메모리 뱅크 충돌)**

* **문제점:** $K$ 행렬을 로드할 때 `order=(0, 1, 3, 2)`를 사용하여 메모리상에서 실시간 전치(Transpose)를 수행하고 있습니다. 이는 Global Memory 읽기 효율을 떨어뜨리거나, Shared Memory에 기록할 때 심각한 Bank Conflict를 유발할 수 있습니다.
* **ncu 검증 지표:** 프로파일링 결과에서 `Shared Memory Bank Conflicts` 카운트와 `L1/TEX Cache Throughput`을 확인해야 합니다.

**가설 B: Software Pipelining (비동기 데이터 로드) 부족**

* **문제점:** 코드에 `latency=2`, `latency=4` 힌트가 있지만, 명시적인 더블 버퍼링(Double Buffering) 구조가 보이지 않습니다. 즉, 현재 타일의 $K, V$를 연산하는 동안 다음 타일의 $K, V$를 미리 HBM에서 당겨오는(Prefetch) 파이프라인이 최적화되지 않았을 가능성이 큽니다.
* **ncu 검증 지표:** `Warp State Statistics`에서 `Stall Long Scoreboard` (메모리 로딩 대기 시간) 비율이 얼마나 높은지 확인해야 합니다.

---

### 🚀 다음 단계 (Action Plan for Executor)

우리의 폐루프(Closed Loop) 실험을 본격적으로 가동하기 위해, Executor(실험실) 환경에서 다음 3단계를 수행해 주시기 바랍니다.

1. **정확도 및 성능 베이스라인 측정 (시간 측정)**
먼저 `ncu` 없이 순수 파이썬으로 스크립트를 실행하여 PyTorch FlashAttention(cuDNN) 대비 어느 정도의 속도가 나오는지, 그리고 Autotuner가 선택한 최적의 `TILE_M`, `TILE_N` 조합이 무엇인지 터미널 로그를 확인합니다.
```bash
python3 AttentionFMHA.py --correctness-check

```

2. **ncu 프로파일링 추출 (Macro ➔ Micro)**
시간 측정이 끝났다면, Autotuning 과정은 빼고 특정 타일 사이즈(예: 128x128)를 고정한 상태에서 `fmha_kernel`만 정확히 캡처합니다. (앞서 만든 `run_matmul.sh`와 동일한 방식으로 `run_fmha.sh`를 만들어 우회 실행하는 것을 잊지 마세요.)
```bash
ncu --set full -k regex:fmha_kernel -o fmha_baseline_rtx4060 ./run_fmha.sh

```

3. **결과 피드백**
생성된 `.ncu-rep` 파일을 로컬 PC로 가져와 Nsight Compute GUI로 열어보신 후, **SOL(Speed of Light)의 Compute와 Memory Throughput 수치**, 그리고 Autotuner가 선택한 **Tuned Config 결과**를 저에게 알려주세요.
실측 데이터가 도착하는 대로, 가설을 검증하고 코드를 어떻게 수정(예: Swizzling 적용, 타일 사이즈 미세 조정)해야 할지 시스템 모델을 업데이트하겠습니다!