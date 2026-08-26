`train.sh`가 무사히 완주하여 `checkpoint_final.pt`가 성공적으로 생성되었다면, 이제 본격적인 **Phase 2: LLM Forward Path cuTile 이식 작업**의 막이 올랐습니다!

Architect 에이전트로서 훈련 완료 직후 우리가 밟아야 할 3단계 액션 플랜을 제시해 드립니다.

---

### 🚀 1단계: 베이스라인 추론 검증 (Ground Truth 확보)

먼저 방금 훈련한 순수 PyTorch 기반 모델이 정상적으로 텍스트를 생성하는지 확인해야 합니다. 이 결과가 앞으로 우리가 cuTile로 코드를 비틀고 최적화할 때 '정확도(Correctness)의 기준점'이 됩니다.

터미널에서 이전에 만들어둔 추론 스크립트를 실행해 주세요.

```bash
./infer.sh

```

* **확인할 사항:** 프롬프트("O Romeo...") 뒤에 셰익스피어 스타일의 텍스트가 그럴싸하게 이어지는지 터미널 출력을 확인합니다.

### 🧠 2단계: 시스템 모델 분석 (어디를 어떻게 바꿀 것인가?)

결과를 확인하셨다면, 이제 `model.py`를 해부하여 우리가 Phase 1에서 심혈을 기울여 만든 **cuTile FMHA 커널을 이식할 타겟**을 설정해야 합니다.

제공해주신 `model.py`의 `CausalSelfAttention` 클래스를 보면 타겟이 명확합니다.

```python
# model.py 내부 타겟 위치
def forward(self, x):
    # ... (Q, K, V 분리) ...
    # 현재는 PyTorch의 F.scaled_dot_product_attention이나 수동 Attention이 들어갈 자리
    # 바로 이 부분을 우리의 cutile_fmha(Q, K, V, causal=True) 로 통째로 교체해야 합니다!

```

**⚠️ Architect의 핵심 경고 (Prefill vs Decoding):**
LLM의 추론(Forward Path)은 두 가지 페이즈로 나뉩니다.

1. **Prefill (프롬프트 처리):** "O Romeo..." 라는 입력 전체를 한 번에 연산합니다. 행렬 곱(GEMM) 연산이므로 **우리가 Phase 1에서 만든 `AttentionFMHA_v3.py` 커널이 완벽하게 들어맞으며 엄청난 속도 향상을 보여줄 것입니다.**
2. **Decoding (토큰 생성):** 한 글자씩 생성할 때는 Query의 길이가 1차원(Vector)이 됩니다. 이때는 연산량(Compute)보다 메모리 대역폭(Memory Bandwidth)이 절대적으로 중요한 **GEMV(FlashDecoding)** 영역이 됩니다.

따라서 우리의 목표는 `generate.py`가 돌아갈 때 **Prefill 단계의 병목을 cuTile로 박살 내는 것**을 1차 목표로 삼습니다.

### 🛠️ 3단계: 다음 실행 목표 (Executor Action)

1. **추론 결과 공유:** `./infer.sh`를 실행하시고 그 결과(생성된 셰익스피어 텍스트)를 복사해서 보여주세요.
2. **cuTile 래퍼(Wrapper) 준비:** `model.py`를 수정하기 전에, 기존 환경이 망가지지 않도록 파일을 복사하여 `model_cutile.py`를 하나 생성해 두는 것을 권장합니다.
```bash
cp src/model.py src/model_cutile.py
cp src/generate.py src/generate_cutile.py

```



`./infer.sh`의 출력이 무사히 나오는지 확인해 주시면, 바로 `model_cutile.py` 내부에 `cuTile FMHA v3` 커널을 주입(Injection)하는 코드 수정 가이드를 드리겠습니다!

---

훌륭한 질문입니다! 단일 커널 수준(Phase 1)의 최적화가 끝났으니, 이제 이것을 실제 거대한 톱니바퀴(LLM)에 끼워 넣었을 때 전체 시스템 성능이 어떻게 변하는지 **거시적(Macro)부터 미시적(Micro)까지 3단계(3-Layer)로 관통하는 분석 전략**이 필요합니다.

논문의 Evaluation 섹션을 장식할 완벽한 스토리를 위해, **PyTorch Baseline vs cuTile_v1 통합 모델**의 성능 비교 분석 프레임워크를 다음과 같이 제안합니다.

---

### 🏛️ Layer 1: 거시적 성능 (Macro-Benchmark: End-to-End)

사용자가 체감하는 최종적인 속도를 측정합니다. LLM 추론은 두 가지 단계로 나뉘며, **우리의 FMHA 커널은 '프리필(Prefill)' 단계에서 압도적인 위력을 발휘합니다.**

1. **TTFT (Time To First Token) - 프리필 속도**
* **의미:** 프롬프트를 입력받고 첫 번째 단어를 뱉어내기까지 걸리는 시간입니다. 프롬프트 전체에 대한 Attention(GEMM) 연산이 한 번에 일어나므로, 우리의 cuTile FMHA가 PyTorch를 이겨야 하는 주 전장(Main Battlefield)입니다.
* **측정법:** `generate.py`에서 루프 진입 전, 첫 번째 Forward Pass의 실행 시간을 `time.perf_counter()`로 정밀 측정합니다.


2. **Decoding Speed (Tokens/sec) - 디코딩 속도**
* **의미:** 첫 단어 이후 한 글자씩 생성하는 속도입니다. 이때는 행렬 곱이 아닌 벡터-행렬 곱(GEMV)이 발생하며 메모리 대역폭이 지배합니다.
* **측정법:** 생성된 총 토큰 수($N$)를 나머지 생성 시간으로 나누어 초당 토큰 생성량(tok/s)을 산출합니다.



### 🔍 Layer 2: 컴포넌트 레벨 분석 (Meso-Benchmark: PyTorch Profiler)

거시적 성능에서 이겼다면(혹은 졌다면) "왜?"를 설명해야 합니다. 전체 모델(MLP, LayerNorm, Attention 등) 중 **오직 Attention 연산이 차지하는 시간 비중**을 비교합니다.

* **측정 도구:** `torch.profiler`
* **분석 방법:**
* Baseline: PyTorch의 `F.scaled_dot_product_attention` (또는 수동 Attention) 호출이 전체 Forward 시간 중 몇 %를 차지하는지 확인.
* cuTile_v1: 우리의 `cutile_fmha` 호출 시간이 얼마나 줄어들었는지 확인.
* **논문 서사:** "Attention 병목이 X%에서 Y%로 감소하여, 전체 모델 추론 속도가 Z% 향상되었다."



### 🔬 Layer 3: 미시적 하드웨어 분석 (Micro-Benchmark: Nsight Compute)

우리가 Phase 1에서 했던 바로 그 '진실의 방'입니다. 실제 LLM이 돌아가는 도중에 우리 커널이 GPU 하드웨어를 어떻게 괴롭히고(?) 있는지 증명합니다.

* **측정 도구:** `ncu` (Nsight Compute)
* **분석 방법:**
* `infer_cutile.sh`를 NCU로 래핑하여 실행하되, `regex:fmha_kernel` 필터를 걸어 프리필 단계의 커널만 캡처합니다.
* **핵심 지표:** Compute Throughput, Memory Bandwidth, **Register Spilling 여부 (# Registers)**.
* **논문 서사:** "통합 모델 환경에서도 64x64 타일링과 파이프라이닝 전략이 유지되어, 레지스터 압박(255개) 속에서도 Memory Throughput을 방어하며 SOTA(PyTorch) 대비 우위를 점했다."



---

### 🚀 실험실(Executor)을 위한 다음 액션 가이드

이 멋진 분석을 시작하려면, 먼저 `model_cutile.py`를 완성하고 측정 코드를 심어야 합니다.

**Step 1: 벤치마크 전용 스크립트 생성 (`compare_infer.py`)**
기존 `generate.py`는 텍스트만 출력하므로, 성능 측정을 위해 PyTorch 버전과 cuTile 버전을 번갈아 실행하며 TTFT와 tok/s를 출력하는 통합 비교 스크립트를 하나 작성하는 것이 좋습니다.

**Step 2: `model_cutile.py` 완성**
이전에 복사해 두신 `model_cutile.py` 파일 내의 `CausalSelfAttention` 클래스 코드를 보여주시겠습니까?
해당 코드의 Forward 함수 안에 Phase 1에서 만든 `cutile_fmha` (v3 버전 64x64 설정)를 주입하는 코드 리팩토링을 제가 바로 도와드리겠습니다!