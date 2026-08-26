# 강의 02: nano-vllm 프로젝트 전체 구조

> '4계층 아키텍처'라는 한 장의 그림으로 저장소를 간략히 파악하기: `LLM` 진입점부터 `Scheduler`/`BlockManager`, 그리고 `ModelRunner`와 `layers`까지, 마지막으로 vLLM과 비교하며 축소 범위와 면접 멘트를 명확히 합니다.

## 본 강의 목표

1. nano-vllm의 **디렉터리 구조**와 각 최상위 디렉터리의 역할을 **숙지**합니다.
2. **인터페이스 계층 → 엔진 계층 → 스케줄링 계층 → 실행 계층**의 호출 관계를 그릴 수 있게 됩니다.
3. **사용자 입력부터 토큰 출력까지**의 **데이터 흐름**을 한 문단으로 설명할 수 있습니다(면접관과 화이트보드에도 동일하게 전달 가능).
4. **nano-vllm과 vLLM**의 **코드량, 기능 범위**의 차이를 추상적인 표현 없이 답변할 수 있습니다.

## 핵심 개념

### 전체 디렉터리 구조 (저장소와 일치)

```
nanovllm/
├── __init__.py          # LLM, SamplingParams 내보내기
├── config.py            # Config dataclass
├── llm.py               # LLM(LLMEngine) 얇은 래퍼
├── sampling_params.py   # SamplingParams dataclass
├── engine/
│   ├── llm_engine.py    # 엔진 메인 루프
│   ├── scheduler.py     # Prefill/Decode 스케줄링
│   ├── sequence.py      # 시퀀스 상태 관리
│   ├── block_manager.py # KV Cache 블록 관리
│   └── model_runner.py  # GPU 모델 실행
├── layers/
│   ├── attention.py     # FlashAttention + Triton KV 저장
│   ├── linear.py        # 열/행/QKV 병렬 Linear
│   ├── embed_head.py    # 어휘 병렬 Embedding + LMHead
│   ├── layernorm.py     # RMSNorm
│   ├── activation.py    # SiluAndMul (SwiGLU)
│   ├── rotary_embedding.py
│   └── sampler.py       # temperature 샘플링
├── models/
│   └── qwen3.py         # Qwen3 모델 구현
└── utils/
    ├── context.py       # 전역 추론 컨텍스트
    └── loader.py        # safetensors 가중치 로딩
```

### 4계층 아키텍처 분석

**제1계층: 인터페이스 계층 (`llm.py` + `sampling_params.py` + `config.py`)**

- 사용자가 접촉하는 것은 오직 **`LLM`**과 **`SamplingParams`** 뿐입니다.
- **`Config`**는 **GPU 메모리, 배치, TP, KV 블록** 등 전역 제약을 집약하며, 엔진 초기화 시 한 번에 효력을 발생합니다.

**제2계층: 엔진 계층 (`engine/llm_engine.py`)**

- **메인 루프**: `Scheduler`에서 실행 가능한 배치를 가져와 **`ModelRunner`**를 호출하고, 다시 **샘플링**, **시퀀스 상태 업데이트**를 수행합니다.
- **비즈니스 로직의 심장**: 스케줄링과 모델 실행을 연결합니다.

**제3계층: 스케줄링 계층 (`scheduler.py` + `sequence.py` + `block_manager.py`)**

- **`Scheduler`**: 이번 라운드가 **prefill**인지 **decode**인지 결정하고, **batch**를 구성하며, `max_num_batched_tokens` 등의 제약을 받습니다.
- **`Sequence`**: 각 요청의 **토큰, 단계 수, 종료 여부**를 추적합니다.
- **`BlockManager`**: KV를 위한 **물리 블록**을 할당(Paged 개념)하며, `Attention` 내의 `block_table`과 호응합니다.

**제4계층: 실행 계층 (`engine/model_runner.py` + `models/qwen3.py` + `layers/*`)**

- **`ModelRunner`**: **입력 텐서 준비**, **전역 `context`** 설정(`is_prefill`, `cu_seqlens_q` 등), **`Qwen3`** 호출.
- **`layers`**: 연산자 레벨 구현; **`attention.py`**가 **KV 쓰기**와 **FlashAttention 순전파**를 동시에 담당합니다.

### 각 모듈 역할 요약표

| 모듈 | 한 줄 역할 |
|------|------------|
| `config.py` | 하이퍼파라미터와 HF 설정 정렬, max length 및 batch 제약 |
| `llm.py` | 대외 `generate` 제공, 내부적으로 `LLMEngine`으로 변환 |
| `llm_engine.py` | 추론 루프: 스케줄링 → 순전파 → 샘플링 → 업데이트 |
| `scheduler.py` | 시퀀스 선택, batch 구성, prefill/decode 전략 |
| `sequence.py` | 단일 요청 상태 머신 |
| `block_manager.py` | KV 블록 할당 및 회수 |
| `model_runner.py` | CUDA 측 실행 및 context 주입 |
| `attention.py` | Triton KV 저장 + FlashAttention 두 경로 |
| `embed_head.py` | 어휘 병렬 임베딩 및 LMHead |
| `qwen3.py` | 전체 Transformer 스택 |

## 소스 코드 분석 (전체 소스 코드 및 줄별 주석 포함)

아래는 **최소한의 개략 코드**로 네 계층이 어떻게 연결되는지 보여줍니다(실제 구현은 저장소 기준이며, 여기서는 **호출 방향**을 강조합니다):

```python
# 개념 연결: 인터페이스 -> 엔진 -> 스케줄링 -> 실행 (의사 코드)

class LLM:
    def __init__(self, model_path, **kwargs):
        self.engine = LLMEngine(Config(model=model_path, **kwargs))

    def generate(self, prompts, sampling_params):
        return self.engine.generate(prompts, sampling_params)


class LLMEngine:
    def generate(self, prompts, sampling_params):
        # 1) tokenize (엔진 외부 혹은 내부에서, 예제는 주로 외부)
        # 2) scheduler가 관리하는 시퀀스에 추가
        while not all_finished():
            batch = self.scheduler.schedule()   # 스케줄링 계층: 누가 prefill / decode를 실행할지
            logits = self.model_runner.run(batch)  # 실행 계층: 순전파
            self.sample_and_update(logits, sampling_params)
        return decoded_texts
```

줄별 설명:

1. **`LLM`**: **`LLMEngine`**을 생성하고, **`Config`**를 주입합니다.
2. **`generate`**: 엔진이 **모든 시퀀스가 끝날 때까지** 루프를 담당합니다.
3. **`scheduler.schedule()`**: **이번 라운드 실행 가능한 서브셋**과 **단계 표시**(prefill/decode)를 반환합니다.
4. **`model_runner.run`**: **`get_context()`**에 필요한 필드를 설정하고, **`Qwen3`**를 실행합니다.

## 그림 설명 (텍스트/ASCII 설명)

**데이터 흐름 (사용자 입력 → 출력)**:

```
사용자 문자열
    |
    v
Tokenizer (HF AutoTokenizer, 보통 비즈니스 스크립트 내)
    |
    v
token id 시퀀스  -->  LLM.generate
    |
    v
LLMEngine: Sequence 등록, KV 블록 할당 (BlockManager)
    |
    v
Scheduler: 이번 라운드 prefill 또는 decode의 배치 구성
    |
    v
ModelRunner: context 채우기 (is_prefill, cu_seqlens_q, block_tables, ...)
    |
    v
Qwen3 forward -> Attention (KV 쓰기 + flash_attn_*)
    |
    v
ParallelLMHead -> logits -> Sampler (temperature)
    |
    v
다음 토큰을 Sequence에 기록; 끝나지 않았다면 다음 decode 라운드로 진입
    |
    v
출력 토큰 이어붙이기 -> 문자열
```

**4계층 접이식 그림**:

```
 [ LLM.generate ]          <-- 인터페이스 계층
       |
 [ LLMEngine 루프 ]        <-- 엔진 계층
       |
 [ Scheduler + Seq + BM ]  <-- 스케줄링 계층
       |
 [ ModelRunner + Qwen3 ]   <-- 실행 계층
```

## vLLM과의 비교 (코드량, 기능 범위)

| 차원 | nano-vllm | vLLM |
|------|-----------|------|
| **코드량** | ~1,200줄 수준, 단일 저장소 가독 | 수만 줄 + 다중 모듈 |
| **핵심 개념** | Paged KV, 연속 배치, FlashAttention, TP | 동급 + 더 많은 프로덕션 특성 |
| **기능 범위** | Qwen3 경로 + 교습 목적 최소 폐루프 | 다중 모델, 양자화, API, 분산 등 |
| **학습 가치** | **설명 가능한 소스 코드 맵**을 빠르게 구축 | 실무서비스에 대응되나, 읽기 부담 높음 |

면접 표현 제안: **nano-vllm은 vLLM을 대체하기 위한 것이 아니라, vLLM의 핵심 알고리즘과 엔지니어링 경로를 가독 좋은 범위로 압축한 것입니다**.

## 면접 출제 포인트

- **4계층 아키텍처**를 1분 안에 그릴 수 있는가.
- **`Scheduler`와 `ModelRunner`의 역할 분담**: 누가 '누구를 실행할지'를 결정하고, 누가 '어떻게 텐서로 실행할지'를 결정하는가.
- **`BlockManager`와 `attention` 내 `block_table`의 관계**.
- **vLLM과 비교할 때 '열등함'이 아닌 '축소/교육적 목적'임을 강조**할 수 있는가.

## 자주 나오는 면접 질문

1. **기능을 하나 추가한다면, 어떤 파일을 수정하겠습니까?**  
   답: 먼저 분류합니다 — 스케줄링 전략 변경은 `scheduler`; 새로운 연산자는 `layers`; 새 모델 구조는 `models`; 진입 파라미터 변경은 `config`/`llm`.

2. **KV는 어디에 저장되며, 누가 할당합니까?**  
   답: 실제 물리 텐서는 **`Attention`의 `k_cache`/`v_cache`** (또는 runner 초기화 시 바인딩)에 있습니다. **논리 블록 매핑**은 **`BlockManager`**에 있으며, **`block_table`**을 통해 FlashAttention에 전달됩니다.

3. **연속 배치 처리는 어떤 계층에서 이루어집니까?**  
   답: 주로 **`Scheduler` + `LLMEngine` 루프**에서: 동적으로 시퀀스를 추가/삭제하고, 매 단계 배치 구성이 달라질 수 있습니다.

## 요약

nano-vllm은 **명확한 4계층 구분**을 통해 추론 엔진을 해체합니다: **인터페이스**는 극히 단순하게, **엔진**은 루프를 관리, **스케줄링**은 배치와 KV를 관리, **실행**은 GPU 텐서와 연산자를 담당합니다. **vLLM**과 대비하면, **작은 코드량으로 핵심을 모두 커버**하는 학습 표본입니다.

**학습 후 과제 (필기 권장)**:

1. `LLMEngine`의 한 번의 루프 안에서 호출하는 **핵심 함수명 3~5개**를 그려 보세요(비워 두고 18강에서 채워도 좋습니다).
2. **왜 `BlockManager`가 layers가 아닌 스케줄링 계층에 속하는지** 한 문장으로 설명해 보세요(힌트: 자원 할당과 생애주기를 관리하고, 직접 행렬 곱셈을 하지 않음).

## 다음 강의 예고

다음 강의 **《03-강의03-설정과 진입점》** 에서는 **`Config` / `SamplingParams`**를 필드별로 분석하고, **`example.py`**와 결합하여 로컬 모델 경로부터 **`llm.generate`**까지의 첫 번째 '실행 가능한 인지'를 완성할 것입니다.