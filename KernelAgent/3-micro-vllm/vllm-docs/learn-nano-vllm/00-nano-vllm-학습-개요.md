# 가이드: nano-vllm 프로젝트 개요

> 약 1,200줄의 파이썬 코드로 산업용 LLM 추론 엔진의 핵심을 파악하기: 연속 배치 처리, PagedAttention부터 FlashAttention과 TP까지, 면접 대비를 위한 원스톱 학습 지도.

## 본 강의 목표

이 가이드를 모두 읽으면 다음을 할 수 있어야 합니다:

1. **nano-vllm의 포지셔닝**(누가 만들었는지, 코드 줄 수, GitHub에서의 인기와 의의)을 명확히 설명할 수 있다.
2. **20개의 튜토리얼 전체 맥락**과 권장 학습 순서를 이해하고, 스스로 학습 계획을 세울 수 있다.
3. 면접에서 한 문장으로 **nano-vllm과 vLLM의 관계**를 명확히 설명할 수 있다: 어떤 사상을 계승했고, 어떤 엔지니어링 디테일을 생략했는지.
4. **본 튜토리얼의 면접 가치**를 명확히 이해한다: 어떤 지식 포인트를 평가할 수 있고, 어떻게 이력서/프로젝트 경험과 결합시킬지.

## 핵심 개념

### nano-vllm이란?

**nano-vllm**은 **DeepSeek 엔지니어**가 오픈소스로 공개한 초경량 LLM 추론 엔진 구현체입니다. 의도적으로 '가독성'에 초점을 맞추고 있습니다:

- **코드 규모**: 약 **1,200줄**의 파이썬 (주석과 공백을 제외해도 비슷한 수준)으로, 설정, 스케줄링, KV 관리부터 모델 순전파까지 전체 파이프라인을 다룹니다.
- **커뮤니티 인기**: GitHub에서 약 **12.6k개 이상의 스타**(시간에 따라 증가, '널리 인정받은 교육용 프로젝트'라는 인상을 주기에 충분한 수치)를 받았으며, 이는 **vLLM 사상을 실행 가능하게 축소한 버전**으로서 수많은 개발자에게 학습용으로 사용되고 있음을 보여줍니다.

vLLM을 대체하여 프로덕션에 사용하기 위한 것이 아니라, **유지보수 가능한 코드량으로 추론 시스템의 핵심을 명확히 설명**하기 위한 것이며, 특히 다음에 적합합니다:

- **Serving / Inference**를 처음 시스템적으로 학습하는 개발자
- **LLM 추론 엔지니어, 인프라 엔지니어** 면접을 준비하며 '소스 코드 수준의 디테일을 설명할 수 있는' 지원자

### 왜 배울 가치가 있는가 (nanoGPT와의 비교)

| 프로젝트 | 대응하는 '완전판' | 교육적 중점 |
|------|----------------|------------|
| nanoGPT | 대규모 언어 모델 학습 | 학습 루프, Transformer 블록 |
| **nano-vllm** | **vLLM 등 추론 엔진** | **배치 처리, KV, 스케줄링, 저지연 실행** |

이미 `forward`를 작성할 줄 안다면, 다음 단계는 **배치가 어떻게 구성되는지, KV를 어떻게 배치하는지, 언제 prefill하고 언제 decode하는지**를 이해하는 것입니다—nano-vllm이 바로 이 흐름을 명쾌하게 보여줍니다.

## 소스 코드 분석 (전체 소스 코드 및 줄별 주석 포함)

가이드 단계에서는 특정 모듈에 깊이 들어가지 않고, **패키지 진입점**만 보여주어 '디렉터리 트리'와 '외부 API'를 연결시킬 수 있도록 돕습니다 (실제 프로젝트는 로컬의 `nanovllm/__init__.py` 기준입니다).

```python
# nanovllm/__init__.py (from nanovllm import LLM, SamplingParams가 가능하게 하는 일반적인 export)
from nanovllm.llm import LLM
from nanovllm.sampling_params import SamplingParams

__all__ = ["LLM", "SamplingParams"]
```

줄별 설명:

1. **`LLM`**: 사용자 측에서 신경 써야 할 유일한 클래스, 내부적으로 `LLMEngine`을 소유하며 `generate`를 캡슐화합니다.
2. **`SamplingParams`**: temperature, `max_tokens`, EOS 무시 여부 등 샘플링 하이퍼파라미터, 엔진 스케줄링과 분리되어 있습니다.
3. **`__all__`**: `from nanovllm import *` 시 심볼을 제한하여, '외부 API 최소화' 설계 스타일을 보여줍니다 — 교육용 프로젝트에서도 명확한 경계를 유지합니다.

제공된(혹은 저장소의) **소스 코드 디렉터리 구조**와 대조하여, 파일명과 역할 간의 매핑을 정립하세요:

```
nanovllm/
├── __init__.py          # LLM, SamplingParams 내보내기
├── config.py            # Config dataclass
├── llm.py               # LLM(LLMEngine) 빈 하위 클래스 혹은 얇은 래퍼
├── sampling_params.py   # SamplingParams dataclass
├── engine/
│   ├── llm_engine.py    # 엔진 메인 루프
│   ├── scheduler.py     # Prefill/Decode 스케줄링
│   ├── sequence.py      # 시퀀스 상태 관리
│   ├── block_manager.py # KV Cache 블록 관리
│   └── model_runner.py  # GPU 상에서 모델 실행
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
    ├── context.py       # 전역 추론 컨텍스트 (prefill/decode, cu_seqlens 등)
    └── loader.py        # safetensors 가중치 로딩
```

## 그림 설명 (텍스트/ASCII 설명)

**매크로 데이터 흐름 (후속 강의와 일치)**:

```
사용자 문자열
    → Tokenizer (예제 코드에서는 HuggingFace 사용)
    → LLM.generate
    → LLMEngine 루프: Scheduler가 배치 선택 → ModelRunner.run
    → 샘플링으로 다음 토큰 획득 → EOS 혹은 max_tokens 도달까지 반복
    → 텍스트 반환
```

**세 계층 멘탈 모델**:

```
        +------------------+
        |   사용자 API: LLM  |
        +--------+---------+
                 |
        +--------v---------+
        | LLMEngine 메인 루프 |
        | Scheduler / Seq    |
        | BlockManager       |
        +--------+---------+
                 |
        +--------v---------+
        | ModelRunner +     |
        | Qwen3 + Attention |
        +-------------------+
```

## 학습 경로 계획 (20강 개요)

아래 번호와 파일명은 튜토리얼 목차와 일치합니다. **처음 6강**은 본 문서 세트에서 이미 상세히 다룬 부분입니다.

| 강의 | 주제 (요약) |
|------|----------------|
| 01 | LLM 추론 기초: Prefill/Decode, 자기회귀, 병목 |
| 02 | nano-vllm 전체 구조: 4계층 아키텍처와 데이터 흐름 |
| 03 | 설정과 진입점: `Config` / `SamplingParams` / `example.py` |
| 04 | Tokenizer와 어휘 병렬 Embedding, LMHead |
| 05 | Attention: FlashAttention, KV Triton, prefix 분기 |
| 06 | RoPE, LayerNorm, SwiGLU (저장소 후반부 챕터와 연계 가능) |
| 07 | Linear와 TP, 가중치 로딩 |
| 08 | Qwen3 블록 레벨 구조 |
| 09 | KV Cache와 블록, Paged 개념 |
| 10 | BlockManager와 블록 할당 |
| 11 | Sequence와 요청 생애주기 |
| 12 | Scheduler: 연속 배치 처리 |
| 13 | 연속 배치 처리와 스케줄링 전략 (심화) |
| 14 | ModelRunner: 컨텍스트와 텐서 준비 |
| 15 | 텐서 병렬 처리 심화 |
| 16 | CUDA Graph / `enforce_eager` |
| 17 | Triton Kernel 작성 핵심 |
| 18 | LLMEngine 추론 루프 종합 |
| 19 | 성능과 벤치마크 |
| 20 | 전체 프로젝트 관통 및 면접 표현 |

## 권장 학습 순서

1. **개념 먼저, 코드 나중에**: 01 → 02. `scheduler`에 바로 뛰어들어 prefill과 decode의 차이도 모르는 상황을 피하세요.
2. **데이터 흐름 따라가기**: 03(진입점) → 04(임베딩) → 05(어텐션) → … → engine.
3. **엔진을 단일 연산자보다 우선시**: `generate`가 어떻게 스케줄링에서 `model_runner`로 이어지는지 설명할 수 있게 된 후, FlashAttention 공식을 다시 보는 것이 훨씬 수월합니다.
4. **면접 전**: 02 + 18 + 20 강의로 '아키텍처 구술 설명'을 연습하고, 05, 09, 12 강의로 '난점 심층 파고들기'를 준비하세요.

## 면접 가치 설명

- **체계성**: "요청이 들어온 후의 코드 경로"에 답할 수 있습니다 — 이는 추론 엔지니어 직무의 빈출 문제입니다.
- **깊이 조절 가능**: 1,200줄의 코드는 1~2주 안에 '주요 함수를 식별할 수 있는' 수준까지 읽을 수 있어, 이력서에 기재하고 STAR 기법으로 답변을 준비하기에 적합합니다.
- **산업계와의 정합성**: vLLM의 핵심 개념(continuous batching, paged KV, flash-attn)이 모두 포함되어 있어, 면접관이 vLLM을 물을 때 nano-vllm의 구체적인 구현으로 답변을 전환할 수 있습니다.

## vLLM과의 관계 및 차이점

**관계**:

- 사상의 동일한 기원: **블록화된 KV, 연속 배치 처리, FlashAttention, TP** 등은 vLLM과 맥을 같이 합니다.
- nano-vllm은 **vLLM의 축소 교육용 버전**으로 볼 수 있습니다: 최소한의 의존성으로 Qwen3 계열 모델 추론을 실행합니다.

**차이점 (간략히)**:

| 차원 | vLLM | nano-vllm |
|------|------|-----------|
| 코드량 | 방대함, 멀티프로세스/멀티노드 특성 완비 | ~1,200줄, 단일 프로세스 교육 중심 |
| 기능 | 양자화, API 서버, 더 많은 모델 생태계 등 | 핵심에 집중, 가독성 높음 |
| 적용 시나리오 | 프로덕션 Serving | **학습, 면접, 2차 실험** |

## 면접 출제 포인트

- nano-vllm **개발자 배경과 코드 규모** (정보 출처에 대한 관심도를 보여줌).
- **Prefill vs Decode**와 자원 병목 (연산 vs 메모리 접근).
- **vLLM과의 비교** (어느 한쪽을 폄하하지 말고, 교육용과 프로덕션용의 역할 분담을 강조할 것).
- **왜 이 프로젝트를 읽었는가** (명확한 학습 경로 + 모듈 역할을 설명할 수 있는 능력).

## 자주 나오는 면접 질문

1. **nano-vllm과 HuggingFace `generate`를 직접 호출하는 것의 본질적인 차이는 무엇인가요?**  
   답변: **배치 처리 스케줄링, KV 저장 방식, 그리고 다중 요청 Serving 시나리오 최적화 여부**에 차이가 있습니다. nano-vllm은 엔진 계층을 구현한 반면, 대부분의 HF 호출은 단일 요청 또는 소규모 배치 API입니다.

2. **왜 '추론이 병목'이라고 하나요?**  
   답변: **GPU 메모리(KV가 시퀀스에 따라 증가)**, **연산(긴 prefill)**, **지연 시간(decode 스텝이 많음)** 세 가지 측면에서 답변하고, 처리량과 지연 시간 간의 트레이드오프를 언급합니다.

3. **면접관에게 이 오픈소스 프로젝트를 어떻게 소개하시겠습니까?**  
   답변: 세 문장으로 — **포지셔닝**(DeepSeek, 초경량 vLLM), **무엇을 읽었는지**(엔진 루프 + Attention + Block), **얻은 것**(데이터 흐름을 그릴 수 있고, 한 부분을 수정해 실험할 수 있음).

## 요약

nano-vllm은 **약 1,200줄의 코드**와 **수만 개의 스타 규모의 커뮤니티 인정**을 바탕으로, LLM 추론 엔진의 핵심을 명확하게 설명합니다. 본 가이드의 **20강 로드맵**에 따라 학습하면, 면접에서 안정적으로 **아키텍처 수준의 설명 + 주요 모듈 소스 코드 수준의 디테일**을 전달할 수 있습니다.

**자가 점검 리스트 (가이드를 읽은 후 구두로 수행 가능해야 함)**:

1. 디렉터리 트리에서 `engine/`과 `layers/`의 파일명 각 세 개를 쓰고 역할을 설명할 수 있는가?
2. Prefill과 Decode 중 어느 쪽이 연산 중심이고 어느 쪽이 메모리 접근 중심인지 30초 안에 설명할 수 있는가?
3. vLLM과 비교: '사상의 계승'과 '엔지니어링적 축소'를 각각 한 가지씩 말할 수 있는가?

## 다음 강의 예고

다음 강의 **《01-강의01-대규모 모델 추론 이해》** 에서는 처음부터 **Prefill / Decode, 자기회귀, 처리량과 지연 시간** 등의 개념을 확립하여, `scheduler`와 `attention`을 읽기 위한 언어적·직관적 기초를 다질 것입니다.