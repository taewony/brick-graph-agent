# STAR 면접 스크립트: nano-vllm 프로젝트 면접 준비

> **읽기 목표**: STAR(Situation–Task–Action–Result)를 사용하여 **자기소개, 프로젝트 소개, 기술 심층 분석, 행동 면접**을 구성하고, 1분/3분/5분 버전으로 전환할 수 있도록 합니다.  
> **설명**: 아래 텍스트의 숫자(예: 1200줄, 스타 수)는 저장소와 커뮤니티를 기준으로 삼으세요; 본인의 경력은 'XXX'와 학교/회사 정보로 대체하세요.

---

## 0. STAR 기법 속람

| 알파벳 | 의미 | 면접에서 말할 내용 |
|------|------|----------------|
| **S** | 배경 | 산업/학습 동기: 추론의 어려움, vLLM은 너무 무거움, 읽을 수 있는 완전한 경로 필요 |
| **T** | 과제 | 당신의 목표: 어떤 모듈을 숙달하고, 어떤 의문을 해결할 것인지 |
| **A** | 행동 | **구체적 동작**: 어떤 파일을 읽었는지, 어떤 실험을 했는지, 어떻게 검증했는지 |
| **R** | 결과 | 능력/데이터: 원리를 설명할 수 있음, 벤치마크 결과, 후속 계획 |

---

## 1. 자기소개 템플릿 (1분 버전)

골격을 바로 암송하고, 당신의 실제 정보로 바꾸면 됩니다:

```
면접관님 안녕하세요, 저는 XXX입니다. XXX 대학교 XXX 전공을 나왔습니다. 저는 대규모 모델 추론 최적화 방향에 매우 관심이 있으며,
최근 nano-vllm 프로젝트를 깊이 있게 학습했습니다—이것은 약 1200줄 코드 규모의 경량 vLLM 유사 구현체입니다.
소스 코드 수준의 학습을 통해, 저는 PagedAttention, 연속 배치 처리, 텐서 병렬, CUDA Graph 등 추론 엔진
핵심 기술을 숙달했습니다. 저는 소스 코드 수준에서 이러한 기술의 구현 방식을 설명할 수 있으며, 그 이면의 설계적 선택을 이해합니다.
저는 이 지식들을 실제 추론 최적화와 공학적 구현에 적용하고 싶습니다.
```

**시간 제어**: 약 180–220자 구두 = 1분; 만약 영어 면접이 요구된다면, 동일 구조의 영문 버전을 준비하세요.

---

## 2. 프로젝트 소개 템플릿 (3분 버전 · STAR 기법)

### S (상황)

대규모 모델 추론은 AI 현장 적용의 핵심 병목 중 하나입니다: GPU 메모리는 컨텍스트에 따라 증가하고, 온라인 서비스는 높은 처리량과 제어 가능한 지연을 필요로 합니다. 완전한 산업용 프레임워크(예: vLLM)는 코드 양이 방대하여, 단기간 내에 '스케줄링—메모리—계산—통신'의 전체 그림을 구축하기에 불리합니다.

### T (과제)

nano-vllm 추론 엔진을 깊이 학습하여, **프로덕션급 추론 프레임워크 핵심 모듈**에 대한 소스 코드 수준의 이해를 구축하고, 벤치마크와 디버깅을 결합하여 각 최적화가 코드에서 어디에 위치하는지 설명할 수 있도록 합니다.

### A (행동)

- 약 1200줄 규모의 소스 코드를 체계적으로 읽으며, **스케줄링**(`scheduler.py`), **페이징 KV**(`block_manager.py`), **실행 및 KV 텐서**(`model_runner.py`), **어텐션 및 Triton KV 쓰기**(`layers/attention.py`) 등 핵심 경로를 커버합니다.  
- **PagedAttention** 분석: `BlockManager`의 `allocate` / `may_append` / `deallocate`, `block_table`, xxhash 프리픽스 캐시와 `hash_to_block_id`.  
- **연속 배치 처리** 이해: `Scheduler.schedule`에서 먼저 가능한 한 Prefill(`waiting` 큐), 그렇지 않으면 Decode로 진입; `can_append`와 `preempt`의 메모리 부족 시 동작.  
- **텐서 병렬** 학습: 열/행 병렬 Linear, `QKVParallelLinear` 등과 NCCL 초기화(`model_runner.py`의 `init_process_group`).  
- **CUDA Graph** 분석: `capture_cudagraph` 다중 batch size 녹화, `run_model` 내 `graph.replay()` 분기.  
- `bench.py`(또는 프로젝트 내 벤치마크 스크립트)를 통해 성능 테스트를 재현하고, **Prefill / Decode** 처리량 표시 로직을 구분합니다(`llm_engine.py`의 `num_tokens` 부호).

### R (결과)

- `add_request` → `step` → `schedule` → `ModelRunner.run`의 완전한 경로를 구술할 수 있으며, **KV 쓰기**(Triton `store_kvcache`)와 **FlashAttention 인터페이스**(varlen vs kvcache)의 분업을 지적할 수 있습니다.  
- 재복습 가능한 기술 노트/면접 질문 답변을 형성합니다(예: 본 튜토리얼 `24-면접질문전집-STAR답변.md`).  
- 이 코드베이스 상에서 2차 개발을 수행할 수 있는 기초를 갖추었습니다(새 모델 클래스, 스케줄링 전략 실험, Profiling 등).

---

## 3. 기술 심층 분석 준비 (각 기술 포인트의 STAR)

### 1. PagedAttention

- **S**: 전통적인 최대 길이 기준 연속 KV 할당은 파편화와 낭비가 심하여, 동시성을 제한합니다.  
- **T**: nano-vllm이 어떻게 **블록**으로 KV를 관리하고, 어떻게 논리 블록 테이블로 물리 블록을 매핑하는지 이해합니다.  
- **A**: `block_manager.py` 읽기: `allocate` 내 프리픽스 해시와 `cache_miss` 분기; `may_append`가 `len(seq) % block_size == 1`일 때 새 블록 신청; `deallocate`가 **역방향**으로 `block_table`을 순회하며 참조 카운트 감소 수행(`deallocate` 루프 참조).  
- **R**: '페이징이 파편화를 줄인다', '프리픽스 적중 시 중복 계산을 건너뛴다', '참조 카운트가 0이 되어야 빈 풀로 반환된다'를 설명할 수 있습니다.

### 2. 연속 배치 처리 (Continuous Batching)

- **S**: 정적 배치는 동일 길이 또는 대량의 패딩이 필요하고, 배치 경계가 고정되어 있어 GPU가 쉽게 공회전합니다.  
- **T**: `waiting` / `running` 두 큐와 단계별 `schedule`이 어떻게 배치를 구성하는지 이해합니다.  
- **A**: `scheduler.py` 대조: `schedule` 첫 번째 구간 `while self.waiting`으로 Prefill 수행, 만약 `scheduled_seqs`가 비어 있지 않으면 **바로 반환** `is_prefill=True`; 그렇지 않으면 두 번째 구간에서 `running`에 대해 Decode 수행, `can_append` 실패 시 `preempt`.  
- **R**: '한 단계 내에서 Prefill과 Decode는 상호 배타적임', '선점으로 KV를 해제한 후 시퀀스는 `waiting`으로 되돌아간다'를 설명할 수 있습니다.

### 3. 텐서 병렬 (Tensor Parallel)

- **S**: 대형 모델은 단일 카드에 가중치와 KV를 수용할 수 없어, 계산과 통신을 분할해야 합니다.  
- **T**: Column / Row / QKV 등 병렬 레이어의 샤딩과 collective 통신 의미론을 이해합니다.  
- **A**: `layers/linear.py`(및 Qwen3에서의 사용법), `model_runner` 내 멀티프로세스 + NCCL 읽기; `ParallelLMHead`가 prefill 시 어떻게 시퀀스당 마지막 hidden만 취하는지 대조(`embed_head.py` 내 `cu_seqlens_q`).  
- **R**: TP 하에서 한 층 Transformer의 데이터 흐름과 통신 횟수를 그릴 수 있습니다.

### 4. CUDA Graph

- **S**: Decode 단계별 연산자 시퀀스는 비교적 고정되어 있지만, 파이썬 + 다중 launch 오버헤드 비중이 높습니다.  
- **T**: 왜 Prefill은 Graph를 쓰기 어렵고, Decode는 어떻게 그래프를 선택하여 재생하는지 이해합니다.  
- **A**: `capture_cudagraph` 읽기: 여러 `bs` 녹화; `run_model` 내 `is_prefill` 또는 `enforce_eager` 또는 `input_ids.size(0) > 512`이면 eager 경로, 그렇지 않으면 `graph_bs` 중 현재 bs 이상의 최소 그래프를 선택하여 `replay()`.  
- **R**: 'shape이 열거 가능하다', '자리 차지 tensor + copy_ 후 replay'를 설명할 수 있습니다.

### 5. Triton Kernel (KV 쓰기)

- **S**: 어텐션 레이어에서 새로 계산된 K/V를 페이징 주소에 따라 전역 `k_cache`/`v_cache`에 기록해야 합니다.  
- **T**: `slot_mapping`과 물리 slot의 일대일 대응을 이해합니다.  
- **A**: `store_kvcache_kernel` 읽기: `slot == -1`은 건너뜀; 그렇지 않으면 `slot * D`로 펼쳐진 캐시에 기록.  
- **R**: `prepare_prefill` / `prepare_decode`에서 구성된 `slot_mapping`과 어떻게 연결되는지 설명할 수 있습니다.

---

## 4. 행동 면접 질문 (STAR 준비)

### "가장 큰 기술적 도전은 무엇이었나요?"

- **S/T**: 처음 접할 때, **스케줄링 + 블록 테이블 + slot_mapping + FlashAttention 파라미터**가 서로 맞물리기 어려웠습니다.  
- **A**: 실제 막혔던 지점을 하나 선택하고(예를 들어 prefix cache 분기 하에서 `block_tables`가 언제 `set_context`에 전달되는지), 작은 규모의 예시를 손으로 계산하거나 중간 텐서 shape을 출력하여 검증했습니다.  
- **R**: '인덱스와 물리 저장'에 대한 명확한 멘탈 모델을 형성하고, 다른 사람에게 설명할 수 있게 되었습니다.

### "새로운 코드베이스를 어떻게 학습하나요?"

- **A**: 진입점(`LLMEngine.generate` / `step`)에서 시작 → 호출 그래프 그리기 → 각 모듈에서 하나의 **대표 API** 선택(예: `schedule`, `allocate`) → 디버거 또는 로그로 최소 예제 실행.  
- **R**: 노트나 마인드맵을 출력하고, 벤치마크를 재현할 수 있습니다.

### "이 프로젝트에 어떤 개선 구상이 있나요?"

- **R 방향 예시**: Chunked Prefill, 추측적 디코딩, 가중치 양자화, API Server, 더 세분화된 metrics; **변경 지점**이 대략 `scheduler` / `model_runner` / 새로운 `Model` 클래스에 있음을 설명합니다.  
- 피할 것: 막연한 '성능 최적화'; **모듈명 + 트레이드오프**를 말해야 합니다.

---

## 5. 다양한 시간 버전 속용

### 1분 (프로젝트 버전, 프로젝트만)

"저는 nano-vllm이라는 간소화된 추론 엔진을 학습하여, 페이징 KV, 연속 배치 처리, FlashAttention 두 단계, CUDA Graph와 Triton KV 쓰기를 커버했습니다. 주요 경로를 설명할 수 있고 벤치마크 테스트를 실행할 수 있습니다."

### 3분

본 문서의 **2절** 전체 STAR를 사용합니다.

### 5분

3분 기반 위에, **두 개의 기술 심층 분석**을 추가합니다: **PagedAttention + CUDA Graph**를 선택하는 것을 권장하며, 각 1분씩, vLLM과의 차이(기능 범위, 엔지니어링 정도)에 대한 한 마디를 삽입합니다.

---

## 6. 면접 대화 팁

1. **결론 먼저, 디테일 나중에**: 예를 들어 "Prefill 단계에서는 CUDA Graph를 사용하지 않습니다, 왜냐하면…" 이후 `run_model` 분기를 전개합니다.  
2. **정직한 경계**: 구현되지 않은 양자화/추측적 디코딩은 직접 말할 수 있으며, "만약 한다면 어떤 모듈을 움직일지"를 보충합니다.  
3. **능동적으로 JD에 연결**: 상대가 "KV / 스케줄링 / 분산"이라고 썼다면, 마지막에 "그래서 저는 `block_manager` / `scheduler` / TP를 중점적으로 읽었습니다"라고 한 마디 덧붙입니다.

---

## 7. 튜토리얼 문서 인덱스

- 이력서 작성법: **`22-프로젝트이력서작성가이드.md`**  
- 문제별 STAR 풀이: **`24-면접질문전집-STAR답변.md`**  
- 빈출 문제 체계: **`21-면접빈출문제대전.md`**