# 03. 스케줄링: 요청의 여정

LLM 서비스에 수많은 사용자가 동시에 접속하면 어떻게 될까요? `nano-vllm`의 **Scheduler**는 교통경찰처럼 각 요청의 순서를 정하고 메모리를 배분합니다.

---

## 1. Sequence: 요청의 상태 변화

하나의 채팅 요청은 `Sequence` 객체로 관리되며, 다음과 같은 상태를 거칩니다.

1.  **WAITING**: 요청이 들어왔지만 아직 처리를 시작하지 않은 상태 (대기열).
2.  **RUNNING**: 모델이 돌면서 답변을 생성하고 있는 상태.
3.  **FINISHED**: 답변 생성이 완료되었거나 최대 길이에 도달한 상태.

```python
class SequenceStatus(Enum):
    WAITING = auto()   # "나 좀 처리해줘!"
    RUNNING = auto()   # "열심히 답변 만드는 중..."
    FINISHED = auto()  # "완료! 메모리 반납할게."
```

## 2. 두 가지 추론 단계: Prefill vs Decode

스케줄러는 추론 과정을 크게 두 단계로 나누어 처리합니다.

### A. Prefill (최초 입력 처리)
사용자가 보낸 질문 전체를 한꺼번에 읽고 KV Cache를 처음으로 만드는 단계입니다.
*   **특징**: 계산량이 많지만, 한 번만 수행하면 됩니다.
*   **스케줄링**: 메모리가 충분하면 `WAITING`에 있는 요청을 `RUNNING`으로 옮깁니다.

### B. Decode (다음 토큰 생성)
이전 단계의 결과를 바탕으로 단어를 하나씩 생성하는 단계입니다.
*   **특징**: 계산량은 적지만, 답변이 끝날 때까지 여러 번 반복해야 합니다.
*   **스케줄링**: 이미 `RUNNING` 상태인 요청들에게 다음 단어를 만들 기회를 줍니다.

## 3. Preemption: 메모리가 부족하면 어떡하죠?

추론 도중에 답변이 너무 길어져서 GPU 메모리(KV Cache)가 꽉 차버리면 어떻게 될까요? `nano-vllm`은 **Preemption(선점)** 기법을 사용합니다.

1.  메모리가 부족하면 `RUNNING` 중인 요청 중 하나를 골라 **강제로 중단**시킵니다.
2.  그 요청이 쓰던 메모리를 뺏어서(Deallocate) 다른 급한 요청에게 줍니다.
3.  쫓겨난 요청은 다시 `WAITING` 맨 앞으로 돌아가서 나중에 처음부터 다시 시작합니다.

> **비유**: 식당(GPU)에 자리가 없으면, 음식을 다 먹어가는 손님(Sequence)에게 양해를 구하고 잠시 대기실로 모시는 것과 비슷합니다.

## 4. Scheduler의 핵심 로직 (`engine/scheduler.py`)

```python
def schedule(self):
    # 1. Prefill 가능한 요청이 있는지 확인
    while self.waiting:
        if self.block_manager.can_allocate(seq):
            # 메모리 할당 후 실행 목록에 추가
            ...
    
    # 2. 만약 Prefill이 없다면, 기존 실행 중인 것들(Decode) 처리
    while self.running:
        if not self.block_manager.can_append(seq):
            # 메모리 부족! 다른 요청을 쫓아냄 (Preempt)
            self.preempt(last_seq)
```

---

## 📝 학생들을 위한 요약
1.  **상태 관리**: 모든 요청은 `Sequence`라는 객체로 상태가 관리됩니다.
2.  **효율적 배분**: 스케줄러는 `Prefill`과 `Decode`를 구분하여 처리 효율을 극대화합니다.
3.  **안정성**: 메모리가 꽉 차도 시스템이 멈추지 않고, `Preemption`을 통해 순차적으로 처리합니다.

다음 장에서는 모델의 뼈대인 **Transformer 레이어**들이 어떻게 코드로 구현되어 있는지 살펴보겠습니다.
