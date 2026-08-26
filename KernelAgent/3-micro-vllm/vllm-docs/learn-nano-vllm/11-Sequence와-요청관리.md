이 파일은 이미 한국어로 제공되었습니다. 아래 내용은 원본 파일을 그대로 보여드린 것입니다.

---

# 강의 11: Sequence와 요청 관리

> **학습 목표**: nano-vllm에서 요청의 핵심 데이터 구조인 `Sequence`를 깊이 이해합니다. `SequenceStatus` 상태 머신의 변환 논리를 파악합니다. 속성별, 메서드별로 소스 코드를 분석합니다. block_table과 KV Cache의 매핑 관계를 이해합니다. 직렬화 최적화 기법을 숙달합니다. 후속 스케줄러와 연속 배치 처리 학습을 위한 탄탄한 기초를 다집니다.

---

## 1. Sequence 클래스가 필요한 이유

### 1.1 추론 엔진에서의 "요청"

대규모 모델 추론 서비스에서, 각 사용자 입력(prompt)과 후속 생성 token은 하나의 **시퀀스(Sequence)**를 구성합니다. 추론 엔진은 각 시퀀스에 대해 다음 정보를 추적해야 합니다:

- **상태**: 이 시퀀스는 현재 대기 중인지, 실행 중인지, 아니면 완료되었는지?
- **Token 리스트**: 원본 prompt의 token과 이미 생성된 completion token을 포함합니다.
- **KV Cache 매핑**: 시퀀스의 Key/Value 캐시 데이터가 어떤 물리 블록에 분산되어 저장되는지?
- **샘플링 파라미터**: temperature(온도), max_tokens(최대 생성 길이), ignore_eos(EOS 무시 여부) 등.
- **캐시 정보**: 몇 개의 token에 대한 KV Cache가 이미 계산되어 캐시되었는지?

통일된 데이터 구조로 이러한 정보를 관리하지 않으면, 스케줄러(Scheduler)와 모델 실행기(ModelRunner)가 효율적으로 협업할 수 없습니다. 따라서 nano-vllm은 `Sequence` 클래스를 엔진 전 생애주기를 관통하는 **핵심 데이터 구조**로 설계했습니다.

### 1.2 비유로 이해하기

추론 엔진을 병원에 비유할 수 있습니다:

| 병원 | 추론 엔진 |
|------|---------|
| 환자 | Sequence (사용자 요청) |
| 등록 정보 | token_ids, sampling_params |
| 진료 카드 | block_table (KV Cache 매핑) |
| 진료 상태 (대기/진료 중/완료) | SequenceStatus |
| 접수처 | Scheduler |
| 진료실 | ModelRunner (GPU) |

### 1.3 시스템 내에서 Sequence의 위치

```
사용자 요청(텍스트)
  ↓  tokenize
LLMEngine.add_request()
  ↓  Sequence 객체 생성
Scheduler.add(seq)         ← seq가 waiting 큐에 진입
  ↓
Scheduler.schedule()       ← seq가 스케줄링되고, KV Cache block 할당됨
  ↓
ModelRunner.run(seqs, ...)  ← seq의 token_ids / block_table 읽음
  ↓
Scheduler.postprocess()     ← 새 token 추가, 종료 여부 판단
  ↓
seq.status == FINISHED → 사용자에게 결과 반환
```

생성부터 종료까지, `Sequence` 인스턴스는 **큐 추가 → 스케줄링 → 순전파 추론 → 후처리 → 완료**의 전체 과정을 관통합니다. 이는 엔진 내 모든 모듈이 통신하는 "공통 언어"입니다.

---

## 2. SequenceStatus 상태 머신

### 2.1 세 가지 상태 정의

소스 코드 경로: `nanovllm/engine/sequence.py`

```python
class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()
```

| 상태 | 의미 | 진입 조건 | 이탈 조건 |
|------|------|---------|---------|
| `WAITING` | 대기열에서 스케줄링을 기다림 | 신규 생성 / 선점(preempt)됨 | `schedule()`에 선택되어 block 할당됨 |
| `RUNNING` | 추론에 참여 중 (prefill 또는 decode) | `schedule()`이 block 할당 후 | 생성 완료(EOS / max_tokens) / 선점됨 |
| `FINISHED` | 생성 완료, 회수 대기 | EOS 도달 또는 max_tokens 도달 | 종료 상태, 더 이상 변화 없음 |

### 2.2 상태 변환 다이어그램

```
              schedule()에 선택됨
  ┌────────┐  block 할당   ┌────────┐
  │WAITING │──────────────→│RUNNING │
  └────────┘               └────────┘
       ↑                     │    │
       │    preempt()        │    │ EOS 도달 또는
       │    block 해제       │    │ max_tokens 도달
       └─────────────────────┘    │
                                  ↓
                             ┌──────────┐
                             │ FINISHED │
                             └──────────┘
```

### 2.3 상태 변환의 트리거 위치

| 변환 | 트리거 위치 | 코드 조각 |
|------|---------|---------|
| WAITING → RUNNING | `Scheduler.schedule()` | `seq.status = SequenceStatus.RUNNING` |
| RUNNING → WAITING | `Scheduler.preempt()` | `seq.status = SequenceStatus.WAITING` |
| RUNNING → FINISHED | `Scheduler.postprocess()` | `seq.status = SequenceStatus.FINISHED` |

### 2.4 왜 문자열 대신 Enum을 사용할까?

문자열(예: `"waiting"`) 대신 `Enum`을 사용하는 이점:

1. **타입 안전성**: 오타 발생 시 접근 시점에 즉각 오류 보고, 조용한 실패 방지
2. **성능**: 열거형 비교는 정수 비교이므로 문자열 비교보다 빠름
3. **IDE 지원**: 자동 완성, 리팩터링 시 모든 참조 추적 가능
4. **가독성**: `SequenceStatus.RUNNING`이 `"running"`보다 의미가 더 명확

### 2.5 왜 PREEMPTED 상태가 없을까?

선점된 시퀀스에 별도의 `PREEMPTED` 상태가 필요하지 않을까 의문이 들 수 있습니다.

nano-vllm 설계에서는, 선점된 시퀀스는 **바로 WAITING으로 돌아갑니다**. 이렇게 하면 상태 머신이 단순해집니다—스케줄러는 waiting 큐만 확인하면 되며, "새로운 것"과 "선점된 것"을 구분할 필요가 없습니다. 선점된 시퀀스는 `appendleft`를 통해 waiting 큐의 맨 앞에 배치되어, 우선적으로 재스케줄링되도록 보장됩니다.

> **면접 팁**: vLLM의 정식 버전에는 CPU 메모리로 교환된 시퀀스를 구분하기 위한 `SWAPPED` 상태가 있습니다. nano-vllm은 이 설계를 단순화했습니다.

---

## 3. Sequence 클래스 소스 코드 완전 분석

### 3.1 클래스 레벨 속성

```python
class Sequence:
    block_size = 256
    counter = count()
```

| 속성 | 타입 | 설명 |
|------|------|------|
| `block_size` | int (클래스 변수) | KV Cache 블록 크기, 기본값 256 토큰. 각 물리 블록이 몇 개의 토큰에 대한 KV 데이터를 수용할 수 있는지 나타냄 |
| `counter` | itertools.count (클래스 변수) | 전역 증가 카운터, 고유한 `seq_id` 생성을 위해 사용 |

**왜 block_size는 클래스 변수인가?**

모든 시퀀스가 동일한 물리 블록 관리 시스템을 공유하므로, 블록 크기가 일관되어야 합니다. 클래스 변수로 설정하면 모든 인스턴스가 동일한 block_size를 사용하도록 보장되며, 한 곳만 수정하면 전역적으로 효력이 발생합니다.

**왜 `itertools.count()`를 사용할까?**

`count()`는 무한 반복자이며, (CPython GIL 하에서) 스레드 안전하고, 고유한 증가 ID를 매우 간결하게 생성합니다:

```python
from itertools import count
c = count()
next(c)  # 0
next(c)  # 1
next(c)  # 2
```

### 3.2 생성자 `__init__`

```python
def __init__(self, token_ids, sampling_params=SamplingParams()):
    self.seq_id = next(Sequence.counter)
    self.status = SequenceStatus.WAITING
    self.token_ids = copy(token_ids)
    self.last_token = token_ids[-1]
    self.num_tokens = len(self.token_ids)
    self.num_prompt_tokens = len(token_ids)
    self.num_cached_tokens = 0
    self.block_table = []
    self.temperature = sampling_params.temperature
    self.max_tokens = sampling_params.max_tokens
    self.ignore_eos = sampling_params.ignore_eos
```

속성별 상세 설명:

| 속성 | 타입 | 초기값 | 설명 |
|------|------|--------|------|
| `seq_id` | int | 증가값 | 전역 고유 식별자, 서로 다른 시퀀스 구분 |
| `status` | SequenceStatus | WAITING | 초기 상태: 새로 생성된 시퀀스는 반드시 대기 상태 |
| `token_ids` | list[int] | prompt tokens의 복사본 | prompt + 이미 생성된 completion token 포함 |
| `last_token` | int | prompt의 마지막 token | decode 단계에서 다음 단계 입력으로 사용 (리스트 끝을 조회하는 오버헤드 회피) |
| `num_tokens` | int | len(token_ids) | 현재 총 토큰 수 (prompt + completion) |
| `num_prompt_tokens` | int | len(token_ids) | prompt의 토큰 수, **생성 후 불변** |
| `num_cached_tokens` | int | 0 | 이미 KV Cache에 캐시된 토큰 수 (프리픽스 캐싱용) |
| `block_table` | list[int] | [] | 물리 블록 인덱스 리스트, KV Cache 저장 위치 기록 |
| `temperature` | float | sampling_params에서 | 샘플링 온도: 0이면 greedy, 클수록 무작위적 |
| `max_tokens` | int | sampling_params에서 | 최대 생성 토큰 수 |
| `ignore_eos` | bool | sampling_params에서 | EOS token 무시 여부 (강제로 max_tokens까지 생성) |

### 3.3 왜 `copy(token_ids)`를 사용할까?

```python
self.token_ids = copy(token_ids)
```

`copy()`로 token_ids의 얕은 복사본을 생성하여, 외부에서 원본 리스트를 수정할 때 Sequence 내부 상태에 영향을 미치는 것을 방지합니다. 이는 방어적 프로그래밍(defensive programming)의 전형적인 실천입니다.

### 3.4 `num_prompt_tokens` vs `num_tokens`

이 두 속성의 차이를 이해하는 것이 Sequence를 이해하는 핵심입니다:

```
생성 시: num_prompt_tokens = 5, num_tokens = 5
토큰 1개 생성 후: num_prompt_tokens = 5, num_tokens = 6
토큰 2개 생성 후: num_prompt_tokens = 5, num_tokens = 7
...
```

이로부터 이미 생성된 completion token 수를 유도할 수 있습니다:

```python
@property
def num_completion_tokens(self):
    return self.num_tokens - self.num_prompt_tokens
```

이 값은 `max_tokens` 제한에 도달했는지 판단하는 데 사용됩니다.

---

## 4. Block 관련 속성과 메서드

### 4.1 block_table의 역할

`block_table`은 Sequence와 KV Cache 물리 저장소 사이의 가교입니다. 정수 리스트로, 각 원소는 **물리 블록의 인덱스 번호**입니다.

```
가정: block_size = 256, 시퀀스가 600개의 토큰을 가짐:

block_table = [3, 7, 15]  # 3개의 물리 블록

Block 3: token 0-255의 KV Cache
Block 7: token 256-511의 KV Cache
Block 15: token 512-599의 KV Cache (가득 차지 않음)
```

**핵심 이해**: block_table은 **논리 블록에서 물리 블록으로의 매핑**을 구현합니다. 논리적으로 시퀀스의 0번째 블록은 물리 메모리의 3번째 블록에 대응될 수 있고, 1번째 논리 블록은 물리 7번째 블록에 대응되며, 이와 같이 진행됩니다. 이것이 PagedAttention의 핵심 사상입니다—운영체제의 가상 메모리처럼 KV Cache를 관리하는 것입니다.

### 4.2 num_blocks 속성

```python
@property
def num_blocks(self):
    return (self.num_tokens + self.block_size - 1) // self.block_size
```

이것은 고전적인 **올림 나눗셈** 공식으로, 현재 시퀀스에 필요한 블록 수를 계산합니다:

| num_tokens | block_size | num_blocks | 계산 과정 |
|-----------|-----------|-----------|---------|
| 1 | 256 | 1 | (1+255)//256 = 1 |
| 256 | 256 | 1 | (256+255)//256 = 1 |
| 257 | 256 | 2 | (257+255)//256 = 2 |
| 512 | 256 | 2 | (512+255)//256 = 2 |
| 600 | 256 | 3 | (600+255)//256 = 3 |

왜 `math.ceil(n / d)` 대신 `(n + d - 1) // d`를 사용할까? 정수 연산이 부동소수점 연산보다 더 빠르고 정확하기 때문이며, 고빈도 호출 시나리오에서 이 작은 차이가 누적됩니다.

### 4.3 last_block_num_tokens 속성

```python
@property
def last_block_num_tokens(self):
    return self.num_tokens - (self.num_blocks - 1) * self.block_size
```

마지막 블록에 몇 개의 토큰이 있는지 계산합니다. 이 정보는 decode 단계의 `slot_mapping` 계산에 매우 중요합니다.

```
예시: num_tokens=600, block_size=256, num_blocks=3
last_block_num_tokens = 600 - 2 * 256 = 88

의미: 마지막 블록에 88개의 토큰이 있으며, 다음 새 토큰은 slot 88에 기록되어야 함
```

**용도**: `ModelRunner.prepare_decode()`에서 새 토큰의 KV Cache가 기록될 물리 위치를 계산하는 데 사용됩니다:

```python
slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
```

### 4.4 block(i) 메서드

```python
def block(self, i):
    return self.token_ids[i*self.block_size: (i+1)*self.block_size]
```

`i`번째 논리 블록에 해당하는 token_ids 슬라이스를 가져옵니다. 주로 **프리픽스 캐싱(Prefix Caching)**에 사용됩니다: 두 시퀀스의 동일 위치 블록의 토큰 내용을 비교하여, 동일한 프리픽스를 공유하는지 판단하고 KV Cache를 재사용합니다.

```
시퀀스 A: [101, 202, 303, ..., 256개 토큰, 401, 402, ...]
시퀀스 B: [101, 202, 303, ..., 256개 토큰, 501, 502, ...]

block(0) 동일 → Block 0의 KV Cache 공유 가능
block(1) 다름 → Block 1은 독립적으로 계산 필요
```

---

## 5. append_token 흐름

### 5.1 소스 코드 분석

```python
def append_token(self, token_id):
    self.token_ids.append(token_id)
    self.last_token = token_id
    self.num_tokens += 1
```

이 메서드는 decode 단계마다 `Scheduler.postprocess()`에 의해 호출되어, 새로 생성된 토큰을 시퀀스에 추가합니다.

### 5.2 실행 흐름

```
호출 전:
  token_ids = [1, 2, 3, 4, 5]  (prompt)
  last_token = 5
  num_tokens = 5

append_token(100):
  token_ids = [1, 2, 3, 4, 5, 100]
  last_token = 100
  num_tokens = 6

append_token(200):
  token_ids = [1, 2, 3, 4, 5, 100, 200]
  last_token = 200
  num_tokens = 7
```

### 5.3 왜 last_token과 token_ids를 둘 다 유지할까?

겉보기에는 중복이지만, 실제로는 성능을 고려한 것입니다:

1. **decode 단계**에서는 마지막 토큰만 입력으로 필요하며, `last_token`을 직접 읽는 것은 O(1) 연산입니다
2. 매번 `token_ids[-1]`에서 가져오는 것도 파이썬 리스트 인덱싱이 O(1)이지만, `last_token`은 독립 속성으로 직렬화 전송 시 전체 token_ids 리스트를 전송하지 않아도 됩니다
3. `__getstate__`에서, 시퀀스에 이미 completion token이 있으면 완전한 `token_ids` 대신 `last_token`만 전송합니다

### 5.4 append_token은 block_table을 갱신하지 않습니다

주의할 점은 `append_token`이 **`block_table`을 수정하지 않는다는 것**입니다. 블록 할당은 `BlockManager`의 책임이며, `Scheduler.schedule()`에서 `block_manager.may_append()`를 통해 완료됩니다. Sequence는 단지 데이터 소유자일 뿐, 자원 관리를 책임지지 않습니다—이는 **관심사 분리(Separation of Concerns)** 설계 원칙을 반영합니다.

---

## 6. `__len__` 및 기타 보조 속성

### 6.1 `__len__` 메서드

```python
def __len__(self):
    return self.num_tokens
```

Sequence가 `len(seq)` 구문을 지원하도록 하여, 스케줄러 코드에서 빈번하게 사용됩니다. 예를 들어:

```python
num_batched_tokens += len(seq) - seq.num_cached_tokens
```

### 6.2 num_completion_tokens 속성

```python
@property
def num_completion_tokens(self):
    return self.num_tokens - self.num_prompt_tokens
```

생성 종료 여부를 판단하는 데 사용됩니다:

```python
if seq.num_completion_tokens == seq.max_tokens:
    seq.status = SequenceStatus.FINISHED
```

### 6.3 is_prefill / is_finished 속성

```python
@property
def is_prefill(self):
    return self.num_completion_tokens == 0

@property
def is_finished(self):
    return self.status == SequenceStatus.FINISHED
```

- `is_prefill`: 아직 completion token을 생성하지 않았다면, 이 시퀀스는 prefill 단계에 있음
- `is_finished`: 시퀀스가 완료되었는지 판단

---

## 7. 직렬화 최적화: `__getstate__`와 `__setstate__`

### 7.1 왜 직렬화 최적화가 필요할까?

멀티프로세스(텐서 병렬) 시나리오에서, rank 0 프로세스는 스케줄링 결과를 다른 worker에게 브로드캐스트해야 합니다. 각 단계마다 추론에 참여하는 모든 Sequence 객체를 전송해야 합니다. 최적화하지 않으면, 매번 완전한 `token_ids` 리스트(수천 개의 토큰이 있을 수 있음)를 직렬화해야 하므로, 막대한 통신 오버헤드가 발생합니다.

### 7.2 `__getstate__` 소스 코드

```python
def __getstate__(self):
    return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens,
            self.block_table,
            self.token_ids if self.num_completion_tokens == 0 else self.last_token)
```

**정교함**:

1. **Prefill 단계**(`num_completion_tokens == 0`): 완전한 `token_ids`를 전송, worker가 모든 prompt token을 필요로 하여 KV Cache를 계산해야 하기 때문
2. **Decode 단계**(`num_completion_tokens > 0`): `last_token`만 전송, worker가 입력으로 마지막 토큰만 필요하기 때문

### 7.3 전송 데이터 양 비교

어떤 시퀀스에 1000개의 prompt token이 있고, 이미 500개의 token이 생성되었다고 가정:

| 방식 | 전송 데이터 | 크기 추정 |
|------|---------|---------|
| 미최적화 | 완전한 객체 (모든 속성 + 1500 token_ids) | ~12KB |
| `__getstate__` 최적화 | num_tokens + num_prompt_tokens + num_cached_tokens + block_table + last_token | ~100B |

decode 단계의 전송량이 약 **100배** 감소했습니다!

### 7.4 `__setstate__` 소스 코드

```python
def __setstate__(self, state):
    self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table, token_data = state
    if isinstance(token_data, list):
        self.token_ids = token_data
    else:
        self.last_token = token_data
```

수신 측은 `token_data`의 타입에 따라 현재 단계를 판단합니다:

- 만약 `list`이면: prefill 단계, `token_data`가 완전한 `token_ids`
- 만약 `int`이면: decode 단계, `token_data`가 `last_token`

### 7.5 시스템 내 직렬화 사용 시나리오

```
Rank 0 (주 프로세스)                     Rank 1/2/3 (Worker 프로세스)
     │                                      │
     │  scheduler.schedule()                │
     │  → [seq1, seq2, ...] 획득            │
     │                                      │
     │  pickle.dumps(seqs)                  │
     │  → __getstate__ 호출됨               │
     │  → 슬림화된 데이터                    │
     │                                      │
     │  ════ SharedMemory 전송 ═══════>     │
     │                                      │
     │                           pickle.loads(data)
     │                           → __setstate__ 호출됨
     │                           → Sequence 객체 복원
     │                                      │
     │                           model.forward(seqs)
```

---

## 8. Sequence와 다른 모듈 간의 상호작용

### 8.1 Scheduler와의 상호작용

```python
# Scheduler 생성 시점
scheduler.add(seq)  # seq가 waiting 큐에 추가됨

# Scheduler 스케줄링 시
scheduler.schedule()
    → seq.status = SequenceStatus.RUNNING
    → block_manager.allocate(seq)  # seq.block_table 채움

# Scheduler 후처리 시
scheduler.postprocess(seqs, token_ids)
    → seq.append_token(token_id)
    → seq.num_completion_tokens == seq.max_tokens 확인
```

### 8.2 ModelRunner와의 상호작용

```python
# Prefill 준비
model_runner.prepare_prefill(seqs)
    → seq.token_ids[seq.num_cached_tokens:] 읽음  # 이미 캐시된 토큰 건너뜀
    → seq.block_table 읽음  # KV Cache 기록 위치 획득

# Decode 준비
model_runner.prepare_decode(seqs)
    → seq.last_token 읽음  # 마지막 토큰만 필요
    → seq.block_table[-1] 읽음  # 마지막 블록
    → seq.last_block_num_tokens 읽음  # slot_mapping 계산
```

### 8.3 BlockManager와의 상호작용

```python
# 블록 할당
block_manager.allocate(seq)
    → seq.num_blocks 계산
    → 물리 블록 할당
    → seq.block_table 채움

# 블록 해제
block_manager.deallocate(seq)
    → seq.block_table 내 물리 블록 회수
    → seq.block_table = []

# 블록 추가
block_manager.may_append(seq)
    → 마지막 블록이 가득 찼다면, 새 블록 할당
    → seq.block_table에 추가
```

---

## 9. num_cached_tokens의 의미와 역할

### 9.1 Prefix Caching이란?

Prefix Caching(프리픽스 캐싱)은 최적화 기술입니다: 두 요청이 동일한 프리픽스(예: 동일한 시스템 프롬프트)를 가지면, 이미 계산된 KV Cache를 재사용하여 중복 계산을 방지합니다.

```
요청 A: "당신은 AI 어시스턴트입니다. 베이징을 소개해 주세요."
요청 B: "당신은 AI 어시스턴트입니다. 상하이를 소개해 주세요."

공통 프리픽스: "당신은 AI 어시스턴트입니다."
→ 이 부분의 KV Cache는 한 번만 계산하면 됨
```

### 9.2 num_cached_tokens의 역할

`num_cached_tokens`는 시퀀스에서 얼마나 많은 토큰의 KV Cache가 **이미 사용 가능한지**(프리픽스 캐시 또는 이전 계산에서 비롯됨)를 기록합니다.

```
num_cached_tokens = 100, num_tokens = 500이라고 가정

prefill 시 실제 계산이 필요한 토큰 수: 500 - 100 = 400
→ 100개 토큰의 계산량 절감
```

`ModelRunner.prepare_prefill()`에서의 사용:

```python
input_ids.extend(seq[seq.num_cached_tokens:])  # 캐시되지 않은 토큰만 취함
positions.extend(list(range(seq.num_cached_tokens, len(seq))))  # 위치는 캐시 종료 지점에서부터 시작
```

`Scheduler.schedule()`에서 실제 배치 처리 토큰 수 계산:

```python
num_batched_tokens += len(seq) - seq.num_cached_tokens  # 실제 계산이 필요한 토큰만 계산
```

### 9.3 선점 후 num_cached_tokens

시퀀스가 선점(preempt)될 때, 그 KV Cache는 해제되고, `num_cached_tokens`는 0으로 초기화됩니다(BlockManager가 처리). 재스케줄링 시 모든 토큰의 KV Cache를 다시 계산해야 합니다.

---

## 10. SamplingParams 샘플링 파라미터

### 10.1 파라미터 설명

```python
@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 256
    ignore_eos: bool = False
```

| 파라미터 | 기본값 | 설명 |
|------|--------|------|
| `temperature` | 1.0 | 샘플링 온도. 0 = greedy decoding, 1.0 = 표준 샘플링, > 1.0 = 더 무작위적 |
| `max_tokens` | 256 | 최대 생성 토큰 수 |
| `ignore_eos` | False | EOS 토큰 무시 여부. True이면 EOS를 만나도 계속 생성 |

### 10.2 temperature의 수학적 원리

샘플링 전에, logits가 temperature로 나눠집니다:

\[
p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
\]

- \( T \to 0 \): 확률 분포가 one-hot에 가까워짐, 항상 최대 확률의 토큰 선택(greedy)
- \( T = 1 \): 표준 softmax 분포
- \( T > 1 \): 분포가 더 평탄해지고, 더 많은 "창의성"을 보이지만 통제력은 떨어짐

### 10.3 왜 Sequence가 샘플링 파라미터를 직접 저장할까?

`SamplingParams`의 참조를 유지하는 대신:

1. 직렬화 시 복잡한 객체 대신 단순 타입(float, int, bool)만 전송하면 됨
2. SamplingParams 객체가 외부에서 수정되어 발생하는 부작용 방지
3. Sequence는 최소 데이터 단위로서, 필요한 모든 정보를 자체 포함해야 함

---

## 11. 설계 패턴과 엔지니어링 실천

### 11.1 데이터 클래스(Data Class) 패턴

`Sequence`는 본질적으로 데이터 클래스이며, 주요 역할은 **데이터를 보유**하는 것이지 복잡한 로직을 실행하는 것이 아닙니다. 복잡한 비즈니스 로직은 Scheduler와 ModelRunner가 완료합니다. 이는 **빈약한 모델(Anemic Model)** 설계 스타일을 따릅니다(논란의 여지는 있지만, 고성능 시스템에서는 매우 흔합니다).

### 11.2 불변 vs 가변

| 속성 | 가변 여부 | 수정 시점 |
|------|---------|---------|
| `seq_id` | 불변 | 생성 시 결정 |
| `num_prompt_tokens` | 불변 | 생성 시 결정 |
| `temperature` | 불변 | 생성 시 결정 |
| `max_tokens` | 불변 | 생성 시 결정 |
| `status` | 가변 | Scheduler가 수정 |
| `token_ids` | 가변 | append_token이 추가 |
| `num_tokens` | 가변 | append_token이 증가 |
| `block_table` | 가변 | BlockManager가 수정 |
| `num_cached_tokens` | 가변 | BlockManager가 수정 |

### 11.3 전역 고유 ID 설계

`itertools.count()`를 사용하여 전역 고유 `seq_id`를 생성합니다. 이는 UUID보다 더 가벼우며, 단일 프로세스 환경에서 충분합니다. 분산 시나리오에서는 Sequence가 rank 0에서만 생성되므로, ID 충돌 문제가 존재하지 않습니다.

---

## 12. 소스 코드 대조 요약

완전한 Sequence 소스 코드와 엔진 각 모듈의 사용 시나리오를 대조합니다:

| Sequence 속성/메서드 | 생성/수정자 | 사용자 | 용도 |
|-------------------|-----------|--------|------|
| `seq_id` | `__init__` | Engine | 고유 식별 |
| `status` | `__init__`, Scheduler | Scheduler, Engine | 생애주기 관리 |
| `token_ids` | `__init__`, `append_token` | ModelRunner | prefill 입력 |
| `last_token` | `__init__`, `append_token` | ModelRunner | decode 입력 |
| `num_tokens` | `__init__`, `append_token` | Scheduler, ModelRunner | 배치 처리 계산 |
| `num_prompt_tokens` | `__init__` | Scheduler | prefill/decode 판단 |
| `num_cached_tokens` | `__init__`, BlockManager | Scheduler, ModelRunner | 프리픽스 캐싱 |
| `block_table` | `__init__`, BlockManager | ModelRunner | KV Cache 주소 지정 |
| `num_blocks` | 계산 속성 | BlockManager | 블록 할당 |
| `last_block_num_tokens` | 계산 속성 | ModelRunner | slot_mapping 계산 |
| `block(i)` | 메서드 | BlockManager | 프리픽스 캐시 매칭 |
| `append_token()` | 메서드 | Scheduler.postprocess | 생성 토큰 추가 |
| `__getstate__` | 메서드 | pickle(멀티프로세스 통신) | 직렬화 슬림화 |
| `__setstate__` | 메서드 | pickle(멀티프로세스 통신) | 역직렬화 복원 |

---

## 13. 면접 출제 포인트

### 포인트 1: Sequence의 상태 머신과 변환 조건을 설명하세요.

**표준 답변**: Sequence에는 WAITING, RUNNING, FINISHED 세 가지 상태가 있습니다. 새로 생성되면 WAITING, 스케줄러에 선택되어 KV Cache 블록을 할당받으면 RUNNING, 생성이 완료되면(EOS 도달 또는 max_tokens 충족) FINISHED가 됩니다. GPU 자원이 부족하면 RUNNING 상태의 시퀀스가 WAITING 상태로 선점될 수 있으며(KV Cache 블록 해제), 이후 재스케줄링을 기다립니다.

### 포인트 2: block_table이란 무엇인가요? 왜 KV Cache를 순차적으로 저장하지 않나요?

**표준 답변**: block_table은 논리 블록에서 물리 블록으로의 매핑 테이블로, 운영체제의 페이지 테이블과 유사합니다. PagedAttention의 핵심 사상을 구현합니다—KV Cache를 고정 크기의 블록으로 나누어, GPU 메모리에 비연속적으로 저장합니다. 이점: (1) 메모리 파편화 방지, (2) 동적 확장 지원(매번 하나의 블록만 추가), (3) 프리픽스 캐싱 지원(서로 다른 시퀀스가 동일 내용의 블록 공유 가능).

### 포인트 3: `__getstate__`의 직렬화 최적화 전략은 무엇인가요? 왜 이렇게 설계했나요?

**표준 답변**: decode 단계에서는 완전한 `token_ids` 리스트 대신 `last_token`만 전송합니다. decode 단계에서는 worker가 입력으로 마지막 토큰만 필요하며, 완전한 과거 토큰은 필요하지 않기 때문입니다. prefill 단계에서는 완전한 `token_ids`를 전송하여, worker가 모든 prompt 토큰의 KV Cache를 계산할 수 있도록 합니다. 이 최적화로 decode 단계의 통신량을 두 자릿수 이상 줄일 수 있습니다.

### 포인트 4: num_cached_tokens의 역할은 무엇인가요?

**표준 답변**: `num_cached_tokens`는 시퀀스에서 이미 KV Cache에 캐시된 토큰 수를 기록하며, 프리픽스 캐싱 최적화에 사용됩니다. prefill 시에는 `token_ids[num_cached_tokens:]` 부분의 토큰에 대해서만 KV Cache를 계산하면 되며, 이미 캐시된 부분은 건너뛸 수 있습니다. 배치 처리 토큰 수를 계산할 때도 `num_cached_tokens`를 차감하는데, 이 부분은 계산 자원을 점유하지 않기 때문입니다.

### 포인트 5: 왜 Sequence는 token_ids와 last_token을 동시에 유지하나요?

**표준 답변**: 이는 공간으로 시간을 교환하는 최적화입니다. decode 단계에서, 모델 입력은 마지막 토큰만 필요합니다. 독립된 `last_token` 속성을 유지하면: (1) O(1) 빠른 접근, (2) 직렬화 전송 시 전체 리스트 대신 하나의 정수만 전송, (3) 의미가 더 명확하고 코드 가독성이 더 좋습니다.

### 포인트 6: Sequence의 block_size가 왜 클래스 변수인가요?

**표준 답변**: 전체 시스템의 KV Cache가 통일된 블록 크기를 사용하기 때문에, 모든 시퀀스가 동일한 block_size를 사용해야 BlockManager와 올바르게 상호작용할 수 있습니다. 클래스 변수로 설정하면 일관성을 보장하며, 한 곳만 수정하면 전역적으로 효력이 발생합니다.

### 포인트 7: 만약 프로덕션급 Sequence 클래스를 설계한다면, 어떤 개선을 하시겠습니까?

**참고 방향**:
1. `SWAPPED` 상태 추가, KV Cache를 (바로 폐기하는 대신) CPU 메모리로 교환하는 기능 지원
2. beam search 지원, `parent_seq_id`와 `fork()` 메서드 추가
3. `LoRA adapter ID` 추가, 다중 LoRA 추론 지원
4. 요청 수준의 `priority` 필드 추가, 우선순위 스케줄링 지원
5. `arrival_time` 추가, 대기 시간 기반 공정 스케줄링 지원
6. EOS뿐만 아니라 `stop_sequences`(정지어 리스트) 지원

---

## 14. 요약

| 지식 포인트 | 핵심 이해 |
|--------|---------|
| Sequence 포지셔닝 | 추론 엔진의 핵심 데이터 구조, 요청 전 생애주기 관통 |
| 상태 머신 | WAITING → RUNNING → FINISHED, WAITING으로의 선점 회귀 지원 |
| block_table | 논리 블록에서 물리 블록으로의 매핑, PagedAttention 구현 |
| num_cached_tokens | 프리픽스 캐싱의 핵심 필드, 중복 계산 감소 |
| append_token | 가벼운 추가 연산, 블록 할당 불포함 |
| 직렬화 최적화 | decode 단계에서 last_token만 전송, 100배 통신량 감소 |
| 설계 원칙 | 관심사 분리, 방어적 복사, 공간으로 시간 교환 |

**다음 강의 예고**: 우리는 Scheduler 스케줄러로 깊이 들어가, 이 Sequence의 속성들을 어떻게 활용하여 효율적인 prefill 우선 스케줄링과 선점 메커니즘을 구현하는지 살펴볼 것입니다.

---

> **학습 제안**: Sequence가 생성부터 완료까지의 전 과정을 종이에 그려보고, 각 단계에서 각 속성의 변화를 표시해 보세요. 이 연습은 면접의 화이트보드 문제에 매우 도움이 됩니다.