# 강의 12: Scheduler 스케줄러

> **학습 목표**: nano-vllm 스케줄러의 완전한 작동 메커니즘을 깊이 이해합니다. waiting / running 이중 큐 모델을 숙달합니다. prefill 우선 스케줄링 전략과 decode 라운드 로빈 스케줄링의 설계 사상을 이해합니다. 선점(preempt) 메커니즘의 구현 디테일을 파악합니다. postprocess 후처리 흐름을 이해합니다. 면접에서 다양한 스케줄링 전략의 장단점을 비교 분석할 수 있습니다.

---

## 1. 스케줄러의 역할과 책임

### 1.1 왜 스케줄러가 필요할까

대규모 모델 추론 엔진은 보통 여러 사용자 요청을 동시에 서비스합니다. 각 요청은 서로 다른 시간에 도착하며, 서로 다른 길이의 prompt 처리와 서로 다른 수의 token 생성을 필요로 합니다. GPU의 메모리와 연산 능력은 유한하므로, 모든 요청을 동시에 처리할 수 없습니다.

스케줄러의 핵심 책임:

1. **각 단계마다 어떤 시퀀스를 실행할지 결정** (선택 + 정렬)
2. **prefill과 decode 단계 구분** (서로 다른 단계의 자원 특징이 뚜렷하게 다름)
3. **KV Cache 자원 관리** (BlockManager를 통해 물리 블록 할당 / 해제)
4. **자원 부족 처리** (낮은 우선순위의 시퀀스를 선점하고, 높은 우선순위의 시퀀스를 위해 블록 해제)
5. **후처리** (token 추가, 종료 조건 판단, 완료된 시퀀스 정리)

### 1.2 비유로 이해하기

스케줄러를 레스토랑의 지배인에 비유할 수 있습니다:

| 레스토랑 | 추론 엔진 |
|------|---------|
| 지배인 | Scheduler |
| 대기 중인 손님 | waiting 큐 |
| 식사 중인 손님 | running 큐 |
| 테이블 (유한 자원) | KV Cache 블록 |
| 자리 안내 | schedule() - 블록 할당 |
| 계산 재촉하여 자리 비우게 함 | preempt() - 선점 |
| 음식 서빙 + 식사 완료 확인 | postprocess() |

### 1.3 시스템 내 스케줄러의 위치

```
LLMEngine.step()  ← 엔진의 "심장 박동"
    │
    ├── scheduler.schedule()             ← 이번 단계에 참여할 시퀀스 선택 + 블록 할당
    │       ↓ (seqs, is_prefill) 반환
    ├── model_runner.run(seqs, is_prefill)  ← 순전파 추론
    │       ↓ token_ids 반환
    └── scheduler.postprocess(seqs, token_ids)  ← token 추가, 종료 판단
```

각 `step()` 호출은 schedule → run → postprocess의 한 라운드를 구성하며, 엔진의 **심장 박동 루프**를 이룹니다. 스케줄러는 이 루프의 **첫 번째 고리**로, 전체 시스템의 효율을 결정합니다.

---

## 2. Scheduler의 데이터 구조

### 2.1 생성자

소스 코드 경로: `nanovllm/engine/scheduler.py`

```python
class Scheduler:
    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
```

### 2.2 주요 속성 상세 설명

| 속성 | 타입 | 설명 |
|------|------|------|
| `max_num_seqs` | int | 한 단계에서 최대 처리할 수 있는 시퀀스 수, 배치 크기 제한. 동시에 너무 많은 요청을 처리하여 요청당 지연이 높아지는 것을 방지 |
| `max_num_batched_tokens` | int | 한 단계에서 최대 처리할 수 있는 토큰 총 수, 계산량 제한. GPU의 단일 단계 최대 작업 부하를 결정 |
| `eos` | int | EOS 토큰 ID, 생성이 자연스럽게 종료되었는지 판단하는 데 사용 |
| `block_manager` | BlockManager | KV Cache 물리 블록 관리자, 블록 할당 및 회수 책임 |
| `waiting` | deque[Sequence] | 대기 큐: 아직 시작되지 않았거나 선점된 시퀀스 보관 |
| `running` | deque[Sequence] | 실행 큐: 추론에 참여 중인 시퀀스 보관 |

### 2.3 왜 list 대신 deque를 사용할까

`deque`(양단 큐) vs `list`의 성능 비교:

| 연산 | deque | list |
|------|-------|------|
| 왼쪽 추가 `appendleft` | O(1) | O(n) |
| 왼쪽 팝 `popleft` | O(1) | O(n) |
| 오른쪽 추가 `append` | O(1) | 균등 분할 O(1) |
| 오른쪽 팝 `pop` | O(1) | O(1) |
| 임의 접근 `[i]` | O(n) | O(1) |
| 중간 삭제 `remove` | O(n) | O(n) |

스케줄러의 핵심 연산은 **큐의 앞에서 시퀀스를 꺼내는 것**과 **앞/뒤에 시퀀스를 추가하는 것**이며, 이러한 연산들은 deque에서 모두 O(1)입니다.

### 2.4 두 큐의 관계

```
                   schedule()
  ┌──────────┐    선택 및 할당    ┌──────────┐
  │ waiting  │──────────────→ │ running  │
  │   큐     │                │   큐     │
  └──────────┘                └──────────┘
       ↑                         │  │
       │      preempt()          │  │ postprocess()
       │      자원 부족으로 회귀   │  │ 완료 후 제거
       └─────────────────────────┘  │
                                    ↓
                              시퀀스 완료, running에서 제거
```

---

## 3. schedule() 메서드의 완전한 흐름

### 3.1 메서드 시그니처와 반환값

```python
def schedule(self):
    # 반환: (scheduled_seqs: list[Sequence], is_prefill: bool)
```

- `scheduled_seqs`: 이번 단계에 추론에 참여할 시퀀스 리스트
- `is_prefill`: True이면 이번 단계는 prefill 실행, False이면 decode 실행

### 3.2 핵심 설계 원칙: Prefill 우선

nano-vllm의 스케줄링 전략은 **prefill 우선**입니다: waiting 큐에 스케줄링 가능한 시퀀스가 있기만 하면, 그것들을 우선적으로 처리합니다(running 큐에 decode를 기다리는 시퀀스가 있더라도).

**왜 prefill이 우선일까?**

1. **사용자 경험**: 새 요청은 prefill을 먼저 완료해야 생성을 시작할 수 있으며, prefill이 빠를수록 사용자가 첫 토큰을 기다리는 시간(TTFT, Time To First Token)이 짧아집니다
2. **계산 효율**: prefill은 계산 집약적(compute-bound)이므로, GPU 연산 능력을 효율적으로 활용할 수 있습니다
3. **기아 방지**: decode가 우선되면, 새 요청이 장시간 처리되지 못할 수 있습니다

### 3.3 Prefill 스케줄링 단계 (소스 코드와 일치)

아래는 `nano-vllm-main/nanovllm/engine/scheduler.py`의 `schedule()` 중 **prefill 분기**와 일치합니다:

```python
def schedule(self):
    # ---------- prefill ----------
    scheduled_seqs = []
    num_seqs = 0
    num_batched_tokens = 0
    while self.waiting and num_seqs < self.max_num_seqs:
        seq = self.waiting[0]
        if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
            break
        num_seqs += 1
        self.block_manager.allocate(seq)
        num_batched_tokens += len(seq) - seq.num_cached_tokens
        seq.status = SequenceStatus.RUNNING
        self.waiting.popleft()
        self.running.append(seq)
        scheduled_seqs.append(seq)
    if scheduled_seqs:
        return scheduled_seqs, True
    # ---------- decode는 다음 절 참조 ----------
```

**줄별 해석**:

1. **`num_seqs = 0`、`num_batched_tokens = 0`**: 이번 단계에서 prefill 분기로 진입하면, 카운트는 **오직 waiting에서 이제 막 꺼내려는 시퀀스**부터 계산합니다; 이미 `running`에서 decode 중인 시퀀스는 이번 단계의 배치에 포함시키지 **않습니다**(다음 항목 참조).
2. **`len(seq)`**: waiting에 있는 시퀀스의 경우, 일반적으로 프롬프트 전체 길이(첫 번째 prefill)입니다; 만약 선점되어 waiting으로 돌아왔다면, 길이에는 이미 생성된 부분이 포함되며, 스케줄링 로직은 여전히 `len(seq)`와 `num_cached_tokens`를 BlockManager와 함께 사용합니다.
3. **`len(seq) - seq.num_cached_tokens`**: 프리픽스 캐시 적중 시, **캐시되지 않은** 토큰만 이번 단계의 계산량에 포함시켜, 실제 FLOPs / 메모리 쓰기에 부합하도록 합니다.
4. **두 개의 break 조건 (AND 연산을 한 줄로 풀어씀)**:
   - `max_num_batched_tokens`: 단일 단계 총 토큰 상한;
   - `can_allocate(seq)`: 물리 KV 블록이 전체 시퀀스의 현재 필요 블록 수를 충족하는지 여부.
5. **Prefill 우선**: `waiting`에 위 조건을 만족하는 시퀀스가 **있기만 하면**, 이 함수는 **바로** `..., True`를 반환하며, **이번 단계는 decode를 실행하지 않습니다**. 따라서 nano-vllm은 **단일 `schedule()` 호출 내에서** prefill과 decode를 혼합하지 않습니다; 연속 배치 처리는 **여러 번의** `step()` 교대 실행에 반영됩니다(먼저 waiting의 prefill을 한 무더기 실행한 후, 후속 단계에서 running의 decode를 실행).

### 3.4 흔한 오해: running의 길이를 prefill 카운트에 병합하지 않음

이전 버전 강의에서는 `num_seqs = len(self.running)`이라고 잘못 기술한 적이 있습니다. 현재 저장소 구현에서, **prefill 분기는** 이미 running에서 디코딩 중인 시퀀스를 `num_seqs` / `num_batched_tokens`에 포함시키지 **않습니다**. 만약 waiting이 비어 있지 않고 자원이 허용된다면, 이번 단계는 **오직** waiting → prefill만 스케줄링합니다. decode는 **waiting에서 이번 라운드에 어떤 시퀀스도 스케줄링할 수 없을 때**만, 다음 분기에서 처리됩니다.

### 3.5 Decode 스케줄링 단계 (소스 코드와 일치)

```python
    # ---------- decode（위의 prefill이 반환되지 않은 경우에만 실행）----------
    while self.running and num_seqs < self.max_num_seqs:
        seq = self.running.popleft()
        while not self.block_manager.can_append(seq):
            if self.running:
                self.preempt(self.running.pop())
            else:
                self.preempt(seq)
                break
        else:
            num_seqs += 1
            self.block_manager.may_append(seq)
            scheduled_seqs.append(seq)
    assert scheduled_seqs
    self.running.extendleft(reversed(scheduled_seqs))
    return scheduled_seqs, False
```

**줄별 해석**:

1. **`num_seqs`는 prefill 구간 종료 시의 값을 그대로 사용**: prefill에서 어떤 시퀀스도 스케줄링되지 않았다면, 여기서는 여전히 `0`입니다. 미래에 코드가 변경되어 두 구간이 카운트를 공유하게 된다면, 저장소를 기준으로 삼아야 합니다.
2. **`self.running.popleft()`**: running **큐의 앞쪽**에서 시퀀스를 가져와, **FIFO** 라운드 로빈 디코딩을 보장합니다.
3. **내부 `while not can_append`**: KV를 더 이상 append할 수 없을 때(예: 블록 부족), **LIFO** `running.pop()`으로 **큐의 뒤쪽** 시퀀스를 선점하며, append가 가능해지거나 현재 `seq`만 선점하고 `break`할 때까지 반복합니다.
4. **`while ... else`**: 내부 while이 `break`로 빠져나가지 **않은** 경우에만 `else`로 진입합니다: `may_append`를 실행하고, `scheduled_seqs`에 `append`하며, `num_seqs += 1`합니다.
5. **`extendleft(reversed(scheduled_seqs))`**: 이번 단계에서 선택된 시퀀스들을 원래의 상대적 순서대로 **running 큐의 앞쪽**에 다시 넣어, 다음 단계에서도 여전히 큐의 앞쪽에서 꺼내어 **라운드 로빈**을 구현합니다.
6. **`assert scheduled_seqs`**: decode 경로는 최소한 하나의 시퀀스가 있어야 합니다; running이 비어 있지 않은데 자원 교착 상태로 인해 실패한다면, assertion이 실패할 수 있습니다—이는 용량 설정 문제이며, 면접에서 언급할 수 있습니다.

### 3.6 while...else 구문 상세 해설

이것은 파이썬에서 흔하지는 않지만 매우 우아한 구문입니다:

```python
while condition:
    ...
    if some_check:
        break
else:
    # while이 정상 종료(condition이 False가 됨)된 경우에만 실행
    # break로 빠져나간 경우에는 실행되지 않음
    ...
```

decode 스케줄링에서:
- 만약 충분한 공간을 성공적으로 확보했다면(`can_append`가 True가 되어 while 정상 종료) → else 실행, 시퀀스를 스케줄링에 추가
- 만약 선점할 수 있는 시퀀스가 더 이상 없다면(break 탈출) → else 실행 안 함, 해당 시퀀스는 이번 단계에 참여하지 않음

---

## 4. 선점 메커니즘 (Preempt)

### 4.1 왜 선점이 필요할까

decode 단계에서, 각 시퀀스는 단계마다 하나의 토큰을 생성하며, KV Cache에 새로운 KV 쌍을 기록해야 합니다. 어떤 시퀀스의 마지막 블록이 가득 찼다면, 새로운 물리 블록을 할당해야 합니다. 그러나 물리 블록은 유한합니다—이미 다 소진되었다면, 다른 시퀀스를 **선점(preempt)**하여 블록을 해제해야 합니다.

### 4.2 preempt() 소스 코드

```python
def preempt(self, seq):
    seq.status = SequenceStatus.WAITING
    self.block_manager.deallocate(seq)
    self.waiting.appendleft(seq)
```

세 단계 연산:

1. **상태 회귀**: 시퀀스 상태를 RUNNING에서 WAITING으로 변경
2. **자원 해제**: BlockManager를 통해 해당 시퀀스가 점유한 모든 물리 블록 해제
3. **재대기**: `appendleft`를 사용하여 시퀀스를 waiting 큐의 **앞쪽**에 배치

### 4.3 왜 appendleft를 사용할까

선점된 시퀀스는 `append`(뒤쪽에 배치) 대신 `appendleft`(앞쪽에 배치)를 사용하여, **공정성을 보장**합니다:

- 선점된 시퀀스는 이미 일정 시간 대기했으므로, 새 요청보다 뒤로 밀려서는 안 됩니다
- 앞쪽에 배치하면 다음 라운드 스케줄링에서 **우선적으로 재스케줄링**됩니다
- 이는 "기아" 문제를 방지합니다—어떤 시퀀스가 반복적으로 선점되면서 영원히 완료되지 못하는 현상

### 4.4 선점 전략: LIFO

```python
self.preempt(self.running.pop())  # pop()은 뒤쪽에서 꺼냄
```

nano-vllm은 **LIFO (Last In First Out)** 선점 전략을 사용합니다—running 큐에 가장 나중에 추가된 시퀀스가 가장 먼저 선점됩니다.

**왜 LIFO를 선택할까?**

1. **낭비 최소화**: 가장 나중에 추가된 시퀀스는 이제 막 생성을 시작했을 수 있으며, 이를 선점하면 낭비되는 계산량이 가장 적습니다
2. **자원 해제량이 큼**: 가장 나중에 추가된 시퀀스에 긴 프롬프트가 있다면, 그 블록이 더 많아 해제 후 공간 요구를 충족할 가능성이 더 높습니다
3. **단순하고 효율적**: deque.pop()은 O(1) 연산입니다

### 4.5 선점의 대가

선점은 공짜가 아닙니다:

```
시퀀스 A가 running에 있고, prefill 완료(1000 토큰), 200개의 토큰 생성
  → ceil(1200/256) = 5개의 블록 점유

A를 선점:
  → 5개 블록 해제 (KV Cache 전부 손실)
  → A는 waiting 큐로 복귀
  → A를 재스케줄링할 때, 1000 토큰의 prefill을 다시 수행해야 함
  → 이전의 모든 계산이 낭비됨
```

이것이 nano-vllm의 간소화된 설계—**recompute 전략**—입니다: 선점된 시퀀스는 완전히 재계산해야 합니다. vLLM의 완전한 버전에는 **swap 전략**도 있습니다: KV Cache를 GPU에서 CPU 메모리로 교환하여, 중복 계산을 방지합니다.

### 4.6 자신을 선점하는 시나리오

```python
if self.running:
    self.preempt(self.running.pop())
else:
    self.preempt(seq)  # 자기 자신을 선점
    break
```

running 큐가 비어 있고(다른 모든 시퀀스가 선점됨), 현재 시퀀스도 여전히 충분한 블록을 얻을 수 없을 때는, **자기 자신을 선점**할 수밖에 없습니다. 이러한 상황은 다음을 의미합니다:

- 해당 시퀀스가 필요로 하는 블록 수가 시스템 총 용량을 초과합니다
- 또는 심각한 메모리 파편화가 존재합니다

자기 자신을 선점한 후, 시퀀스는 waiting 큐로 돌아가, 다른 시퀀스가 완료되어 더 많은 블록을 해제할 때까지 기다립니다.

---

## 5. Decode 스케줄링의 자원 검사

### 5.1 can_append vs can_allocate

| 메서드 | 호출 시점 | 검사 내용 |
|------|---------|---------|
| `can_allocate(seq)` | Prefill 스케줄링 | 시퀀스에 `num_blocks`개의 새 블록이 필요하며, 충분한 빈 블록이 있는지 |
| `can_append(seq)` | Decode 스케줄링 | 시퀀스의 마지막 블록이 가득 찼는지, 가득 찼다면 1개의 빈 블록이 있는지 |

### 5.2 may_append의 조건부 할당

```python
def may_append(self, seq):
    if seq.last_block_num_tokens == self.block_size:
        # 마지막 블록이 가득 참, 새 블록 할당 필요
        new_block = self.allocate_block()
        seq.block_table.append(new_block)
```

마지막 블록이 정확히 가득 찼을 때만, 새로 할당이 필요합니다. 대부분의 경우, 마지막 블록에는 아직 빈 자리가 있어, 아무런 작업도 필요하지 않습니다.

### 5.3 Decode 스케줄링의 완전한 흐름도

```
decode 스케줄링 시작
    │
    ▼
running 큐 앞쪽에서 시퀀스 seq 가져오기
    │
    ▼
can_append(seq)?
    │
    ├── Yes ──→ may_append(seq) → scheduled_seqs에 추가 → 다음 시퀀스 가져오기
    │
    └── No ──→ running에 다른 시퀀스가 있는가?
                    │
                    ├── Yes ──→ preempt(running.pop()) → can_append 재검사
                    │
                    └── No ──→ preempt(seq) → break（이번 단계에서 이 시퀀스를 스케줄링할 수 없음）
```

---

## 6. postprocess 후처리

### 6.1 소스 코드 해석

```python
def postprocess(self, seqs, token_ids):
    for seq, token_id in zip(seqs, token_ids):
        seq.append_token(token_id)
        
        if (not seq.ignore_eos and token_id == self.eos) or \
           seq.num_completion_tokens == seq.max_tokens:
            seq.status = SequenceStatus.FINISHED
            self.block_manager.deallocate(seq)
            self.running.remove(seq)
```

### 6.2 처리 흐름

이번 단계에 추론에 참여한 각 시퀀스에 대해:

1. **새 토큰 추가**: `seq.append_token(token_id)`를 호출하여 시퀀스 상태 갱신
2. **종료 조건 검사**:
   - 조건 1: EOS 토큰을 만나고 `ignore_eos`가 설정되지 않음
   - 조건 2: 이미 생성된 completion 토큰 수가 `max_tokens` 상한에 도달
3. **종료 시**:
   - 상태를 `FINISHED`로 표시
   - 해당 시퀀스의 모든 KV Cache 블록 해제
   - running 큐에서 제거

### 6.3 종료 조건의 논리식

```python
(not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens
```

진리표로 분석:

| ignore_eos | token == EOS | completion == max | 종료 여부 |
|-----------|-------------|------------------|---------|
| False | True | - | **예**（자연 종료） |
| False | False | True | **예**（상한 도달） |
| False | False | False | 아니오 |
| True | True | False | 아니오（EOS 무시됨） |
| True | True | True | **예**（상한 도달） |
| True | False | True | **예**（상한 도달） |
| True | False | False | 아니오 |

핵심 논리: `max_tokens`는 하드 상한으로, 어떤 경우에도 초과할 수 없습니다; `EOS`는 소프트 종료로, `ignore_eos`에 의해 덮어쓰기될 수 있습니다.

### 6.4 왜 `self.running.remove(seq)`를 사용할까

완료된 시퀀스가 반드시 큐의 앞쪽에 있는 것은 아니기 때문입니다—배치 내 어떤 위치의 시퀀스든 먼저 완료될 수 있습니다. `remove()`는 값에 따라 찾아 삭제하며, 시간 복잡도는 O(n)이지만, running 큐는 보통 매우 짧기 때문에(수십 개의 시퀀스), 성능 병목이 되지 않습니다.

### 6.5 postprocess는 waiting 큐의 시퀀스를 처리하지 않음

`postprocess`는 이번 단계에 추론에 참여한 시퀀스만 처리하며(`seqs` 파라미터), waiting 큐에는 손대지 않습니다. 이는 waiting에 있는 시퀀스는 추론에 참여하지 않았으므로, 새 토큰이 생성되지 않기 때문입니다.

---

## 7. add 메서드

### 7.1 소스 코드

```python
def add(self, seq: Sequence):
    self.waiting.append(seq)
```

극도로 단순합니다: 새 시퀀스를 waiting 큐의 뒤쪽에 추가합니다. FIFO 순서는 먼저 도착한 요청이 먼저 처리되도록 보장합니다.

### 7.2 add 호출 시점

```python
# LLMEngine 내
def add_request(self, prompt, sampling_params=SamplingParams()):
    token_ids = self.tokenizer.encode(prompt)
    seq = Sequence(token_ids, sampling_params)
    self.scheduler.add(seq)
    return seq
```

### 7.3 is_running / has_unfinished 속성

```python
@property
def is_running(self):
    return bool(self.running)

@property
def has_unfinished(self):
    return bool(self.waiting) or bool(self.running)
```

- `is_running`: 실행 중인 시퀀스가 있는지 여부 (`LLMEngine`이 계속 스텝을 진행해야 하는지 판단하는 데 사용)
- `has_unfinished`: 완료되지 않은 시퀀스가 있는지 여부 (대기 중인 것과 실행 중인 것 포함)

---

## 8. 스케줄러의 완전한 생애주기 예시

### 8.1 시나리오 설정

시스템 파라미터 가정:
- `max_num_seqs = 4`
- `max_num_batched_tokens = 1024`
- `block_size = 256`
- 총 10개의 물리 블록

### 8.2 실행 타임라인

```
T=0: 요청 A 도착 (prompt 300 토큰)
     waiting: [A(300)]
     running: []

T=1: schedule() — prefill 단계
     A는 2개의 블록 필요, 할당 성공
     waiting: []
     running: [A]
     → ([A], is_prefill=True) 반환
     → ModelRunner가 A의 prefill 실행

T=1: postprocess()
     A가 token_301 생성
     A 미종료
     running: [A(301)]

T=2: 요청 B 도착 (prompt 500 토큰)
     waiting: [B(500)]
     running: [A(301)]

T=2: schedule() — prefill 우선
     B는 2개의 블록 필요, 할당 성공
     waiting: []
     running: [A(301), B(500)]
     → ([B], is_prefill=True) 반환

T=2: postprocess()
     B가 token_501 생성
     running: [A(301), B(501)]

T=3: schedule() — waiting 없음, decode 진입
     A: can_append? Yes (블록이 가득 차지 않음)
     B: can_append? Yes (블록이 가득 차지 않음)
     → ([A, B], is_prefill=False) 반환

T=3: postprocess()
     A가 token_302 생성, B가 token_502 생성
     running: [A(302), B(502)]

... 정상 decode 계속 ...

T=10: A가 EOS 만남
      postprocess: A.status = FINISHED, 2개 블록 해제
      running: [B(510)]

T=11: 요청 C 도착
      waiting: [C]
      schedule(): prefill 우선, C 스케줄링
```

---

## 9. 스케줄링 전략의 심층 비교

### 9.1 FCFS (선착순)

nano-vllm의 waiting 큐는 본질적으로 FCFS입니다—먼저 도착한 요청이 먼저 스케줄링됩니다.

**장점**: 단순하고 공정함
**단점**: 요청 우선순위를 구분할 수 없으며, 긴 프롬프트가 짧은 프롬프트를 막을 수 있음

### 9.2 Prefill 우선 vs Decode 우선

| 전략 | TTFT | TPOT | 처리량 | 구현 복잡도 |
|------|------|------|--------|-----------|
| Prefill 우선 | 낮음 | 높을 수 있음 | 중간 | 낮음 |
| Decode 우선 | 높음 | 낮음 | 중간 | 낮음 |
| 혼합 스케줄링 | 중간 | 중간 | 높음 | 높음 |

- **TTFT** (Time To First Token): 사용자가 첫 출력 토큰을 기다리는 시간
- **TPOT** (Time Per Output Token): 각 출력 토큰의 생성 시간

nano-vllm이 prefill 우선을 선택한 이유는 TTFT가 사용자 경험에 더 큰 영향을 미치기 때문입니다—사용자는 "응답 속도"보다 "언제부터 응답이 시작되는지"를 더 신경 씁니다.

### 9.3 chunked prefill

더 발전된 시스템에서는, 긴 프롬프트의 prefill을 여러 개의 chunk로 나누어, decode 시퀀스와 교대로 실행할 수 있습니다. 이렇게 하면 긴 프롬프트가 decode 시퀀스를 막지 않으면서도, 낮은 TTFT를 유지할 수 있습니다.

```
전통적 prefill 우선:
Step 1: [A(prefill 2000 tokens)]  ← decode 시퀀스가 막힘
Step 2: [B(decode), C(decode)]

chunked prefill:
Step 1: [A(prefill chunk1 512 tokens), B(decode), C(decode)]
Step 2: [A(prefill chunk2 512 tokens), B(decode), C(decode)]
Step 3: [A(prefill chunk3 512 tokens), B(decode), C(decode)]
Step 4: [A(prefill chunk4 464 tokens), B(decode), C(decode)]
```

nano-vllm의 간소화된 버전은 chunked prefill을 구현하지 않았습니다.

### 9.4 우선순위 스케줄링 (Priority scheduling)

프로덕션 환경에서는, 서로 다른 사용자/요청이 서로 다른 우선순위를 가질 수 있습니다(예: 유료 사용자 > 무료 사용자). 이는 waiting 큐에서 단순한 FIFO 대신 우선순위 큐(예: heap)를 구현해야 함을 의미합니다.

---

## 10. BlockManager 상호작용

### 10.1 스케줄러와 BlockManager의 협업

```python
# 스케줄러는 물리 블록을 직접 관리하지 않고, BlockManager에 위임합니다

# Prefill 시:
self.block_manager.can_allocate(seq)   # 질문: 이 시퀀스에 충분한 블록이 있는가?
self.block_manager.allocate(seq)       # 실행: 블록 할당 및 seq.block_table 채움

# Decode 시:
self.block_manager.can_append(seq)     # 질문: 토큰 하나를 추가할 수 있는가?
self.block_manager.may_append(seq)     # 실행: 필요 시, 새 블록 할당

# 선점/완료 시:
self.block_manager.deallocate(seq)     # 실행: 해당 시퀀스의 모든 블록 해제
```

### 10.2 자원 관리의 2단계 검사

nano-vllm은 **먼저 검사하고 나중에 실행하는** 패턴을 채택했습니다:

1. `can_xxx` 메서드: 읽기 전용 쿼리, 상태를 수정하지 않음
2. `allocate/deallocate/may_append` 메서드: 실제 자원 연산 수행

이러한 설계는 스케줄러가 의사 결정 단계에서 안전하게 자원 상황을 "탐색"할 수 있게 하며, 검사 연산으로 인한 부작용이 발생하지 않도록 합니다.

---

## 11. 스케줄러의 핵심 설계 원칙

### 11.1 Prefill과 Decode 혼합 금지

동일 단계에서는, 모두 prefill을 하거나 모두 decode를 합니다. 이는 ModelRunner의 구현을 단순화합니다—동일한 배치에서 서로 다른 두 가지 계산 모드를 혼합할 필요가 없습니다.

```python
if scheduled_seqs:
    return scheduled_seqs, True   # prefill이 있음, 이번 단계는 prefill만 수행
# ...
return scheduled_seqs, False      # 그렇지 않으면 decode 수행
```

하지만 이는 decode 중인 시퀀스가 새 prefill 요청이 도착했을 때 **한 단계 멈추게** 된다는 의미이기도 합니다.

### 11.2 보수적 스케줄링

스케줄러는 보수적인 경향이 있습니다—시퀀스를 몇 개 덜 스케줄링하는 한이 있더라도, 자원 부족으로 인한 시스템 붕괴를 초래하지 않습니다:

- 토큰 수 검사: `num_batched_tokens + len(seq) > self.max_num_batched_tokens` → break
- 블록 검사: `not self.block_manager.can_allocate(seq)` → break

일단 스케줄링할 수 없는 시퀀스를 만나면, 후속 시퀀스를 스케줄링하는 것을 즉시 중단합니다. 비록 후속 시퀀스가 더 작을 수 있어도 말이죠. 이는 FCFS의 특성입니다—큐의 앞쪽에 있는 큰 요청을 건너뛰고 뒤쪽의 작은 요청을 먼저 스케줄링하지 않습니다.

### 11.3 단일 단계 원자성

각 `schedule()` 호출은 하나의 완전한 스케줄링 결과를 생성합니다. 스케줄링 과정 중의 모든 연산(블록 할당, 상태 수정, 큐 이동)은 전부 성공하거나 롤백해야 합니다. nano-vllm에서는 단일 스레드 실행이므로, 동시성 문제가 발생하지 않습니다.

---

## 12. 스케줄러의 잠재적 개선점

### 12.1 Swap 지원 (CPU로 교환)

현재의 preempt 전략은 **전량 해제**(recompute)이며, 선점된 시퀀스는 prefill을 다시 수행해야 합니다. 개선 방안은 KV Cache를 GPU에서 CPU 메모리로 교환하여, 후속 복구 시 CPU에서 GPU로 다시 전송하기만 하면 되므로, 재계산을 방지합니다.

### 12.2 우선순위 스케줄링 지원

waiting 큐를 deque에서 우선순위 큐로 변경하여, 우선순위 또는 대기 시간에 기반한 스케줄링을 지원합니다.

### 12.3 Chunked Prefill 지원

긴 프롬프트의 prefill을 분할하여, decode와 교대로 실행함으로써, TTFT와 TPOT의 균형을 맞춥니다.

### 12.4 Speculative Decoding 지원

투기적 디코딩은 스케줄러가 "초안 모델 + 검증"의 2단계 실행 모드를 지원해야 합니다.

---

## 13. 소스 코드 대조 요약

| Scheduler 메서드/속성 | 호출자 | 목적 |
|---------------------|--------|------|
| `__init__` | LLMEngine | 스케줄러와 BlockManager 초기화 |
| `add(seq)` | LLMEngine.add_request | 새 시퀀스 큐잉 |
| `schedule()` | LLMEngine.step | 이번 단계의 시퀀스 선택, (seqs, is_prefill) 반환 |
| `postprocess(seqs, token_ids)` | LLMEngine.step | 토큰 추가, 종료 판단 |
| `preempt(seq)` | schedule() 내부 | 시퀀스를 선점하여 자원 해제 |
| `is_running` | LLMEngine | 활성 시퀀스 여부 판단 |
| `has_unfinished` | LLMEngine | 모든 작업 완료 여부 판단 |

---

## 14. 면접 출제 포인트

### 포인트 1: nano-vllm 스케줄러의 schedule() 메서드의 완전한 흐름을 설명하세요.

**표준 답변**: schedule()은 두 단계로 실행됩니다. 첫 번째 단계는 waiting 큐에서 prefill 시퀀스를 스케줄링하려고 시도합니다: FCFS 순서로 하나씩 꺼내면서, 토큰 수 제한과 블록 가용성을 검사하고, 검사를 통과하면 블록을 할당하고, 상태를 RUNNING으로 수정하고, running 큐로 이동시킵니다. prefill 시퀀스 스케줄링에 성공하면, 바로 반환합니다. 두 번째 단계는 decode를 처리합니다: running 큐에서 하나씩 꺼내면서, 새 토큰을 추가할 수 있는지(새 블록이 필요할 수 있음) 검사하고, 블록이 부족하면 LIFO 전략으로 다른 시퀀스를 선점하여 자원을 해제합니다.

### 포인트 2: 왜 prefill 우선 전략을 채택했나요? 이 전략의 장단점은 무엇인가요?

**표준 답변**: Prefill 우선은 TTFT(첫 토큰 지연)를 낮추어, 사용자 경험을 향상시킵니다. Prefill은 compute-bound 연산이므로, GPU 연산 능력을 충분히 활용할 수 있습니다. 단점은 decode 중인 시퀀스가 새 prefill 요청이 있을 때 한 단계 멈추게 되어, TPOT가 증가한다는 것입니다. 개선 방안은 chunked prefill로, 긴 프롬프트를 분할하여 decode와 교대로 실행하는 것입니다.

### 포인트 3: 선점(preempt) 메커니즘의 구현을 설명하고, 왜 LIFO 전략을 사용하는지 설명하세요.

**표준 답변**: decode 단계에서 새 블록이 필요하지만 빈 블록이 없을 때, 스케줄러는 running 큐의 뒤쪽에서 시퀀스를 꺼내 선점합니다—모든 KV Cache 블록을 해제하고, 상태를 WAITING으로 되돌리며, waiting 큐의 앞쪽에 배치합니다. LIFO 전략을 사용하는 이유는 가장 나중에 추가된 시퀀스가 생성한 토큰이 가장 적어, 선점 시 낭비되는 계산량이 가장 적기 때문입니다. 선점된 시퀀스를 waiting의 앞쪽에 배치하는 것은 공정성을 보장하고, 기아를 방지하기 위함입니다.

### 포인트 4: nano-vllm의 선점 전략은 recompute이며, swap 전략과는 어떤 차이가 있나요?

**표준 답변**: Recompute 전략은 선점된 시퀀스의 KV Cache를 직접 폐기하며, 재스케줄링 시 prefill을 다시 수행해야 하므로, 계산 낭비가 크지만 구현이 단순하고 추가 메모리가 필요하지 않습니다. Swap 전략은 KV Cache를 GPU에서 CPU 메모리로 교환하며, 복구 시에는 GPU로 다시 전송하기만 하면 되므로, 중복 계산을 방지하지만, 추가적인 CPU 메모리와 PCIe 대역폭이 필요하고, 구현도 더 복잡합니다. vLLM은 두 전략을 동시에 지원합니다.

### 포인트 5: postprocess의 종료 조건은 무엇인가요? ignore_eos는 어떻게 처리하나요?

**표준 답변**: 종료 조건은 두 가지입니다: (1) EOS 토큰을 생성했고 `ignore_eos=False`, (2) 이미 생성된 completion 토큰 수가 `max_tokens` 제한에 도달. `ignore_eos=True`이면 EOS 검사를 건너뛰고, `max_tokens`까지 강제로 생성합니다. `max_tokens`는 하드 상한으로, `ignore_eos` 설정 여부와 관계없이 효력이 발생합니다.

### 포인트 6: nano-vllm 스케줄러를 개선한다면, 어떤 측면에서 접근하시겠습니까?

**참고 방향**:
1. chunked prefill 구현, TTFT와 TPOT의 균형
2. swap 전략 추가로 선점 낭비 감소
3. 우선순위 스케줄링 지원 (priority queue)
4. prefix-aware 스케줄링 지원, 프리픽스를 공유하는 요청을 함께 스케줄링하여 캐시 적중 극대화
5. 투기적 디코딩의 2단계 스케줄링 지원
6. 공정성 보장 추가 (대기 시간 기반 우선순위 상승)

### 포인트 7: 왜 prefill과 decode를 동일 단계에 혼합 실행하지 않나요?

**표준 답변**: Prefill과 decode의 계산 모드는 다릅니다—prefill은 여러 개의 연속 토큰을 처리하며, 가변 길이 시퀀스의 어텐션 계산을 사용합니다; decode는 시퀀스당 하나의 토큰만 처리하며, KV Cache 가속 어텐션 계산을 사용합니다. 혼합 실행하지 않으면 ModelRunner의 구현과 CUDA 커널 선택이 단순해집니다. 실제로 vLLM과 같은 고성능 시스템은 GPU 활용률을 높이기 위해 이미 혼합 실행을 지원합니다.

---

## 15. 요약

| 지식 포인트 | 핵심 이해 |
|--------|---------|
| 스케줄러 역할 | 엔진의 "두뇌", 각 단계마다 누가 계산에 참여할지 결정 |
| 이중 큐 모델 | waiting(대기) + running(실행), deque로 구현 |
| Prefill 우선 | 새 요청 우선 처리, TTFT 감소 |
| Decode 라운드 로빈 | FIFO로 running 내 시퀀스 순차 처리 |
| 선점 메커니즘 | LIFO 전략 + recompute 전략, 선점된 시퀀스는 waiting 앞쪽으로 복귀 |
| postprocess | 토큰 추가 + 종료 검사 + 자원 회수 |
| 자원 관리 | 스케줄러는 블록을 직접 관리하지 않고, BlockManager에 위임 |

**다음 강의 예고**: 우리는 스케줄러의 시점에서 시스템 수준으로 올라가, **연속 배치 처리(Continuous Batching)**—스케줄러와 ModelRunner가 협업하여 구현하는 핵심 최적화 전략—를 깊이 이해할 것입니다.

---

> **학습 제안**: 종이 위에서 3-4개의 요청이 있는 스케줄링 시나리오를 시뮬레이션하며, 각 단계마다 waiting과 running 큐의 변화, 그리고 블록의 할당과 해제 과정을 그려보세요. 이는 스케줄러를 이해하는 데 매우 중요합니다.