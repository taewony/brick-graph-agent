# 강의 10: PagedAttention과 BlockManager

> 운영체제가 물리 페이지 프레임을 가상 페이지에 할당하듯, **KV Cache 풀**을 고정 블록으로 나누고, **블록 테이블**로 각 시퀀스의 논리 주소 매핑을 기술하며, **해시 프리픽스 재사용**으로 프롬프트를 공유하여, 다중 요청 동시 처리 시 메모리를 절약하면서도 적중률을 높입니다.

## 본 강의 목표

- **가상 메모리 페이징 비유**로 PagedAttention이 해결하는 문제(파편화, 공유, 필요 기반 할당)를 이해합니다.
- **`Block`**, **`block_table`**, **`free_block_ids` / `used_block_ids`** 의 역할을 숙달합니다.
- **`xxhash` + prefix**가 어떻게 '전체 블록 토큰'에 대해 재사용 가능한 지문을 생성하는지 이해합니다.
- **`allocate`**의 cache hit / miss 분기, 그리고 **`may_append`**가 언제 블록을 확장하고 언제 블록에 해시를 기록하는지 구술할 수 있습니다.
- **`ref_count`**와 **deallocate**의 참조 카운팅 의미 및 면접 확장을 설명합니다.

## 핵심 개념

### 1. '대형 연속 KV 텐서' 위에 PagedAttention이 추가로 필요한 이유

이전 강의의 `allocate_kv_cache`는 **물리 풀**(블록 수 × 블록 크기 × 레이어 × …)을 할당했습니다. 각 시퀀스가 **연속된 고정 길이** 구간을 점유한다면 다음과 같은 문제가 발생합니다:

- **내부 파편화**: 시퀀스 길이가 블록 크기의 배수가 아닐 때 꼬리 부분 낭비;
- **외부 파편화/스케줄링 어려움**: 여러 시퀀스를 동적으로 추가·삭제할 때 빽빽하게 재사용하기 어려움;
- **프리픽스 공유 어려움**: 동일한 시스템 프롬프트에 대해 여러 요청이 각자 KV를 복사하면 메모리 낭비.

PagedAttention의 사고방식: **논리적으로** 블록 테이블로 'i번째 블록 토큰이 어떤 물리 `block_id`에 위치하는지'를 기록하고, 물리 블록은 풀 내의 임의의 빈 슬롯에서 가져올 수 있습니다. 필요에 따라 여러 시퀀스가 **공유**하는 동일 물리 블록(읽기 전용 공유 + 참조 카운팅).

### 2. Block: 물리 슬롯의 메타데이터

각 `Block`은 다음을 가집니다:

- **`block_id`**: 풀 내의 인덱스;
- **`ref_count`**: 몇 개의 시퀀스/논리적 참조가 이 블록을 사용하는지 (0이 되면 `free_block_ids`로 회수 가능);
- **`hash`**: 블록이 가득 찬 후 `(token_ids, prefix_hash)`로 계산된 다이제스트, `-1`은 봉인되지 않았거나 가득 차지 않았음을 의미;
- **`token_ids`**: 이 블록이 현재 담고 있는 토큰 리스트 (해시 테이블 충돌이나 잘못된 적중을 검증하는 데 사용).

### 3. `hash_to_block_id`: 전역 해시 테이블

**내용 해시**에서 **물리 블록 id**로 매핑하여, "동일한 토큰 시퀀스를 이전에 본 적 있고 블록 내용이 일치하면 물리 블록을 재사용"을 구현합니다.

### 4. xxhash와 `prefix` 파라미터

```python
@classmethod
def compute_hash(cls, token_ids, prefix=-1):
    h = xxhash.xxh64()
    if prefix != -1:
        h.update(prefix.to_bytes(8, "little"))
    h.update(np.array(token_ids).tobytes())
    return h.intdigest()
```

- **`prefix`**: 보통 **이전 블록의 해시**(체인 방식). 이렇게 하면 전체 블록 내용이 이 블록의 토큰뿐만 아니라 윗문장에도 의존하게 되어, '다른 컨텍스트에서 우연히 토큰이 동일한' 충돌 위험을 낮춥니다(단순히 이 블록의 bare 토큰만 해시하는 것보다 안정적).
- **xxhash를 사용하는 이유**: 극도로 빠르며, 비암호학적 해시로 충분; 런타임 키에 적합.

### 5. `allocate`: 전체 시퀀스의 초기 점유

사용자가 제공한 소스 코드 로직(`nanovllm/engine/block_manager.py`와 일치)의 요점:

- 시퀀스의 **각 블록 인덱스** `i`에 대해, `seq.block(i)`의 `token_ids`를 꺼냅니다.
- **오직** `len(token_ids) == block_size`일 때만, 체인 해시 `h`를 계산합니다; 그렇지 않으면 `h = -1`(가득 차지 않은 블록은 전역 재사용 인덱스에 참여하지 않음).
- `hash_to_block_id.get(h)`를 조회; 존재하지 않거나 블록 내 `token_ids`가 불일치 → **cache miss**, `free_block_ids`에서 새 물리 블록을 가져옵니다.
- **cache hit** 시: `num_cached_tokens`를 증가시키고(적중한 전체 블록의 토큰 수), **참조 카운트를 증가**시킬 수 있습니다(블록 공유).
- 마지막으로 선택된 `block_id`를 `seq.block_table`에 추가합니다.

### 6. `deallocate`: 역순 참조 해제

`block_table`을 **역방향**으로 순회하며, 각 단계마다 `ref_count -= 1`, 0이 되면 `_deallocate_block`으로 빈 큐에 반환합니다. 역순은 할당 순서와 대칭되어 디버깅 및 특정 일관성 제약에 용이합니다.

### 7. `can_append`와 `may_append`: 디코딩 성장

```python
def can_append(self, seq):
    return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)
```

파이썬에서 `(len(seq) % self.block_size == 1)`은 불리언 값입니다; 정수와 비교할 때 `True`는 `1`, `False`는 `0`으로 간주됩니다.

- **길이를 블록 크기로 나눈 나머지가 1**일 때, 다음 `may_append` 단계에서 **새 물리 블록이 필요**하므로(아래 `may_append` 분기 참조), `free_block_ids`가 최소 **1개** 필요합니다.
- 그렇지 않으면 새 블록을 예약할 필요가 없으며, `>= 0`은 항상 성립합니다(빈 블록 수가 음이 아닌 경우).

**`may_append`**(시퀀스에 이미 하나의 '현재 꼬리 블록'이 있음):

- **`len(seq) % block_size == 1`**: 방금 새 블록 경계 상태에 진입, **assert**로 이전 블록에 유효한 `hash`가 있음을 단언, 빈 큐에서 **새로운 `block_id`를 하나 더 할당**하여 `block_table`에 추가합니다(새 블록을 먼저 점유하고, 후속 토큰을 기록).
- **`len(seq) % block_size == 0`**: 블록 하나를 꼭 맞게 채움, **이 블록**의 `token_ids`를 가져와, **이전 블록**의 `hash`를 `prefix`로 사용하여 `h`를 계산, `last_block.update(h, token_ids)`로 기록하고 `hash_to_block_id`에 등록.
- **그 외의 경우**: 꼬리 블록이 가득 차지 않음, `assert last_block.hash == -1`(가득 차지 않은 블록은 전역 재사용 해시를 기록하지 않음).

(구체적인 내용은 `Sequence.block`의 인덱싱 규칙과 결합하여 읽으세요.)

### 8. 운영체제 페이징과의 비유

| OS 개념 | PagedAttention |
|--------|----------------|
| 물리 페이지 프레임 | `Block(block_id)` 슬롯 |
| 가상 페이지 테이블 | `seq.block_table` |
| 페이지 공유 (공유 라이브러리) | 동일 프리픽스 토큰 블록 해시 적중, 여러 시퀀스 `ref_count++` |
| 페이지 부재 할당 | cache miss → `_allocate_block` |

---

## 소스 코드 분석

### `Block.reset`과 `_allocate_block`

소스 코드에서 `_allocate_block`은 `block.reset()`을 호출합니다: `ref_count = 1`, `hash`와 `token_ids`를 비우고, `free_block_ids`에서 제거하여 `used_block_ids`에 추가합니다.

### `can_allocate`

(전체 소스 코드에는 `can_allocate`도 있습니다: 빈 블록이 **시퀀스에 필요한 블록 수** `seq.num_blocks`보다 **적지 않은지** 확인하여, 스케줄러가 새 요청을 수락하기 전에 수용 가능 여부를 판단하는 데 사용됩니다.)

### 참조 카운트가 +1되는 시점

- **allocate 적중**이고 해당 블록이 **이미 used 집합에 있음**: 여러 시퀀스가 공유 중임을 의미, `ref_count += 1`.
- **신규 할당**: `reset`이 `ref_count`를 1로 설정.

---

## 그림 설명

### allocate: cache hit vs miss (간략화)

```text
[각 블록 i 순회] → {len token_ids == block_size?}
    → 아니오: h = -1, 보통 miss 경로로 새 블록 점유
    → 예: 체인 해시 h 계산
        → {해시 적중이고 토큰 일치?}
            → 예: block_id 재사용, ref++ / num_cached_tokens += block_size
            → 아니오: free에서 새 블록 가져옴
    → block_table에 추가
```

### may_append: 세 가지 모듈로 경우 (개념)

```text
len % B == 1  -->  새 물리 블록 필요, block_table 확장
len % B == 0  -->  가득 찬 블록, hash 기록 및 hash_to_block_id 등록
그 외           -->  가득 차지 않은 블록, 전역 hash 등록 안 함
```

---

## 면접 출제 포인트

### PagedAttention 논문과 vLLM

면접에서 간략히 설명 가능: **블록 레벨 KV, 블록 테이블, 필요 기반 할당, 프리픽스 공유**; nano-vllm은 교육용/간소화 구현으로, 세부사항은 프로덕션 vLLM과 차이가 있을 수 있지만 사상은 일관됩니다.

### 해시 충돌이 일어나면 어떻게 하나

비암호학적 해시는 충돌 확률이 존재합니다. 구현에서는 **블록 내 토큰 리스트로 2차 검증**(`token_ids !=`이면 miss로 간주), `hash_to_block_id`는 단지 가속 인덱스로만 사용됩니다.

### 왜 가득 차지 않은 블록은 `h = -1`인가

가득 차지 않은 블록의 내용은 여전히 생성에 따라 변하므로, **안정적인 전역 키로 사용할 수 없습니다**. 가득 찰 때까지 기다렸다 등록합니다.

### 동시성과 스레드 안전성

단일 프로세스 추론 스케줄링이 싱글 스레드로 BlockManager를 실행하면 보통 락이 필요 없습니다. 멀티스레드 시 추가 동기화가 필요합니다(본 저장소 논의 범위를 벗어남).

---

## 자주 나오는 면접 질문

1. **PagedAttention과 연속 KV Cache의 차이는?**  
   연속: 단순하지만 파편화와 공유에 취약; 페이징: 블록 테이블 매핑, 공유와 재사용이 용이.

2. **`ref_count`가 0이라는 것은 무엇을 의미하나?**  
   어떤 시퀀스도 이 물리 블록을 더 이상 참조하지 않으며, 빈 연결 리스트로 회수 가능합니다.

3. **prefix로 이전 블록 hash를 사용하는 목적은?**  
   체인 지문, '다른 컨텍스트에서 동일한 단락'의 거짓 적중을 줄입니다.

4. **`can_append`는 왜 `len % block_size == 1`과 결합되어 있나?**  
   이 조건에서 다음 성장은 **새 물리 블록으로 넘어가게** 되므로, 빈 블록이 여전히 있는지 확인해야 합니다.

5. **BlockManager와 ModelRunner.kv_cache의 관계는?**  
   ModelRunner는 **텐서 메모리**를 할당하고, BlockManager는 **논리 블록 id**와 매핑을 할당하며, Attention 커널은 `block_table`에 따라 토큰을 올바른 물리 슬롯에 기록합니다.

6. **최악의 경우 메모리는?**  
   공유 없이, 각 시퀀스가 블록 테이블을 독점하면, 연속 할당에 근접합니다. 공유가 잘 될수록 더 절약됩니다.

---

## 요약

PagedAttention은 KV 저장을 블록화하고, `block_table`로 시퀀스에서 물리 블록으로의 매핑을 기술합니다. `xxhash` 체인 해시는 프리픽스 블록 재사용을 지원하며, `allocate`는 첫 테이블 구축과 적중/미스를 처리하고, `may_append`는 생성 과정에서 블록 확장과 해시 봉인을 수행합니다. `ref_count`는 공유 생애주기를 관리합니다. OS 페이징 비유는 머릿속 모형을 빠르게 구축하는 데 도움이 됩니다.

## 다음 강의 예고

목차 순서대로 학습한다면, **스케줄러 / 연속 배치 처리** 등의 장으로 계속 진행하여, '블록 레벨 KV'와 '배치 내 다중 시퀀스 스케줄링'을 완전한 추론 시스템으로 연결할 수 있습니다.

---

### 부록: `Sequence`와의 협업 (읽기 팁)

실제 공학에서, `seq.num_blocks`, `seq.block(i)`, `len(seq)`는 모두 **`Sequence`** 클래스에 정의되어 있습니다. 전자는 현재 토큰 수와 `block_size`로부터 유도되며, 후자는 i번째 논리 블록 내의 토큰 슬라이스를 반환합니다. `nanovllm/engine/sequence.py`를 읽으면 본 강의의 '블록 인덱스'와 '사용자에게 보이는 프롬프트+생성 문자열'을 대응시킬 수 있어, BlockManager만 기억하고 단절되는 것을 피할 수 있습니다.

### 부록: 디버깅 체크리스트 (자가 점검)

1. 동일한 프롬프트에 대한 두 요청이 앞쪽 몇 개의 `block_id`를 공유할 수 있는가? (기대: 적중 시 `ref_count>1`이거나 동일한 id를 재사용.)
2. 생성이 새 블록 경계에 도달했을 때, 빈 블록이 부족하면 `can_append`가 거짓을 반환하는가? (스케줄러가 대기하거나 선점해야 함.)
3. `deallocate` 후 `block_table`이 비워지고, `num_cached_tokens`가 0으로 돌아가는가?

이상의 질문에 답할 수 있다면, **메모리 풀 + 논리 매핑 + 생애주기**를 폐쇄 루프로 연결한 것입니다.