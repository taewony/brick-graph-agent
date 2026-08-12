---
type: Module
id: module.prefix_caching
title: Advanced Prefix Caching (고급 프리픽스 캐싱)
description: Content-Addressable Memory 기반의 KV Cache 블록 공유 시스템으로, 동일한 프롬프트 프리픽스를 가진 다중 요청 간 블록을 재사용하여 메모리 사용량을 획기적으로 절감하고 Prefill 연산을 생략
status: draft
generated:
  by: agent:builder/1.0
  at: 2026-08-06T14:00:00Z
verified:
  - by: human:curator
    at: 2026-08-06T14:00:00Z
prerequisites:
  - module.memory_management
  - composite.block_manager
  - atomic.paged_kv_cache
composes_into:
  - composite.distributed_serving
sources:
  - id: deepwiki_prefix_caching
    resource: https://deepwiki.com/liguodongiot/nano-vllm/6.3-prefix-caching
    title: "nano-vLLM Prefix Caching Implementation"
  - id: cnblogs_prefix_cache
    resource: https://www.cnblogs.com/llm-inference/articles/20050689
    title: "Nano-vLLM 源码解读 - 5. Prefix Cache"
  - id: vllm_docs_prefix_caching
    resource: https://docs.vllm.ai/en/latest/design/prefix_caching/
    title: "vLLM Automatic Prefix Caching Documentation"
---

# Advanced Prefix Caching (고급 프리픽스 캐싱)

## 📌 개요

Module 5까지 우리는 **Memory Pool**을 통해 GPU 메모리를 블록 단위로 관리하고, **Physical Block Manager**를 통해 블록의 할당/해제 및 스와핑을 처리하는 **물리적 메모리 인프라**를 구축했습니다.

Module 6은 이 인프라 위에 **Content-Addressable Memory(내용 기반 주소 지정)** 개념을 도입하여, **동일한 프롬프트 프리픽스(Prefix)를 가진 다중 요청 간에 KV Cache 블록을 공유**함으로써 메모리 사용량을 획기적으로 절감하고 Prefill 연산을 생략하는 최적화 계층을 구현합니다.

### Prefix Caching의 핵심 가치

| 측면 | 설명 |
|---|---|
| **메모리 절감** | 동일 프리픽스를 가진 N개 요청에서 KV Cache를 1/N로 절감 |
| **Prefill 생략** | 캐시된 프리픽스에 대한 Prefill 연산을 완전히 생략하여 **First Token Latency 감소** |
| **Content-Addressable** | 블록의 내용(토큰)을 해시(Hash)로 식별하여 **O(1) 조회**로 캐시 히트 판정 |

> 💡 **실제 사례**: 시스템 프롬프트가 동일한 100개의 채팅 요청이 들어올 때, Prefix Caching이 없다면 각 요청마다 시스템 프롬프트에 대한 Prefill(수백~수천 tokens)을 **100번 반복**해야 합니다. Prefix Caching이 있다면 **단 1번의 Prefill**로 모든 요청이 공유할 수 있습니다.

---

## 🧩 포함 원자적 개념

- `atomic.content_hash` — 블록의 토큰 내용을 식별하는 xxHash 기반 해시 값
- `atomic.hash_chain` — 이전 블록의 해시를 포함하는 **체인 해시(Chain Hash)**로 프리픽스의 연속성 보장
- `atomic.hash_to_block_id` — 해시값 → 물리적 블록 ID를 매핑하는 **역인덱스(Reverse Index)** 테이블
- `atomic.ref_count` — 블록을 참조 중인 Sequence의 개수를 추적하는 **참조 카운터(Reference Counter)** 
- `atomic.cache_miss_propagation` — 캐시 미스 발생 시 이후 모든 블록이 미스 처리되는 **연쇄 미스 전파** 최적화

## 🏗️ 복합 개념 (Composite Concept)

- `composite.prefix_cache_manager` — Content-Addressable Block System 전체를 통합 관리하는 프리픽스 캐시 관리자

## 🔗 관련 관계

- **PREREQUISITES**: `memory_pool`, `block_manager`, `paged_kv_cache`
- **PREREQUISITE_OF**: `distributed_serving` (Module 7)

---

## 🧱 1. 핵심 문제: 중복 Prefill의 비용

### 1.1 문제 정의

LLM 추론에서 **Prefill Phase**는 전체 프롬프트에 대한 Attention 연산을 수행하는 **Compute-Bound** 단계입니다.

- 프롬프트가 길수록(수천~수만 tokens) Prefill에 드는 시간과 비용이 선형적으로 증가
- **동일한 프롬프트 프리픽스**(예: 시스템 프롬프트, Few-shot 예제)가 반복될 경우, **동일한 연산이 중복** 수행됨

```
요청 A: [System Prompt] + [User Query A] → Prefill 2000 tokens
요청 B: [System Prompt] + [User Query B] → Prefill 2000 tokens (동일한 1500 tokens 중복!)
요청 C: [System Prompt] + [User Query C] → Prefill 2000 tokens (또 중복!)
```

### 1.2 중복의 규모

| 사용 패턴 | 프리픽스 비중 | 낭비되는 연산 |
|---|---|---|
| **챗봇 서비스** | 시스템 프롬프트 (수백 tokens) | 전체 프롬프트의 20~40% |
| **Few-shot 학습** | 예제들 (수천 tokens) | 전체 프롬프트의 50~80% |
| **RAG (검색 증강)** | 고정 검색 결과 컨텍스트 | 전체 프롬프트의 60~90% |

---

## 🧱 2. 해결책: Content-Addressable Block System

### 2.1 핵심 아이디어

Prefix Caching은 **"동일한 내용의 블록은 물리적 메모리를 공유한다"** 는 원칙에 기반합니다.

1. 각 블록의 **토큰 내용**을 **해시(Hash)**로 식별
2. 해시값을 키로 하는 **`hash_to_block_id`** 역인덱스 테이블 유지
3. 새 Sequence 할당 시, 각 블록의 해시를 계산하여 **이미 존재하는 블록이면 재사용**
4. **참조 카운터(Ref Count)** 로 여러 Sequence가 동일 블록을 공유하는 상황 관리

### 2.2 체인 해시 (Chain Hash) — 프리픽스 연속성 보장

단순히 블록 내 토큰만 해시하면, **동일한 토큰 조합이 다른 위치에서 나타날 때** 잘못된 공유가 발생할 수 있습니다.

**해결책**: 이전 블록의 해시를 **시드(Seed)**로 포함하는 **체인 해시(Chain Hash)** 사용

```
hash(block_i) = xxhash( hash(block_{i-1}) + token_ids(block_i) )
```

**예시** (`block_size = 4`):

| Sequence | Block 0 | Block 1 | Block 2 | Block 3 |
|---|---|---|---|---|
| A | [1,2,3,4] | [5,6,7,8] | [9,10,11,12] | [13,14,15,16] |
| B | [1,2,3,4] | [5,6,7,8] | [9,10,99,100] | [101,102,103,104] |

- Block 0, 1은 **동일한 프리픽스** → 해시 동일 → **공유 가능**
- Block 2는 **내용이 다름** → 해시가 달라짐 → **공유 불가** (이후 블록들도 자동으로 미스 처리)

> 💡 **체인 해시의 효과**: Block 2에서 해시가 달라지면, Block 3은 Block 2의 해시를 시드로 사용하므로 **자동으로 해시가 달라집니다**. 따라서 "프리픽스가 동일한 경우에만 공유"라는 조건이 자연스럽게 보장됩니다.

### 2.3 자료구조 (Data Structures)

Prefix Caching은 다음 4가지 핵심 자료구조를 사용합니다:

| 자료구조 | 설명 |
|---|---|
| **`blocks: List[Block]`** | 모든 물리적 블록의 메타데이터(block_id, ref_count, hash, token_ids)를 저장 |
| **`free_block_ids: deque[int]`** | 사용 가능한(Free) 블록 ID 큐 (LRU 시간순 유지) |
| **`used_block_ids: set[int]`** | 현재 할당 중인 블록 ID 집합 |
| **`hash_to_block_id: dict[hash → block_id]`** | **해시값 → 물리적 블록 ID** 역인덱스 (캐시 조회의 핵심) |

### 2.4 블록 할당 플로우 (Allocation Flow)

```
[새 Sequence 할당 요청: num_blocks = 3]
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 1: can_allocate (캐시 가능 블록 수 계산)              │
│   - Block 0: hash 계산 → hash_to_block_id에서 조회 → HIT!  │
│   - Block 1: hash 계산 (이전 해시 포함) → 조회 → HIT!      │
│   - Block 2: hash 계산 → hash_to_block_id에 없음 → MISS!   │
│   - 반환: num_cached_blocks = 2, num_new_blocks = 1        │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: allocate (실제 할당 실행)                           │
│   - Block 0: HIT → ref_count += 1 (기존 블록 공유)         │
│   - Block 1: HIT → ref_count += 1 (기존 블록 공유)         │
│   - Block 2: MISS → free_block_ids에서 새 블록 할당        │
│              → hash_to_block_id에 새 해시 등록              │
│              → ref_count = 1                                │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
[Sequence의 Block Table에 물리적 블록 ID 기록]
```

---

## 🔗 Module 5 Block Manager와의 통합

Prefix Caching은 **Module 5의 Physical Block Manager를 확장**하여 구현됩니다.

### 확장된 Block Manager 인터페이스

```python
# Module 5 Block Manager + Prefix Caching 확장
class PhysicalBlockManager:
    def __init__(self):
        # Module 5: 기존 자료구조
        self.memory_pool = MemoryPool(...)
        self.free_block_ids = deque([...])
        self.used_block_ids = set()
        
        # Module 6: Prefix Caching 추가 자료구조
        self.hash_to_block_id = {}  # hash → block_id 역인덱스
    
    def compute_hash(self, token_ids: List[int], prev_hash: int) -> int:
        """체인 해시 계산 (xxHash64)"""
        # 이전 블록의 해시를 시드로 포함
        ...
    
    def can_allocate(self, seq_token_ids: List[int]) -> int:
        """Sequence의 프리픽스 중 캐시 hit 가능한 블록 수 계산"""
        num_cached = 0
        prev_hash = -1
        for i in range(0, len(seq_token_ids), BLOCK_SIZE):
            block_tokens = seq_token_ids[i:i+BLOCK_SIZE]
            block_hash = self.compute_hash(block_tokens, prev_hash)
            if block_hash in self.hash_to_block_id:
                num_cached += 1
                prev_hash = block_hash
            else:
                break  # 첫 MISS 이후 모든 후속 블록은 MISS
        return num_cached
    
    def allocate(self, seq_id: str, token_ids: List[int]):
        """Sequence에 블록 할당 (캐시 hit 시 재사용)"""
        prev_hash = -1
        for i in range(0, len(token_ids), BLOCK_SIZE):
            block_tokens = token_ids[i:i+BLOCK_SIZE]
            block_hash = self.compute_hash(block_tokens, prev_hash)
            
            if block_hash in self.hash_to_block_id:
                # HIT: 기존 블록 재사용
                block_id = self.hash_to_block_id[block_hash]
                self.blocks[block_id].ref_count += 1
            else:
                # MISS: 새 블록 할당
                block_id = self.free_block_ids.popleft()
                self.blocks[block_id] = Block(
                    block_id=block_id,
                    ref_count=1,
                    hash=block_hash,
                    token_ids=block_tokens
                )
                self.hash_to_block_id[block_hash] = block_id
                self.used_block_ids.add(block_id)
            
            # Sequence의 Block Table에 추가
            self.block_tables[seq_id].append(block_id)
            prev_hash = block_hash
```

---

## 📊 성능 영향

| 지표 | Prefix Caching 없음 | Prefix Caching 적용 |
|---|---|---|
| **메모리 사용량** | 모든 Sequence가 독립적 KV Cache 보유 | **공통 프리픽스는 1/N로 절감** |
| **Prefill 연산** | 모든 요청마다 전체 프롬프트 Prefill | **캐시된 프리픽스는 Prefill 생략** |
| **First Token Latency** | 프롬프트 길이에 비례 | **프리픽스 길이만큼 단축** |
| **캐시 Hit 시 처리량** | 기준 (1x) | **최대 수백% 향상** (공유율에 따라) |

---

## 📝 Brick-by-Brick 학습 관점

Module 6은 지금까지 구축한 **물리적 메모리 인프라 위에 최적화 계층**을 추가합니다.

```
[Module 1] KV Cache 존재 인지
     │
     ▼
[Module 2] 연속 할당의 한계(단편화) 인지
     │
     ▼
[Module 3] Continuous Batching (계산 효율)
     │
     ▼
[Module 4] PagedAttention (논리-물리 매핑)
     │
     ▼
[Module 5] Memory Pool + Block Allocator + Swap Manager (물리적 자원 운영)
     │
     ▼
[Module 6] Prefix Caching ★★★★★
     → Content-Addressable Memory 도입
     → xxHash 기반 체인 해시로 프리픽스 식별
     → hash_to_block_id 역인덱스로 O(1) 캐시 조회
     → Ref Count로 블록 공유 관리
     → **동일 프리픽스 요청 간 KV Cache 공유로 메모리/연산 절감**
     │
     ▼
[Module 7] Distributed Serving (이 위에 분산 확장 계층 추가)
```

> 💡 **최종 인사이트**: Module 6의 Prefix Caching은 **"이미 계산한 것은 다시 계산하지 말고, 이미 할당한 메모리는 다시 할당하지 말자"** 는 가장 직관적인 최적화 원칙을 **Content-Addressable Memory**라는 시스템적 개념으로 구현합니다. 이 최적화는 특히 **챗봇, Few-shot, RAG** 등 반복적인 프리픽스가 발생하는 모든 실제 서비스에서 **가장 큰 성능 향상**을 제공하는 핵심 기술입니다.