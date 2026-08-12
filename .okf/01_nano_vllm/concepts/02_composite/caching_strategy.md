---
type: CompositeConcept
id: composite.prefix_cache_manager
title: Prefix Caching Strategy & Manager (프리픽스 캐싱 전략 및 관리자)
description: Content-Addressable Memory 기반의 프리픽스 캐싱을 구현하는 통합 관리자로, xxHash 체인 해시를 통해 프리픽스를 식별하고, 해시-블록 역인덱스와 참조 카운터를 유지하며, 캐시 히트 시 KV Cache 블록을 재사용하고 미스 시에만 새로운 블록을 할당하여 메모리와 연산을 최적화
status: draft
generated:
  by: agent:builder/1.0
  at: 2026-08-06T16:00:00Z
verified:
  - by: human:curator
    at: 2026-08-06T16:00:00Z
components:
  - atomic.content_hash
  - atomic.hash_chain
  - atomic.hash_to_block_id
  - atomic.ref_count
  - atomic.cache_miss_propagation
prerequisites:
  - module.memory_management
  - composite.block_manager
composes_into:
  - composite.distributed_serving (Module 7)
sources:
  - id: deepwiki_prefix_caching
    resource: https://deepwiki.com/liguodongiot/nano-vllm/6.3-prefix-caching
    title: "nano-vLLM Prefix Caching Implementation"
  - id: vllm_docs_prefix_caching
    resource: https://docs.vllm.ai/en/latest/design/prefix_caching/
    title: "vLLM Automatic Prefix Caching Documentation"
---

# Prefix Caching Strategy & Manager (프리픽스 캐싱 전략 및 관리자)

## 📌 개요

**Prefix Caching Strategy & Manager**는 Module 5의 `Physical Block Manager` 위에 구축된 **지능형 캐싱 계층**입니다.

Module 5는 "물리적 블록을 어떻게 할당/해제하고 스왑할까?"라는 **물리적 자원 운영**에 집중했습니다. Module 6의 `Prefix Cache Manager`는 여기에 **"어떤 블록이 이미 계산되었는지 식별하고, 동일한 계산을 반복하지 않게 하려면 어떻게 해야 할까?"**라는 **논리적 최적화**를 추가합니다.

**이 관리자의 핵심 책임**:
1. **Content-Addressable 식별**: xxHash 기반 체인 해시로 블록의 고유성을 판별
2. **캐시 조회 및 히트/미스 판정**: `hash_to_block_id` 역인덱스를 통한 O(1) 조회
3. **참조 기반 공유 관리**: `ref_count`로 여러 Sequence가 동일 블록을 공유하는 상황 추적
4. **캐시 미스 전파**: 프리픽스 중간에 미스가 발생하면 이후 블록은 모두 미스 처리하여 불필요한 해시 계산 방지
5. **캐시 만료(Eviction)**: 참조 카운터가 0이 된 블록을 LRU 정책으로 해제하여 메모리 확보

---

## 🧱 구성 요소 (Components)

| 구성 요소 | 설명 |
|---|---|
| `atomic.content_hash` | xxHash64 기반 블록 내용 해시 계산기 |
| `atomic.hash_chain` | 이전 블록의 해시를 시드로 포함하는 **체인 해시**로 프리픽스 연속성 보장 |
| `atomic.hash_to_block_id` | 해시값 → 물리적 블록 ID를 매핑하는 **전역 역인덱스 (Global Reverse Index)** |
| `atomic.ref_count` | 각 블록을 참조 중인 Sequence의 수를 추적하는 참조 카운터 |
| `atomic.cache_miss_propagation` | 중간 블록에서 캐시 미스 발생 시 이후 모든 블록을 미스 처리하는 최적화 전략 |

---

## 🏗️ 통합 아키텍처 (Integrated Architecture)

Prefix Cache Manager는 `Physical Block Manager`를 **래핑(Wrapping)** 하여, 기존 할당/해제 로직 앞에 **캐시 조회 단계**를 추가합니다.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Scheduler (Module 3)                           │
│  - "Sequence A의 Token IDs를 할당해줘"                                 │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ (Token IDs + num_blocks)
┌─────────────────────────────────────────────────────────────────────────┐
│              ★ Prefix Cache Manager (Module 6) ★                       │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │  Step 1: can_allocate (캐시 가능 블록 수 계산)                 │  │
│  │    - Block 0: 체인 해시 계산 → hash_to_block_id 조회 (HIT/MISS) │  │
│  │    - Block 1: 체인 해시 계산 (이전 해시 포함) → 조회          │  │
│  │    - 중간에 MISS 발생 시 → cache_miss_propagation 적용          │  │
│  │    - 반환: num_cached_blocks, num_new_blocks                   │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                              │                                        │
│                              ▼ (캐시 HIT 블록: ref_count 증가)       │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │  Step 2: allocate (실제 할당 실행)                             │  │
│  │    - HIT: hash_to_block_id에서 block_id 조회 → ref_count++     │  │
│  │    - MISS: Physical Block Manager에 새 블록 할당 요청          │  │
│  │           → 새 block_id 반환 → ref_count=1로 초기화           │  │
│  │           → hash_to_block_id에 (hash → block_id) 등록          │  │
│  └─────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ (MISS 시에만 호출)
┌─────────────────────────────────────────────────────────────────────────┐
│                    Physical Block Manager (Module 5)                   │
│  - Memory Pool에서 물리적 블록 ID 할당                                 │
│  - Free List 업데이트                                                  │
│  - Swap Manager 호출 (메모리 부족 시)                                  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      GPU HBM (Memory Pool)                             │
│  - 블록 7, 3, 15 (물리적 메모리)                                      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🔗 핵심 동작 메커니즘 (Core Mechanism)

### 1. 체인 해시 계산 (Chain Hash Computation)

프리픽스의 연속성을 보장하기 위해, 각 블록의 해시는 **이전 블록 해시를 시드(Seed)로 포함**합니다.

```python
def compute_block_hash(token_ids: List[int], prev_hash: int) -> int:
    """
    이전 블록의 해시를 포함한 체인 해시 계산.
    - BLOCK_SIZE = 16 (고정)
    - xxHash64 알고리즘 사용
    """
    hasher = xxhash.xxh64(seed=prev_hash)  # 이전 해시를 시드로 사용
    hasher.update(bytes(token_ids))
    return hasher.intdigest()
```

**예시** (`BLOCK_SIZE = 4`):

| 블록 | 토큰 ID | 이전 해시 | 계산된 해시 |
|---|---|---|---|
| Block 0 | [1, 2, 3, 4] | 0 (초기값) | `H0` |
| Block 1 | [5, 6, 7, 8] | `H0` | `H1` |
| Block 2 | [9, 10, 11, 12] | `H1` | `H2` |

> 💡 **이점**: 동일한 `[5, 6, 7, 8]` 블록이라도 이전 블록 해시가 다르면(`H0` vs `H100`) **완전히 다른 해시값**이 생성됩니다. 따라서 **프리픽스가 완전히 일치하는 경우에만** 캐시 히트가 발생합니다.

### 2. 캐시 조회 및 히트/미스 판정 (Cache Lookup)

```python
def can_allocate(self, seq_token_ids: List[int]) -> Tuple[int, int]:
    """
    Sequence의 블록 중 캐시 히트 가능한 개수와 새로 할당해야 할 개수를 계산.
    """
    num_blocks = len(seq_token_ids) // BLOCK_SIZE
    num_cached = 0
    prev_hash = -1
    
    for i in range(0, len(seq_token_ids), BLOCK_SIZE):
        block_tokens = seq_token_ids[i:i+BLOCK_SIZE]
        block_hash = self.compute_hash(block_tokens, prev_hash)
        
        if block_hash in self.hash_to_block_id:
            num_cached += 1
            prev_hash = block_hash
        else:
            # ★ 캐시 미스 발생! 이후 모든 블록은 미스 처리 ★
            break  
    
    num_new = num_blocks - num_cached
    return num_cached, num_new
```

> ⚠️ **cache_miss_propagation**: 중간 블록에서 미스가 발생하면, 그 이후 블록들은 **무조건 미스**로 간주합니다. 이는 프리픽스가 달라지면 이후 모든 내용이 달라진다는 LLM의 특성에 기반한 최적화입니다.

### 3. 참조 기반 공유 관리 (Reference Counting)

각 블록은 `ref_count`를 유지하여 몇 개의 Sequence가 해당 블록을 참조 중인지 추적합니다.

| 동작 | ref_count 변화 |
|---|---|
| **캐시 히트 (Cache Hit)** | `ref_count += 1` (기존 블록 재사용) |
| **새 블록 할당 (Cache Miss)** | `ref_count = 1` (새 블록 생성) |
| **Sequence 종료 (Free)** | `ref_count -= 1` (참조 해제) |
| **ref_count == 0** | 블록이 사용되지 않음 → **Eviction 후보** (해시 테이블에서 제거 및 메모리 반환) |

```python
def free_blocks(self, seq_id: str):
    """Sequence의 모든 블록 참조를 해제"""
    for block_id in self.block_tables[seq_id]:
        self.blocks[block_id].ref_count -= 1
        if self.blocks[block_id].ref_count == 0:
            # 참조가 0이면 해시 테이블에서 제거 및 메모리 해제
            block_hash = self.blocks[block_id].hash
            del self.hash_to_block_id[block_hash]
            self.physical_block_manager.free_block(block_id)
```

### 4. 캐시 만료 정책 (Eviction Policy)

Prefix Cache Manager는 `ref_count == 0`인 블록 중에서 **LRU(Least Recently Used)** 정책으로 캐시를 만료(Evict)합니다.

- **LFU (Least Frequently Used)**: 사용 빈도가 낮은 블록 우선 해제
- **FIFO (First In First Out)**: 가장 오래된 블록 우선 해제
- **vLLM 기본**: **LRU** (구현 단순, 실제 워크로드에 가장 효과적)

---

## 🔗 Module 5 Physical Block Manager와의 통합 (Integration)

Prefix Cache Manager는 **Physical Block Manager를 포함(Composition)** 하여 구현됩니다.

```python
class PrefixCacheManager:
    def __init__(self, physical_block_manager: PhysicalBlockManager):
        self.physical = physical_block_manager  # Module 5 의존성
        
        # Module 6 전용 자료구조
        self.hash_to_block_id = {}  # hash → Physical Block ID
        self.blocks = {}            # block_id → BlockMetadata (ref_count, hash, token_ids)
        self.access_log = []        # LRU eviction을 위한 접근 시간 기록
    
    def allocate(self, seq_id: str, token_ids: List[int]):
        # ... (캐시 조회 로직)
        
        if block_hash in self.hash_to_block_id:
            # HIT: 기존 블록 재사용 (Physical Block Manager 호출 없음!)
            block_id = self.hash_to_block_id[block_hash]
            self.blocks[block_id].ref_count += 1
        else:
            # MISS: Physical Block Manager에 할당 요청
            block_id = self.physical.allocate_block()  # Module 5
            self.blocks[block_id] = BlockMetadata(
                ref_count=1,
                hash=block_hash,
                token_ids=block_tokens
            )
            self.hash_to_block_id[block_hash] = block_id
```

> 💡 **계층 분리의 이점**: Prefix Cache Manager는 **"캐시 히트 시 Physical Block Manager를 전혀 호출하지 않습니다"** (Zero Overhead). 즉, 캐시 히트가 발생하면 메모리 할당/해제 오버헤드가 **완전히 제거**됩니다.

---

## 📊 성능 영향 (Performance Impact)

| 지표 | Prefix Caching 없음 | Prefix Caching 적용 |
|---|---|---|
| **동일 프리픽스 요청 간 메모리 사용** | N개 요청 = N × KV Cache 크기 | N개 요청 = **1 × KV Cache 크기** (공유) |
| **Prefill 연산** | 모든 요청에서 전체 프롬프트 Prefill | **캐시된 프리픽스에 대한 Prefill 생략** |
| **First Token Latency (TTFT)** | 프롬프트 전체 길이에 비례 | **프리픽스 길이만큼 단축** (예: 60% 단축) |
| **캐시 조회 오버헤드** | 없음 | xxHash 계산 + Dict Lookup (수 μs) |
| **메모리 할당/해제 호출** | 매 요청마다 발생 | **캐시 히트 시 전혀 발생하지 않음** |
| **처리량 (Throughput)** | 기준 (1x) | **캐시 히트율에 따라 2~10x 향상** |

---

## 📝 Brick-by-Brick 학습 관점

이 복합 개념은 Module 6의 **정점**으로, 지금까지 구축한 물리적 인프라 위에 **지능적 최적화 계층**을 올립니다.

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
[Module 4] PagedAttention + PagedAttention Manager (논리-물리 매핑)
     │
     ▼
[Module 5] Physical Block Manager (물리적 자원 운영: 할당/해제/스왑)
     │
     ▼
[Module 6 - Composite ★★★★★]
     └── Prefix Cache Manager
          → Physical Block Manager를 감싸는 Wrapper
          → "이미 계산된 프리픽스인가?"를 묻고, 맞으면 재사용
          → xxHash 체인 해시로 프리픽스 식별
          → hash_to_block_id 역인덱스로 O(1) 조회
          → ref_count로 블록 공유 관리
          → ref_count=0인 블록은 LRU로 Eviction
          → **결과: 동일 프리픽스 요청 간 메모리/연산 낭비 제거**
     │
     ▼
[Module 7] Distributed Serving (분산 환경으로 확장)
```

> 💡 **최종 인사이트**: Prefix Cache Manager는 **"이미 계산한 것은 다시 계산하지 말고, 이미 할당한 메모리는 다시 할당하지 말자"** 는 가장 직관적인 최적화 원칙을 시스템적으로 구현합니다. 이 관리자가 없다면 vLLM은 시스템 프롬프트가 동일한 100개의 채팅 요청에 대해 **100번의 동일한 Prefill 연산**을 수행해야 합니다. 이 관리자가 있으면 **단 1번의 Prefill**로 충분합니다. 이것이 바로 vLLM이 실제 프로덕션 환경에서 **압도적인 처리량**을 달성하는 핵심 비결입니다.