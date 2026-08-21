---
type: CompositeConcept
id: composite.block_manager
title: Physical Block Manager (물리적 블록 관리자)
description: Module 5의 핵심 복합 개념으로, Memory Pool, Block Allocator, Swap Manager를 통합하여
  GPU 메모리 블록의 할당, 해제, 스와핑을 총괄하는 물리적 자원 운영 엔진. Scheduler(Module 3)와 PagedAttention
  Manager(Module 4)에게 메모리 상태를 제공하고 블록 할당 인터페이스를 노출
status: draft
generated:
  by: agent:builder/1.0
  at: 2026-08-06 13:00:00+00:00
verified:
- by: human:curator
  at: 2026-08-06 13:00:00+00:00
components:
- atomic.memory_pool
- atomic.block_allocator
- atomic.swap_manager
- atomic.swap_policy
prerequisites:
- atomic.memory_pool
- atomic.paged_kv_cache
- atomic.swap_manager
- composite.paged_attention_manager (Module 4)
composes_into:
- composite.distributed_serving (Module 7)
sources:
- id: original_lab
  resource: https://hackmd.io/9Ivogn3dRwm3WgA4Kl3fJQ
  title: 'nano-vLLM Module 5: Memory Management & Block Manager'
prerequisite_of:
- composite.distributed_serving
- composite.prefix_cache_manager
---

# Physical Block Manager (물리적 블록 관리자)

## 📌 개요

**Physical Block Manager**는 Module 5의 최상위 복합 개념으로, 지금까지 학습한 세 가지 원자적 개념([`memory_pool`](../03_atomic/memory_pool.md), [`block_allocator`](../03_atomic/block_allocator.md), [`swap_manager`](../03_atomic/swap_manager.md))을 하나의 통합된 인터페이스로 묶어줍니다.

이 관리자는 **Scheduler(Module 3)**와 **PagedAttention Manager(Module 4)** 사이에서 중개자(Mediator) 역할을 수행하며, 다음과 같은 실질적인 메모리 운영을 담당합니다.

1. **자원 추상화**: Memory Pool을 통해 GPU HBM을 블록 단위로 추상화
2. **할당 전략 실행**: Block Allocator를 통해 First-Fit/Best-Fit 정책으로 블록 할당
3. **메모리 부족 대응**: Swap Manager를 호출하여 LRU/LFU 기반으로 블록을 CPU로 스왑아웃
4. **상태 모니터링**: 현재 가용 블록 수, 할당된 블록 수, 스왑 중인 블록 수를 실시간 추적

---

## 🧱 구성 요소 (Components)

| 구성 요소 | 설명 |
|---|---|
| [`atomic.memory_pool`](../03_atomic/memory_pool.md) | GPU HBM을 고정 크기 블록들의 풀로 추상화한 물리적 기반 |
| [`atomic.block_allocator`](../03_atomic/block_allocator.md) | First-Fit, Best-Fit, Next-Fit 등 다양한 할당 전략을 구현하는 정책 엔진 |
| [`atomic.swap_manager`](../03_atomic/swap_manager.md) | GPU 메모리 부족 시 블록을 CPU 메모리로 스왑아웃/스왑인하는 관리자 |
| `atomic.swap_policy` | 스왑아웃 대상 선정 정책 (LRU, LFU, FIFO, Priority-Based) |

---

## 🏗️ 통합 아키텍처 및 계층 구조

Physical Block Manager는 **Scheduler → PagedAttention Manager → Physical Block Manager**로 이어지는 계층 구조의 최하단에 위치합니다.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Scheduler (Module 3)                              │
│  - "Sequence A에 블록 3개가 필요해. 할당해줘."                         │
│  - "현재 가용 블록이 몇 개야?"                                        │
└─────────────────────────────────────────────────────────────────────────┘
                                    │ (메모리 요청)
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│               PagedAttention Manager (Module 4)                        │
│  - "요청 받았어. Sequence A의 Block Table을 업데이트해야 해."          │
│  - "Physical Block Manager에게 물리적 블록 ID를 요청할게."             │
└─────────────────────────────────────────────────────────────────────────┘
                                    │ (물리적 블록 ID 요청)
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              ★ Physical Block Manager (Module 5) ★                    │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │  1. Memory Pool에서 가용 블록 확인                              │  │
│  │  2. Block Allocator로 할당 전략 실행 (First-Fit)               │  │
│  │  3. 가용 블록 부족 시 Swap Manager 호출 (LRU로 스왑아웃)       │  │
│  │  4. 할당된 물리적 블록 ID(7, 3, 15)를 PagedAttention Manager에 반환│
│  └─────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      GPU HBM (Memory Pool)                            │
│  Block 7 │ Block 3 │ Block 15 │ ... (실제 물리적 메모리 블록)         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🔗 핵심 동작 흐름 (Sequence Diagram)

### 시나리오 1: 정상 할당 (Normal Allocation)

```
[Scheduler] → [PagedAttention Manager] → [Physical Block Manager] → [Memory Pool]

1. Scheduler가 새 Sequence를 배치에 추가
2. PagedAttention Manager가 Physical Block Manager에게 블록 3개 요청
3. Physical Block Manager가 Memory Pool의 Free List에서 블록 3개를 가져옴
   → (Free List: [0,1,2,3,4,5] → [0,1,2] 할당 → Free List: [3,4,5])
4. 물리적 블록 ID [7, 3, 15]를 PagedAttention Manager에 반환
5. PagedAttention Manager가 Block Table에 (Logical 0→Physical 7) 등록
```

### 시나리오 2: 메모리 부족 시 스와핑 (Swap-Out)

```
[Scheduler] → [Physical Block Manager] → [Swap Manager] → [CPU Memory]

1. Scheduler가 새 Sequence 추가를 위해 블록 5개 요청
2. Physical Block Manager가 Memory Pool 확인 → 가용 블록 2개뿐!
3. Physical Block Manager가 Swap Manager에게 "블록 3개를 스왑아웃해줘" 요청
4. Swap Manager가 Swap Policy(LRU)를 적용하여 가장 오래된 Sequence B의 블록 3개 선택
5. GPU → CPU로 블록 데이터 복사 (cudaMemcpy)
6. Memory Pool의 Free List에 블록 3개 추가 → 가용 블록 5개 확보
7. 새 Sequence에 블록 5개 할당 완료
```

---

## 🧠 Physical Block Manager의 내부 상태 (State)

```python
# 개념적 Python 코드
class PhysicalBlockManager:
    def __init__(self):
        # 1. Memory Pool
        self.memory_pool = MemoryPool(total_memory=80_GB, block_size=16)
        
        # 2. 할당 정보 추적
        self.allocations = {}  # SeqID → List[PhysicalBlockID]
        self.free_blocks = self.memory_pool.get_all_block_ids()
        
        # 3. Swap Manager 참조
        self.swap_manager = SwapManager(swap_policy=LRU())
        
        # 4. 성능 통계
        self.stats = {
            'total_allocated_blocks': 0,
            'total_swapped_out_blocks': 0,
            'total_swapped_in_blocks': 0
        }
    
    def allocate_blocks(self, seq_id: str, num_blocks: int) -> List[int]:
        """Sequence에 num_blocks만큼의 물리적 블록을 할당"""
        available = len(self.free_blocks)
        if available >= num_blocks:
            # 정상 할당
            allocated = self.free_blocks[:num_blocks]
            self.free_blocks = self.free_blocks[num_blocks:]
            self.allocations[seq_id] = allocated
            return allocated
        else:
            # 메모리 부족 → Swap Manager 호출
            shortage = num_blocks - available
            evicted_blocks = self.swap_manager.swap_out(shortage)
            self.free_blocks.extend(evicted_blocks)
            # 재귀적으로 다시 할당 시도 (또는 루프)
            return self.allocate_blocks(seq_id, num_blocks)
    
    def free_blocks(self, seq_id: str) -> None:
        """Sequence의 모든 블록을 해제"""
        if seq_id in self.allocations:
            self.free_blocks.extend(self.allocations[seq_id])
            del self.allocations[seq_id]
```

---

## 🔗 Module 3 + 4와의 통합 인터페이스

| 호출자 (Caller) | Physical Block Manager가 제공하는 인터페이스 | 설명 |
|---|---|---|
| **Scheduler (Module 3)** | `get_num_free_blocks()` | 현재 가용 블록 수를 반환 (배치 크기 결정에 사용) |
| **Scheduler (Module 3)** | `can_allocate(seq_len)` | 특정 Sequence를 할당할 수 있는지 여부를 반환 |
| **PagedAttention Manager (Module 4)** | `allocate_blocks(seq_id, num_blocks)` | 물리적 블록 ID 목록을 반환 (Block Table 매핑에 사용) |
| **PagedAttention Manager (Module 4)** | `free_blocks(seq_id)` | Sequence 종료 시 블록을 메모리 풀로 반환 |

---

## 📊 성능 영향 및 트레이드오프

| 측면 | Physical Block Manager 적용 전 | Physical Block Manager 적용 후 |
|---|---|---|
| **할당 속도** | 느림 (cudaMalloc 호출, 수백 μs) | **매우 빠름** (포인터 연산, 수 ns) |
| **메모리 단편화** | 심각 (외부 단편화 30~50%) | **거의 제로 (Near-Zero)** |
| **메모리 활용률** | 50~70% | **~100%** |
| **스와핑 지연** | 없음 (OOM 발생 시 시스템 중단) | 있음 (CPU-GPU 복사, 수 ms) |
| **최대 동시 요청 수** | Memory Pool 크기에 제한 | **Memory Pool + CPU 스왑 용량** (확장됨) |
| **시스템 안정성** | 낮음 (OOM 위험) | **매우 높음 (Graceful Degradation)** |

> ⚠️ **트레이드오프 인식**: 스와핑은 CPU-GPU 간 메모리 복사를 수반하므로, 과도한 스와핑은 오히려 지연 시간을 증가시킬 수 있습니다. 따라서 Memory Pool 크기를 워크로드에 맞게 적절히 튜닝하고, 스왑 정책(LRU/LFU)을 신중히 선택하는 것이 중요합니다.

---

## 📝 Brick-by-Brick 학습 관점

이 복합 개념은 Module 5의 **정점(Peak)**으로, 지금까지 학습한 모든 메모리 관리 개념을 하나의 운영 가능한 시스템으로 통합합니다.

```
[Module 1] KV Cache 존재 인지
     │
     ▼
[Module 2] 연속 할당의 한계(단편화) 인지 → "메모리를 어떻게 조각 없이 관리할까?"
     │
     ▼
[Module 3] Continuous Batching (계산 효율) → "GPU를 어떻게 쉬지 않게 할까?"
     │
     ▼
[Module 4] PagedAttention + PagedAttention Manager (논리-물리 매핑)
     │  → "주소를 어떻게 변환할까?" (Mapping Layer)
     │
     ▼
[Module 5 - 원자적 개념]
     ├── memory_pool   → "메모리를 어떻게 미리 준비해둘까?" (Foundation)
     ├── block_allocator → "어떤 전략으로 할당할까?" (Strategy)
     └── swap_manager    → "부족하면 어떻게 대처할까?" (Safety Net)
     │
     ▼
[Module 5 - Composite ★★★★★]
     └── Physical Block Manager
          → 위 세 가지를 통합하여 Scheduler와 PagedAttention Manager에게
            **하나의 일관된 메모리 관리 인터페이스**를 제공
          → 이것이 실제 vLLM이 동작하는 **메모리 운영 체제(Memory OS)**임.
     │
     ▼
[Module 6] Prefix Caching (이 위에 캐시 최적화 계층 추가)
     │
     ▼
[Module 7] Distributed Serving (이 위에 분산 확장 계층 추가)
```

> 💡 **최종 인사이트**: Module 4의 `PagedAttention Manager`가 **"주소 변환기(Translator)"**라면, Module 5의 `Physical Block Manager`는 **"자원 관리자(Resource Manager)"**입니다. 전자는 논리적 질문("이 토큰은 어디에 있나?")에 답하고, 후자는 물리적 질문("이 블록을 어디서 가져오고, 부족하면 어떻게 할까?")에 답합니다. 이 두 관리자가 함께 동작할 때 vLLM은 메모리 단편화 없이, OOM 없이, 그리고 최대 처리량으로 동작할 수 있습니다.


> **⚠️ Module 4와 Module 5 Block Manager의 명확한 구분 (중요)**:
> - **Module 4의 [`composite.paged_attention_manager`](paged_attention_manager.md)**: **논리적 주소 변환 엔진** (Block Table 관리, Slot Mapping 계산, 논리→물리 매핑을 담당)
> - **Module 5의 [`composite.block_manager`](block_manager.md)**: **물리적 자원 운영 엔진** (Memory Pool에서 블록을 할당/해제하고, 메모리 부족 시 Swap Manager를 호출하는 **실제 메모리 관리자**)
>
> 즉, Module 4는 **"어떻게 매핑할까?"** (Mapping), Module 5는 **"어떻게 할당/해제하고 부족할 때 대처할까?"** (Allocation & Swapping)를 담당합니다.

---

```markdown