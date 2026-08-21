---
type: CompositeConcept
id: composite.serving_system
title: vLLM 통합 서빙 시스템 (Serving System)
description: Continuous Batching(Module 3)과 PagedAttention(Module 4)이 결합된 vLLM의 핵심
  서빙 아키텍처로, GPU 계산 효율과 메모리 효율을 동시에 최적화하여 10~23배의 처리량 향상을 달성
status: draft
generated:
  by: agent:builder/1.0
  at: 2026-08-06 09:00:00+00:00
verified:
- by: human:curator
  at: 2026-08-06 09:00:00+00:00
components:
- atomic.continuous_batching
- atomic.iteration_level_scheduling
- atomic.paged_kv_cache
- atomic.block_table
- atomic.slot_mapping
- composite.paged_attention_manager
prerequisites:
- atomic.kv_cache
- composite.dynamic_batcher
- module.continuous_batching
- module.paged_attention
sources:
- id: original_lab
  resource: https://hackmd.io/9Ivogn3dRwm3WgA4Kl3fJQ#Module-4
  title: 'nano-vLLM Module 4: PagedAttention - The Logical Abstraction'
---

# vLLM 통합 서빙 시스템 (Serving System)

## 📌 개요

**vLLM Serving System**은 Module 3의 **Continuous Batching**과 Module 4의 **PagedAttention**을 하나의 실행 파이프라인으로 통합한 최상위 복합 개념입니다.

- **Module 3 (Continuous Batching)**: GPU가 쉬지 않고 연산하도록 **계산 워크로드(Compute Workload)** 를 최적화합니다.
- **Module 4 (PagedAttention)**: GPU 메모리를 조각 없이 100% 활용하도록 **메모리 워크로드(Memory Workload)** 를 최적화합니다.

이 두 가지가 함께 작동할 때, vLLM은 기존 시스템 대비 **10~23배의 처리량(Throughput)** 향상을 달성할 수 있습니다.

---

## 🏗️ 통합 아키텍처 (Integrated Architecture)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         [사용자 요청 유입]                              │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 1. Scheduler (Module 3 - Continuous Batching)                         │
│    - 대기 큐(Waiting Queue)에서 실행 가능한 요청을 선택               │
│    - Prefill(Compute-Bound) 우선 정책 적용                            │
│    - 2단계 스케줄링 (Phase 1: Prefill, Phase 2: Decode)              │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ (메모리 할당 요청)
┌─────────────────────────────────────────────────────────────────────────┐
│ 2. Block Manager (Module 4 - PagedAttention)                          │
│    - Paged KV Cache를 고정 크기 블록(예: 16 tokens) 단위로 할당      │
│    - Block Table로 논리적 블록 번호 → 물리적 주소 매핑               │
│    - Slot Mapping으로 정확한 메모리 위치 계산                         │
│    - Sequence 종료 시 블록을 즉시 메모리 풀로 반환 (재사용)           │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ (혼합 배치 구성)
┌─────────────────────────────────────────────────────────────────────────┐
│ 3. GPU Execution Engine (Module 3 + 4 협업)                           │
│    - Continuous Batching으로 Prefill/Decode 혼합 배치 실행            │
│    - PagedAttention 커널로 불연속적 물리 블록에 저장된 KV Cache 접근   │
│    - 배치 실행 완료 후 결과 반환 및 배치 해체(Dissolve)               │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 4. 상태 갱신 및 루프 반복                                              │
│    - 완료된 Sequence는 블록을 Block Manager에 반환                     │
│    - 진행 중인 Sequence는 Block Table 갱신                             │
│    - Scheduler가 즉시 다음 배치 구성 (Iteration-Level Scheduling)      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🔗 두 모듈의 상호 의존성 (Interdependence)

| 관점 | 설명 |
|---|---|
| **Module 3 → Module 4** | Continuous Batching이 **Prefill과 Decode가 혼합된 배치**를 지속적으로 생성합니다. 이 혼합 배치에서 각 요청의 KV Cache는 **동적으로 증가**합니다. PagedAttention은 이 **동적이고 불규칙한** KV Cache 성장을 **효율적으로 관리**하는 방법을 제공합니다. |
| **Module 4 → Module 3** | PagedAttention의 **블록 단위 할당**은 Scheduler가 **메모리 용량을 정확히 예측**하고 **배치 크기를 최적화**할 수 있게 합니다. 즉, 메모리 관리가 **스케줄링의 의사 결정**을 더 정교하게 만듭니다. |

> 💡 **핵심**: Scheduler는 Block Manager에게 "이 Sequence에 블록을 몇 개나 할당할 수 있나요?"라고 묻고, 그 응답에 따라 **배치에 포함시킬 요청의 수를 동적으로 조정**합니다. 이로 인해 메모리 부족(OOM) 없이 GPU를 최대한 활용할 수 있습니다.

---

## 🧱 구성 요소 (Components)

| 구성 요소 | 출처 | 역할 |
|---|---|---|
| [`atomic.continuous_batching`](../03_atomic/continuous_batching.md) | Module 3 | 매 Iteration마다 배치를 새롭게 구성 |
| [`atomic.iteration_level_scheduling`](../03_atomic/iteration_level_scheduling.md) | Module 3 | 요청 단위가 아닌 Iteration 단위 스케줄링 |
| [`atomic.paged_kv_cache`](../03_atomic/paged_kv_cache.md) | Module 4 | KV Cache를 고정 블록으로 분할 |
| [`atomic.block_table`](../03_atomic/block_table.md) | Module 4 | 논리→물리 주소 매핑 |
| [`atomic.slot_mapping`](../03_atomic/slot_mapping.md) | Module 4 | 토큰 위치 → 블록 내 슬롯 변환 |
| [`composite.paged_attention_manager`](paged_attention_manager.md) | Module 4 | 블록 할당/해제/매핑 총괄 |

---

## 📊 통합 성능 지표 (Combined Impact)

| 지표 | Continuous Batching 단독 | PagedAttention 단독 | **통합 (vLLM)** |
|---|---|---|---|
| **GPU 활용률** | 90%+ | 50~70% (메모리 단편화로 인해) | **95%+** |
| **메모리 활용률** | 70~80% | 100% (단편화 제거) | **~100%** |
| **처리량 (Throughput)** | 3~5x 향상 | 2~3x 향상 | **10~23x 향상** |
| **OOM 발생 빈도** | 낮음 | 높음 (연속 할당) | **거의 없음** |

---

## 📝 Brick-by-Brick 학습 관점

이 복합 개념은 지금까지 학습한 모든 개념의 **정점(Peak)** 입니다.

```
[Module 00] inference_only, prefill_phase, decode_phase
     │
     ▼
[Module 01] kv_cache, attention, ffn, sampling
     │
     ▼
[Module 02] static_kv_cache, seq_len_budget (연속 할당의 한계 인지)
     │
     ▼
[Module 03] continuous_batching, iteration_level_scheduling (계산 효율 극대화)
     │
     ▼
[Module 04] paged_kv_cache, block_table, slot_mapping (메모리 효율 극대화)
     │
     ▼
[Composite] serving_system (Module 3 + 4 통합) ★ **여기까지가 vLLM의 핵심**
     │
     ▼
[Module 5+] memory_pool, prefix_cache, distributed_serving (고급 최적화)
```

> 💡 **최종 인사이트**: [`serving_system`](serving_system.md)은 단순히 두 모듈을 합친 것이 아닙니다. Scheduler와 Block Manager가 실시간으로 메모리 상태를 주고받으며 **배치 구성을 최적화**하는 **유기적 협업 시스템**입니다. 이 통합이 바로 vLLM이 다른 엔진과 차별화되는 결정적 요인입니다.
> **Continuous Batching (계산 효율)** 과 **PagedAttention (메모리 효율)** 이 결합되어 vLLM의 완전한 서빙 엔진을 구성하는 방식을 계층적으로 설명합니다.

# PagedAttention Block Manager (페이지드 어텐션 블록 관리자)

## 📌 개요

**PagedAttention Block Manager**는 Module 4의 핵심 복합 개념으로, PagedAttention이 동작하기 위한 **논리-물리 주소 변환 인프라**를 제공합니다.

연속 할당 방식에서는 KV Cache가 단일 연속 메모리 블록으로 존재했기 때문에, Attention 커널은 단순히 `base_address + token_position * head_dim`으로 메모리에 접근할 수 있었습니다.

그러나 PagedAttention에서는 KV Cache가 **수십 개의 불연속적인 물리적 블록**에 흩어져 저장됩니다. 이 복잡성을 Attention 커널로부터 숨기고, **논리적으로는 연속적인 캐시처럼 보이게** 하는 것이 이 Block Manager의 핵심 역할입니다.

---

## 🧱 구성 요소 (Components)

| 구성 요소 | 설명 |
|---|---|
| [`atomic.paged_kv_cache`](../03_atomic/paged_kv_cache.md) | KV Cache를 고정 크기 블록(예: 16 tokens) 단위로 물리적 메모리에 분산 저장 |
| [`atomic.block_table`](../03_atomic/block_table.md) | 각 Sequence별로 `Logical Block ID → Physical Block ID`를 매핑하는 테이블 |
| [`atomic.slot_mapping`](../03_atomic/slot_mapping.md) | 특정 토큰 위치(`token_position`)를 `(physical_block_id, slot_index)` 쌍으로 변환하는 계산기 |

---

## 🏗️ 아키텍처 및 동작 원리

### 1. 논리적 캐시 vs 물리적 캐시

Block Manager는 Sequence가 인식하는 **논리적 캐시(Logical Cache)** 와 GPU 메모리에 실제로 저장된 **물리적 캐시(Physical Cache)** 를 분리합니다.

```
[Sequence가 인식하는 논리적 캐시]
┌──────────────────────────────────────────────────────┐
│ Logical Block 0 │ Logical Block 1 │ Logical Block 2 │
│ (Token 0~15)    │ (Token 16~31)   │ (Token 32~47)   │
└──────────────────────────────────────────────────────┘
         │                │                │
         ▼ (Block Table 매핑)
┌──────────────────────────────────────────────────────┐
│ Physical Block 7 │ Physical Block 3 │ Physical Block 15│
│ (GPU 메모리)     │ (GPU 메모리)     │ (GPU 메모리)      │
└──────────────────────────────────────────────────────┘
```

### 2. Block Table (블록 테이블)

각 Sequence는 자신만의 Block Table을 유지합니다. 이 테이블은 **논리적 블록 번호**를 키로, **물리적 블록 주소(또는 ID)** 를 값으로 저장합니다.

```
Sequence A의 Block Table (BLOCK_SIZE = 16):
┌─────────────────┬──────────────────┐
│ Logical Block 0 │ Physical Block 7 │
│ Logical Block 1 │ Physical Block 3 │
│ Logical Block 2 │ Physical Block 15│
│ Logical Block 3 │ Physical Block 22│
└─────────────────┴──────────────────┘

Sequence B의 Block Table:
┌─────────────────┬──────────────────┐
│ Logical Block 0 │ Physical Block 5 │
│ Logical Block 1 │ Physical Block 11│
└─────────────────┴──────────────────┘
```

### 3. Slot Mapping (슬롯 매핑)

Attention 커널이 특정 토큰의 K/V 텐서를 읽어야 할 때, Block Manager는 **토큰 위치 → 물리적 블록 내 슬롯**을 계산합니다.

**매핑 공식**:

```
logical_block_id  = token_position // BLOCK_SIZE
physical_block_id = block_table[logical_block_id]
slot_index        = token_position % BLOCK_SIZE
```

**예시** (`BLOCK_SIZE = 16`, Sequence A 기준):

| Token Position | logical_block_id | physical_block_id | slot_index |
|---|---|---|---|
| 0 | 0 | 7 | 0 |
| 15 | 0 | 7 | 15 |
| 16 | 1 | 3 | 0 |
| 31 | 1 | 3 | 15 |
| 32 | 2 | 15 | 0 |

> 💡 Attention 커널은 `physical_block_id * BLOCK_SIZE * head_dim + slot_index * head_dim` 오프셋으로 메모리에서 K/V를 읽어옵니다.

---

## 🔗 Module 3 Scheduler와의 협업

Block Manager는 단순히 매핑만 수행하는 것이 아니라, **Scheduler(Module 3)와 긴밀히 협력**하여 메모리 상태를 실시간으로 공유합니다.

| 협업 포인트 | 설명 |
|---|---|
| **가용 블록 조회** | Scheduler가 `get_num_free_blocks()`를 호출하여 현재 할당 가능한 블록 수를 확인 |
| **배치 크기 결정** | 가용 블록 수가 부족하면 Scheduler는 배치에 포함시킬 요청 수를 동적으로 축소 |
| **메모리 할당 요청** | Scheduler가 새 Sequence를 배치에 추가할 때, Block Manager에게 필요한 블록 수만큼 할당 요청 |
| **메모리 해제 알림** | Sequence가 완료되면 Scheduler가 Block Manager에게 해당 블록들을 해제하도록 지시 |

```
┌─────────────────────────────────────────────────────────────┐
│ Scheduler (Module 3)                                       │
│  - "지금 5개 블록이 필요해" → Block Manager에 요청          │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼ (메모리 상태 확인)
┌─────────────────────────────────────────────────────────────┐
│ Block Manager (Module 4)                                   │
│  - "현재 3개 블록밖에 없어. 2개는 Swap Manager로 보낼게."  │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼ (스왑 완료 후)
┌─────────────────────────────────────────────────────────────┐
│ Scheduler (Module 3)                                       │
│  - "확인. 5개 블록 할당받았으니 배치 구성 완료."            │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Block Manager의 성능 기여

| 지표 | 연속 할당 방식 | PagedAttention (Block Manager 적용) |
|---|---|---|
| **메모리 접근 방식** | 연속적 (Contiguous) | 불연속적 (Non-Contiguous) |
| **주소 변환 오버헤드** | 없음 (직접 오프셋 계산) | 약간 있음 (Block Table Lookup + Slot Calc) |
| **메모리 단편화** | 심각 (External Fragmentation) | **거의 제로 (Near-Zero)** |
| **Attention 커널 복잡도** | 낮음 | 중간 (불연속 메모리 읽기 지원 필요) |
| **전체 시스템 처리량** | 1x (기준) | **10~23x 향상** |

> ⚠️ **주의**: Block Manager의 주소 변환은 약간의 오버헤드를 유발하지만, 이는 **메모리 단편화 제거로 인한 이득(10~23배 처리량 향상)에 비해 무시할 수 있는 수준**입니다.

---

## 🧠 블록 관리자의 상태 추적 (State Tracking)

Block Manager는 내부적으로 다음 상태들을 지속적으로 추적합니다.

```python
# Block Manager 내부 상태 (개념적)
class PagedAttentionBlockManager:
    # Sequence ID → Block Table
    block_tables: Dict[SeqID, List[PhysicalBlockID]]
    
    # Physical Block ID → (Sequence ID, Logical Block ID) 역매핑 (디버깅/해제용)
    reverse_mapping: Dict[PhysicalBlockID, Tuple[SeqID, LogicalBlockID]]
    
    # 사용 가능한 물리적 블록 목록 (Free List)
    free_blocks: List[PhysicalBlockID]
    
    # 전체 블록 개수 (GPU Memory Pool 크기)
    total_blocks: int
```

---

## 📝 Brick-by-Brick 학습 관점

Module 4의 Block Manager는 지금까지 학습한 개념들 중 **가장 실용적인 추상화 계층**입니다.

```
[Module 1] KV Cache 존재 인지
     │
     ▼
[Module 2] 연속 할당의 한계(단편화) 인지
     │
     ▼
[Module 3] Continuous Batching (계산 효율 극대화)
     │
     ▼
[Module 4] PagedAttention (논리-물리 분리 개념 도입) ★
     │
     ▼
[Module 4 Block Manager] ★★★★★
     - Block Table 관리
     - Slot Mapping 계산
     - Scheduler와 메모리 상태 공유
     - Attention 커널에 불연속 메모리 접근 인터페이스 제공
     │
     ▼
[Module 5] Memory Pool / Allocator / Swap Manager (운영 인프라 확장)
```

> 💡 **최종 인사이트**: Module 4의 Block Manager는 PagedAttention이라는 **혁신적인 아이디어**와 실제 GPU 커널이 동작하는 **구현 계층(Implementation Layer)** 사이의 **번역기(Translator)** 역할을 합니다. 이 관리자가 없으면 PagedAttention은 이론에 그치고, Attention 커널은 불연속 메모리를 읽을 수 없어 시스템이 동작하지 않습니다. 바로 이 계층이 vLLM을 **"논문의 아이디어"**에서 **"실제로 동작하는 시스템"**으로 전환시키는 마법 같은 접착제입니다.