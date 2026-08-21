---
type: AtomicConcept
id: atomic.paged_kv_cache
title: Paged KV Cache (페이지드 KV 캐시)
description: 'KV Cache를 연속된 하나의 큰 텐서가 아닌, 고정 크기(예: 16 tokens)의 블록(Block) 단위로 GPU 메모리에
  분산 저장하여 외부 단편화를 완전히 제거하고, 메모리 활용률을 끌어올리는 PagedAttention의 물리적 저장소 구조'
status: draft
generated:
  by: agent:builder/1.0
  at: 2026-08-07 09:00:00+00:00
verified:
- by: human:curator
  at: 2026-08-07 09:00:00+00:00
prerequisites:
- atomic.kv_cache
- atomic.static_kv_cache
composes_into:
- composite.paged_attention_manager (Module 04)
sources:
- id: vllm_paged_attention
  resource: https://arxiv.org/abs/2309.06180
  title: Efficient Memory Management for Large Language Model Serving with PagedAttention
prerequisite_of:
- atomic.block_table
- atomic.distributed_kv
- atomic.memory_pool
- atomic.slot_mapping
- composite.block_manager
- composite.paged_attention_manager
---

# Paged KV Cache (페이지드 KV 캐시)

## 📌 개념 정의

**Paged KV Cache**는 기존의 연속된(Contiguous) KV Cache 할당 방식이 초래하는 **외부 단편화(External Fragmentation)** 문제를 해결하기 위해, KV Cache를 **고정 크기(Fixed-Size)의 블록(Block)** 단위로 분할하여 GPU 메모리에 불연속적(Non-Contiguous)으로 저장하는 PagedAttention의 핵심 자료구조입니다.

- 각 블록은 동일한 크기(예: `BLOCK_SIZE = 16 tokens`)를 가지며, 하나의 블록에는 16개 토큰의 Key와 Value 텐서가 저장됩니다.
- Sequence의 길이가 증가하면 필요에 따라 새로운 블록을 동적으로 할당하여 추가합니다.
- Sequence가 종료되면 해당 블록들을 즉시 메모리 풀로 반환하여 다른 Sequence가 재사용할 수 있습니다.

---

## ❗ 왜 Paged KV Cache가 필요한가? (The Problem)

### 기존 Static KV Cache의 한계 (연속 할당)

- 각 Sequence는 자신의 KV Cache를 위해 **연속된 메모리 블록**을 할당받습니다.
- Sequence가 종료되어 메모리를 해제하면, 그 공간은 **조각(Fragment)** 으로 남습니다.
- 시간이 지남에 따라 메모리는 수많은 작은 조각들로 분할되고, **실제로는 충분한 총 메모리가 있음에도 불구하고** 새로운 Sequence가 필요한 연속 공간을 할당받지 못해 **OOM(Out Of Memory)** 오류가 발생합니다.

```
GPU 메모리 상태 (연속 할당):
─────────────────────────────────────────────
[ Seq A (1024) ] [ Free (400) ] [ Seq B (512) ]
                  ↑ 800 tokens 요청이 들어오면?
                  → 연속 공간이 400 tokens 밖에 없어 할당 불가!
                  → OOM 발생 (실제 총 Free는 1400 tokens)
```

### Paged KV Cache의 해결책 (The Solution)

- 모든 블록을 **동일한 크기(예: 16 tokens)** 로 고정합니다.
- Sequence가 더 이상 하나의 큰 연속 공간을 요구하지 않고, **필요한 개수의 블록**을 할당받습니다.
- 블록 크기가 동일하므로, 해제된 블록은 **조각(Fragment) 없이** 즉시 다른 Sequence가 재사용할 수 있습니다.
- 결과적으로 **외부 단편화가 거의 제로(Near-Zero)** 에 수렴하고, 메모리 활용률이 50~70%에서 **~100%** 로 향상됩니다.

---

## 🧱 Paged KV Cache의 내부 구조

### 1. 블록 단위 물리적 저장 (Block-based Physical Storage)

Paged KV Cache는 GPU 메모리 상에 다음과 같이 배치됩니다:

```
GPU HBM 메모리 (PagedAttention):
┌──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
│ Block 7  │ Block 3  │ Block 15 │ Block 22 │ Block 8  │ Block 1  │
│ (Seq A)  │ (Seq A)  │ (Seq B)  │ (Seq A)  │ (Seq C)  │ (Seq B)  │
└──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘
   ↑ 각 블록은 16 tokens의 KV를 저장, 물리적 주소는 불연속적
```

- 각 블록은 **고정된 토큰 수**(`BLOCK_SIZE`)의 Key와 Value 텐서를 저장합니다.
- 블록 크기는 보통 16 또는 32로 설정되며, 이는 내부 단편화(Internal Fragmentation)와 관리 오버헤드 간의 균형을 최적화합니다.

### 2. 블록 테이블을 통한 논리-물리 매핑

Paged KV Cache는 그 자체로는 논리적 연속성을 제공하지 않습니다.  
이를 위해 **Block Table(블록 테이블)** 이 각 Sequence의 논리적 블록 번호를 물리적 블록 ID로 매핑합니다.

```
Sequence A의 Block Table:
┌─────────────────┬──────────────────┐
│ Logical Block 0 │ Physical Block 7 │
│ Logical Block 1 │ Physical Block 3 │
│ Logical Block 2 │ Physical Block 22│
└─────────────────┴──────────────────┘
```

> 💡 **핵심**: Paged KV Cache는 **물리적 저장소(Physical Storage)**의 역할만 담당하며, 논리적 연속성은 [`atomic.block_table`](block_table.md)이 제공합니다. 두 개념은 분리되어 있지만, PagedAttention이 동작하기 위해서는 반드시 함께 사용됩니다.

---

## 📊 성능 영향 (Performance Impact)

| 지표 | Static KV Cache (연속 할당) | Paged KV Cache (블록 할당) |
|---|---|---|
| **외부 단편화** | 심각 (30~50% 낭비) | **거의 제로 (Near-Zero)** |
| **메모리 활용률** | 50~70% | **~100%** |
| **OOM 발생 빈도** | 빈번 | **거의 없음** |
| **할당 속도** | 느림 (cudaMalloc 호출) | **매우 빠름** (Free List에서 Pop) |
| **처리량 (Throughput)** | 제한적 | **10~23배 향상** |

---

## 🔗 관련 관계 (Relationships)

- **PREREQUISITES**: [`atomic.kv_cache`](kv_cache.md) (기본 KV Cache 개념을 확장)
- **COMPOSES_INTO**: [`composite.paged_attention_manager`](../02_composite/paged_attention_manager.md) (Module 04)
- **SYNERGY WITH**:
  - [`atomic.block_table`](block_table.md) (논리-물리 주소 매핑)
  - [`atomic.slot_mapping`](slot_mapping.md) (토큰 위치 → 블록 내 슬롯 변환)
  - [`atomic.memory_pool`](memory_pool.md) (Module 05, 블록을 할당할 물리적 메모리 풀)

---

## 📝 Brick-by-Brick 관점에서의 위치

Paged KV Cache는 **Module 02 (Static KV Cache)의 한계를 인지**하고, **Module 04 (PagedAttention)의 물리적 기반**을 제공하는 핵심 원자 개념입니다.

```
[Module 01] kv_cache (기본 개념)
     │
     ▼
[Module 02] static_kv_cache (연속 할당의 한계 체험)
     │
     ▼
[Module 04] paged_kv_cache ★★★★★
     → "연속성이 아닌 블록 단위로 메모리를 관리하자!"
     → 외부 단편화 제거, 메모리 활용률 극대화
     → PagedAttention의 물리적 저장소 역할
     │
     ▼
[Module 04] block_table, slot_mapping (논리-물리 변환)
```

> 💡 **핵심 인사이트**: Paged KV Cache는 vLLM의 가장 혁신적인 아이디어인 PagedAttention을 가능하게 하는 **물리적 데이터 구조**입니다. 연속 할당의 단점을 극복하고, 메모리 단편화를 완전히 제거함으로써 vLLM이 10~23배의 처리량 향상을 달성할 수 있게 합니다.
```