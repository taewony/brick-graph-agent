---
type: Module
id: module.paged_attention
title: PagedAttention — The Logical Abstraction (페이지드 어텐션)
description: 연속된 물리 메모리 할당의 한계를 극복하고, KV Cache를 고정 크기 블록 단위로 논리-물리 분리하여 관리하는 vLLM의 혁신적 메모리 관리 기법
status: draft
generated:
  by: agent:builder/1.0
  at: 2026-08-06T08:00:00Z
verified:
  - by: human:curator
    at: 2026-08-06T08:00:00Z
prerequisites:
  - module.continuous_batching
  - atomic.kv_cache
  - atomic.paged_kv_cache
composes_into:
  - composite.serving_system
sources:
  - id: original_lab
    resource: https://hackmd.io/9Ivogn3dRwm3WgA4Kl3fJQ#Module-4
    title: "nano-vLLM Module 4: PagedAttention - The Logical Abstraction"
---

# PagedAttention — The Logical Abstraction (페이지드 어텐션)

## 📌 개요

본 모듈은 **PagedAttention**의 핵심 개념을 다룹니다. Module 2에서 확인한 **연속 할당(Contiguous Allocation) 방식의 외부 단편화(External Fragmentation)** 문제를 해결하기 위해, 운영체제의 **가상 메모리 페이징(Virtual Memory Paging)** 개념을 차용하여 KV Cache를 **고정 크기 블록(Fixed-Size Blocks)** 단위로 관리하는 논리적 추상화를 구현합니다.

연속 할당 방식은 각 요청의 KV Cache를 하나의 연속된 GPU 메모리 블록에 할당하여, Sequence가 종료된 후에도 해제된 공간이 조각(Fragment)으로 남아 **실제로는 충분한 메모리가 있어도 할당 가능한 연속 공간이 없어 OOM(Out Of Memory)이 발생**하는 문제를 야기했습니다.

PagedAttention은 **논리적 캐시(Logical Cache)와 물리적 메모리(Physical Memory)를 분리(Decouple)** 하여 이 문제를 해결합니다.

## 🧩 포함 원자적 개념

- `atomic.paged_kv_cache` — KV Cache를 고정 크기 블록으로 분할하여 관리
- `atomic.block_table` — 논리적 블록 번호를 물리적 블록 주소로 매핑하는 테이블
- `atomic.slot_mapping` — 특정 토큰 위치를 물리적 블록 내 슬롯으로 변환

## 🏗️ 복합 개념 (Composite Concept)

- `composite.paged_attention_manager` — 블록 테이블 관리, 할당/해제, 슬롯 매핑을 총괄하는 PagedAttention 관리자

## 🔗 관련 관계

- **PREREQUISITES**: `kv_cache`, `continuous_batching`
- **PREREQUISITE_OF**: `serving_system`

---

## 🧱 1. 핵심 문제: 연속 할당의 한계 (The Problem)

### 1.1 외부 단편화 (External Fragmentation)

전통적인 LLM 서빙 시스템에서는 각 Sequence의 KV Cache를 **연속된(Contiguous) GPU 메모리 블록**으로 할당했습니다.

**문제의 연쇄 과정**:

1. Sequence A (1024 tokens)가 종료되어 메모리 해제
2. Sequence B (512 tokens)가 해제된 공간의 일부를 할당받음
3. Sequence C (800 tokens)가 도착했으나, **남은 연속 공간이 800 tokens 미만** → 할당 실패 (OOM)
4. **실제 남은 총 메모리는 1536 tokens**이지만, **파편화(Fragmentation)** 로 인해 사용 불가

```
GPU 메모리 주소 공간 스냅샷:
────────────────────────────────
████████████████████ Seq 1 (Active)
████████ Free (400) ← 800 tokens 요청에 비해 너무 작음
██████████████ Seq 2 (Active)
████████████████████████ Free (700) ← 가장 큰 연속 공간
██████ Seq 3 (Active)
████ Free (300) ← 너무 작음
────────────────────────────────
총 Free: 1400 tokens, 하지만 3개 조각으로 분할됨!
새 요청 (800 tokens) → ❌ 할당 불가!
```

### 1.2 GPU의 특수성

GPU 메모리는 **고정(Pinned) 할당**으로 운영체제가 자동으로 메모리를 압축(Compaction)할 수 없습니다.또한, 실행 중인 대규모 텐서를 이동시키려면 **전체 GPU 연산을 중단**해야 하므로, 처리량(Throughput)이 심각하게 저하됩니다.

---

## 🧱 2. 해결책: PagedAttention의 논리적 추상화

### 2.1 핵심 통찰: 논리-물리 분리 (Decouple Logical & Physical)

PagedAttention은 **운영체제의 가상 메모리(Virtual Memory) 개념**을 차용합니다.

- **논리적 캐시(Logical Cache)**: Sequence가 자신의 KV Cache를 **연속적인 블록 번호(0, 1, 2, ...)** 로 인식
- **물리적 메모리(Physical Memory)**: GPU 메모리에 **실제로 할당된 블록들** (불연속적, 7, 3, 15, ...)
- **블록 테이블(Block Table)**: 논리적 블록 번호 → 물리적 블록 주소를 매핑

### 2.2 3가지 핵심 추상화

#### ① Logical Block vs Physical Block

| 개념 | 설명 |
|---|---|
| **Logical Block** | 각 Sequence가 논리적으로 자신의 KV Cache를 구성하는 블록 번호 (연속적: 0, 1, 2, ...) |
| **Physical Block** | GPU 메모리에 실제로 할당된 블록의 물리적 주소 (불연속적: 7, 3, 15, ...) |

#### ② Block Table (블록 테이블)

각 Sequence는 자신만의 **Block Table**을 유지합니다.

```
Sequence A의 Block Table:
┌─────────────────┬──────────────────┐
│ Logical Block 0 │ Physical Block 7 │
│ Logical Block 1 │ Physical Block 3 │
│ Logical Block 2 │ Physical Block 15│
│ Logical Block 3 │ Physical Block 22│
└─────────────────┴──────────────────┘
```

#### ③ Slot Mapping (슬롯 매핑)

특정 토큰의 KV Cache가 **어느 물리적 블록의 몇 번째 슬롯(Slot)** 에 저장되는지 계산합니다.

```
계산 공식:
- logical_block = token_position // BLOCK_SIZE   (어느 논리 블록?)
- physical_block = block_table[logical_block]    (물리 블록 주소는?)
- slot = token_position % BLOCK_SIZE             (블록 내 몇 번째 위치?)
```

**예시** (`BLOCK_SIZE = 16`):

| Token Position | Logical Block | Physical Block | Slot |
|---|---|---|---|
| 0 | 0 | 7 | 0 |
| 15 | 0 | 7 | 15 |
| 16 | 1 | 3 | 0 |
| 31 | 1 | 3 | 15 |

---

## 🧱 3. 블록 단위 메모리 관리 (Block-Level Memory Management)

### 3.1 블록 할당 (Block Allocation)

1. 새 Sequence가 도착하면, **초기 KV Cache를 저장할 블록 1개**를 GPU 메모리 풀에서 할당
2. Sequence의 토큰이 증가함에 따라 **필요한 만큼의 블록을 동적으로 추가 할당**
3. Sequence가 종료되면 **모든 블록을 메모리 풀로 반환**

### 3.2 블록 재사용 (Block Reuse)

해제된 블록은 **다른 Sequence가 즉시 재사용**할 수 있습니다.

- 연속 할당 방식에서는 Sequence A가 해제한 1024 tokens 공간이 **조각으로 남아** 다른 Sequence가 사용하기 어려웠습니다.
- PagedAttention에서는 **블록 단위(예: 16 tokens)로 해제되므로**, 작은 블록들이 모여 **거의 100%에 가까운 메모리 활용률**을 달성합니다.

---

## 🧱 4. Module 3과의 연결: 계산 효율 + 메모리 효율

| 관점 | 설명 |
|---|---|
| **Module 3 → Module 4** | Continuous Batching이 **Prefill과 Decode가 혼합된 배치**를 지속적으로 생성합니다. 이 혼합 배치에서 각 요청의 KV Cache는 **동적으로 증가**합니다. PagedAttention은 이 **동적이고 불규칙한** KV Cache 성장을 **효율적으로 관리**하는 방법을 제공합니다. |
| **Module 4 → Module 3** | PagedAttention의 **블록 단위 할당**은 Scheduler가 **메모리 용량을 정확히 예측**하고 **배치 크기를 최적화**할 수 있게 합니다. 즉, 메모리 관리가 **스케줄링의 의사 결정**을 더 정교하게 만듭니다. |

### vLLM의 3대 혁신 (종합)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        vLLM 3대 혁신                              │
├─────────────────────────────────────────────────────────────────────┤
│  1. Continuous Batching (Module 3)                                │
│     → GPU 활용률 30~50% → 90%+                                   │
│     → "계산(compute) 효율"의 혁신                                 │
├─────────────────────────────────────────────────────────────────────┤
│  2. Paged KV Cache (PagedAttention) (Module 4)                    │
│     → 메모리 단편화 제거, near-zero waste                         │
│     → "메모리(memory) 효율"의 혁신                                │
├─────────────────────────────────────────────────────────────────────┤
│  3. Hybrid Scheduling (Module 3 + 4의 결합)                       │
│     → Continuous Batching + PagedAttention이 함께 동작            │
│     → Prefill 우선 + 메모리 인지형 스케줄링                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📊 효과: Near-Zero Waste

| 항목 | 기존 방식 (Contiguous) | PagedAttention |
|---|---|---|
| **메모리 단편화** | 심각한 외부 단편화 (30~50% 낭비) | **거의 제로 (near-zero)** |
| **메모리 활용률** | 50~70% | **~100%** |
| **OOM 발생** | 빈번 | **거의 없음** |
| **처리량 (Throughput)** | 제한적 | **10~23배 향상** |

---

## 📝 요약: Brick-by-Brick 관점

| 단계 | 학습한 개념 | 다음 개념으로의 연결 |
|---|---|---|
| **Module 1** | Autoregressive Loop + KV Cache | KV Cache가 메모리 병목임을 인지 |
| **Module 2** | KV Cache 메모리 문제 (연속 할당 + 단편화) | 메모리 문제 해결의 필요성 확인 |
| **Module 3** | Continuous Batching + Scheduler | 계산 효율 최적화, **동적 KV Cache 성장** 발생 |
| **Module 4** | PagedAttention (논리적 추상화) | **동적 KV Cache를 효율적으로 관리**하는 방법 제시 |
| **Module 5+** | Memory Pool / Allocator / Prefix Caching | PagedAttention 기반 고급 메모리 관리 기법으로 확장 |

> 💡 **핵심 인사이트**: Module 4는 **"GPU 메모리를 어떻게 낭비 없이 쓸까?"** 라는 질문에 답합니다. 연속 할당이 초래한 **외부 단편화**라는 치명적 문제를, 운영체제의 페이징 개념을 차용한 **논리-물리 주소 분리**로 해결함으로써, vLLM이 **10~23배의 처리량 향상**을 달성할 수 있는 메모리 기반을 제공합니다.