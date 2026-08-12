---
type: Document
id: meta.learning_path
title: nano-vLLM 7개 모듈 전체 학습 경로 (Learning Path)
description: 원자적 개념(AtomicConcept)부터 시작하여 복합 시스템(CompositeConcept)으로 쌓아 올려 7개 모듈을 정복하는 Brick-by-Brick 학습 가이드
status: draft
generated:
  by: agent:builder/1.0
  at: 2026-08-05T13:06:00Z
verified:
  - by: human:curator
    at: 2026-08-05T13:06:00Z
---

# nano-vLLM 7개 모듈 전체 학습 경로 (Learning Path)

> **Brick-by-Brick 학습 철학**: 복잡한 LLM 서빙 엔진 아키텍처를 한번에 이해하려 하지 않고, 원자적 개념(Atomic Concept)을 하나씩 마스터(`mastered`)한 뒤 선행 조건(`PREREQUISITE_OF`)을 충족시켜 상위 복합 시스템(Composite Concept)으로 조립합니다.

---

## 🗺️ 전체 계층적 학습 흐름 (Hierarchy Flow)

```
[Module 00: 기본 개념] ──► [Module 01: 자기회귀 생성] ──► [Module 02: 정적 KV 캐시]
                                                                  │
┌─────────────────────────────────────────────────────────────────┘
▼
[Module 03: Continuous Batching] ──► [Module 04: PagedAttention]
                                           │
┌──────────────────────────────────────────┘
▼
[Module 05: Memory Pool / Allocator] ──► [Module 06: Prefix Caching] ──► [Module 07: Multi-GPU / Distributed]
```

---

## 🧱 모듈별 선행조건 & 조립 단계 (Step-by-Step Prerequisites)

### 1단계: Foundational & Autoregressive (Module 00 ~ Module 02)
1. **[prefill_phase](file:///D:/code/brick-graph-agent/.okf/00_nano_vllm/01_atomic_concepts/prefill_phase.md)** & **[decode_phase](file:///D:/code/brick-graph-agent/.okf/00_nano_vllm/01_atomic_concepts/decode_phase.md)** 학습
   - *선행 조건*: 없음 (원자적 개념)
2. **[kv_cache](file:///D:/code/brick-graph-agent/.okf/00_nano_vllm/01_atomic_concepts/kv_cache.md)** 학습
   - *선행 조건*: `prefill_phase`, `decode_phase`
3. **[autoregressive_loop](file:///D:/code/brick-graph-agent/.okf/00_nano_vllm/02_composite_concepts/autoregressive_loop.md)** 조립
   - *조립 구성요소*: `kv_cache`, `attention`, `sampling`

### 2단계: Batching & Memory Paging (Module 03 ~ Module 04)
4. **[continuous_batching](file:///D:/code/brick-graph-agent/.okf/00_nano_vllm/01_atomic_concepts/continuous_batching.md)** 학습
   - *선행 조건*: `autoregressive_loop`
5. **[paged_kv_cache](file:///D:/code/brick-graph-agent/.okf/00_nano_vllm/01_atomic_concepts/paged_kv_cache.md)** 학습
   - *선행 조건*: `kv_cache`, `continuous_batching`

### 3단계: Advanced Systems (Module 05 ~ Module 07)
6. **[block_allocator](file:///D:/code/brick-graph-agent/.okf/00_nano_vllm/01_atomic_concepts/block_allocator.md)** & **[memory_pool](file:///D:/code/brick-graph-agent/.okf/00_nano_vllm/01_atomic_concepts/memory_pool.md)** 학습
7. **[prefix_cache](file:///D:/code/brick-graph-agent/.okf/00_nano_vllm/01_atomic_concepts/prefix_cache.md)** & **[distributed_kv](file:///D:/code/brick-graph-agent/.okf/00_nano_vllm/01_atomic_concepts/distributed_kv.md)** 학습 및 최상위 복합 서빙 엔진 조립
