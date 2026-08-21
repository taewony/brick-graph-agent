---
type: Index
id: index.nano_vllm
title: nano-vLLM OKF 지식 베이스 메인 인덱스
description: nano-vLLM 7개 모듈 및 계층적 개념 체계(Atomic/Composite)를 관리하는 메인 인덱스 문서
status: draft
generated:
  by: agent:builder/1.0
  at: 2026-08-05T13:06:00Z
verified:
  - by: human:curator
    at: 2026-08-05T13:06:00Z
---

# nano-vLLM OKF 지식 베이스 (Knowledge Base)

> **nano-vLLM**: vLLM의 고성능 서빙 아키텍처(PagedAttention, Continuous Batching, Memory Pool, Prefix Caching, Distributed Serving)를 바닥부터(Build-from-Scratch) 구현하며 학습하기 위한 계층적 OKF 지식 체계입니다.

---

## 📌 지식 구조 체계 (Knowledge Structure)

1. **[meta](meta/learning_path.md)**: 전체 학습 경로(`learning_path.md`) 및 주요 용어 사전(`glossary.md`)
2. **[03_atomic](concepts/03_atomic)**: 더 이상 분해되지 않는 기본 개념 단위 (Prefill/Decode, KV Cache, Attention, Block Allocator 등)
3. **[02_composite](concepts/02_composite)**: 원자적 개념들이 `COMPOSED_OF` 관계로 결합된 하위/상위 복합 시스템 (Block Manager, Dynamic Batcher, Serving Engine 등)
4. **[01_module](concepts/01_module)**: Module 00부터 Module 07까지의 실습 및 아키텍처 모듈 명세

---

## 🗺️ nano-vLLM 7개 모듈 맵 (7-Module Map)

| 모듈 | 제목 | 핵심 개념 (Atomic / Composite) | 상태 |
|---|---|---|---|
| **[Module 00](concepts/01_module/module_00_foundational.md)** | Foundational Concepts | [`inference_only`](concepts/03_atomic/inference_only.md), [`prefill_phase`](concepts/03_atomic/prefill_phase.md), [`decode_phase`](concepts/03_atomic/decode_phase.md) | 🟢 완료 |
| **[Module 01](concepts/01_module/module_01_autoregressive.md)** | Autoregressive Generation | [`kv_cache`](concepts/03_atomic/kv_cache.md), [`attention`](concepts/03_atomic/attention.md), [`ffn`](concepts/03_atomic/ffn.md), [`sampling`](concepts/03_atomic/sampling.md) | 🟢 완료 |
| **[Module 02](concepts/01_module/module_02_basic_kv_cache.md)** | Basic KV Cache Engine | [`static_kv_cache`](concepts/03_atomic/static_kv_cache.md), [`seq_len_budget`](concepts/03_atomic/seq_len_budget.md) | 🟢 완료 |
| **[Module 03](concepts/01_module/module_03_continuous_batching.md)** | Continuous Batching | [`continuous_batching`](concepts/03_atomic/continuous_batching.md), [`iteration_level_scheduling`](concepts/03_atomic/iteration_level_scheduling.md) | 🟢 완료 |
| **[Module 04](concepts/01_module/module_04_paged_attention.md)** | PagedAttention Architecture | [`paged_kv_cache`](concepts/03_atomic/paged_kv_cache.md), [`block_table`](concepts/03_atomic/block_table.md), [`slot_mapping`](concepts/03_atomic/slot_mapping.md) | 🟢 완료 |
| **[Module 05](concepts/01_module/module_05_memory_management.md)** | Memory Management & Allocation | [`memory_pool`](concepts/03_atomic/memory_pool.md), [`block_allocator`](concepts/03_atomic/block_allocator.md), [`swap_manager`](concepts/03_atomic/swap_manager.md), [`block_manager`](concepts/02_composite/block_manager.md) | 🟢 완료 |
| **[Module 06](concepts/01_module/module_06_prefix_caching.md)** | Advanced Prefix Caching | [`prefix_cache`](concepts/03_atomic/prefix_cache.md), [`cache_hit_detection`](concepts/03_atomic/cache_hit_detection.md), [`caching_strategy`](concepts/02_composite/caching_strategy.md) | 🟢 완료 |
| **[Module 07](concepts/01_module/module_07_distributed_serving.md)** | Multi-GPU & Distributed Serving | [`distributed_kv`](concepts/03_atomic/distributed_kv.md), [`load_balancer`](concepts/03_atomic/load_balancer.md), [`fault_tolerance`](concepts/03_atomic/fault_tolerance.md), [`distributed_serving`](concepts/02_composite/distributed_serving_system.md) | 🟢 완료 |

---

## 🔗 주요 퀵 링크

- [학습 경로 가이드 (learning_path.md)](meta/learning_path.md)
- [용어 사전 (glossary.md)](meta/glossary.md)
- [변경 이력 (log.md)](log.md)