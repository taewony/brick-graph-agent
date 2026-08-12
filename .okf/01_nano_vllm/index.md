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
| **Module 00** | Foundational Concepts | `inference_only`, `prefill_phase`, `decode_phase` | 🟢 완료 |
| **Module 01** | Autoregressive Generation | `kv_cache`, `attention`, `ffn`, `sampling` | 🟢 완료 |
| **Module 02** | Basic KV Cache Engine | `static_kv_cache`, `seq_len_budget` | 🟢 완료 |
| **Module 03** | Continuous Batching | `continuous_batching`, `iteration_level_scheduling` | 🟢 완료 |
| **Module 04** | PagedAttention Architecture | `paged_kv_cache`, `block_table`, `slot_mapping` | 🟢 완료 |
| **Module 05** | Memory Management & Allocation | `memory_pool`, `block_allocator`, `swap_manager`, `block_manager` | 🟢 완료 |
| **Module 06** | Advanced Prefix Caching | `prefix_cache`, `cache_hit_detection`, `caching_strategy` | 🟢 완료 |
| **Module 07** | Multi-GPU & Distributed Serving | `distributed_kv`, `load_balancer`, `fault_tolerance`, `distributed_serving` | 🟢 완료 |

---

## 🔗 주요 퀵 링크

- [학습 경로 가이드 (learning_path.md)](meta/learning_path.md)
- [용어 사전 (glossary.md)](meta/glossary.md)
- [변경 이력 (log.md)](log.md)
