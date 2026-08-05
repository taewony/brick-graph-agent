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

1. **[00_meta](file:///D:/code/brick-graph-agent/.okf/00_nano_vllm/00_meta)**: 전체 학습 경로(`learning_path.md`) 및 주요 용어 사전(`glossary.md`)
2. **[01_atomic_concepts](file:///D:/code/brick-graph-agent/.okf/00_nano_vllm/01_atomic_concepts)**: 더 이상 분해되지 않는 기본 개념 단위 (Prefill/Decode, KV Cache, Attention, Block Allocator 등)
3. **[02_composite_concepts](file:///D:/code/brick-graph-agent/.okf/00_nano_vllm/02_composite_concepts)**: 원자적 개념들이 `COMPOSED_OF` 관계로 결합된 하위/상위 복합 시스템 (Block Manager, Dynamic Batcher, Serving Engine 등)
4. **[03_modules](file:///D:/code/brick-graph-agent/.okf/00_nano_vllm/03_modules)**: Module 00부터 Module 07까지의 실습 및 아키텍처 모듈 명세

---

## 🗺️ nano-vLLM 7개 모듈 맵 (7-Module Map)

| 모듈 | 제목 | 핵심 개념 (Atomic / Composite) | 상태 |
|---|---|---|---|
| **Module 00** | Foundational Concepts | `inference_only`, `prefill_phase`, `decode_phase` | 🟢 완료 |
| **Module 01** | Autoregressive Generation | `kv_cache`, `attention`, `ffn`, `sampling` | 🟢 완료 |
| **Module 02** | Basic KV Cache Engine | `static_kv_cache`, `seq_len_budget` | 🟢 완료 |
| **Module 03** | Continuous Batching | `continuous_batching`, `iteration_level_scheduling` | ⚪ 대기 |
| **Module 04** | PagedAttention Architecture | `paged_kv_cache`, `block_size_alignment` | ⚪ 대기 |
| **Module 05** | Memory Management & Allocation | `memory_pool`, `block_allocator`, `swap_manager` | ⚪ 대기 |
| **Module 06** | Advanced Prefix Caching | `prefix_cache`, `radix_tree`, `cache_hit_detection` | ⚪ 대기 |
| **Module 07** | Multi-GPU & Distributed Serving | `distributed_kv`, `load_balancer`, `tensor_parallel` | ⚪ 대기 |

---

## 🔗 주요 퀵 링크

- [학습 경로 가이드 (learning_path.md)](file:///D:/code/brick-graph-agent/.okf/00_nano_vllm/00_meta/learning_path.md)
- [용어 사전 (glossary.md)](file:///D:/code/brick-graph-agent/.okf/00_nano_vllm/00_meta/glossary.md)
- [변경 이력 (log.md)](file:///D:/code/brick-graph-agent/.okf/00_nano_vllm/log.md)
