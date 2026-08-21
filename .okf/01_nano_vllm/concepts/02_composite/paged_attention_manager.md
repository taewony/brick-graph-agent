---
type: CompositeConcept
id: composite.paged_attention_manager
title: PagedAttention Manager (페이지드 어텐션 관리자)
description: Paged KV Cache, Block Table, Slot Mapping을 종합 관리하여 GPU 메모리 단편화를 방지하고
  연속적인 어텐션 연산을 보장하는 복합 개념 컴포넌트
status: draft
generated:
  by: agent:builder/1.0
  at: 2026-08-07 09:00:00+00:00
verified:
- by: human:curator
  at: 2026-08-07 09:00:00+00:00
prerequisites:
- atomic.paged_kv_cache
- atomic.block_table
- atomic.slot_mapping
composed_of:
- atomic.paged_kv_cache
- atomic.block_table
- atomic.slot_mapping
composes_into:
- module.paged_attention
sources:
- id: vllm-paper
  resource: https://arxiv.org/abs/2209.06155
  title: vLLM - Easy, Fast and Cheap LLM Serving with PagedAttention
prerequisite_of:
- composite.block_manager
- composite.distributed_serving
---

# PagedAttention Manager (페이지드 어텐션 관리자)

## 📌 개념 정의

**PagedAttention Manager**는 PagedAttention 메커니즘을 구동하기 위한 통합 메모리 컨트롤러로, **물리적 저장소([`paged_kv_cache`](../03_atomic/paged_kv_cache.md))**, **논리-물리 주소 매핑([`block_table`](../03_atomic/block_table.md))**, **토큰-슬롯 인덱서([`slot_mapping`](../03_atomic/slot_mapping.md))**를 결합하여 고성능 GPU 어텐션 연산을 관리합니다.

---

## 🔗 관련 관계 (Relationships)

- **PREREQUISITES**: [`atomic.paged_kv_cache`](../03_atomic/paged_kv_cache.md), [`atomic.block_table`](../03_atomic/block_table.md), [`atomic.slot_mapping`](../03_atomic/slot_mapping.md)
- **COMPOSED_OF**: [`atomic.paged_kv_cache`](../03_atomic/paged_kv_cache.md), [`atomic.block_table`](../03_atomic/block_table.md), [`atomic.slot_mapping`](../03_atomic/slot_mapping.md)
- **COMPOSES_INTO**: [`module.paged_attention`](../01_module/module_04_paged_attention.md)