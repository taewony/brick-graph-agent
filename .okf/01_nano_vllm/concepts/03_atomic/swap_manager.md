---
type: AtomicConcept
id: atomic.swap_manager
title: Swap Manager (스와핑 관리자)
description: "메모리 풀이 포화될 경우, 저우선순위 블록을 디스크(스와프)로 옮겨 실시간 메모리 사용량을 조절하는 보조 컴포넌트."
status: draft
generated:
  by: agent:builder/1.0
  at: 2026-08-06T07:59:21Z
verified: []
sources:
  - id: vllm-paper
    resource: https://arxiv.org/abs/2209.06155
    title: vLLM - Easy, Fast and Cheap LLM Serving with PagedAttention
---

# Swap Manager (스와핑 관리자)

## 📌 개념 정의
**Swap Manager**는 [`memory_pool`](memory_pool.md)이 메모리 한계에 도달했을 때, **덜 중요한 블록**을 디스크에 스와핑(swap)하고 필요
시 다시 읽어오는 메커니즘이다.
- **스와프 스레드**: 백그라운드에서 비동기로 동작.
- **우선순위 정책**: 오래된·사용 빈도가 낮은 블록을 우선 스와핑.

## ⚙️ 주요 흐름
1. **점검** – 현재 풀 사용량 > `max_pool_size`?
2. **선별** – `eviction_priority`에 따라 스와핑 대상 선정.
3. **스와핑** – 블록을 파일 시스템(또는 SSD)으로 이동, 메모리 풀에 빈 슬롯 확보.
4. **복구** – 필요 시 `swap_in(block_id)` 로 메모리 복구.

## 🔗 관련 관계
- **PREREQUISITES**: [`memory_pool`](memory_pool.md), [`block_allocator`](block_allocator.md)
- **PREREQUISITE_OF**: [`composite.block_manager`](../02_composite/block_manager.md) (블록 추적)