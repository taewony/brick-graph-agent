---
type: AtomicConcept
id: atomic.block_allocator
title: Block Allocator (블록 할당기)
description: 메모리 풀에서 고정‑크기 블록을 효율적으로 할당·해제하는 로직. 연속 배치와 KV 캐시가 요구하는 빠른
메모리 접근을 지원한다.
status: draft
generated:
  by: agent:builder/1.0
  at: 2026-08-06T07:59:21Z
verified: []
---

# Block Allocator (블록 할당기)

## 📌 개념 정의
**Block Allocator**는 `memory_pool`에서 미리 할당된 고정‑크기 메모리 블록을 **시점‑기반**으로 가져오고 반환하는
책임을 진다.
- **대기열 없이 즉시 할당**: 배치가 시작될 때 즉시 사용 가능한 블록을 제공.
- **단순 해제**: 사용이 끝난 블록을 풀에 되돌려 재사용을 보장.

## ⚙️ 핵심 메서드
- `allocate()` → 사용 가능한 가장 작은 블록 반환.
- `free(block_id)` → 블록을 풀에 반환, 재사용 대기.

## 🔗 관련 관계
- **PREREQUISITES**: `memory_pool`
- **COMPOSED_OF**: `swap_manager` (스와핑 지원)