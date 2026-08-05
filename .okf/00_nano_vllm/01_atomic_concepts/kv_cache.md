---
type: AtomicConcept
id: atomic.kv_cache
title: Key-Value Cache (KV 캐시)
description: 자기회귀 생성 과정에서 이전 스텝 토큰들의 Key, Value 행렬을 메모리에 보관하여 재계산을 방지하는 필수 기법
status: draft
generated:
  by: agent:builder/1.0
  at: 2026-08-05T13:15:00Z
verified:
  - by: human:curator
    at: 2026-08-05T13:15:00Z
---

# Key-Value Cache (KV 캐시)

## 📌 개념 정의

**KV Cache**는 자기회귀(Autoregressive) 텍스트 생성 과정에서 이미 계산된 이전 토큰들의 Key($K$) 및 Value($V$) 텐서를 메모리에 저장하고, 매 디코드 스텝마다 신규 토큰의 $K, V$만 새로 계산하여 기존 캐시에 병합(Append)하는 기술입니다.

KV Cache가 없을 경우 $t$번째 토큰을 생성할 때 1부터 $t-1$번째 토큰까지의 어텐션을 매번 처음부터 재계산해야 하므로, 복잡도가 $O(N^2)$에서 $O(N)$으로 획기적으로 줄어듭니다.

---

## 📐 KV Cache 텐서 예시

- **Key Cache Shape**: `[Batch Size, Num Key Heads, Current Seq Length, Head Dim]`
- **Value Cache Shape**: `[Batch Size, Num Key Heads, Current Seq Length, Head Dim]`

---

## 🔗 관련 관계 (Relationships)

- **PREREQUISITES**:
  - `prefill_phase`
  - `decode_phase`
- **PREREQUISITE_OF**:
  - `autoregressive_loop`
  - `static_kv_cache` (Module 02)
  - `paged_kv_cache` (Module 04)
