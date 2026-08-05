---
type: CompositeConcept
id: composite.static_cache_manager
title: Static Cache Manager (정적 캐시 관리자)
description: Static KV Cache 버퍼 인덱싱, 토큰 Append, 슬라이스 어텐션 생성을 총괄하는 1세대 캐시 관리 엔진
status: draft
generated:
  by: agent:builder/1.0
  at: 2026-08-05T13:22:00Z
verified:
  - by: human:curator
    at: 2026-08-05T13:22:00Z
---

# Static Cache Manager (정적 캐시 관리자)

## 📌 개념 정의

**Static Cache Manager**는 `static_kv_cache` 버퍼와 `seq_len_budget` 관리 정책이 수신되어 완성된 기초 캐시 관리 컴포넌트입니다.

---

## 🧱 컴포넌트 메커니즘

```
[New Key, Value Tensor (1 Token)]
              │
              ▼
┌─────────────────────────────────────────────────────────┐
│ Static Cache Manager                                    │
│  1. Check seq_len_budget limit                          │
│  2. Insert K,V to static_kv_cache buffer at current_pos │
│  3. Increment current_pos pointer                       │
│  4. Return valid slice buffer for Attention computation │
└─────────────────────────────────────────────────────────┘
              │
              ▼
[Valid Key/Value Slice Matrix for Attention]
```

---

## 🔗 조립 정보 (Composition & Relationships)

- **COMPOSED_OF**:
  - `static_kv_cache`
  - `seq_len_budget`
- **PREREQUISITES**:
  - `autoregressive_loop` (Module 01)
- **PREREQUISITE_OF**:
  - `continuous_batching` (Module 03)
