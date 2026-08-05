---
type: CompositeConcept
id: composite.autoregressive_loop
title: Autoregressive Generation Loop (자기회귀 생성 루프)
description: Decoder Layer, KV Cache, Sampling이 통합되어 토큰을 1개씩 반복 생성하는 순차 제어 루프
status: draft
generated:
  by: agent:builder/1.0
  at: 2026-08-05T13:15:00Z
verified:
  - by: human:curator
    at: 2026-08-05T13:15:00Z
---

# Autoregressive Generation Loop (자기회귀 생성 루프)

## 📌 개념 정의

**Autoregressive Generation Loop**는 Module 01의 핵심 복합 시스템으로, `decoder_layer`를 통한 Forward pass, `kv_cache` 업데이터, `sampling` 알고리즘을 결합하여 `EOS` 토큰이 생성될 때까지 디코드 루프를 반복 구동합니다.

---

## 🧱 시스템 루프 구조

```
[Prefill 토큰 텐서 & KV Cache 초기화]
               │
               ▼
   ┌───► [Decode Step Forward Pass] ◄───┐
   │           │                        │
   │           ▼                        │
   │     [KV Cache Append]              │ Repeat Loop
   │           │                        │ (Until EOS Token)
   │           ▼                        │
   │     [Logits & Sampling]            │
   │           │                        │
   └───────────┴── Next Token Output ───┘
```

---

## 🔗 조립 정보 (Composition & Relationships)

- **COMPOSED_OF**:
  - `decoder_layer`
  - `kv_cache`
  - `sampling`
- **PREREQUISITES**:
  - `inference_model`
- **PREREQUISITE_OF**:
  - `continuous_batching` (Module 03)
