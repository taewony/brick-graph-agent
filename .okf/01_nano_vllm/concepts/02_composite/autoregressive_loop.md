---
type: CompositeConcept
id: composite.autoregressive_loop
title: Autoregressive Generation Loop (자기회귀 생성 루프)
description: Decoder Layer, KV Cache, Sampling이 통합되어 토큰을 1개씩 반복 생성하는 순차 제어 루프
status: draft
generated:
  by: agent:builder/1.0
  at: 2026-08-05 13:15:00+00:00
verified:
- by: human:curator
  at: 2026-08-05 13:15:00+00:00
sources:
- id: vllm-paper
  resource: https://arxiv.org/abs/2209.06155
  title: vLLM - Easy, Fast and Cheap LLM Serving with PagedAttention
prerequisite_of:
- composite.dynamic_batcher
- composite.static_cache_manager
prerequisites:
- atomic.kv_cache
- atomic.sampling
- composite.decoder_layer
---

# Autoregressive Generation Loop (자기회귀 생성 루프)

## 📌 개념 정의

**Autoregressive Generation Loop**는 Module 01의 핵심 복합 시스템으로, [`decoder_layer`](decoder_layer.md)를 통한 Forward pass, [`kv_cache`](../03_atomic/kv_cache.md) 업데이터, [`sampling`](../03_atomic/sampling.md) 알고리즘을 결합하여 `EOS` 토큰이 생성될 때까지 디코드 루프를 반복 구동합니다.

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
  - [`decoder_layer`](decoder_layer.md)
  - [`kv_cache`](../03_atomic/kv_cache.md)
  - [`sampling`](../03_atomic/sampling.md)
- **PREREQUISITES**:
  - [`inference_model`](inference_model.md)
- **PREREQUISITE_OF**:
  - [`continuous_batching`](../03_atomic/continuous_batching.md) (Module 03)