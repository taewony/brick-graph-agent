---
type: CompositeConcept
id: composite.decoder_layer
title: Transformer Decoder Layer (트랜스포머 디코더 레이어)
description: Attention 블록과 FFN 블록이 잔차 연결(Residual Connection) 및 RMSNorm과 결합된 기본 트랜스포머
  구조 단위
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
prerequisites:
- atomic.attention
- atomic.ffn
---

# Transformer Decoder Layer (트랜스포머 디코더 레이어)

## 📌 개념 정의

**Decoder Layer**는 [`attention`](../03_atomic/attention.md) 연산 블록과 [`ffn`](../03_atomic/ffn.md) 블록이 잔차 연결(Residual Connection) 및 층 정규화(RMSNorm/LayerNorm)와 조합되어 수십 층으로 쌓이는 복합 개념 구조입니다.

---

## 🧱 구조적 파이프라인

```
Input x
  │
  ├──► RMSNorm ──► Attention ──(KV Cache Sync)──┐
  │                                             │
  ▼                                             ▼
Add Residual <──────────────────────────────────┘
  │
  ├──► RMSNorm ──► Feed-Forward (FFN) ──────────┐
  │                                             │
  ▼                                             ▼
Add Residual <──────────────────────────────────┘
  │
Output Layer Token Hidden State
```

---

## 🔗 조립 정보 (Composition & Relationships)

- **COMPOSED_OF**:
  - [`attention`](../03_atomic/attention.md)
  - [`ffn`](../03_atomic/ffn.md)
- **PREREQUISITE_OF**:
  - [`autoregressive_loop`](autoregressive_loop.md)