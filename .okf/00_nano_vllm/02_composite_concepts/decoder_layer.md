---
type: CompositeConcept
id: composite.decoder_layer
title: Transformer Decoder Layer (트랜스포머 디코더 레이어)
description: Attention 블록과 FFN 블록이 잔차 연결(Residual Connection) 및 RMSNorm과 결합된 기본 트랜스포머 구조 단위
status: draft
generated:
  by: agent:builder/1.0
  at: 2026-08-05T13:15:00Z
verified:
  - by: human:curator
    at: 2026-08-05T13:15:00Z
---

# Transformer Decoder Layer (트랜스포머 디코더 레이어)

## 📌 개념 정의

**Decoder Layer**는 `attention` 연산 블록과 `ffn` 블록이 잔차 연결(Residual Connection) 및 층 정규화(RMSNorm/LayerNorm)와 조합되어 수십 층으로 쌓이는 복합 개념 구조입니다.

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
  - `attention`
  - `ffn`
- **PREREQUISITE_OF**:
  - `autoregressive_loop`
