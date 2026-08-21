---
type: Module
id: module.01_autoregressive
title: "Module 01: Autoregressive Generation (자기회귀 생성)"
description: KV Cache, Attention, FFN, Sampling 기법을 활용하여 토큰을 자기회귀 방식으로 생성하는 모듈
status: draft
generated:
  by: agent:builder/1.0
  at: 2026-08-05T13:15:00Z
verified:
  - by: human:curator
    at: 2026-08-05T13:15:00Z
---

# Module 01: Autoregressive Generation (자기회귀 생성)

> **모듈 목표**: LLM이 이전 토큰들을 기반으로 다음 토큰 확률 분포를 계산하고 샘플링하여 텍스트를 지속적으로 확장해 나가는 **Autoregressive Loop**를 최소 파이프라인으로 구현합니다.

---

## 🧱 포함된 개념 (Concepts)

### 원자적 개념 (Atomic Concepts)
- **[`kv_cache`](../03_atomic/kv_cache.md)**: 이전 토큰들의 Key, Value 행렬을 메모리에 보관하여 중복 어텐션 계산을 방지하는 캐싱 기법
- **[`attention`](../03_atomic/attention.md)**: Query, Key, Value 텐서 연산을 통해 토큰 간 상호작용 문맥 점수를 산출하는 어텐션 메커니즘
- **[`ffn`](../03_atomic/ffn.md)**: 각 토큰 표현을 독립적으로 변환하는 Feed-Forward Network 레이어
- **[`sampling`](../03_atomic/sampling.md)**: 모델의 Logit 분포에서 Temperature, Top-p, Top-k 등을 적용하여 다음 토큰 ID를 선택하는 기법

### 복합 개념 (Composite Concepts)
- **[`decoder_layer`](../02_composite/decoder_layer.md)**: Attention과 FFN이 수신되어 형성된 Transformer 트랜스포머 블록
- **[`autoregressive_loop`](../02_composite/autoregressive_loop.md)**: Decoder Layer와 Sampling, KV Cache가 통합되어 순차적으로 토큰을 생성하는 반복 루프

---

## 🔗 학습 경로 및 선행 조건 (Learning Flow)

```
[Module 00: inference_model]
          │
          ▼
     [kv_cache]  ──┐
     [attention] ──┼──► [decoder_layer] ──┐
     [ffn]       ──┘                      ├──► [autoregressive_loop] (Composite)
     [sampling]  ─────────────────────────┘
```

---

## 📝 실습 과제 (Lab Assignment)

1. KV Cache 미사용 시와 사용 시의 어텐션 연산 시간 복잡도($O(N^2)$ vs $O(N)$) 비교 구현하기
2. Logits에 Temperature scaling 및 Top-p (Nucleus) 필터링을 적용하는 [`sampling`](../03_atomic/sampling.md) 메서드 작성하기