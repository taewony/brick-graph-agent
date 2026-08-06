---
type: AtomicConcept
id: atomic.attention
title: Scaled Dot-Product Attention (어텐션 메커니즘)
description: Query, Key, Value 텐서를 통해 시퀀스 내 토큰 간 가중 문맥 표현을 계산하는 핵심 어텐션 메커니즘
status: draft
generated:
  by: agent:builder/1.0
  at: 2026-08-05T13:15:00Z
verified:
  - by: human:curator
    at: 2026-08-05T13:15:00Z
---

# Scaled Dot-Product Attention (어텐션 메커니즘)

## 📌 개념 정의

**Scaled Dot-Product Attention**은 트랜스포머 아키텍처의 핵심 연산으로, 입력 시퀀스 토큰 간의 관련성을 계산하여 문맥 정보를 융합합니다.

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

- **Prefill 단계**: $Q, K, V$ 모두 전체 프롬프트 시퀀스 길이를 가져 병렬 GEMM 연산 수행
- **Decode 단계**: $Q$는 길이 1이며, $K, V$는 KV Cache에 보관된 전체 과거 시퀀스와 어텐션 연산(GEMV) 수행

---

## 🔗 관련 관계 (Relationships)

- **PREREQUISITE_OF**:
  - `decoder_layer`
