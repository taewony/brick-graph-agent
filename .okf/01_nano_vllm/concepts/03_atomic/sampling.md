---
type: AtomicConcept
id: atomic.sampling
title: Token Sampling Strategies (토큰 샘플링 기법)
description: 모델의 Logit 분포에서 다음 토큰 ID를 결정하기 위한 확률 필터링 및 추출 기법
status: draft
generated:
  by: agent:builder/1.0
  at: 2026-08-05T13:15:00Z
verified:
  - by: human:curator
    at: 2026-08-05T13:15:00Z
sources:
  - id: vllm-paper
    resource: https://arxiv.org/abs/2209.06155
    title: vLLM - Easy, Fast and Cheap LLM Serving with PagedAttention
---

# Token Sampling Strategies (토큰 샘플링 기법)

## 📌 개념 정의

**Sampling** 기법은 LLM의 최종 Linear Head를 거쳐 나온 Logit 텐서에서 다음 생성될 단어(Token ID)를 결정하는 알고리즘입니다.

### 핵심 제어 파라미터:
1. **Temperature ($T$)**: 확률 분포의 무작위성(Randomness) 조율 ($T \to 0$일수록 Greedy Search)
2. **Top-k Sampling**: Logit 상위 $k$개 토큰 후보만 남기고 나머지 제외
3. **Top-p (Nucleus) Sampling**: 누적 확률 분포가 $p$에 도달할 때까지 상위 토큰들만 동적으로 선별

---

## 🔗 관련 관계 (Relationships)

- **PREREQUISITE_OF**:
  - [`autoregressive_loop`](../02_composite/autoregressive_loop.md)