---
type: AtomicConcept
id: atomic.ffn
title: Feed-Forward Network (FFN / MLP)
description: 어텐션층을 거친 각 토큰의 비선형 특징 표현을 위치별(Position-wise)로 독립적 변환하는 신경망 블록
status: draft
generated:
  by: agent:builder/1.0
  at: 2026-08-05T13:15:00Z
verified:
  - by: human:curator
    at: 2026-08-05T13:15:00Z
---

# Feed-Forward Network (FFN / MLP)

## 📌 개념 정의

**Feed-Forward Network (FFN)** 또는 MLP(Multi-Layer Perceptron) 블록은 어텐션 결과 텐서를 받아 각 토큰별로 비선형 투영(Projection) 및 차원 확장을 수행하는 레이어입니다.

최근 LLM(Llama, Mistral 등)에서는 SwiGLU 또는 GELU 활성화 함수 기반 3개의 선형 레이어(Gate, Up, Down Projection)로 구성됩니다.

$$\text{FFN}(x) = (\text{Swish}(xW_{\text{gate}}) \odot xW_{\text{up}}) W_{\text{down}}$$

---

## 🔗 관련 관계 (Relationships)

- **PREREQUISITE_OF**:
  - `decoder_layer`
