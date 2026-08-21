---
type: AtomicConcept
id: atomic.inference_only
title: Inference-Only Workload (추론 전용 워크로드)
description: 학습(Training)과 달리 역전파 및 가중치 업데이트가 없으며 Forward Pass 및 메모리 읽기 중심인 추론 고유의
  실행 특성
status: draft
generated:
  by: agent:builder/1.0
  at: 2026-08-05 13:11:00+00:00
verified:
- by: human:curator
  at: 2026-08-05 13:11:00+00:00
sources:
- id: vllm-paper
  resource: https://arxiv.org/abs/2209.06155
  title: vLLM - Easy, Fast and Cheap LLM Serving with PagedAttention
prerequisite_of:
- atomic.tensor_parallelism
---

# Inference-Only Workload (추론 전용 워크로드)

## 📌 개념 정의

**Inference-Only Workload**는 이미 학습 완료된 LLM 모델 파라미터를 고정(Freeze)한 상태에서 입력 텍스트에 대한 Forward Pass만을 수행하는 작업입니다.

학습(Training) 단계와 비교했을 때 다음과 같은 핵심 차이점이 존재합니다:
1. **No Gradient Accumulation**: 역전파(Backpropagation)를 위한 활성화 값(Activation) 저장이 불필요함.
2. **Deterministic Weights**: 모델 가중치(Weights)가 불변(Immutable) 상태로 메모리에 유지됨.
3. **KV Cache Dependency**: 이전 단계 토큰들의 연산 재사용을 위해 KV 캐시 관리가 핵심 성능 병목 요소로 작용함.

---

## 📊 연산 특성 비교

| 구분 | Training | Inference-Only |
|---|---|---|
| **Pass** | Forward + Backward Pass | Forward Pass Only |
| **Memory Bottleneck** | Activation memory + Optimizer States | Model Weights + KV Cache |
| **Compute Pattern** | Large GEMM operations | GEMM (Prefill) + GEMV (Decode) |
| **Batching Strategy** | Static Uniform Batching | Dynamic Continuous Batching |

---

## 🔗 관련 관계 (Relationships)

- **PREREQUISITE_OF**:
  - [`prefill_phase`](prefill_phase.md)
  - [`decode_phase`](decode_phase.md)