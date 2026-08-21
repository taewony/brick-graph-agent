---
type: CompositeConcept
id: composite.dynamic_batcher
title: Dynamic Batcher (동적 배치 엔진)
description: Static Batching(정적 배치)의 GPU 유휴 문제를 해결하기 위해, 매 Iteration(Iteration-Level
  Scheduling)마다 배치를 동적으로 재구성하여 GPU 활용률을 30~50%에서 90% 이상으로 끌어올리는 핵심 Composite 개념
status: draft
generated:
  by: agent:builder/1.0
  at: 2026-08-06 07:20:00+00:00
verified:
- by: human:curator
  at: 2026-08-06 07:20:00+00:00
components:
- continuous_batching
- iteration_level_scheduling
prerequisites:
- atomic.iteration_level_scheduling
- atomic.length_aware_scheduler
- atomic.max_batch_size
- composite.autoregressive_loop (Module 01)
- composite.static_cache_manager (Module 02)
composes_into:
- composite.serving_system (Module 03+04 통합)
sources:
- id: vllm-paper
  resource: https://arxiv.org/abs/2209.06155
  title: vLLM - Easy, Fast and Cheap LLM Serving with PagedAttention
---

# Dynamic Batcher (동적 배치 엔진)

## 📌 개념 정의

**Dynamic Batcher**는 [`continuous_batching`](../03_atomic/continuous_batching.md)(연속 배치)과 [`iteration_level_scheduling`](../03_atomic/iteration_level_scheduling.md)(반복 수준 스케줄링)을 결합한 복합 개념입니다.  
기존 Static Batching(정적 배치)이 하나의 긴 요청 때문에 전체 배치가 블로킹(Blocking)되어 GPU가 유휴 상태로 남는 문제를 해결합니다.

---

## 🧱 Core Technical Analysis: Static vs Continuous Batching

### 1. Static Batching (The Baseline & Bottleneck)

**Static Batching**은 고정된 요청 집합으로 배치를 구성하고, 해당 배치의 모든 요청이 현재 단계(Phase)를 완전히 마칠 때까지 처리한 후 다음 배치로 넘어가는 전통적인 방식입니다.

- **Process**: 고정된 요청 집합으로 배치 구성 → 시스템이 각 요청의 현재 단계(전체 Prefill, 전체 Decode)의 모든 토큰을 처리 → 다음 배치로 이동.
- **Inefficiency (비효율성)**: Decode 단계에서 **하나의 긴 요청(Long-running request)**이 전체 배치를 병목(Stall)시킵니다. 짧은 요청들이 기다리는 동안 GPU는 유휴(Idle) 상태로 남게 되어 **GPU 활용률이 30~50%**에 그치고, **긴 꼬리 지연 시간(High tail latency)**이 발생합니다. 자원이 심각하게 낭비됩니다.

```
GPU 시간 축 (Static Batching):
[Batch 1 처리] → [Decode A (긴 요청)만 실행] → [유휴(Idle)] → [Batch 2 처리] ...
▉▉▉▉▉▉▉▉▉▉ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ ████████ ░░░░░░░░░░░░░░░░
          ↑ GPU가 놀고 있음 (30~50% 활용률)
```

### 2. Continuous Batching (Iteration-Level Scheduling)

**Continuous Batching**은 **매 스케줄링 반복(Iteration)**마다 배치를 새롭게 구성하는 방식으로, 배치 구성을 개별 요청의 생명주기(Lifecycle)로부터 분리(Decouple)합니다.

- **Core Idea**: 배치 구성(Composition)을 개별 요청의 생명주기에서 분리하여, 매 반복마다 실행 가능한 요청들로 최적의 배치를 새로 구성합니다.

- **Process (작동 과정)**:
  1. **Selection (선택)**: 스케줄러가 실행 준비가 된(처리할 토큰이 있는) 요청들을 선택합니다.
  2. **Packing (패킹)**: 각 요청의 **다음 계산 단계(Next computational step)**를 하나의 배치로 묶습니다.
     - 새 요청의 **Prefill 단계** (전체 프롬프트 처리)
     - 진행 중인 요청의 **Decode 단계** (다음 토큰 1개 처리)
  3. **Execution (실행)**: GPU가 이 **혼합(Heterogeneous) 배치**를 실행합니다.
  4. **Update & Dissolve (갱신 및 해체)**: 결과를 반환하고, 요청 상태를 갱신하며, 배치는 **즉시 해체(Dissolve)**됩니다. 스케줄러는 **바로 다음 배치**를 구성합니다.

- **Efficiency (효율성)**:
  - GPU가 **지속적으로 포화(Saturated)** 상태를 유지합니다 (활용률 90% 이상).
  - 짧은 요청은 긴 요청에 **차단(Block)되지 않고** 완료되어 자원을 즉시 해제합니다.

```
GPU 시간 축 (Continuous Batching):
[Prefill A,B,C] → [Decode A,B,C] → [Decode A,C + Prefill D] → [Decode A,C,D + Prefill E] → ...
▉▉▉▉▉▉▉▉▉▉ ▉▉▉▉▉ ▉▉▉▉▉▉▉▉▉▉▉ ▉▉▉▉▉▉▉▉▉▉▉▉▉▉ ▉▉▉▉▉▉
          ↑ GPU가 쉬지 않고 일함 (90%+ 활용률)
```

---

## 🛠️ 핵심 구성 요소

| 구성 요소 | 설명 |
|---|---|
| [`continuous_batching`](../03_atomic/continuous_batching.md) | 배치를 실시간으로 재구성하는 연속 배치 처리 엔진 |
| [`iteration_level_scheduling`](../03_atomic/iteration_level_scheduling.md) | 요청 단위가 아닌 Iteration 단위로 스케줄링하는 정책 |
| `dynamic_rebalancer` | 배치 내 토큰 길이 불균형 시 재조정하는 내부 로직 |

---

## 🔗 관련 관계

- **COMPOSED_OF**: [`continuous_batching`](../03_atomic/continuous_batching.md), [`iteration_level_scheduling`](../03_atomic/iteration_level_scheduling.md)
- **PREREQUISITES**: [`composite.static_cache_manager`](static_cache_manager.md) (Module 02), [`composite.autoregressive_loop`](autoregressive_loop.md) (Module 01)
- **PREREQUISITE_OF**: [`composite.serving_system`](serving_system.md) (Module 03+04 통합)