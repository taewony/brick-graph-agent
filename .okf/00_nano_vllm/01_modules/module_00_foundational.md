---
type: Module
id: module.00_foundational
title: "Module 00: Foundational Concepts (기초 개념)"
description: LLM 추론 동작의 두 가지 핵심 단계인 Prefill Phase와 Decode Phase의 특성을 이해하고 구분하는 기초 모듈
status: draft
generated:
  by: agent:builder/1.0
  at: 2026-08-05T13:11:00Z
verified:
  - by: human:curator
    at: 2026-08-05T13:11:00Z
---

# Module 00: Foundational Concepts (기초 개념)

> **모듈 목표**: vLLM 서빙 엔진을 설계하기 전, LLM 추론 과정이 **Prefill Phase(Compute-bound)**와 **Decode Phase(Memory Bandwidth-bound)**라는 두 가지 비대칭적 실행 단계로 나뉜다는 결정적 특성을 이해합니다.

---

## 🧱 포함된 개념 (Concepts)

### 원자적 개념 (Atomic Concepts)
- **[prefill_phase](file:///D:/code/brick-graph-agent/.okf/00_nano_vllm/01_atomic_concepts/prefill_phase.md)**: 입력 프롬프트 전체 토큰을 한 번에 병렬 처리하여 첫 번째 토큰을 생성하고 최초 KV 캐시를 구축하는 단계
- **[decode_phase](file:///D:/code/brick-graph-agent/.okf/00_nano_vllm/01_atomic_concepts/decode_phase.md)**: 생성된 이전 토큰 1개만을 입력으로 받아 다음 토큰을 순차적으로 1개씩 생성하는 단계
- **[inference_only](file:///D:/code/brick-graph-agent/.okf/00_nano_vllm/01_atomic_concepts/inference_only.md)**: 역전파(Backpropagation) 없이 Forward Pass만 수행되는 추론 전용 워크로드 특성

### 복합 개념 (Composite Concepts)
- **[inference_model](file:///D:/code/brick-graph-agent/.okf/00_nano_vllm/02_composite_concepts/inference_model.md)**: Prefill과 Decode 단계를 통합 제어하는 기본 추론 파이프라인

---

## 🔗 학습 경로 및 선행 조건 (Learning Flow)

```
[inference_only]
     │
     ├───► [prefill_phase] (Compute-bound)  ──┐
     │                                        ├──► [inference_model] (Composite)
     └───► [decode_phase]  (Memory-bound)   ──┘
```

---

## 📝 실습 과제 (Lab Assignment)

1. `prefill_phase`와 `decode_phase`의 입력 텐서 차원(Batch Size, Sequence Length) 차이 분석하기
2. GPU 연산 병목(Compute-bound vs Memory Bandwidth-bound) 특성 비교하기
