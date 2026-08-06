---
type: CompositeConcept
id: composite.dynamic_batcher
title: Dynamic Batcher (동적 배치 엔진)
description: 연속 배치를 기반으로 토큰 길이와 자원 사용량을 고려해 배치를 동적으로 재구성하고 스케줄링하는 핵심 컴포지트
status: draft
generated:
  by: agent:builder/1.0
  at: 2026-08-06T07:20:00Z
verified:
  - by: human:curator
    at: 2026-08-06T07:20:00Z
---

# Dynamic Batcher (동적 배치 엔진)

## 📌 개념 정의

**Dynamic Batcher**는 `continuous_batching` 과 `iteration_level_scheduling` 을 조합해 배치 내부 토큰 길이에 따라 배치를 재조정하고, GPU 메모리와 연산량을 균등하게 할당하는 복합 개념입니다. 배치가 가득 차면 즉시 처리하고, 남은 토큰은 새로운 배치에 재배치합니다.

## 🛠️ 핵심 구성 요소

- `continuous_batching` (Atomic): 배치에 토큰을 실시간 추가
- `iteration_level_scheduling` (Atomic): 길이 기반 스케줄링 정책
- `max_batch_size`: 배치당 최대 토큰 수 (예: 2048)
- `dynamic_rebalancer`: 배치 길이 불균형 시 재조정 로직

## 🔗 관련 관계

- **COMPOSED_OF**: `continuous_batching`, `iteration_level_scheduling`
- **PREREQUISITE_OF**: `module_03_continuous_batching` (Module 03)
