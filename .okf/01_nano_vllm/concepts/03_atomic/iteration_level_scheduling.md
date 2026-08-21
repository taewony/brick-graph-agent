---
type: AtomicConcept
id: atomic.iteration_level_scheduling
title: Iteration Level Scheduling (배치 단계 스케줄링)
description: 각 배치 단계에서 토큰 길이와 자원 사용을 고려해 배치를 순차적으로 처리하는 스케줄링 정책
status: draft
generated:
  by: agent:builder/1.0
  at: 2026-08-06T07:16:15Z
verified:
  - by: human:curator
    at: 2026-08-06T07:16:15Z
sources:
  - id: vllm-paper
    resource: https://arxiv.org/abs/2209.06155
    title: vLLM - Easy, Fast and Cheap LLM Serving with PagedAttention
---

# Iteration Level Scheduling (배치 단계 스케줄링)

## 📌 개념 정의

**Iteration Level Scheduling**은 배치 내부 토큰 길이에 따라 우선순위를 부여하고, GPU 메모리/연산량을 균등하게 분배하도록 배치를 순차적으로 처리하는 정책입니다.

## 🔗 관련 관계
- **PREREQUISITES**: [`continuous_batching`](continuous_batching.md)
- **PREREQUISITE_OF**: [`dynamic_batcher`](../02_composite/dynamic_batcher.md) (Composite Concept, Module 03)