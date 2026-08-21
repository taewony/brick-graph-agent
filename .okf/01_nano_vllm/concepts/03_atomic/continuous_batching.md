---
type: AtomicConcept
id: atomic.continuous_batching
title: Continuous Batching (연속 배치)
description: 연속 배치 기법은 입력 토큰을 길이와 자원 사용량에 따라 동적으로 묶어 처리하여 GPU 활용 효율을 극대화하고, 배치 간 레이턴시를
  최소화합니다.
status: draft
generated:
  by: agent:builder/1.0
  at: 2026-08-06 07:31:00+00:00
verified: []
sources:
- id: vllm-paper
  resource: https://arxiv.org/abs/2209.06155
  title: vLLM - Easy, Fast and Cheap LLM Serving with PagedAttention
prerequisite_of:
- atomic.length_aware_scheduler
- atomic.max_batch_size
- atomic.memory_pool
prerequisites:
- composite.autoregressive_loop
---

# Continuous Batching (연속 배치)

## 📌 개념 정의
**Continuous Batching**은 요청이 들어올 때마다 현재 실행 중인 배치에 바로 삽입하거나, 배치가 완료되면 새로운 배치를 시작하는 방식으로, 전통적인 고정 배치와 달리 **동적인 배치 결합**을 제공합니다.

## ⚙️ 핵심 요소
- [`continuous_batching`](continuous_batching.md) – 배치 결합 로직 및 토큰 길이 기반 동적 병합
- [`max_batch_size`](max_batch_size.md) – 하드웨어 제한에 따른 최대 배치 크기 (예: 2048 토큰)
- [`length_aware_scheduler`](length_aware_scheduler.md) – 토큰 길이와 연산량을 고려해 배치를 스케줄링하는 스케줄러

## 🔗 선행 관계
- **PREREQUISITES**: [`kv_cache`](kv_cache.md) (기본 KV 캐시 엔진)
- **PREREQUISITE_OF**: [`iteration_level_scheduling`](iteration_level_scheduling.md) (배치 단계 스케줄링)

## 📚 참고 자료
- Sarathi‑Serve 지속 배치 구현
- vLLM 연속 배치 논문