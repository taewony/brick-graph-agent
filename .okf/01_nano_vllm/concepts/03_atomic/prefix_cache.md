---
type: AtomicConcept
id: atomic.prefix_cache
title: Prefix Cache (프리픽스 캐시)
description: "요청 시 앞부분(프리픽스) 토큰을 미리 캐시해 재사용함으로써, 프리패딩 단계에서 반복 계산을 방지하고 레이턴시를 크게 감소시킨다."
status: draft
generated:
  by: agent:builder/1.0
  at: 2026-08-06T08:07:41Z
verified: []
sources:
  - id: vllm-paper
    resource: https://arxiv.org/abs/2209.06155
    title: vLLM - Easy, Fast and Cheap LLM Serving with PagedAttention
---

# Prefix Cache (프리픽스 캐시)

## 📌 개념 정의
**Prefix Cache**는 동일한 입력 프리픽스(예: 시스템 프롬프트, 첫 N 토큰)를 **핵심 연산 결과**(KV, hidden states
등)와 함께 메모리에 저장하고, 이후 요청이 동일 프리픽스를 가질 경우 재사용한다.
- **재사용 범위**: 동일 프리픽스·모델·배치 설정 조건 하에 재활용.
- **효과**: 프리패딩 연산을 한 번만 수행 → GPU 사용량·레이트 감소.

## ⚙️ 핵심 흐름
1. **프리픽스 추출** – 요청에서 첫 `P` 토큰을 식별.
2. **캐시 탐색** – `hash(prefix)` 으로 기존 캐시 존재 여부 확인.
3. **캐시 히트** → 저장된 KV/hidden 상태를 바로 로드.
4. **캐시 미스** → 프리패딩 수행 후 결과를 캐시에 저장.

## 🔗 관련 관계
- **PREREQUISITES**: [`kv_cache`](kv_cache.md)
- **COMPOSED_OF**: [`cache_hit_detection`](cache_hit_detection.md), [`caching_strategy`](../02_composite/caching_strategy.md)