---
type: AtomicConcept
id: atomic.length_aware_scheduler
title: Length Aware Scheduler (길이‑인식 스케줄러)
description: 토큰 길이와 연산량을 고려해 배치를 동적으로 스케줄링하는 정책. 연속 배치 시 GPU 메모리와 연산량을 균형 있게 배분한다.
status: draft
generated:
  by: agent:builder/1.0
  at: 2026-08-06T07:38:00Z
verified: []
---

# Length Aware Scheduler (길이‑인식 스케줄러)

## 📌 개념 정의
**Length‑Aware Scheduler**는 각 요청의 토큰 길이와 예상 연산량을 기반으로 배치에 할당할 우선순위를 결정합니다.
- 짧은 요청은 빠르게 처리되어 레이턴시를 감소시킵니다.
- 긴 요청은 GPU 메모리 한계에 맞춰 적절히 분할하거나 대기열에 배치됩니다.

## ⚙️ 핵심 동작
- **길이 기반 가중치**: `weight = 1 / token_length` 로 우선순위 계산.
- **자원 사용량 체크**: 현재 배치의 추정 메모리 사용량이 `max_batch_size` 를 초과하면 새 배치를 시작합니다.
- **동적 재배치**: 실행 중인 배치에 새로운 짧은 요청이 들어오면 즉시 삽입하고, 긴 요청은 다음 배치에 할당합니다.

## ⚙️ 핵심 요소
- length_aware_scheduler – 토큰 길이와 연산량 기반 스케줄링 로직  
- max_batch_size – 하드웨어 제한에 따른 배치 상한 (예: 2048 토큰) 

## 🔗 선행 관계
- **PREREQUISITES**: continuous_batching  
- **PREREQUISITE_OF**: dynamic_batcher (Composite Concept, Module 03)
- **COMPOSED_OF**: `max_batch_size` (Maximum batch size parameter)

## 📚 참고 자료
- Sarathi‑Serve 연속 배치 구현  
- vLLM 연속 배치 논문
