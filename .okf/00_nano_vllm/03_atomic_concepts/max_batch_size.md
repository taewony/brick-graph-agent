---
type: AtomicConcept
id: atomic.max_batch_size
title: Max Batch Size (최대 배치 크기)
description: 한 번에 처리할 수 있는 최대 토큰 수 또는 시퀀스 길이를 정의하는 파라미터. 배치 크기를 제한해 메모리 사용량을 제어하고, 연속 배치에서 GPU 메모리 초과를 방지한다.
status: draft
generated:
  by: agent:builder/1.0
  at: 2026-08-06T07:31:00Z
verified: []
---

# Max Batch Size (최대 배치 크기)

## 📌 개념 정의
**Max Batch Size**는 연속 배치 엔진이 동시에 처리할 수 있는 토큰/시퀀스의 상한을 의미한다. 설정 값보다 큰 요청은 대기열에 넣어 순차적으로 처리한다.

## 🔗 관련 관계
- **PREREQUISITES**: continuous_batching
- **PREREQUISITE_OF**: dynamic_batcher (Composite Concept, Module 03)
