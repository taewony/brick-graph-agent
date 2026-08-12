---
type: Module
id: module.continuous_batching
title: Continuous Batching (연속 배치)
description: 토큰 시퀀스를 배치에 동적으로 할당해 메모리 효율을 높이고 배치 내부 길이 불균형을 최소화하는 핵심 엔진
status: draft
generated:
  by: agent:builder/1.0
  at: 2026-08-06T07:20:00Z
verified:
  - by: human:curator
    at: 2026-08-06T07:20:00Z
---

# Continuous Batching (연속 배치)

## 📌 개요

본 모듈은 **Continuous Batching**과 **Iteration Level Scheduling** 두 개의 원자적 개념을 조합하여 구현됩니다. 배치 내부 토큰 길이 불균형을 완화하고, GPU 메모리와 연산량을 효율적으로 사용하도록 설계되었습니다.

## 🧩 포함 원자적 개념

- `continuous_batching`
- `iteration_level_scheduling`

## 🏗️ Composite Concept

- `dynamic_batcher` (Composite Concept) – 두 원자적 개념을 `COMPOSED_OF` 관계로 연결한 구현 체계.

## 🔗 관련 관계

- **PREREQUISITES**: `kv_cache`
- **PREREQUISITE_OF**: `dynamic_batcher` (Composite Concept, Module 03)
