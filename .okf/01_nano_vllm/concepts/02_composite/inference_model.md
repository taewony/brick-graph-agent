---
type: CompositeConcept
id: composite.inference_model
title: Inference Model Pipeline (추론 파이프라인)
description: Prefill Phase와 Decode Phase를 연동하여 완전한 LLM 텍스트 생성을 수행하는 기본 복합 추론 시스템
status: draft
generated:
  by: agent:builder/1.0
  at: 2026-08-05T13:11:00Z
verified:
  - by: human:curator
    at: 2026-08-05T13:11:00Z
---

# Inference Model Pipeline (추론 파이프라인)

## 📌 개념 정의

**Inference Model Pipeline**은 원자적 개념인 `prefill_phase`와 `decode_phase`가 결합(`COMPOSED_OF`)하여 입력 프롬프트를 최종 텍스트 응답으로 변환하는 기본 추론 모델 시스템입니다.

---

## 🧱 구조적 파이프라인 (Structural Pipeline)

```
[입력 프롬프트]
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│ 1. Prefill Phase (Compute-bound)                        │
│    - Prompt Processing (Parallel GEMM)                  │
│    - First Token & Initial KV Cache Build               │
└─────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│ 2. Decode Phase Loop (Memory Bandwidth-bound)           │
│    - Token-by-Token Autoregressive Generation           │
│    - Read/Append KV Cache & GEMV Attention              │
│    - Stop Condition Check (EOS or Max Length)           │
└─────────────────────────────────────────────────────────┘
      │
      ▼
[최종 생성 텍스트]
```

---

## 🔗 조립 정보 (Composition & Relationships)

- **COMPOSED_OF**:
  - `prefill_phase`
  - `decode_phase`
  - `inference_only`
- **PREREQUISITE_OF**:
  - `autoregressive_loop` (Module 01)
