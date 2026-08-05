---
type: AtomicConcept
id: atomic.prefill_phase
title: Prefill Phase (프리필 단계)
description: 입력 프롬프트 전체 토큰을 병렬 처리하여 첫 번째 토큰을 생성하고 Initial KV Cache를 구축하는 Compute-bound 연산 단계
status: draft
generated:
  by: agent:builder/1.0
  at: 2026-08-05T13:11:00Z
verified:
  - by: human:curator
    at: 2026-08-05T13:11:00Z
---

# Prefill Phase (프리필 단계)

## 📌 개념 정의

**Prefill Phase**(또는 Prompt Processing Phase)는 사용자가 입력한 $N$개의 프롬프트 토큰 전체를 LLM 모델에 한 번에 주입하여 첫 번째 출력 토큰(First Token)을 생성하는 단계입니다.

이 단계는 다음과 같은 기술적 특징을 가집니다:
1. **Parallel Execution**: $N$개 토큰에 대한 행렬 곱 연산을 GPU Tensor Core에서 한 번에 병렬 처리함.
2. **Compute-Bound Operation**: Arithmetic Intensity(FLOP/Byte)가 높아 GPU 연산 성능(TFLOPS)이 주 병목 요인임.
3. **KV Cache Initialization**: $N$개 토큰 각각에 해당하는 Key, Value 벡터들을 계산하여 최초 KV 캐시 블록에 할당 및 대치함.

---

## 📐 연산 차원 (Tensor Shape)

- **Input Tensor Shape**: `[Batch Size, Prompt Length, Hidden Size]`
- **Attention Map Matrix**: `[Batch Size, Num Heads, Prompt Length, Prompt Length]`
- **Output KV Cache Shape**: `[Batch Size, Num Heads, Prompt Length, Head Dim]`

---

## 🔗 관련 관계 (Relationships)

- **PREREQUISITES**:
  - `inference_only`
- **PREREQUISITE_OF**:
  - `decode_phase`
  - `inference_model`
