---
type: AtomicConcept
id: atomic.decode_phase
title: Decode Phase (디코드 단계)
description: 이전 단계 토큰 1개만을 받아 다음 토큰을 순차적으로 1개씩 생성하는 Memory Bandwidth-bound 연산 단계
status: draft
generated:
  by: agent:builder/1.0
  at: 2026-08-05T13:11:00Z
verified:
  - by: human:curator
    at: 2026-08-05T13:11:00Z
---

# Decode Phase (디코드 단계)

## 📌 개념 정의

**Decode Phase**(또는 Token Generation Phase)는 Prefill Phase 이후 `EOS(End of Sequence)` 토큰이 나오거나 최대 생성 길이에 도달할 때까지 토큰을 1개씩 자기회귀(Autoregressively) 방식으로 반복 생성하는 단계입니다.

이 단계는 다음과 같은 기술적 특징을 가집니다:
1. **Sequential Token-by-Token Generation**: 이전 스텝에서 생성된 토큰 단 1개만 입력으로 사용함.
2. **Memory Bandwidth-Bound Operation**: 매 스텝마다 전체 모델 가중치(Weights)와 누적된 KV 캐시를 GPU HBM에서 SRAM으로 불러와야 하므로 메모리 대역폭(GB/s)이 주 병목임.
3. **KV Cache Traversal**: 새 토큰의 K, V 벡터 1개를 계산한 후, 기존 저장된 전체 KV 캐시와 어텐션 연산(GEMV)을 수행함.

---

## 📐 연산 차원 (Tensor Shape)

- **Input Tensor Shape**: `[Batch Size, 1, Hidden Size]`
- **Attention Map Matrix**: `[Batch Size, Num Heads, 1, Current Seq Length]`
- **Updated KV Cache Shape**: `[Batch Size, Num Heads, Current Seq Length, Head Dim]`

---

## 🔗 관련 관계 (Relationships)

- **PREREQUISITES**:
  - `prefill_phase`
- **PREREQUISITE_OF**:
  - `inference_model`
