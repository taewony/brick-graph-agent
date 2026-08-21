---
type: Module
id: module.02_basic_kv_cache
title: "Module 02: Basic KV Cache Engine (기초 KV 캐시 엔진)"
description: 고정된 시퀀스 예산(Sequence Budget)을 기반으로 정적 메모리를 할당하여 KV 캐시를 효율적으로 관리하는 기초 캐시 엔진 모듈
status: draft
generated:
  by: agent:builder/1.0
  at: 2026-08-05T13:22:00Z
verified:
  - by: human:curator
    at: 2026-08-05T13:22:00Z
---

# Module 02: Basic KV Cache Engine (기초 KV 캐시 엔진)

> **모듈 목표**: 동적 메모리 관리(PagedAttention)로 진입하기 전, 미리 정의된 최대 시퀀스 길이(Max Sequence Length) 예산을 바탕으로 연속된 정적 메모리 버퍼를 사전 할당하여 어텐션 연산을 가속하는 **Static KV Cache Engine**을 구축합니다.

---

## 🧱 포함된 개념 (Concepts)

### 원자적 개념 (Atomic Concepts)
- **[`static_kv_cache`](../03_atomic/static_kv_cache.md)**: 최대 시퀀스 길이에 맞춰 사전 할당된 연속형 텐서 버퍼 구조
- **[`seq_len_budget`](../03_atomic/seq_len_budget.md)**: 메모리 파편화를 방지하기 위해 요청별로 사전 정의하는 시퀀스 예산 관리 정책

### 복합 개념 (Composite Concepts)
- **[`static_cache_manager`](../02_composite/static_cache_manager.md)**: 정적 텐서 버퍼의 인덱싱, 토큰 추가(Append), 오버플로우 검사를 담당하는 기초 캐시 관리자

---

## 🔗 학습 경로 및 선행 조건 (Learning Flow)

```
[Module 01: kv_cache]
          │
          ▼
     [static_kv_cache]  ──┐
                          ├──► [static_cache_manager] (Composite)
     [seq_len_budget]   ──┘
```

---

## 📝 실습 과제 (Lab Assignment)

1. `max_seq_len` 크기의 Tensor를 사전 할당(`torch.zeros`)하고 디코드 스텝마다 인덱스를 1씩 증가시키며 $K, V$ 슬라이싱 업데이트 구현하기
2. [`seq_len_budget`](../03_atomic/seq_len_budget.md) 초과 시 처리할 예외(Sequence Length Overflow Error) 메커니즘 설계하기