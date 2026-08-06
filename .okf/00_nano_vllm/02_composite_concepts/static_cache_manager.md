---
type: CompositeConcept
id: composite.static_cache_manager
title: Static Cache Manager (정적 KV 캐시 관리자)
description: Static KV Cache 버퍼(연속 할당)와 Sequence Length Budget 정책을 통합하여, 토큰 Append 및 어텐션 슬라이스 생성을 총괄하는 1세대 캐시 관리 엔진. (Static Batching과는 무관)
status: draft
generated:
  by: agent:builder/1.0
  at: 2026-08-05T13:22:00Z
verified:
  - by: human:curator
    at: 2026-08-05T13:22:00Z
components:
  - static_kv_cache
  - seq_len_budget
prerequisites:
  - composite.autoregressive_loop (Module 01)
composes_into:
  - composite.dynamic_batcher (Module 03)
---

# Static Cache Manager (정적 KV 캐시 관리자)

## 📌 개념 정의

**Static Cache Manager**는 `static_kv_cache` (연속된 정적 버퍼)와 `seq_len_budget` (시퀀스 길이 예산 정책)을 결합한 기초 캐시 관리 컴포넌트입니다.  
**Module 02**에서 다루는 개념으로, PagedAttention(Module 04) 이전의 1세대 KV Cache 관리 방식을 구현합니다.

> 🚨 **주의: Static Batching과 혼동하지 마십시오!**  
> 이 문서는 **"Static KV Cache" (메모리 할당 방식)**에 대한 설명입니다.  
> **"Static Batching" (정적 배치 처리 방식)**은 `composite.dynamic_batcher` 문서에서 다룹니다.  
> 두 개념은 완전히 다르며, 이름이 비슷하지만 전혀 다른 계층(Layer)의 최적화입니다.

---

## 🧱 컴포넌트 메커니즘

```
[New Key, Value Tensor (1 Token)]
              │
              ▼
┌─────────────────────────────────────────────────────────┐
│ Static Cache Manager                                    │
│  1. Check seq_len_budget limit (예산 초과 여부 확인)    │
│  2. Insert K,V to static_kv_cache buffer at current_pos │
│  3. Increment current_pos pointer (포인터 증가)         │
│  4. Return valid slice buffer for Attention computation │
└─────────────────────────────────────────────────────────┘
              │
              ▼
[Valid Key/Value Slice Matrix for Attention]
```

### 1. Static KV Cache (정적 KV 캐시)
- 시스템 초기화 시 `max_seq_len`에 해당하는 **연속된(Contiguous) 메모리 버퍼**를 GPU에 사전 할당(Pre-allocate)합니다.
- 디코드 단계에서 새 토큰의 K/V 텐서가 들어오면, 현재 위치(`current_pos`)에 덮어쓰거나 추가(Append)합니다.
- **장점**: 메모리 할당/해제 오버헤드가 없고, 포인터 연산이 단순합니다.
- **단점**: **외부 단편화(External Fragmentation)**가 발생하고, 실제 사용량보다 큰 메모리를 점유하여 낭비(Internal Fragmentation)가 심합니다.

### 2. Seq Length Budget (시퀀스 길이 예산)
- 각 요청이 생성할 수 있는 최대 토큰 수(`max_tokens`)를 제한합니다.
- 버퍼 오버플로우(Overflow)를 방지하고, 시스템 메모리를 보호하는 안전 장치(Safety Guard)입니다.

---

## 🔗 조립 정보 (Composition & Relationships)

- **COMPOSED_OF**:
  - `atomic.static_kv_cache`
  - `atomic.seq_len_budget`
- **PREREQUISITES**:
  - `composite.autoregressive_loop` (Module 01)
- **PREREQUISITE_OF**:
  - `composite.dynamic_batcher` (Module 03)

---

## ⚠️ Static Cache Manager의 한계 (PagedAttention으로의 연결)

이 관리자는 구현이 단순하고 빠르지만, 다음과 같은 치명적 한계로 인해 **Module 04 (PagedAttention)**에서 대체됩니다.
1. **외부 단편화**: Sequence 종료 후 해제된 공간이 조각(Fragment)으로 남아 메모리 활용률이 50~70%에 그칩니다.
2. **확장성 부족**: 긴 프롬프트나 대규모 배치 처리 시 메모리 낭비가 심각해집니다.
3. **OOM 취약**: 연속된 큰 메모리 블록을 찾지 못해 Out Of Memory 오류가 빈번히 발생합니다.

> 💡 **다음 학습 단계**: Module 04의 `composite.paged_attention_manager`에서 이 문제를 어떻게 해결하는지 학습하세요.
