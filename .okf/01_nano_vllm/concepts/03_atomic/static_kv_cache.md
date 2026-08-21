---
type: AtomicConcept
id: atomic.static_kv_cache
title: Static KV Cache (정적 KV 캐시 버퍼)
description: 최대 시퀀스 길이에 맞추어 메모리를 연속 텐서로 사전 할당(Pre-allocation)하는 캐시 구조
status: draft
generated:
  by: agent:builder/1.0
  at: 2026-08-05T13:22:00Z
verified:
  - by: human:curator
    at: 2026-08-05T13:22:00Z
sources:
  - id: vllm-paper
    resource: https://arxiv.org/abs/2209.06155
    title: vLLM - Easy, Fast and Cheap LLM Serving with PagedAttention
---

# Static KV Cache (정적 KV 캐시 버퍼)

## 📌 개념 정의

**Static KV Cache**는 요청 처리 전, 생성 가능한 최대 토큰 수(`max_seq_len`)에 해당하는 연속적인 메모리 공간을 GPU에 동적으로 확장하지 않고 한 번에 미리 할당(`Pre-allocate`)해 두는 캐싱 방식입니다.

동적 메모리 재할당(Reallocation) 및 메모리 복사 오버헤드를 방지하여 디코드 속도가 안정적이지만, 실제 생성 토큰 수가 적을 경우 메모리 낭비(Internal Fragmentation)가 심하다는 단점이 있습니다.

---

## 📐 텐서 구조 예시

```python
# 사전 할당 텐서 차원
key_buffer = torch.zeros(batch_size, num_heads, max_seq_len, head_dim)
value_buffer = torch.zeros(batch_size, num_heads, max_seq_len, head_dim)

# t번째 토큰 슬라이스 업데이트
key_buffer[:, :, current_pos, :] = new_key
value_buffer[:, :, current_pos, :] = new_value
```

---

## 🔗 관련 관계 (Relationships)

- **PREREQUISITES**:
  - [`kv_cache`](kv_cache.md) (Module 01)
- **PREREQUISITE_OF**:
  - [`static_cache_manager`](../02_composite/static_cache_manager.md)
  - [`paged_kv_cache`](paged_kv_cache.md) (Module 04 - 한계 극복 대상)