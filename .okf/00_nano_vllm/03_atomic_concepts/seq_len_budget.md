---
type: AtomicConcept
id: atomic.seq_len_budget
title: Sequence Length Budget (시퀀스 예산 정책)
description: 요청별 maximum context length를 지정하여 버퍼 오버플로우를 제어하는 메모리 예산 관리 규칙
status: draft
generated:
  by: agent:builder/1.0
  at: 2026-08-05T13:22:00Z
verified:
  - by: human:curator
    at: 2026-08-05T13:22:00Z
---

# Sequence Length Budget (시퀀스 예산 정책)

## 📌 개념 정의

**Sequence Length Budget**은 추론 요청 생성 시 `max_tokens` 또는 `max_context_length` 값을 상한선으로 설정하여, 정적 KV 캐시 버퍼가 허용 범위 내에서만 인덱싱되도록 제어하는 안전 규칙(Budget Rule)입니다.

현재 시퀀스 길이가 할당된 예산을 초과하면 캐시 오버플로우 예외를 발생시키거나 요청을 거부(Reject)합니다.

---

## 🔗 관련 관계 (Relationships)

- **PREREQUISITE_OF**:
  - `static_cache_manager`
