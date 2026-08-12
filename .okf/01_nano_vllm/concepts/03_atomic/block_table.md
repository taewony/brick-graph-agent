---
type: AtomicConcept
id: atomic.block_table
title: Block Table (블록 테이블)
description: "각 시퀀스의 논리적 블록 번호를 GPU HBM 상의 물리적 블록 ID에 매핑하여 Paged KV Cache의 논리적 연속성을 보장하는 자료구조"
status: draft
generated:
  by: agent:builder/1.0
  at: 2026-08-07T09:00:00Z
verified:
  - by: human:curator
    at: 2026-08-07T09:00:00Z
prerequisites:
  - atomic.paged_kv_cache
composes_into:
  - composite.paged_attention_manager
sources:
  - id: vllm_paged_attention
    resource: https://arxiv.org/abs/2309.06180
    title: "Efficient Memory Management for Large Language Model Serving with PagedAttention"
---

# Block Table (블록 테이블)

## 📌 개념 정의

**Block Table**은 Virtual Memory 시스템의 Page Table과 같이, 각 Sequence의 **논리적 블록 번호(Logical Block Number)**를 GPU HBM 상에 분산 저장된 **물리적 블록 ID(Physical Block ID)**에 매핑해주는 자료구조입니다.

- Sequence의 토큰들이 물리적으로 불연속한 블록들에 저장되더라도, Block Table을 통해 연속적인 시퀀스로 참조할 수 있습니다.
- Prefix Caching이나 Prompt Sharing 시 여러 Sequence가 동일한 물리적 블록을 공유할 수 있도록 해줍니다.

---

## 🧱 논리-물리 매핑 예시

```
Sequence A Block Table:
┌─────────────────┬──────────────────┐
│ Logical Block 0 │ Physical Block 7 │
│ Logical Block 1 │ Physical Block 3 │
│ Logical Block 2 │ Physical Block 22│
└─────────────────┴──────────────────┘
```

---

## 🔗 관련 관계 (Relationships)

- **PREREQUISITES**: `atomic.paged_kv_cache`
- **COMPOSES_INTO**: `composite.paged_attention_manager`
- **SYNERGY WITH**: `atomic.slot_mapping`