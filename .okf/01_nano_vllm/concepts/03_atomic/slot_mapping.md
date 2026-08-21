---
type: AtomicConcept
id: atomic.slot_mapping
title: Slot Mapping (슬롯 매핑)
description: PagedAttention에서 특정 토큰 위치(Token Position)를 논리적 블록 ID와 물리적 블록 ID로 변환하고,
  해당 블록 내부의 슬롯 인덱스(Slot Index)를 계산하여 Attention 커널이 불연속적 GPU 메모리에 저장된 KV Cache에 O(1)
  시간에 접근할 수 있게 하는 주소 변환 공식
status: draft
generated:
  by: agent:builder/1.0
  at: 2026-08-07 09:30:00+00:00
verified:
- by: human:curator
  at: 2026-08-07 09:30:00+00:00
prerequisites:
- atomic.paged_kv_cache
- atomic.block_table
composes_into:
- composite.paged_attention_manager (Module 04)
sources:
- id: vllm_paged_attention
  resource: https://arxiv.org/abs/2309.06180
  title: Efficient Memory Management for Large Language Model Serving with PagedAttention
prerequisite_of:
- composite.paged_attention_manager
---

# Slot Mapping (슬롯 매핑)

## 📌 개념 정의

**Slot Mapping**은 PagedAttention에서 **특정 토큰의 Key/Value 텐서가 GPU 메모리의 어느 물리적 블록(Physical Block)의 몇 번째 슬롯(Slot)에 저장되어 있는지**를 계산하는 핵심 주소 변환 공식입니다.

Paged KV Cache는 각 Sequence의 KV 텐서를 여러 개의 불연속적인 물리적 블록에 분산 저장합니다. Attention 커널이 특정 토큰 위치(예: `token_position = 25`)의 KV를 읽으려면, 해당 토큰이 저장된 **물리적 블록 ID**와 **블록 내부의 오프셋(슬롯)**을 알아야 합니다. 이때 사용되는 것이 바로 Slot Mapping입니다.

---

## 🧮 매핑 공식 (Mapping Formula)

Slot Mapping은 다음의 간단한 정수 연산(산술/논리 연산)으로 구성됩니다:

```
BLOCK_SIZE = 16 (고정, 예시)

1. 논리적 블록 ID 계산:
   logical_block_id = token_position // BLOCK_SIZE

2. 물리적 블록 ID 조회:
   physical_block_id = block_table[logical_block_id]

3. 블록 내 슬롯 인덱스 계산:
   slot_index = token_position % BLOCK_SIZE
```

### 📊 예시 (Example)

**가정**: `BLOCK_SIZE = 16`, Sequence A의 Block Table이 다음과 같다고 가정합니다.

| Logical Block ID | Physical Block ID |
| :---: | :---: |
| 0 | 7 |
| 1 | 3 |
| 2 | 15 |

**`token_position = 25`인 토큰의 KV를 찾는 과정:**

1. **논리적 블록 ID**: `25 // 16 = 1`
2. **물리적 블록 ID**: Block Table[1] → **`3`**
3. **슬롯 인덱스**: `25 % 16 = 9`

> 🎯 **결과**: `token_position = 25`의 KV는 **물리적 블록 3**의 **슬롯 9**에 저장되어 있습니다.

---

## ⚙️ Attention 커널의 메모리 주소 계산

실제 GPU 커널(CUDA/Triton)은 위에서 계산된 `physical_block_id`와 `slot_index`를 사용하여 다음과 같이 메모리 오프셋을 계산하고 KV 텐서를 읽어옵니다.

```cpp
// 의사 코드 (CUDA 커널 내부)
// key_cache: [num_physical_blocks, BLOCK_SIZE, num_heads, head_dim] 형태의 텐서
// head_idx: 현재 계산 중인 Head의 인덱스

// 1. 블록의 시작 주소 계산
void* block_ptr = key_cache[physical_block_id];

// 2. 슬롯(토큰)의 시작 주소 계산
//    각 슬롯은 (num_heads * head_dim) 만큼의 크기를 가짐
int slot_offset = slot_index * num_heads * head_dim;

// 3. 특정 Head의 Key 벡터 주소 계산
int head_offset = head_idx * head_dim;

// 4. 최종 메모리 주소
void* key_ptr = block_ptr + slot_offset + head_offset;
```

> 💡 **연산 비용**: Slot Mapping은 단순한 **정수 나눗셈(Division)과 나머지 연산(Modulo), 그리고 배열 참조(Array Lookup)**에 불과합니다. 따라서 Attention 연산(GEMV/GEMM)에 비해 **무시할 수 있을 정도의 오버헤드(마이크로초 미만)** 만을 발생시킵니다.

---

## ⚠️ 왜 Slot Mapping이 필수적인가?

| 구분 | Static KV Cache (연속 할당) | PagedAttention (Slot Mapping) |
| :--- | :--- | :--- |
| **메모리 접근** | `base_address + token_position * head_dim` (단순 포인터 연산) | `block_table[token // BLOCK_SIZE] * BLOCK_SIZE + token % BLOCK_SIZE` (블록 조회 + 슬롯 계산) |
| **단편화** | 심각한 외부 단편화 발생 | 단편화 제로 (Zero) |
| **주소 변환 비용** | 없음 (직접 접근) | **매우 작음** (단순 연산) |
| **처리량 효과** | 낮음 (OOM 위험) | **10~23배 향상** (단편화 제거 이득 >>> 주소 변환 비용) |

> 💡 **핵심 트레이드오프**: Slot Mapping은 단순한 정수 연산(수 나노초~마이크로초)을 추가하는 대신, 메모리 단편화를 완전히 제거하여 수십 배의 처리량 향상을 얻습니다. 이는 **지불할 가치가 충분히 큰 비용(Totally Worthwhile Trade-off)**입니다.

---

## 🔗 관련 관계 (Relationships)

- **PREREQUISITES**:
  - [`atomic.paged_kv_cache`](paged_kv_cache.md) (물리적 블록이 실제로 존재해야 함)
  - [`atomic.block_table`](block_table.md) (논리-물리 매핑 정보가 필요함)
- **COMPOSES_INTO**: [`composite.paged_attention_manager`](../02_composite/paged_attention_manager.md) (Module 04)
- **SYNERGY WITH**: [`atomic.paged_kv_cache`](paged_kv_cache.md), [`atomic.block_table`](block_table.md) (이 세 가지가 PagedAttention의 논리-물리 주소 변환 계층을 완성)

---

## 📝 Brick-by-Brick 관점에서의 위치

Slot Mapping은 PagedAttention의 **"주소 계산기(Address Calculator)"** 역할을 합니다.

```
[Module 01] kv_cache (기본 KV 저장 개념 인지)
     │
     ▼
[Module 02] static_kv_cache (연속 할당의 한계 인지)
     │
     ▼
[Module 04] paged_kv_cache (불연속 블록 저장소 도입)
     │
     ▼
[Module 04] block_table (논리-물리 매핑 정보 도입)
     │
     ▼
[Module 04] slot_mapping ★★★★★
     → "token_position을 physical_block_id와 slot_index로 변환하자!"
     → Attention 커널이 불연속 메모리를 읽을 수 있게 하는 마지막 퍼즐 조각
     → 단순한 나눗셈과 나머지 연산으로 O(1) 주소 변환 달성
     │
     ▼
[완성] composite.paged_attention_manager (논리-물리 변환 시스템 완성)
```

> 💡 **최종 인사이트**: Slot Mapping은 PagedAttention의 **"마지막 1%의 논리"**입니다. Paged KV Cache가 "무엇을" 저장할지 정의하고, Block Table이 "어디에" 있는지 매핑한다면, Slot Mapping은 "어떻게" 그 주소를 계산할지 정의합니다. 이 단순한 공식 덕분에 GPU 커널은 메모리 단편화 없이도 빠르게 KV 데이터를 찾을 수 있으며, 이것이 바로 vLLM이 실제 하드웨어에서 놀라운 속도를 낼 수 있는 이유입니다.