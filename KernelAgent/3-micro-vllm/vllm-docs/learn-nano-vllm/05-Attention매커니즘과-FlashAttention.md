# 강의 05: Attention 메커니즘과 FlashAttention

> nano-vllm에서 Attention 순전파는 두 가지 경로로 나뉩니다: **Prefill**은 가변 길이 FlashAttention으로 프롬프트 전체를 처리합니다(그리고 paged block table을 포함할 수 있습니다); **Decode**는 KV cache가 포함된 커널로 단일 단계를 이어서 씁니다. 중간에 **Triton**을 사용하여 계산된 K/V를 slot에 따라 전역 캐시에 기록합니다—이 강의를 읽고 나면 IO, prefix, 두 단계의 차이를 명확히 설명할 수 있습니다.

## 본 강의 목표

1. **`Attention.forward`**에서 **KV를 먼저 쓰고, 그 다음 attention을 하는** 순서와 그 이유를 명확히 설명합니다.
2. **`flash_attn_varlen_func`(prefill)**와 **`flash_attn_with_kvcache`(decode)**의 입력값과 적용 시나리오를 비교합니다.
3. **`store_kvcache_kernel`**을 이해합니다: 각 program이 하나의 토큰 위치를 처리하며, `slot_mapping`에 따라 물리 캐시에 기록합니다.
4. **prefix cache 분기**를 설명합니다: `block_tables is not None`일 때 **K/V가 캐시에서** 와서 varlen attention을 수행하는 것의 의미.
5. **FlashAttention의 IO 최적화 사상**을 구술합니다(타일링, 재계산 vs HBM).

## 핵심 개념

### 표준 Attention과 KV Cache

단일 레이어에 대해, 현재 단계의 **Q, K, V**(멀티헤드 shape 생략)가 주어졌을 때, attention:

\[
\mathrm{softmax}\left(\frac{Q K^\top}{\sqrt{d}}\right) V
\]

**추론 시**: 과거 위치의 K, V는 이미 캐시되어 있고, **새로운 토큰의 K, V**를 캐시에 추가한 후, **현재 Q**와 **모든 과거 K, V**로 attention을 수행하면 됩니다(인과 마스크 포함).

### FlashAttention (직관)

전통적 구현: 완전한 **\(S \times S\)** attention 행렬을 구체화하여, **HBM 읽기/쓰기**가 병목이 됩니다.  
**FlashAttention**은 **타일링(tiling)**을 통해, **SRAM** 상에서 softmax 리덕션을 완료하여, **HBM 접근량**을 줄이고, **긴 시퀀스**에서 현저히 가속합니다. **IO 복잡도** 관점은 면접 가산점 요소입니다.

### `flash_attn_varlen_func` (Prefill)

- **여러 가변 길이 시퀀스가 이어붙여진** 텐서를 입력받아, **`cu_seqlens_*`**와 함께 경계를 설명합니다.
- **프롬프트 단계**에 적합: 한 번에 **여러 시퀀스, 여러 토큰**을 처리하며, **인과 마스크**는 True입니다.
- 선택적 **`block_table`**: **PagedAttention**과 일관되게, 논리적 토큰을 **물리 블록**에 매핑하여, **비연속적** KV 저장을 지원합니다.

### `flash_attn_with_kvcache` (Decode)

- **이미 존재하는 KV cache**, 현재 단계의 **짧은 Q**(보통 시퀀스당 1 토큰)에 최적화되어 있습니다.
- **k_cache / v_cache**에서 이력을 읽고, **cache_seqlens**는 현재 캐시된 길이 등을 설명합니다.
- **Decode** 경로에서는 **`q.unsqueeze(1)`**가 **길이 차원**을 도입하여, 커널이 기대하는 shape에 맞춥니다.

### Prefix cache (`block_tables is not None`이며 prefill일 때)

**일부 프리픽스의 KV가 이미 캐시에 존재**할 때(예: **공유 시스템 프롬프트**, **여러 턴의 대화 재사용**), 이번 라운드에서는 **프리픽스 K/V를 다시 계산할 필요가 없습니다**:

- 코드 경로: `if context.is_prefill`이면서 **`context.block_tables is not None`**이면, **`k, v = k_cache, v_cache`**로 하여, **캐시에 이미 있는 KV**와 현재 **Q**를 바로 **varlen attention**에 사용합니다.
- 새로 계산된 K/V는 여전히 **`store_kvcache`**를 통해 **새로 할당된 slot**에 기록될 수 있습니다(스케줄러가 할당한 블록과 일치).

## 소스 코드 분석 (전체 소스 코드 및 줄별 주석 포함)

### `attention.py` 전체 소스 코드

```python
import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        return o
```

### Triton `store_kvcache_kernel` 줄별 설명

| 줄 | 의미 |
|----|------|
| `idx = tl.program_id(0)` | 1차원 grid, **idx번째 토큰 위치**(펼쳐진 기준) |
| `slot = tl.load(slot_mapping_ptr + idx)` | 해당 토큰의 KV가 기록될 **물리 slot**; **-1은 기록하지 않음**(자리 채우기/건너뛰기) |
| `if slot == -1: return` | **유효하지 않은 위치**는 바로 반환, 캐시 오염 방지 |
| `key_offsets` / `value_offsets` | **입력 K/V 텐서**에서 **idx**와 **stride**에 따라 벡터를 가져옵니다(head 차원을 D로 펼침) |
| `tl.load(...)` | **이 토큰**의 K, V 벡터를 읽음 |
| `cache_offsets = slot * D + arange(D)` | 물리 캐시는 **slot을 큰 덩어리**로, **D = num_heads * head_dim**을 연속 저장 |
| `tl.store(k_cache_ptr + cache_offsets, key)` | **K 캐시**에 기록; V도 동일 |

**`store_kvcache` 파이썬 래퍼**:

- **연속된 메모리 stride**를 검증하여, Triton 어드레싱이 정확하도록 보장합니다.
- Grid 크기 **`N`** = 토큰 수, 각 program은 **한 쌍의 K/V를 기록**합니다.

### `Attention.forward` 두 분기 설명

| 단계 | 역할 |
|------|------|
| `get_context()` | **현재 단계**의 prefill/decode, cu_seqlens, block_tables, slot_mapping 등을 가져옴 |
| `if k_cache.numel() and v_cache.numel()` | 캐시 텐서가 이미 할당되어 있다면, 이번 단계에서 계산된 K/V를 **먼저 기록** |
| `store_kvcache(...)` | **`slot_mapping`**에 따라 **이 레이어**의 K/V를 **전역 블록 캐시**에 기록 |
| **prefill + `block_tables is not None`** | **prefix cache**: attention에 사용되는 **K/V를 캐시에서 직접** 가져옴(이미 프리픽스 포함), 프리픽스 재계산 회피 |
| `flash_attn_varlen_func(...)` | **가변 길이 prefill**; `causal=True`; **`block_table`**이 paged KV 지원 |
| **decode** | `flash_attn_with_kvcache`: **캐시 읽기** + **현재 Q**; `cache_seqlens=context.context_lens` |
| `q.unsqueeze(1)` | **seqlen=1** 차원을 추가하여 API에 부합시키기 |

## 그림 설명 (텍스트/ASCII 설명)

**Prefill vs Decode**:

```
Prefill:
  프롬프트 토큰  -->  Q,K,V 선형 투영  -->  store_kvcache 기록
                    -->  flash_attn_varlen (전체 혹은 prefix가 포함된 cache)

Decode:
  1개의 새로운 토큰      -->  Q ；K,V 새 벡터를 캐시에 기록
                    -->  flash_attn_with_kvcache (큰 덩어리의 캐시 읽기)
```

**Slot 기록**:

```
토큰 인덱스 idx ----slot_mapping----> 물리 slot s
K/V 벡터 ----------store-----------> k_cache[s*sD:(s+1)*sD]
```

**FlashAttention IO (구술용 도해)**:

```
단순 구현: HBM <--> 대형 attention 행렬 <--> HBM
Flash: SRAM 상에서 타일링으로 softmax/리덕션 계산, 큰 행렬 구체화 방지
```

## 면접 출제 포인트

- **store를 먼저 하고 attention**: prefill 시 **먼저 이번 단계의 K/V를 캐시에 내려놓고**, 그 다음 **varlen** 또는 **prefix** 경로에 참여시킵니다; 순서는 **slot 할당**과 일관됩니다.
- **prefix 분기**: `k, v = k_cache, v_cache` — **attention 입력을 캐시로 변경**하며, 현재 선형 레이어가 방금 계산한 **완전한 k,v 텐서**를 사용하지 않습니다(현재 단계 Q는 여전히 새 것).
- **`slot == -1`**: 패딩 또는 **이번 단계 기록에 속하지 않는** 위치.
- **Decode에서 `flash_attn_with_kvcache` 사용**: **저지연 단일 단계**, **HBM 내 paged cache** 재사용.
- **FlashAttention**: **HBM 읽기/쓰기 감소**, **수치적으로 안정된 online softmax**(한 마디 언급 가능).

## 자주 나오는 면접 질문

1. **Prefill이 왜 compute-bound일 수 있나요?**  
   답변: **프롬프트가 길고, 행렬이 커서**, GEMM과 attention **FLOPs가 높음**; FlashAttention이 IO를 낮춘 후에도 연산 능력을 충분히 점유할 수 있습니다.

2. **Decode가 왜 더 memory-bound인가요?**  
   답변: 단계당 **유효 계산이 작고**, **KV 읽기**, **KV 쓰기**, 커널 런칭이 차지하는 비중이 높습니다.

3. **Triton store 없이, PyTorch 인덱싱만으로 캐시에 기록해도 되나요?**  
   답변: 가능하지만 **느립니다**; 커스텀 커널이 **융합된 쓰기**, **stride 제어 가능**으로, **Paged** 레이아웃에 유리합니다.

4. **`block_table`이 두 API에서 각각 어떤 역할을 하나요?**  
   답변: **논리적 시퀀스 위치**를 **물리 블록**에 매핑하여, **KV가 비연속적으로 저장**되어도 FlashAttention 커널이 주소를 찾을 수 있게 합니다.

5. **GQA(num_kv_heads < num_heads)는 어디에서 드러나나요?**  
   답변: **Qwen3/Linear 투영**과 **flash-attn**의 head 레이아웃에서 드러납니다; 이 파일에서 **Attention**이 받는 **k,v**는 이미 모델 구현에 따라 처리된 것입니다.

## 요약

- **Triton `store_kvcache_kernel`**: **토큰 병렬**로 K/V를 **slot 정렬된** 물리 캐시에 **scatter**합니다.
- **Prefill**: **`flash_attn_varlen_func`**가 **가변 길이 배치**를 처리하며, **`block_table`**을 지원합니다; 만약 **캐시에 이미 프리픽스가 있다면**, **`k,v = k_cache,v_cache`**의 **prefix** 최적화를 사용합니다.
- **Decode**: **`flash_attn_with_kvcache`**가 **단일 단계 이어쓰기**와 **KV 재사용**에 집중합니다.
- **FlashAttention**: **알고리즘+IO** 양 측면에서 긴 시퀀스 attention을 최적화하며, **추론 엔진**과 **면접** 모두에서 높은 빈도로 등장하는 주제입니다.

## 다음 강의 예고

다음 강의는 보통 **RoPE, LayerNorm, SwiGLU** 또는 **Qwen3 블록 레벨 구조**로 이어지며(튜토리얼 목차 기준), **위치 인코딩과 FFN**을 **Attention**과 결합하여 완전한 Decoder 레이어를 구성한 후, **KV 블록과 Scheduler**로 연결됩니다.