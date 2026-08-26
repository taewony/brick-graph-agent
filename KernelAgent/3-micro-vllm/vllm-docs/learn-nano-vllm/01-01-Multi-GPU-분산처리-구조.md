네, 질문하신 “L40S 4장이 꽂힌 GPU 서버에서 nano-vllm을 설치하고 하나의 SLM을 4장 모두로 서빙한다”는 상황은 바로 **`Config`에서 `tensor_parallel_size=4`로 설정**하고 엔진을 띄우는 경우입니다. nano-vllm은 (그리고 vLLM도 마찬가지로) **메가트론(Megatron) 스타일의 텐서 병렬(TP)** 을 사용해 모델 가중치와 연산을 GPU 여러 장에 분산시킵니다. 아래에 4장으로 분산 처리되는 주요 기능과 동작 구조를 순차적으로 설명드리겠습니다.

---

## 1. 전체적인 분산 구조 (TP=4)

4장의 GPU는 하나의 **TP 그룹**을 형성하며, 각 GPU는 모델의 **일부 가중치만 메모리에 보유**합니다. 동일한 입력 배치가 모든 GPU로 브로드캐스트되고, 각 GPU가 자신이 맡은 부분 연산을 수행한 뒤 필요할 때마다 **통신(all-reduce, gather)** 으로 전체 결과를 복원합니다.

```
[입력 토큰] → 모든 GPU에 같은 데이터
    ↓
GPU0: embedding shard 0 | Q/K/V heads 0-7 ...
GPU1: embedding shard 1 | Q/K/V heads 8-15 ...
GPU2: embedding shard 2 | Q/K/V heads 16-23 ...
GPU3: embedding shard 3 | Q/K/V heads 24-31 ...
    ↓
통신(all-reduce) → 부분 결과 합치기
    ↓
[다음 층] → 다시 분산 연산 반복
```

---

## 2. 어휘 임베딩 (VocabParallelEmbedding) – `layers/embed_head.py`

- **가중치 분할**: 전체 어휘 크기 `V`를 `tp_size=4`로 나누어 각 GPU는 `V/4` 행, 은닉 차원 `D` 전체를 저장합니다.
  - GPU0: 토큰 id 0 ~ V/4-1  
  - GPU1: 토큰 id V/4 ~ 2V/4-1  
  - GPU2: 토큰 id 2V/4 ~ 3V/4-1  
  - GPU3: 토큰 id 3V/4 ~ V-1
- **순전파**:
  - 각 GPU는 전체 입력 토큰 id 텐서를 받고, **자신이 가진 구간에 속한 id만** 로컬 행 번호로 변환해 임베딩을 룩업합니다.  
  - 나머지 구간의 id는 마스킹(masking)으로 **0 벡터**를 만듭니다.
  - 이후 **`all_reduce`(SUM)** 를 호출해 4장의 부분 임베딩을 더하면 **완전한 `D` 차원 임베딩**이 모든 GPU에 동일하게 복원됩니다. (코드: `dist.all_reduce(y)`)

> ✅ **4장에 분산되는 작업**: 어휘 테이블의 행 분할 저장 + 마스킹된 룩업 + all-reduce로 결과 통합.

---

## 3. QKV 선형 투영 (ColumnParallelLinear) – `layers/linear.py`

- **가중치 분할**: Attention의 Q, K, V 가중치는 **열 방향**으로 나뉩니다.  
  예를 들어 원래 가중치가 `(D, 3 * num_heads * head_dim)`라면, 두 번째 차원(출력 차원)을 4등분하여 각 GPU는 `(D, 3 * (num_heads/4) * head_dim)` 만큼의 가중치를 갖습니다.
- **동작**: 입력 `x` (shape `[total_tokens, D]`)는 모든 GPU에서 동일합니다.  
  각 GPU는 `x @ W_shard`를 계산해 **자신에게 할당된 헤드들에 해당하는 Q, K, V**만 생성합니다.  
  이 단계에서는 **통신이 전혀 없습니다** (column parallel은 입력 공유, 출력 분할).

> ✅ **4장에 분산되는 작업**: 전체 헤드 중 1/4씩 나누어 Q, K, V를 생성.

---

## 4. Attention 연산과 KV Cache

- **Attention (FlashAttention)**:
  - 각 GPU는 자신이 만든 Q, K, V (자신의 헤드 구간만)를 가지고 **독립적으로** attention을 수행합니다.  
  - Prefill에서는 `flash_attn_varlen_func`, Decode에서는 `flash_attn_with_kvcache`가 사용되며, 각 GPU는 **자기 헤드에 대한 attention 출력**만 산출합니다.
- **KV Cache 저장 (`store_kvcache`)**:
  - KV 캐시도 헤드 차원을 따라 파티션되어 있습니다.  
  - `k_cache`, `v_cache`의 shape은 대략 `(num_blocks, block_size, num_kv_heads/tp, head_dim)`입니다.  
  - `slot_mapping`과 `block_tables`는 **모든 GPU에서 동일**하며, 각 GPU는 `store_kvcache_kernel`을 통해 **자신이 보유한 헤드들의 K, V만** 물리 캐시에 기록합니다.

> ✅ **4장에 분산되는 작업**: attention 계산 자체가 헤드별로 나뉘어 수행되고, KV 캐시도 각 GPU가 자기 헤드 분량만 저장. 통신은 attention 내부에서는 발생하지 않음.

---

## 5. 출력 선형 투영 (RowParallelLinear) – `layers/linear.py`

- **가중치 분할**: Attention 출력을 다시 은닉 차원으로 투영하는 가중치는 **행 방향**으로 나뉩니다.  
  원래 `(num_heads * head_dim, D)`인 행렬을 **첫 번째 차원(입력 차원)**에서 4분할하여 각 GPU가 `( (num_heads/4)*head_dim , D)` 만큼 보유합니다.
- **동작**:  
  - 각 GPU는 앞서 attention에서 얻은 **부분 출력**(자기 헤드들만의 결과)을 `x_shard @ W_shard`로 변환합니다.  
  - 이 결과는 아직 전체의 일부이므로 **`all_reduce`(SUM)** 로 4장의 결과를 합쳐 **완전한 `D` 차원 히든 벡터**를 얻습니다.  
  - 이 `all_reduce`는 Transformer의 각 디코더 층마다 한 번씩 일어납니다.

> ✅ **4장에 분산되는 작업**: 각 GPU가 자기 헤드의 출력을 처리한 후 all-reduce로 합침. 이 통신이 층마다 발생하며, TP의 주요 오버헤드 중 하나입니다.

---

## 6. LM Head (ParallelLMHead) – `layers/embed_head.py`

- **Prefill**: 마지막 디코더 층의 출력에서 **각 시퀀스의 마지막 토큰**만 추출한 뒤(`cu_seqlens_q` 활용), 임베딩과 공유된 가중치로 logits를 계산합니다.  
  - 이 가중치도 vocab 차원으로 4분할되어 있으므로, 각 GPU는 **자신의 어휘 조각에 대한 logits**만 계산합니다 (`F.linear(x, weight)`).
- **통신**: TP > 1일 때, 전체 어휘 logits를 모아야 샘플링이 가능하므로 **`dist.gather`를 rank 0에 수행**하여 모든 GPU의 logits를 마지막 차원에 연결(cat)합니다. 나머지 rank는 `None`을 반환합니다.  
  - 통상 rank 0에서만 샘플링을 수행하고, 필요시 다음 토큰 id를 브로드캐스트합니다.

> ✅ **4장에 분산되는 작업**: 로짓 계산이 어휘 조각별로 이루어지고, gather로 rank 0에 전체 로짓 집결.

---

## 7. 추가: 가중치 로딩 및 통신 라이브러리

- **가중치 로딩**: `weight_loader` 함수가 `loaded_weight.narrow(0, start_idx, shard_size)`로 전체 체크포인트에서 각 TP rank에 맞는 가중치 조각을 잘라 `copy_`합니다.  
- **통신 백엔드**: `model_runner.py`에서 NCCL을 통해 all-reduce, gather 등이 수행되며, `tensor_parallel_size=4`로 초기화 시 자동으로 프로세스 그룹이 구성됩니다.

---

## 정리: 4장 GPU의 분산 처리 맵

| 구성 요소 | 분산 방식 | 통신 시점 |
|-----------|-----------|-----------|
| **Vocab Embedding** | 어휘 행 분할 → 마스킹 룩업 → all-reduce | 입력 직후 1회 (전체 벡터 통합) |
| **QKV Projection** | 열 병렬 → 헤드 분할 | 통신 없음 |
| **Attention** | 헤드별 독립 수행, KV 캐시도 헤드 분할 저장 | 통신 없음 |
| **Output Projection** | 행 병렬 → 부분 변환 후 all-reduce | 각 Transformer 층마다 1회 |
| **LM Head** | 어휘 분할 로짓 계산 → gather to rank 0 | 출력 직전 1회 (전체 로짓 취합) |
| **KV Cache** | 헤드 차원으로 파티션, slot_mapping은 공유 | store 시 통신 없음 |

이렇게 4장의 L40S가 협력함으로써 하나의 큰 모델을 메모리에 나누어 올리고, 연산도 병렬화하여 처리량과 최대 배치 크기를 확장할 수 있습니다. nano-vllm은 코드가 단순하여 이러한 TP 메커니즘을 약 1200줄 속에서도 핵심만 명쾌하게 보여주고 있습니다.