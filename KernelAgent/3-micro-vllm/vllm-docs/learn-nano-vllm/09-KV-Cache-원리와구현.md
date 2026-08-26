# 강의 09: KV Cache 원리와 구현

> 자기회귀 디코딩은 매 단계마다 하나의 토큰만 새로 추가하지만, 만약 매번 처음부터 시퀀스 전체의 K/V를 다시 계산한다면 \(O(T^2)\) 수준의 중복 연산이 낭비됩니다. KV Cache는 과거 위치의 K, V를 저장해 두어, 디코딩 단계가 대략 \(O(T)\)로 증가하도록 만듭니다.

## 본 강의 목표

- **KV Cache가 왜 필요한지**('매 단계 전체 재계산'과 비교하여) 명확히 설명할 수 있습니다.
- nano-vllm 텐서 shape과 일치하는 **메모리 추정 공식**을 암기하고 유도할 수 있습니다.
- **`allocate_kv_cache`** 를 읽고 이해합니다: 블록 수, 가용 메모리, `torch.empty` 6차원 텐서.
- **Prefill**과 **Decode** 단계에서 KV Cache 읽기/쓰기 패턴을 구분합니다.
- **면접 빈출 후속 질문**(양자화, 다중 요청, GQA가 공식에 미치는 영향)을 정리합니다.

## 핵심 개념

### 1. attention에서 중복 계산이 발생하는 이유

길이 \(T\)인 시퀀스에 대해, \(t\)번째 단계에서 attention을 처음부터 계산한다면, 모든 위치 \(1..t\)의 K, V가 필요합니다. 하지만 **\(t\)번째 단계에서 새로 추가되는 것은 위치 \(t\)의 query뿐**이며, 위치 \(1..t-1\)의 K, V는 이전 단계에 비해 **변하지 않습니다**(모델 가중치와 이미 생성된 토큰이 고정되어 있을 때).

따라서 **과거 모든 단계에서 이미 계산된 K, V**를 GPU에 캐시해 두면, 이번 단계에서는 현재 토큰의 K, V만 계산하여 캐시에 **추가**한 후, 이력과 함께 attention을 수행할 수 있습니다(보통 인과 마스크 또는 '과거에만 attend'하는 구현과 함께 사용).

**절약되는 것**: 과거 토큰에 대해 K/V 투영과 (일부 구현에서는) 중간 결과 쓰기를 반복하는 것을 방지합니다. 복잡도는 '매 단계 긴 시퀀스 prefill을 하는 것'에서 '현재 컨텍스트에 대해 상수 시간 또는 선형 시간'의 점진적 업데이트로 줄어듭니다(구체적인 상수는 커널과 헤드 설정에 따라 다름).

### 2. 전형적인 메모리 추정 공식 (강의와 일치)

각 레이어, 각 시퀀스 위치마다 **K**와 **V** 두 텐서를 저장해야 하며, shape은 `num_kv_heads`, `head_dim`과 관련됩니다.

**총 KV Cache 바이트 수**를 대략 계산하면 (아래 nano-vllm 구현 차원과 일치하게 쓸 경우):

\[
\text{KV_bytes} \approx 2 \times L \times T \times H*{kv} \times D \times S*{\mathrm{dtype}} \times B
\]

여기서:

- **2**: K와 V 두 개;
- **\(L\)**: `num_hidden_layers`;
- **\(T\)**: 시퀀스 길이(또는 현재 점유된 토큰 수 상한);
- **\(H\_{kv}\)**: `num_key_value_heads` (주의: **텐서 병렬 후 rank별 로컬 헤드 수로 취해야 함**);
- **\(D\)**: `head_dim`;
- **\(S\_{\mathrm{dtype}}\)**: 원소 당 바이트 수 (fp16=2, bf16=2 등);
- **\(B\)**: 배치 또는 병렬 시퀀스 수 (시스템의 공유/슬롯 여부에 따라).

면접 시 강조할 점: **GQA는 \(H\_{kv}\)를 사용하며 \(H\)가 아님**, 이는 MHA 공식과의 중요한 차이입니다.

### 3. nano-vllm에서의 '블록'과 전역 텐서

nano-vllm은 '각 요청마다 연속된 KV를 malloc'하는 단순 모드가 아니라, **고정된 블록 수 × 블록 크기**의 큰 풀을 미리 할당하고, **BlockManager**가 매핑을 관리합니다(다음 강의). `allocate_kv_cache`는 **풀 자체**의 메모리와 **레이어별 바인딩**을 담당하여 각 attention 모듈의 `k_cache` / `v_cache` 뷰로 연결합니다.

### 4. Prefill vs Decode

- **Prefill(프롬프트 단계)**: 프롬프트를 한 번에 처리하여, 여러 토큰의 Q/K/V를 병렬로 계산하고, KV Cache에 **연속된 구간을 기록**합니다; 계산 형태는 보통 '큰 텐서, 높은 병렬도'.
- **Decode(생성 단계)**: 단계마다 보통 **1개의 새 토큰**(또는 소량)만 처리하며, KV Cache에 **점진적 추가**; 계산 형태는 보통 '작은 배치, 메모리 대역폭에 민감'.

KV Cache의 이점은 decode 단계에서 가장 큽니다: 캐시가 없다면, 매 단계마다 전체 길이에 대해 과거 K/V를 재계산해야 하므로 지연 시간이 폭발적으로 증가합니다.

---

## 소스 코드 분석: `ModelRunner.allocate_kv_cache`

아래는 `nano-vllm-main/nanovllm/engine/model_runner.py`와 일치합니다.

```python
def allocate_kv_cache(self):
    config = self.config
    hf_config = config.hf_config
    free, total = torch.cuda.mem_get_info()
    used = total - free
    peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
    current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
    num_kv_heads = hf_config.num_key_value_heads // self.world_size
    head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
    block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize
    config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
    assert config.num_kvcache_blocks > 0
    self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
    layer_id = 0
    for module in self.model.modules():
        if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
            module.k_cache = self.kv_cache[0, layer_id]
            module.v_cache = self.kv_cache[1, layer_id]
            layer_id += 1
```

### 메모리 여유분: `total * gpu_memory_utilization - used - peak + current`

- **`mem_get_info`**: 현재 디바이스의 '여유/전체' 메모리.
- **`used = total - free`**: 여유가 아닌 부분(프레임워크 캐시 등 포함, 시맨틱은 CUDA 런타임 기준).
- **`peak` / `current`**: 할당자 통계의 피크와 현재 할당량. '워밍업으로 이미 할당되었지만 상주하지 않을 수 있는' 차이를 보정하는 데 사용.

전체 의도: **사용자 설정 이용률을 초과하지 않는 범위 내**에서, 완전한 **KV 블록**을 얼마나 더 수용할 수 있는지 추산합니다.

### `block_bytes`의 의미

단일 블록, 단일 레이어, 단일 rank의 KV 이중선? 공식을 주의하세요:

```text
2 * num_layers * block_size * num_kv_heads * head_dim * itemsize
```

이것은 **하나의 KV cache 블록 slot**이 차지하는 바이트 수입니다: **모든 레이어**를 가로지르며(\(2 \times L\) 인자가 K/V와 레이어 수를 모두 '블록당 비용'으로 접음), 그 결과 `num_kvcache_blocks = 가용 바이트 // block_bytes`로 **블록 슬롯 수**를 구합니다.

(차원으로 이해한다면: `self.kv_cache`의 첫 번째 차원 2는 K/V; 두 번째는 layer; 블록은 세 번째, 네 번째 차원에 있습니다. `block_bytes`는 '한 레이어의 한 블록'을 '동일 block_id에 대한 모든 레이어의 총 점유'로 확장하며, `empty` shape과 일관됩니다.)

### 6차원 텐서 `self.kv_cache` shape 분석

```text
(2, num_hidden_layers, num_kvcache_blocks, block_size, num_kv_heads, head_dim)
```

| 차원                   | 의미                                                       |
| ---------------------- | ---------------------------------------------------------- |
| **2**                  | K와 V 두 풀 (인덱스 0/1)                                   |
| **num_hidden_layers**  | 레이어별 독립 서브 텐서, 레이어별로 모듈에 바인딩하기 편리 |
| **num_kvcache_blocks** | PagedAttention의 블록 개수                                 |
| **block_size**         | 블록 당 수용 가능한 토큰 슬롯 수                           |
| **num_kv_heads**       | 이 rank 상의 KV 헤드 수 (`world_size`로 이미 나눈 값)      |
| **head_dim**           | 헤드 당 차원                                               |

### 레이어별 바인딩

`self.model.modules()`를 순회하며, `k_cache`, `v_cache`를 동시에 가진 모듈(각 레이어 Attention)이 있으면, **해당 레이어**에 대응하는 `layer_id`의 뷰를 가리킵니다:

```text
k_cache = kv_cache[0, layer_id]   # shape에서 K/V와 layer 앞 두 차원 제거
v_cache = kv_cache[1, layer_id]
```

이렇게 하면 순전파 시 모듈이 직접 자신의 레이어 슬라이스에 쓰며, 레이어마다 별도로 `torch.empty`를 할 필요가 없습니다.

---

## 그림 설명

### 시간에 따른 KV 추가 (개념)

```text
step 0:  [K0]
step 1:  [K0, K1]
...
step t:  [K0 ... Kt]
```

V도 동일; 구현 상으로는 단순한 벡터 추가가 아닌 블록 풀의 이산 블록들에 기록됩니다.

### Prefill vs Decode (대비)

```text
Prefill:   한 번에 여러 토큰의 KV 기록 (병렬도 높음)
Decode:    단계당 1 토큰 기록 (대역폭 민감, 캐시에 강하게 의존)
```

### 블록 관리자와의 관계 (예고)

```text
allocate_kv_cache  -->   하나의 큰 물리 풀
BlockManager         -->   논리 블록 <-> 시퀀스 토큰의 매핑 테이블
```

---

## 면접 출제 포인트

### 왜 공식에 `num_heads`가 아니라 `num_kv_heads`를 쓰는가

GQA/MQA 하에서는 여러 query 헤드가 KV 헤드를 공유하므로, 캐시는 **물리 KV 헤드**만 저장합니다.

### 텐서 병렬이 공식에 어떻게 들어가는가

각 rank는 **자기 분할**의 KV 헤드만 저장합니다: `num_kv_heads // world_size` (코드 변수명 `num_kv_heads`가 이미 나눠져 있음).

### KV Cache 양자화 (후속 질문)

INT8/FP8 등은 \(S\_{\mathrm{dtype}}\)을 낮추지만, 역양자화 또는 전용 커널이 필요합니다. 공식 구조는 변하지 않으며, **원소 당 바이트 수**와 **정밀도 손실** 논의로 변경됩니다.

### `assert config.num_kvcache_blocks > 0`

`max_model_len`을 과도하게 설정하거나, 이용률이 너무 높거나, 메모리가 너무 부족하면 0이 될 수 있습니다. 공학적으로는 사용자에게 파라미터 조정을 알리는 오류를 발생시켜야 합니다.

---

## 자주 나오는 면접 질문

1. **KV Cache만 있고, Q Cache는 없나요?**  
   각 단계는 현재 위치의 Q만 필요합니다. 과거 Q는 현재 단계 attention의 '과거 토큰과의 매칭'에 참여하지 않으므로 전체 과거 Q를 저장할 필요가 없습니다(표준 자기회귀 디코딩).

2. **KV Cache가 그래디언트와 함께 역전파되나요?**  
   추론 경로에는 그래디언트가 없습니다. 학습 시에는 보통 FlashAttention 등의 변형을 사용하며, 캐시 시맨틱이 다릅니다.

3. **블록 크기 `block_size`는 무엇에 영향을 주나요?**  
   단위 크기 vs 파편화: 작은 블록은 더 유연하지만 메타데이터 오버헤드가 큽니다. 큰 블록은 꼬리 공간을 낭비할 수 있습니다.

4. **왜 KV를 할당하기 전에 `warmup_model`을 해야 하나요?**  
   먼저 피크 할당과 cudnn/cublas 작업 공간을 트리거한 후, `peak`를 차감하여, 블록 수 추정을 실제 실행에 더 가깝게 만듭니다(소스 코드 순서와 일관).

5. **배치가 커지면 KV 메모리가 선형적으로 증가하나요?**  
   여러 시퀀스가 각각 슬롯을 차지하므로, 총 점유량은 동시 시퀀스 수에 따라 증가합니다. 구체적으로 선형인지 여부는 프리픽스 공유 여부, 페이징 여부 등에 따라 다릅니다.

---

## 요약

KV Cache는 과거 토큰에 대한 K/V 중복 계산을 방지하여, 저지연 디코딩의 핵심입니다. 메모리는 '레이어 × 길이 × KV 헤드 × 헤드 차원 × 정밀도 × 2'로 추정할 수 있습니다. nano-vllm은 **단일 대형 6차원 텐서 + 레이어별 뷰**로 풀을 관리하며, `allocate_kv_cache`가 GPU 여유분과 블록 비용에 따라 **가용 블록 수**를 계산합니다. Prefill은 배치 쓰기, Decode는 점진적 쓰기를 하며, 둘은 시스템 병목(연산 vs 대역폭)에 미치는 영향이 다릅니다.

## 다음 강의 예고

다음 강의 **PagedAttention과 BlockManager**: 운영체제 페이징 비유, xxhash 프리픽스 블록 재사용, `allocate`/`may_append`와 참조 카운팅, "블록 풀"을 실제로 "다중 요청 동시 처리"에 연결합니다.
