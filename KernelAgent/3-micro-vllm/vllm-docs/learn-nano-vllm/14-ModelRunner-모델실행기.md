# 강의 14: ModelRunner 모델 실행기

본 강의에서는 nano-vllm의 **ModelRunner**를 심층적으로 분석합니다. ModelRunner는 **스케줄러와 GPU 사이의 가교** 역할을 하며, 스케줄러가 선택한 시퀀스를 GPU가 실행할 수 있는 텐서 입력으로 변환하고, 모델 순전파를 구동한 후 샘플링 결과를 반환합니다. ModelRunner를 이해하는 것은 추론 엔진의 "실행 계층"을 이해하는 핵심입니다.

---

## 1. ModelRunner의 역할

### 1.1 개요

ModelRunner는 추론 엔진에서 **모델 실행**에 관한 모든 책임을 집니다.

```
Scheduler.schedule()
    ↓ 반환값: (seqs: list[Sequence], is_prefill: bool)
ModelRunner.run(seqs, is_prefill)
    ├── prepare_prefill(seqs) / prepare_decode(seqs)   ← 입력 텐서 구성
    ├── run_model(input_ids, positions, is_prefill)     ← 순전파
    └── sampler(logits, temperatures)                   ← 다음 토큰 샘플링
    ↓ 반환값: token_ids: list[int]
Scheduler.postprocess(seqs, token_ids)
```

### 1.2 6가지 책임

| 책임 | 설명 | 해당 메서드/단계 |
|------|------|-----------------|
| 모델 로딩 | HuggingFace 가중치를 GPU로 로드 | `__init__` |
| KV 캐시 할당 | GPU 메모리를 기반으로 KV 캐시 공간 계산 및 할당 | `allocate_kv_cache()` |
| 입력 준비 | Sequence 리스트를 GPU 텐서로 변환 | `prepare_prefill()` / `prepare_decode()` |
| 모델 실행 | 모델 순전파 호출 (eager 또는 CUDA Graph) | `run_model()` |
| 샘플링 | logits에서 다음 토큰 샘플링 | `sampler()` |
| 다중 GPU 통신 | TP > 1일 때 SharedMemory로 시퀀스 정보 동기화 | `write_shm()` / `loop()` |

---

## 2. 초기화 과정

### 2.1 생성자 전체 보기

```python
class ModelRunner:
    def __init__(self, config: Config, rank: int, event: Event):
        self.rank = rank
        self.world_size = config.tensor_parallel_size

        # 1. 분산 통신 초기화
        dist.init_process_group("nccl", "tcp://localhost:2333",
                                world_size=self.world_size, rank=rank)

        # 2. 모델 로드
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)

        # 3. 샘플러 초기화
        self.sampler = Sampler()

        # 4. 모델 웜업
        self.warmup_model()

        # 5. KV 캐시 할당
        self.allocate_kv_cache()

        # 6. CUDA Graph 캡처 (강제 eager 모드가 아닐 경우)
        if not self.enforce_eager:
            self.capture_cudagraph()

        # 7. 다중 GPU일 때, rank-0가 아니면 이벤트 루프 진입
        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
            else:
                self.shm = SharedMemory(name="nanovllm")
                self.loop()  # 비 rank-0은 여기서 영구 대기
```

### 2.2 초기화 순서에 대한 설계 고려사항

초기화 단계의 순서는 신중하게 설계되었습니다.

**1단계: NCCL 초기화**

```python
dist.init_process_group("nccl", "tcp://localhost:2333",
                        world_size=self.world_size, rank=rank)
```

모델을 로드하기 전에 반드시 완료되어야 합니다. 텐서 병렬 처리를 하는 선형 계층(예: `ColumnParallelLinear`)은 초기화 시 가중치 분할 방식을 결정하기 위해 `tp_rank`와 `tp_size`를 알아야 하기 때문입니다.

**2단계: 모델 로드**

```python
self.model = Qwen3ForCausalLM(hf_config)
load_model(self.model, config.model)
```

먼저 모델 구조를 생성하고(이때 가중치는 무작위 값), 그다음 HuggingFace 가중치 파일에서 파라미터를 로드합니다. `load_model`은 각 계층의 `weight_loader` 메서드를 호출하며, 텐서 병렬 시 가중치 분할을 처리합니다.

**3단계: 모델 웜업(Warmup)**

```python
self.warmup_model()
```

임의의 입력으로 한 번 순전파를 수행합니다. 목적은 다음과 같습니다.
- PyTorch/CUDA의 JIT 컴파일 유도 (Triton 커널 컴파일 등)
- CUDA 할당자가 메모리 풀을 미리 할당하도록 함
- 이후 연산에서 첫 컴파일로 인한 지연 변동 방지

**4단계: KV 캐시 할당**

반드시 웜업 후에 실행해야 합니다. 웜업이 GPU 메모리를 일부 사용하기 때문에, KV 캐시 할당 시 **남은** 가용 메모리를 정확히 알아야 합니다.

**5단계: CUDA Graph 캡처**

KV 캐시 할당 이후에 실행되어야 합니다. CUDA Graph로 캡처되는 순전파에서 KV 캐시 텐서를 사용하기 때문입니다.

### 2.3 KV 캐시 할당: allocate_kv_cache()

ModelRunner의 가장 중요한 초기화 단계 중 하나입니다.

```python
def allocate_kv_cache(self):
    # 1. GPU 메모리 상태 조회
    free, total = torch.cuda.mem_get_info()

    # 2. KV 캐시 블록 하나당 바이트 수 계산
    num_kv_heads = hf_config.num_key_value_heads // self.world_size
    block_bytes = (2                           # K와 V
                  * hf_config.num_hidden_layers # 모든 Transformer 계층
                  * self.block_size             # 블록당 토큰 수
                  * num_kv_heads                # KV 어텐션 헤드 수
                  * head_dim                    # 헤드당 차원
                  * hf_config.torch_dtype.itemsize)  # 데이터 타입 바이트 수

    # 3. 할당 가능한 블록 수 계산
    config.num_kvcache_blocks = int(
        total * config.gpu_memory_utilization   # 전체 메모리 중 사용 가능 비율
        - used                                  # 사용 중 메모리
        - peak + current                        # 최고점 예약
    ) // block_bytes

    # 4. KV 캐시 텐서 할당
    self.kv_cache = torch.empty(
        2,                            # K와 V
        hf_config.num_hidden_layers,  # 계층 수
        config.num_kvcache_blocks,    # 블록 수
        self.block_size,              # 블록당 토큰 수
        num_kv_heads,                 # KV 헤드 수
        head_dim                      # 헤드 차원
    )
```

#### KV 캐시 텐서의 형태 해석

```
kv_cache는 6차원 텐서입니다:
  0번 축: [K, V]                     → 2
  1번 축: [layer_0, ..., layer_N]    → num_hidden_layers
  2번 축: [block_0, ..., block_M]    → num_kvcache_blocks
  3번 축: [token_0, ..., token_B]    → block_size
  4번 축: [head_0, ..., head_H]      → num_kv_heads
  5번 축: [dim_0, ..., dim_D]        → head_dim
```

`l`번째 계층, `b`번째 블록, `t`번째 토큰의 K 벡터 접근:

```python
k = kv_cache[0, l, b, t, :, :]  # shape: [num_kv_heads, head_dim]
```

#### 블록이 2번째 축인 이유

어텐션 계산 시 `block_table`을 통해 물리 블록을 인덱싱합니다. 블록 차원을 앞쪽에 두면 `torch.index_select`나 CUDA 커널에서 연속적인 메모리 접근 패턴을 활용하여 메모리 접근 효율을 높일 수 있습니다.

#### 메모리 사용량 계산

Qwen3-7B 예시 (FP16, block_size=256):

```
num_hidden_layers = 32
num_kv_heads = 4 (GQA, KV 헤드 4개)
head_dim = 128
block_bytes = 2 × 32 × 256 × 4 × 128 × 2 = 16,777,216 바이트 = 16 MB/블록

GPU 80GB, gpu_memory_utilization=0.9, 모델 점유 약 14 GB:
가용 = 80 × 0.9 - 14 = 58 GB
num_kvcache_blocks = 58 GB / 16 MB ≈ 3,625 블록
전체 토큰 용량 = 3,625 × 256 ≈ 928,000 토큰
```

이는 KV 캐시가 약 93만 개의 토큰을 동시에 수용할 수 있음을 의미하며, 수백 개의 동시 요청을 지원하기에 충분합니다.

---

## 3. prepare_prefill 상세 설명

### 3.1 메서드 시그니처와 역할

```python
def prepare_prefill(self, seqs: list[Sequence]) -> tuple[list[int], list[int]]:
```

prefill할 시퀀스 목록을 모델 순전파에 필요한 입력 텐서로 변환합니다.

### 3.2 핵심 데이터 구조

```python
def prepare_prefill(self, seqs):
    input_ids = []       # 모든 시퀀스의 토큰 ID를 1차원으로 연결
    positions = []       # 각 토큰의 위치 인코딩 인덱스
    cu_seqlens_q = [0]   # Q의 누적 시퀀스 길이
    cu_seqlens_k = [0]   # K의 누적 시퀀스 길이
    slot_mapping = []    # 각 토큰에 대응하는 KV 캐시 물리 위치
```

### 3.3 시퀀스별 구성

```python
for seq in seqs:
    seqlen = len(seq)
    seqlen_q = seqlen - seq.num_cached_tokens   # Q 길이 (계산해야 할 부분)
    seqlen_k = seqlen                            # K 길이 (캐시 포함)

    # 1. input_ids 구성: 캐시되지 않은 토큰만 포함
    input_ids.extend(seq[seq.num_cached_tokens:])

    # 2. positions 구성: num_cached_tokens부터 시작
    positions.extend(list(range(seq.num_cached_tokens, seqlen)))

    # 3. 누적 시퀀스 길이 갱신
    cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
    cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)

    # 4. slot_mapping 구성
    for i in range(seq.num_cached_blocks, seq.num_blocks):
        block_id = seq.block_table[i]
        num_tokens = seq.last_block_num_tokens if i == seq.num_blocks - 1 else self.block_size
        for j in range(num_tokens):
            slot_mapping.append(block_id * self.block_size + j)
```

### 3.4 핵심 개념: cu_seqlens

`cu_seqlens`(cumulative sequence lengths)는 FlashAttention의 가변 길이 인터페이스의 핵심 입력으로, 각 시퀀스가 연결된 텐서 내에서 차지하는 경계를 표시합니다.

```
3개 시퀀스의 Q 길이가 각각 [100, 150, 200]인 경우:
cu_seqlens_q = [0, 100, 250, 450]

연결된 텐서에서:
- 시퀀스 0의 Q: 위치 [0, 100)
- 시퀀스 1의 Q: 위치 [100, 250)
- 시퀀스 2의 Q: 위치 [250, 450)
```

**cu_seqlens_q와 cu_seqlens_k가 달라질 수 있는 이유는?**

프리픽스 캐시 히트가 있을 때:

```
시퀀스 길이 = 300, num_cached_tokens = 256
seqlen_q = 300 - 256 = 44   (새 토큰 44개에 대한 Q만 계산)
seqlen_k = 300               (어텐션은 모든 300개 위치의 K에 접근)
```

Q가 K보다 짧은 이유: 캐시된 부분의 KV는 이미 KV 캐시에 있으므로 Q를 다시 계산할 필요는 없지만, 어텐션은 여전히 그들과 상호작용해야 합니다. 이때 FlashAttention은 `block_table`을 이용한 페이징 어텐션 경로를 사용합니다.

### 3.5 핵심 개념: slot_mapping

`slot_mapping`은 각 토큰을 KV 캐시의 물리 위치(slot)에 매핑합니다.

```
물리 위치 = block_id × block_size + 블록 내 오프셋

예: block_table = [5, 12], block_size = 256
- 토큰 0~255 → 블록 5, 슬롯 5×256+0 ~ 5×256+255
- 토큰 256~299 → 블록 12, 슬롯 12×256+0 ~ 12×256+43
```

어텐션 계층의 순전파에서 새로 계산된 K와 V는 `slot_mapping`이 지정한 위치에 기록됩니다.

```python
# attention.py
cache_k = kv_cache[0][layer_idx]  # [num_blocks, block_size, num_kv_heads, head_dim]
cache_k.view(-1, num_kv_heads, head_dim)[slot_mapping] = k
```

### 3.6 set_context 호출

모든 입력을 구성한 후, 메타데이터를 전역 컨텍스트에 설정하여 어텐션 계층이 읽을 수 있게 합니다.

```python
set_context(True,                   # is_prefill = True
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            slot_mapping=slot_mapping,
            block_tables=block_tables)
```

어텐션 계층은 `get_context()`를 통해 이 정보를 얻어 어떤 어텐션 계산 경로를 사용할지 결정합니다.

### 3.7 전체 예시

2개의 시퀀스가 있다고 가정합니다.
- seq_0: token_ids=[10,20,30,40], block_table=[5], num_cached_tokens=0
- seq_1: token_ids=[50,60,70], block_table=[8], num_cached_tokens=0

구성 결과:

```
input_ids  = [10, 20, 30, 40, 50, 60, 70]
positions  = [0, 1, 2, 3, 0, 1, 2]
cu_seqlens_q = [0, 4, 7]
cu_seqlens_k = [0, 4, 7]
slot_mapping = [5×256+0, 5×256+1, 5×256+2, 5×256+3,
                8×256+0, 8×256+1, 8×256+2]
```

---

## 4. prepare_decode 상세 설명

### 4.1 메서드 구현

```python
def prepare_decode(self, seqs):
    input_ids = []
    positions = []
    slot_mapping = []
    context_lens = []
    block_tables = []

    for seq in seqs:
        # 1. 입력은 1개의 토큰: 마지막으로 생성된 토큰
        input_ids.append(seq.last_token)

        # 2. 위치는 시퀀스 전체 길이 - 1
        positions.append(len(seq) - 1)

        # 3. slot_mapping: 새 토큰을 마지막 블록의 다음 위치에 기록
        slot_mapping.append(
            seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
        )

        # 4. context_lens: 어텐션이 접근해야 할 히스토리 길이
        context_lens.append(len(seq))

        # 5. block_tables: 해당 시퀀스의 모든 물리 블록 ID
        block_tables.append(seq.block_table)
```

### 4.2 prepare_prefill과의 주요 차이점

| 특성 | prepare_prefill | prepare_decode |
|------|----------------|----------------|
| 시퀀스당 토큰 수 | len(seq) - num_cached_tokens | 1 (last_token) |
| 위치 인코딩 | range(num_cached_tokens, len(seq)) | len(seq) - 1 |
| slot_mapping 의미 | 기록해야 할 모든 슬롯 | 1개의 새 슬롯 |
| cu_seqlens | 필요 (가변 길이 어텐션) | 불필요 |
| context_lens | 불필요 (cu_seqlens_k가 암시) | 필요 (시퀀스별 KV 길이) |
| 어텐션 경로 | flash_attn_varlen_func | flash_attn_with_kvcache |

### 4.3 slot_mapping 계산

```python
slot_mapping.append(
    seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
)
```

이 코드는 새 토큰의 KV가 KV 캐시의 어느 위치에 기록될지 결정합니다.

```
seq.block_table = [5, 12], block_size = 256, num_tokens = 300인 경우
last_block_num_tokens = 300 - 1×256 = 44
slot = 12 × 256 + 44 - 1 = 12 × 256 + 43 = 3115

append_token이 postprocess에서 호출될 때 num_tokens는 이미 새 토큰을 포함합니다.
따라서 last_block_num_tokens는 이미 새 토큰을 포함하고, slot은 새 토큰의 위치를 가리킵니다.
```

잠깐, 여기에는 미묘한 부분이 있습니다. `prepare_decode`는 `append_token` **이전에** 호출됩니다. 이때 `num_tokens`는 아직 새 토큰을 포함하지 않은 상태입니다.

실제로 nano-vllm에서는 `may_append`가 `schedule()`에서 새 토큰을 위한 블록을 미리 할당합니다(필요 시). 그리고 `slot_mapping` 계산에는 **현재** `last_block_num_tokens`가 사용되며, 이는 현재의 마지막 채워진 위치를 가리킵니다. 즉, 새 토큰이 기록될 다음 위치입니다.

구체적인 흐름:

```
1. 이전 단계 postprocess: append_token → num_tokens = N
2. 현재 단계 schedule: may_append (N % block_size == 1이면 새 블록 할당)
3. prepare_decode: slot = block_table[-1] × block_size + last_block_num_tokens - 1
   = block_table[-1] × block_size + (N - (num_blocks-1) × block_size) - 1
```

여기서 `last_block_num_tokens - 1`은 **0-기반 인덱스**에서 마지막으로 채워진 위치입니다. 새 KV가 이 위치에 기록됩니다.

### 4.4 block_tables 처리

```python
block_tables.append(seq.block_table)
```

`set_context`에서 모든 시퀀스의 `block_table`은 하나의 2차원 텐서로 병합되어 어텐션 커널에 전달됩니다.

```python
# 모든 시퀀스의 block_table을 동일한 행렬로 패딩 (최대 길이까지)
block_tables_tensor = torch.zeros(num_seqs, max_num_blocks, dtype=torch.int32)
for i, bt in enumerate(block_tables):
    block_tables_tensor[i, :len(bt)] = torch.tensor(bt)
```

어텐션 커널은 `block_tables_tensor[seq_idx]`를 통해 해당 시퀀스의 물리 블록을 찾습니다.

---

## 5. run_model 메서드

### 5.1 Eager 모드 vs CUDA Graph 모드

```python
def run_model(self, input_ids, positions, is_prefill):
    if is_prefill or self.enforce_eager:
        logits = self.model(input_ids, positions)   # 모델 직접 호출
    else:
        logits = self.graph_runners[len(input_ids)].run(input_ids, positions)
```

**Eager 모드**: PyTorch 모델 순전파를 직접 호출합니다. 매번 모든 CUDA 커널을 재실행하므로 CPU→GPU 런치 오버헤드가 발생합니다. Prefill 단계는 **항상 Eager 모드**를 사용합니다. Prefill의 배치 형태(전체 토큰 수)는 크게 변동되어 CUDA Graph를 미리 캡처하기 어렵기 때문입니다.

**CUDA Graph 모드**: Decode 단계에서 사용됩니다. 미리 캡처된 CUDA Graph에는 모든 커널 호출 시퀀스가 포함되어 있어, 실행 시 단 한 번의 `graph.replay()`만으로 완료되며 CPU→GPU 런치 오버헤드를 제거합니다. Decode처럼 단일 토큰 연산량은 적지만 커널 수가 많은 시나리오에서 CUDA Graph의 가속 효과가 뚜렷합니다.

### 5.2 Prefill에 CUDA Graph를 사용하지 않는 이유

1. **고정되지 않은 입력 길이**: 서로 다른 요청의 프롬프트 길이는 수십에서 수천 토큰까지 차이가 커서 가능한 모든 입력 형태를 열거할 수 없습니다.
2. **연산량이 큼**: Prefill의 커널 실행 시간이 런치 오버헤드보다 훨씬 크기 때문에 CUDA Graph의 이점이 미미합니다.
3. **FlashAttention 제한**: 가변 길이 어텐션(`flash_attn_varlen_func`)은 CUDA Graph 지원이 제한적입니다.

### 5.3 CUDA Graph의 배치 사이즈 선택

```python
def capture_cudagraph(self):
    for bs in capture_batch_sizes():
        self.graph_runners[bs] = CUDAGraphRunner(self.model, bs)
```

`capture_batch_sizes()`는 미리 캡처해야 할 배치 사이즈 리스트를 반환합니다. 보통 1부터 `max_num_seqs`까지의 값들(또는 자주 사용되는 값들)을 포함합니다.

실행 시 실제 배치 사이즈가 미리 캡처된 목록에 없으면 Eager 모드로 폴백합니다.

---

## 6. run 메서드 전체 흐름

### 6.1 메서드 구현

```python
def run(self, seqs, is_prefill):
    # 1. 입력 구성
    input_ids, positions = (
        self.prepare_prefill(seqs) if is_prefill
        else self.prepare_decode(seqs)
    )

    # 2. 순전파
    logits = self.run_model(input_ids, positions, is_prefill)

    # 3. 샘플링 (rank 0만 수행)
    token_ids = (
        self.sampler(logits, temperatures).tolist()
        if self.rank == 0
        else None
    )

    return token_ids
```

### 6.2 흐름도

```
run(seqs, is_prefill)
    │
    ├── is_prefill == True
    │       └── prepare_prefill(seqs)
    │               ├── input_ids 구성 (모든 시퀀스의 캐시되지 않은 토큰 연결)
    │               ├── positions 구성 (위치 인코딩 인덱스)
    │               ├── cu_seqlens_q, cu_seqlens_k 구성 (시퀀스 경계)
    │               ├── slot_mapping 구성 (KV 캐시 기록 위치)
    │               └── set_context(is_prefill=True, ...)
    │
    ├── is_prefill == False
    │       └── prepare_decode(seqs)
    │               ├── input_ids 구성 (시퀀스당 1개 last_token)
    │               ├── positions 구성 (시퀀스당 1개 위치)
    │               ├── slot_mapping 구성 (1개 새 슬롯)
    │               ├── context_lens, block_tables 구성
    │               └── set_context(is_prefill=False, ...)
    │
    ├── run_model(input_ids, positions, is_prefill)
    │       ├── Prefill/Eager → self.model(input_ids, positions)
    │       └── Decode/Graph  → self.graph_runners[bs].run(...)
    │       └── logits 반환: [num_tokens, vocab_size]
    │
    └── sampler(logits, temperatures)
            ├── logits / temperature
            ├── softmax → 확률 분포
            ├── multinomial 샘플링
            └── token_ids 반환: [num_seqs]
```

### 6.3 rank 0만 샘플링하는 이유

```python
token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
```

텐서 병렬 처리에서는 모든 랭크가 동일한 모델 순전파를 실행합니다(단지 각 랭크가 서로 다른 어텐션 헤드/FFN 분할을 처리). 최종 logits는 `lm_head`(`VocabParallelEmbedding`)의 `forward`에서 `all_gather`를 통해 모든 랭크로 모입니다.

하지만 **샘플링은 한 번만 수행**하면 됩니다. rank 0이 수행하고 `SharedMemory`나 다른 메커니즘을 통해 결과를 다른 랭크에 전달합니다. 이렇게 하면 다음을 방지할 수 있습니다.
1. 중복 계산 (샘플링은 빠르지만 불필요하게 반복할 필요가 없음)
2. 무작위 샘플링 결과의 불일치 (다른 랭크의 랜덤 시드가 다를 수 있음)

### 6.4 Sampler 구현

```python
class Sampler:
    def __call__(self, logits, temperatures):
        # 1. 온도 스케일링
        logits = logits / temperatures.unsqueeze(-1)
        # 2. Softmax로 확률 변환
        probs = torch.softmax(logits, dim=-1)
        # 3. 다항 분포 샘플링
        token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return token_ids
```

온도의 역할:
- `temperature = 1.0`: 표준 샘플링
- `temperature > 1.0`: 분포가 평평해져 출력이 더 무작위적
- `temperature → 0`: argmax(탐욕적)에 가까워지나, nano-vllm은 temperature=0을 허용하지 않음

---

## 7. 다중 프로세스 통신 (SharedMemory)

### 7.1 SharedMemory가 필요한 이유

텐서 병렬(TP > 1)에서는 각 GPU가 독립적인 프로세스를 실행합니다. 스케줄러는 rank 0 프로세스에서 동작하며, 시퀀스 정보를 다른 랭크의 `ModelRunner`에 전달해야 합니다.

```
Rank 0 (주 프로세스):
  LLMEngine → Scheduler → ModelRunner (rank 0)
                              │
                         SharedMemory
                              │
Rank 1 (하위 프로세스):       ModelRunner (rank 1)
Rank 2 (하위 프로세스):       ModelRunner (rank 2)
```

### 7.2 통신 메커니즘

**Rank 0이 SharedMemory에 쓰기**:

```python
def write_shm(self, method_name, *args):
    data = pickle.dumps([method_name, *args])
    n = len(data)
    self.shm.buf[0:4] = n.to_bytes(4, "little")   # 앞 4바이트에 길이 저장
    self.shm.buf[4:n+4] = data                      # 그 뒤에 직렬화 데이터
```

**Rank 1+가 SharedMemory 읽기**:

```python
def loop(self):
    while True:
        # 신호 대기
        event.wait()
        event.clear()

        # SharedMemory에서 읽기
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])

        # 해당 메서드 호출
        getattr(self, method_name)(*args)
```

### 7.3 call 메서드: 프로세스 간 메서드 호출

```python
def call(self, method_name, *args):
    if self.world_size > 1:
        self.write_shm(method_name, *args)
        self.event.set()   # 다른 랭크에 알림
    return getattr(self, method_name)(*args)
```

이는 **간단한 RPC**를 구현합니다.
1. Rank 0이 메서드 이름과 인자를 직렬화하여 SharedMemory에 기록
2. `Event`를 통해 다른 랭크에 알림
3. 모든 랭크가 동일한 메서드(예: `run(seqs, is_prefill)`)를 호출
4. 각 랭크가 독립적으로 순전파를 실행 (텐서 병렬이 가중치 분할과 AllReduce를 자동 처리)

### 7.4 Sequence 직렬화 최적화

SharedMemory를 통해 전달될 때 `Sequence` 객체는 `pickle.dumps()`로 직렬화되어야 합니다. 바로 이때 `__getstate__` 최적화가 빛을 발합니다.

```python
# sequence.py
def __getstate__(self):
    return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens,
            self.block_table,
            self.token_ids if self.num_completion_tokens == 0 else self.last_token)
```

Decode 단계에서는 5개의 값만 전달하므로(전체 토큰 리스트 대신) 직렬화 오버헤드를 크게 줄입니다. batch_size=256, 평균 시퀀스 길이 1000인 시나리오에서:

```
최적화 미적용: 256 × 1000 × 4 bytes ≈ 1 MB 직렬화 데이터
최적화 적용: 256 × 5 × 8 bytes ≈ 10 KB 직렬화 데이터
100배 감소!
```

### 7.5 SharedMemory vs NCCL vs gRPC

| 통신 방식 | 적합한 시나리오 | 장점 | 단점 |
|---------|--------------|------|------|
| SharedMemory | 동일 노드 프로세스 간 통신 | 무복사, 저지연 | 단일 노드로 제한 |
| NCCL | GPU 간 텐서 통신 | 고대역폭, 다중 노드 지원 | 텐서만 전달 가능 |
| gRPC | 크로스 노드 RPC | 유연함, 다중 언어 | 지연 시간이 비교적 높음 |

nano-vllm의 설계:
- **NCCL**: 텐서 병렬에서 AllReduce에 사용 (모델 순전파의 텐서 통신)
- **SharedMemory**: 시퀀스 메타 정보(메서드명, Sequence 객체 등 비텐서 데이터) 전달에 사용

---

## 8. 웜업과 메모리 관리

### 8.1 warmup_model의 역할

```python
def warmup_model(self):
    dummy_input_ids = torch.zeros(self.max_num_batched_tokens, dtype=torch.long, device="cuda")
    dummy_positions = torch.zeros(self.max_num_batched_tokens, dtype=torch.long, device="cuda")
    self.model(dummy_input_ids, dummy_positions)
    torch.cuda.synchronize()
```

웜업의 목적:

1. **Triton 커널 컴파일**: 최초 실행 시 Triton이 커스텀 어텐션 커널, LayerNorm 커널 등을 JIT 컴파일합니다. 컴파일에 수 초가 소요될 수 있으며, 웜업은 이 지연을 초기화 단계로 앞당깁니다.
2. **CUDA 메모리 풀 초기화**: PyTorch CUDA 메모리 할당자는 첫 할당 시 메모리 풀을 구축합니다. 웜업 후에는 풀에서 할당받아 더 빨라집니다.
3. **cuBLAS 핸들 초기화**: 행렬 곱셈 라이브러리 cuBLAS는 첫 호출 시 초기화가 필요하며, 웜업이 이 오버헤드를 사전에 처리합니다.
4. **메모리 최고점 확인**: 웜업 후 `torch.cuda.max_memory_allocated()`를 호출하면 순전파의 메모리 최고점을 얻을 수 있으며, 이후 KV 캐시 할당 계산에 사용됩니다.

### 8.2 GPU 메모리 예산 할당

GPU 메모리는 다음 우선순위에 따라 할당됩니다.

```
GPU 전체 메모리 (예: 80 GB)
├── 모델 가중치 (예: 14 GB)          ← 고정
├── 순전파 활성값 최고점 (예: 2 GB)  ← 웜업 후 결정
├── CUDA 런타임 오버헤드 (예: 1 GB)  ← 고정
├── KV 캐시 (예: 58 GB)             ← allocate_kv_cache에서 할당
└── 여유 마진 (예: 5 GB)            ← gpu_memory_utilization < 1.0
```

`gpu_memory_utilization` 매개변수(기본값 0.9)는 KV 캐시가 전체 메모리의 몇 퍼센트까지 사용할 수 있는지를 제어합니다. 0.9로 설정하면 10%의 메모리 여유를 남겨 OOM을 방지합니다.

---

## 9. 면접 빈출 핵심

### Q1: ModelRunner의 prepare_prefill과 prepare_decode의 차이점은 무엇인가요?

**참고 답안**:

| 차원 | prepare_prefill | prepare_decode |
|------|----------------|----------------|
| 입력 토큰 수 | 시퀀스당 seqlen - num_cached_tokens | 시퀀스당 1 (last_token) |
| 데이터 형식 | 가변 길이 연결 + cu_seqlens | 고정 길이 (시퀀스당 1토큰) |
| slot_mapping | 여러 슬롯 (계산할 각 토큰당 하나) | 1개 슬롯 |
| 어텐션 경로 | flash_attn_varlen_func | flash_attn_with_kvcache |
| CUDA Graph | 미사용 (형태 고정 안 됨) | 사용 (형태는 batch_size로 결정) |

### Q2: KV 캐시 할당이 웜업 이후에 이루어져야 하는 이유는 무엇인가요?

**참고 답안**:

웜업이 GPU 메모리를 소비하고(Triton 커널 컴파일, CUDA 메모리 풀 구축 등), 웜업 과정에서 발생하는 메모리 최고점이 순전파에 필요한 최대 작업 공간을 결정하기 때문입니다. 웜업 이후에야 "KV 캐시에 할당할 수 있는 남은 메모리"를 정확히 알 수 있습니다. 웜업 전에 할당하면 너무 많이 할당하여 OOM이 발생하거나, 너무 적게 할당하여 낭비가 생길 수 있습니다.

### Q3: nano-vllm은 다중 GPU 추론에서 어떻게 통신을 구현하나요?

**참고 답안**:

두 가지 통신 메커니즘을 사용합니다.
1. **NCCL AllReduce**: 모델 순전파의 텐서 통신에 사용됩니다. `RowParallelLinear`와 `VocabParallelEmbedding`에서 각 랭크가 부분 결과를 계산한 후 AllReduce로 취합합니다. 이는 높은 대역폭의 GPU 간 통신입니다.
2. **SharedMemory + pickle**: 시퀀스 메타 정보를 전달하는 데 사용됩니다. Rank 0이 메서드 이름과 Sequence 객체를 직렬화하여 공유 메모리에 쓰고, Event를 통해 다른 랭크에 알립니다. `Sequence.__getstate__` 최적화를 통해 decode 단계에서는 5개의 기본 값만 전달합니다.

### Q4: cu_seqlens의 의미와 가변 길이 어텐션에서의 역할을 설명해주세요.

**참고 답안**:

`cu_seqlens`(cumulative sequence lengths)는 FlashAttention 가변 길이 인터페이스의 핵심 입력입니다. 길이가 `num_seqs + 1`인 1차원 정수 배열로, 각 시퀀스가 연결된 텐서에서 시작되는 누적 위치를 기록합니다.

예: 3개 시퀀스 길이 [100, 150, 200] → `cu_seqlens = [0, 100, 250, 450]`. FlashAttention은 `cu_seqlens[i]`와 `cu_seqlens[i+1]`로 i번째 시퀀스의 범위를 결정하여 패딩을 피합니다.

프리픽스 캐시 시나리오에서는 `cu_seqlens_q`와 `cu_seqlens_k`가 달라질 수 있습니다. Q는 캐시되지 않은 토큰만 포함하고 K는 모든 토큰(캐시된 부분 포함)을 포함하기 때문입니다.

### Q5: CUDA Graph가 Decode 단계에서 가져오는 가속 효과는 어느 정도이며, Prefill에 사용하지 않는 이유는 무엇인가요?

**참고 답안**:

가속 효과: Decode 단계는 계산량이 적지만(시퀀스당 1토큰) CUDA 커널 수가 많습니다(계층마다 여러 커널). CPU→GPU 런치 오버헤드가 전체 시간의 30-50%를 차지할 수 있습니다. CUDA Graph는 모든 커널을 하나의 replay로 묶어 런치 오버헤드를 없애므로 보통 1.5~2배 가속됩니다.

Prefill에 사용하지 않는 이유:
1. Prefill의 입력 길이가 고정되지 않아 가능한 모든 형태를 캡처할 수 없음
2. Prefill의 커널 실행 시간이 런치 오버헤드보다 훨씬 길어 CUDA Graph의 한계 이득이 작음
3. FlashAttention의 가변 길이 인터페이스가 CUDA Graph 지원이 제한적임

### Q6: slot_mapping은 Prefill과 Decode 단계에서 각각 무엇을 의미하나요?

**참고 답안**:

`slot_mapping`은 토큰을 KV 캐시의 물리 위치로 매핑합니다. 계산식: `slot = block_id × block_size + 블록 내 오프셋`.

- **Prefill**: 계산이 필요한 각 토큰마다 슬롯을 생성합니다. 프리픽스 캐시가 있으면 캐시되지 않은 토큰에 대해서만 슬롯을 생성합니다(`num_cached_blocks`부터 시작). 새로 계산된 모든 K/V가 이 슬롯들에 일괄 기록됩니다.
- **Decode**: 시퀀스당 1개의 슬롯만 가지며, 마지막 블록의 현재 쓰기 위치를 가리킵니다. 새로 생성된 K/V가 이 하나의 슬롯에 기록됩니다.

### Q7: ModelRunner를 최적화한다면 어떤 측면에서 접근하시겠습니까?

**참고 답안**:

1. **청크 Prefill (Chunked Prefill)**: 긴 Prefill을 청크로 나누어 Decode와 혼합 실행하여 Decode 대기 시간 감소
2. **더 효율적인 직렬화**: pickle 대신 struct.pack을 사용하여 SharedMemory 전송량을 더욱 감소
3. **동적 CUDA Graph**: 가변 배치 사이즈를 지원하는 CUDA Graph, 혹은 CUDA Graph 조건 노드 사용
4. **비동기 KV 캐시 관리**: 순전파와 동시에 블록 할당/해제를 비동기로 처리
5. **추측적 디코딩 (Speculative Decoding)**: 드래프트 모델을 통합하여 대형 모델의 Decode 단계 수 감소
6. **KV 캐시 양자화**: KV 캐시를 FP16에서 INT8/FP8로 압축하여 수용 가능한 토큰 수 증가

---

## 10. 요약

| 핵심 | 내용 |
|------|------|
| **핵심 역할** | Sequence 리스트를 GPU 실행 가능한 텐서 입력으로 변환하고 모델 순전파를 구동 |
| **초기화 순서** | NCCL → 모델 로드 → 웜업 → KV 캐시 할당 → CUDA Graph 캡처 |
| **prepare_prefill** | 가변 길이 연결 + cu_seqlens + slot_mapping, 프리픽스 캐시 지원 |
| **prepare_decode** | 시퀀스당 1 토큰 + block_tables + context_lens |
| **실행 모드** | Prefill = Eager, Decode = CUDA Graph |
| **다중 GPU 통신** | NCCL(텐서) + SharedMemory(메타 정보), Sequence 직렬화 최적화 |
| **KV 캐시 할당** | GPU 잔여 메모리로 블록 수 계산, 6차원 텐서에 저장 |

**핵심 암기구**:

> Prefill은 **1차원으로 연결**, cu_seqlens로 **경계를 긋고**;  
> Decode는 **마지막 토큰을 뽑아**, slot_mapping으로 **위치를 정하며**;  
> Eager는 **큰 행렬**을 계산하고, Graph는 **작은 벡터**를 실행하며;  
> SharedMemory는 **메타 정보**를 전달하고, NCCL은 **텐서를 나른다**.

**다음 강의 예고**: ModelRunner가 모델을 어떻게 실행하는지 이해했으니, 이제 **텐서 병렬(Tensor Parallelism)**을 깊이 파고들어 ColumnParallelLinear, RowParallelLinear, QKVParallelLinear가 어떻게 여러 GPU에 걸쳐 가중치와 연산을 분할하는지 알아보겠습니다.