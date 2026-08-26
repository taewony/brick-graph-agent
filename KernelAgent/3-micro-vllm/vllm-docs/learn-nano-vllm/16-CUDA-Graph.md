# 제16강: CUDA Graph 최적화

초보자를 위한 설명: 본 강의는 'CUDA Graph란 무엇인가'부터 nano-vllm에서 그래프를 녹화하고 재사용하여 **decode(토큰 단위 생성)** 단계를 가속하는 방법까지 다루며, 소스 코드를 줄 단위로 해석하고 면접 문제를 통해 이해를 굳힙니다.

다음 코드는 `nanovllm/engine/model_runner.py`의 로직과 일치하므로 함께 읽기에 좋습니다.

```python
@torch.inference_mode()
def capture_cudagraph(self):
    config = self.config
    hf_config = config.hf_config
    max_bs = min(self.config.max_num_seqs, 512)
    max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
    input_ids = torch.zeros(max_bs, dtype=torch.int64)
    positions = torch.zeros(max_bs, dtype=torch.int64)
    slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
    context_lens = torch.zeros(max_bs, dtype=torch.int32)
    block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
    outputs = torch.zeros(max_bs, hf_config.hidden_size)
    self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
    self.graphs = {}
    self.graph_pool = None
    for bs in reversed(self.graph_bs):
        graph = torch.cuda.CUDAGraph()
        set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
        outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # warmup
        with torch.cuda.graph(graph, self.graph_pool):
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # capture
        if self.graph_pool is None:
            self.graph_pool = graph.pool()
        self.graphs[bs] = graph
        torch.cuda.synchronize()
        reset_context()
    self.graph_vars = dict(input_ids=input_ids, positions=positions, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables, outputs=outputs)

def run_model(self, input_ids, positions, is_prefill):
    if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
        return self.model.compute_logits(self.model(input_ids, positions))
    else:
        bs = input_ids.size(0)
        context = get_context()
        graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
        graph_vars = self.graph_vars
        graph_vars["input_ids"][:bs] = input_ids
        graph_vars["positions"][:bs] = positions
        graph_vars["slot_mapping"].fill_(-1)
        graph_vars["slot_mapping"][:bs] = context.slot_mapping
        graph_vars["context_lens"].zero_()
        graph_vars["context_lens"][:bs] = context.context_lens
        graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
        graph.replay()
        return self.model.compute_logits(graph_vars["outputs"][:bs])
```

---

## 1. 개념 설명

### 1.1 CUDA Graph란 무엇인가

일반적인 PyTorch / CUDA 추론에서 **각 연산**(행렬 곱, 정규화, 어텐션 등)은 대부분 **CPU가 GPU에 커널을 제출(kernel launch)**하는 과정을 거칩니다. "launch" 한 번은 CPU가 작업 지시서를 작성해 GPU 드라이버에 전달하여 대기열에 넣는 과정이라고 볼 수 있습니다.

**CUDA Graph**는 NVIDIA가 제공하는 메커니즘으로, **이미 결정된 GPU 연산 시퀀스 전체**를 **사전 녹화(capture)**하여 **DAG(방향성 비순환 그래프)**로 만든 후, 실행 시에는 단순히 **`replay()` 재생**만 하면 되도록 합니다. 매 단계마다 전체 launch 경로를 반복할 필요가 없습니다.

직관적인 비유:

- **Graph 미사용**: 감독이 매 장면마다 모든 배우에게 대사를 직접 말해줍니다(CPU의 빈번한 개입).
- **Graph 사용**: 리허설 때 '최종 영상' 하나로 녹화해 두고, 실제 공연 때 바로 재생합니다(한 번 제출, GPU가 그래프에 따라 실행).

PyTorch에서는 보통 `torch.cuda.CUDAGraph()`와 `torch.cuda.graph()` 컨텍스트 매니저로 녹화와 재생을 수행합니다.

### 1.2 CUDA Graph의 저수준 원리

CUDA Graph는 두 부분으로 구성됩니다.

1. **그래프 정의(Graph Definition)**: 여러 **노드(Node)**와 **에지(Edge)**로 이루어진 방향성 비순환 그래프입니다. 각 노드는 하나의 GPU 작업(커널 런치, 메모리 복사 등)을, 에지는 의존 관계를 나타냅니다.
2. **그래프 인스턴스(Graph Instance)**: 그래프 정의로부터 생성된 실행 가능한 객체로, GPU에서 직접 제출하여 실행할 수 있습니다.

기존 CUDA 실행 모델에서는 CPU 측 **드라이버**가 각 커널을 하나씩 검사, 대기, 제출해야 했습니다. 이 과정에는 다음과 같은 오버헤드가 수반됩니다.

| 오버헤드 발생원 | 설명 |
|---------------|------|
| API 호출 오버헤드 | `cudaLaunchKernel` 호출 시마다 드라이버 수준 파라미터 검증 |
| 동기화 및 의존성 검사 | 선행 작업 완료를 확인한 후 후속 작업 제출 |
| CPU-GPU 통신 | PCIe/NVLink를 통한 명령 버퍼 전달 |
| Python 인터프리터 오버헤드 | PyTorch의 Python dispatch, autograd 등 여러 계층 감싸기 |

CUDA Graph를 사용하면 위 오버헤드는 **capture 단계**에서 단 한 번만 발생하며, 이후 `replay()`는 **단 한 번의 `cudaGraphLaunch` 호출**만 필요하고 GPU가 미리 녹화된 DAG대로 모든 연산을 스스로 수행합니다.

### 1.3 Decode 단계에 CUDA Graph가 특히 적합한 이유

대규모 언어 모델 추론은 **Prefill**과 **Decode** 두 단계로 나뉩니다(1강 복습).

| 특징 | Prefill | Decode |
|------|--------|--------|
| 스텝당 토큰 수 | \(L_p\) (수천까지 가능) | 시퀀스당 1개 |
| 계산 그래프 형태 | **가변 길이** (프롬프트 길이에 의존) | **비교적 고정** (매 스텝 구조 일정) |
| 연산 특성 | 연산 중심 (compute-bound) | 메모리 대역폭 중심 (memory-bound) |
| 주요 병목 | GPU 연산 능력 | CPU launch + 메모리 대역폭 |

Decode 단계의 핵심 특징:

1. **계산 그래프 구조 고정**: 스텝마다 각 시퀀스가 토큰 1개만 처리하며, Transformer 각 레이어의 행렬 형태는 완전히 예측 가능합니다.
2. **작은 규모의 고빈도 실행**: 스텝당 연산량은 크지 않지만, 커널 런치 횟수는 많아(LayerNorm, QKV 프로젝션, 어텐션, FFN 등) 런치 오버헤드 비중이 큽니다.
3. **CPU가 병목**: GPU는 한 스텝의 행렬 연산을 금방 끝내지만, CPU는 다음 스텝의 커널을 제출하느라 바빠서 GPU가 유휴 상태로 기다리게 됩니다.

따라서 decode 순전파 전체를 CUDA Graph로 녹화하면 **수십 번의 커널 런치를 단 한 번의 그래프 런치로 압축**하여 CPU 측 오버헤드를 크게 줄이고 GPU 활용률을 높일 수 있습니다.

### 1.4 CUDA Graph의 한계

CUDA Graph가 만능은 아니라는 점을 주의해야 합니다.

- **CPU 로직 포함 불가**: 녹화 중에는 Python 조건 분기, 루프 등의 동적 제어 흐름이 있을 수 없습니다.
- **동적 shape 연산 포함 불가**: 텐서 모양은 capture 시점에 확정되어야 합니다.
- **CPU-GPU 동기화 포함 불가**: `torch.cuda.synchronize()`, `.item()` 등의 연산은 그래프 내부에 있을 수 없습니다.
- **메모리 주소 바인딩**: capture 시 사용된 텐서 주소가 그래프에 묶이므로, replay 시에는 **동일한 메모리**에 데이터를 써야 합니다.

이러한 제약 때문에 nano-vllm은 **shape가 안정적인 decode 단계에만 Graph를 사용**하고, Prefill은 eager로 처리하는 전략을 취합니다.

---

## 2. 소스 코드 분석: `capture_cudagraph` 전체 흐름

구현은 `nanovllm/engine/model_runner.py`에 있습니다. 핵심은 **초기화 시 capture 여부**, **capture 구체적 단계**, **순전파 시 replay 여부**입니다.

### 2.1 CUDA Graph 사용 조건

`ModelRunner.__init__`에서 KV Cache 할당 후, **`enforce_eager`가 설정되지 않았다면** `capture_cudagraph()`를 호출합니다. `enforce_eager=True`이면 완전히 **eager(즉시 실행)** 경로로만 동작하며 그래프를 녹화하지 않습니다(디버깅이 쉬움).

### 2.2 `capture_cudagraph` 줄 단위 해석

```python
def capture_cudagraph(self):
    config = self.config
    hf_config = config.hf_config

    # ① 최대 배치 크기 상한 결정
    max_bs = min(self.config.max_num_seqs, 512)

    # ② KV Cache에서 시퀀스 하나당 최대 블록 수 계산
    max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size

    # ③ 최대 규모의 입출력 버퍼 사전 할당 (graph_vars)
    input_ids = torch.zeros(max_bs, dtype=torch.int64)
    positions = torch.zeros(max_bs, dtype=torch.int64)
    slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
    context_lens = torch.zeros(max_bs, dtype=torch.int32)
    block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
    outputs = torch.zeros(max_bs, hf_config.hidden_size)

    # ④ 캡처할 배치 크기 단계 정의
    self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
    self.graphs = {}
    self.graph_pool = None

    # ⑤ 큰 것부터 작은 것 순으로 하나씩 녹화
    for bs in reversed(self.graph_bs):
        graph = torch.cuda.CUDAGraph()

        # decode 컨텍스트 설정 (어텐션 레이어에게 현재 decode 모드임을 알림)
        set_context(False, slot_mapping=slot_mapping[:bs],
                    context_lens=context_lens[:bs],
                    block_tables=block_tables[:bs])

        # 웜업: CUDA 컨텍스트, JIT 등 초기화 완료 보장
        outputs[:bs] = self.model(input_ids[:bs], positions[:bs])

        # 정식 캡처
        with torch.cuda.graph(graph, self.graph_pool):
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])

        # 첫 번째 그래프 생성 후 메모리 풀 획득, 이후 그래프들 공유
        if self.graph_pool is None:
            self.graph_pool = graph.pool()

        self.graphs[bs] = graph
        torch.cuda.synchronize()
        reset_context()

    # ⑥ 참조 저장, replay 시 이 버퍼들을 통해 데이터 전달
    self.graph_vars = dict(
        input_ids=input_ids, positions=positions,
        slot_mapping=slot_mapping, context_lens=context_lens,
        block_tables=block_tables, outputs=outputs,
    )
```

아래에서 하나씩 핵심 설계를 살펴보겠습니다.

### 2.3 `max_bs` 상한 설계

```python
max_bs = min(self.config.max_num_seqs, 512)
```

- `max_num_seqs`는 스케줄러가 허용하는 최대 동시 시퀀스 수입니다.
- 하드코딩된 상한 512는 공학적 경험치입니다. 배치가 512를 넘으면 연산 자체로 GPU를 충분히 활용할 수 있어 launch 오버헤드 비중이 낮아지고 Graph의 이득이 줄어듭니다.
- 동시에, 그래프 개수가 많아질수록 캡처 시간과 GPU 메모리 오버헤드가 커지므로 512가 합리적인 절충 상한입니다.

### 2.4 `graph_bs` 단계 설계: `[1, 2, 4, 8, 16, 32, ...]`

```python
self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
```

모든 배치 크기마다 그래프를 녹화하지 않는 이유:

1. **메모리 비용**: 그래프 하나마다 GPU 메모리가 고정되므로 그래프가 너무 많으면 GPU 메모리를 낭비합니다.
2. **캡처 시간**: 그래프마다 웜업 + 캡처가 필요하므로 개수가 많을수록 초기화가 느려집니다.

단계 설계 전략:

- **작은 배치(1-8)는 촘촘하게**: 작은 배치에서는 패딩으로 낭비되는 계산 비율이 높습니다. 예를 들어 bs=3을 bs=4에 매핑하면 25%만 낭비되지만, bs=16만 있다면 bs=3은 80% 이상 낭비됩니다.
- **큰 배치(16 이상)는 16 간격으로**: 큰 배치에서는 패딩 낭비 비율이 낮아지므로 16 단계가 적절한 절충입니다.

### 2.5 `reversed`로 큰 것부터 작은 것 순으로 녹화하는 이유

```python
for bs in reversed(self.graph_bs):
```

CUDA Graph의 메모리 풀(`graph_pool`)은 첫 사용 시 필요에 따라 메모리를 할당합니다. **가장 큰 배치**부터 녹화를 시작하면 메모리 풀이 처음부터 충분히 큰 공간을 확보하게 됩니다. 이후 작은 배치는 이 공간의 일부만 사용하므로 추가 확장 없이 단편화를 줄일 수 있습니다.

### 2.6 웜업의 필요성

```python
outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # warmup
```

처음 실행 시 다음과 같은 일이 발생할 수 있습니다.

- **CUDA 컨텍스트 초기화**: 디바이스 메모리 할당, 드라이버 연결 설정.
- **cuDNN/cuBLAS 알고리즘 선택**: 첫 행렬 곱셈에서 auto-tuning 발생.
- **`torch.compile`의 JIT 컴파일**: 모델에 `@torch.compile`이 있다면 첫 호출 시 컴파일 트리거.

이런 불안정한 동작이 그래프 안에 녹화되면 replay 시 오류 또는 크래시가 발생할 수 있습니다. 따라서 **반드시 웜업 후 캡처**해야 합니다.

### 2.7 `graph_pool` 공유 메모리 풀

```python
with torch.cuda.graph(graph, self.graph_pool):
    ...
if self.graph_pool is None:
    self.graph_pool = graph.pool()
```

여러 개의 CUDAGraph가 각자 독립적으로 메모리를 할당하면 GPU 메모리 단편화가 심해집니다. PyTorch는 첫 번째 그래프의 `graph.pool()`을 이후 `torch.cuda.graph(..., pool=)`에 넘겨 여러 그래프가 동일한 메모리 풀을 공유하도록 허용합니다.

이점:
- **GPU 메모리 단편화 감소**: 모든 그래프가 동일한 사전 할당 영역을 재사용.
- **전체 GPU 메모리 사용량 감소**: 서로 다른 배치 크기의 그래프 임시 버퍼가 겹쳐서 사용될 수 있습니다(동시에 replay되는 그래프는 하나뿐이므로).

### 2.8 `graph_vars`: 입출력 매핑 브릿지

```python
self.graph_vars = dict(
    input_ids=input_ids, positions=positions,
    slot_mapping=slot_mapping, context_lens=context_lens,
    block_tables=block_tables, outputs=outputs,
)
```

CUDA Graph 내부에 기록되는 것은 **텐서 객체가 아닌 텐서의 메모리 주소**입니다. 따라서 replay 전에 실제 데이터를 캡처 시점과 **동일한 메모리**에 복사해 넣어야 합니다. `graph_vars`는 이 "브릿지" 텐서들의 참조를 저장합니다.

| 변수 | 의미 | 데이터 흐름 |
|------|------|----------|
| `input_ids` | 현재 decode 스텝의 토큰 ID | 입력 → 모델 |
| `positions` | 각 토큰의 절대 위치 | 입력 → RoPE |
| `slot_mapping` | 토큰이 KV Cache에 기록될 타겟 슬롯 | 입력 → Attention |
| `context_lens` | 각 시퀀스의 현재 컨텍스트 길이 | 입력 → FlashAttention |
| `block_tables` | 시퀀스 KV 블록 매핑 테이블 | 입력 → FlashAttention |
| `outputs` | 모델 마지막 레이어의 hidden states | 모델 → 출력 |

---

## 3. 소스 코드 분석: `run_model`의 CUDA Graph 분기

```python
def run_model(self, input_ids, positions, is_prefill):
    # 분기 1: Prefill / eager / 초대형 배치 → 직접 순전파
    if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
        return self.model.compute_logits(self.model(input_ids, positions))

    # 분기 2: Decode + Graph 사용 가능 → replay
    else:
        bs = input_ids.size(0)
        context = get_context()

        # bs보다 작지 않은 최소 단계 선택
        graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
        graph_vars = self.graph_vars

        # 실제 데이터를 graph_vars로 복사
        graph_vars["input_ids"][:bs] = input_ids
        graph_vars["positions"][:bs] = positions
        graph_vars["slot_mapping"].fill_(-1)        # 먼저 비움
        graph_vars["slot_mapping"][:bs] = context.slot_mapping
        graph_vars["context_lens"].zero_()           # 먼저 0으로
        graph_vars["context_lens"][:bs] = context.context_lens
        graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables

        # 순전파 전체를 한 번에 재생
        graph.replay()

        return self.model.compute_logits(graph_vars["outputs"][:bs])
```

### 3.1 세 갈래 판단 로직

`run_model`의 첫 `if` 문은 **Graph를 타지 않는** 세 가지 경우를 처리합니다.

1. **`is_prefill`**: Prefill 단계는 토큰 수가 가변적이어서 고정 shape Graph에 부적합.
2. **`self.enforce_eager`**: 사용자가 명시적으로 eager 실행 요청(주로 디버깅용).
3. **`input_ids.size(0) > 512`**: 배치가 캡처 상한을 초과하여 해당하는 그래프 없음.

### 3.2 단계 선택 전략

```python
graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
```

예: `graph_bs = [1, 2, 4, 8, 16, 32, ...]`

| 실제 bs | 선택된 단계 | 패딩 낭비 |
|--------|-----------|----------|
| 1 | 1 | 0% |
| 3 | 4 | 25% |
| 5 | 8 | 37.5% |
| 10 | 16 | 37.5% |
| 20 | 32 | 37.5% |

패딩 부분의 계산은 낭비이긴 하지만 **모든 커널을 다시 런치하는 오버헤드보다 훨씬 작습니다**.

### 3.3 `slot_mapping.fill_(-1)` 트릭

`slot_mapping`의 길이는 캡처 시점에 `max_bs`이지만, 실제 요청은 `bs`개의 유효 토큰만 존재할 수 있습니다. 초기화하지 않으면 이전 replay에서 `[bs:]` 위치에 남은 오래된 슬롯 값이 후속 커널에 잘못 읽힐 수 있습니다.

먼저 **`fill_(-1)`**로 전체를 무효 표시한 후, **`[:bs] = context.slot_mapping`**으로 이번 스텝의 유효 매핑을 씁니다. Attention 내부의 Triton `store_kvcache_kernel`은 `slot == -1`일 경우 바로 `return`하므로 이 약속과 일치합니다(17강 참조).

마찬가지로 `context_lens`도 `zero_()`로 초기화 후 앞쪽 `bs`개만 채워 이전 잔류값 오염을 방지합니다.

### 3.4 `graph.replay()` 실행 메커니즘

`graph.replay()` 호출 시:

1. **CPU 측**: 단 한 번의 `cudaGraphLaunch` 호출로 전체 DAG를 GPU에 제출합니다.
2. **GPU 측**: 캡처 시 녹화된 순서와 의존 관계에 따라 모든 커널을 순차적으로 실행합니다.
3. **데이터 전달**: 그래프 내부 노드들은 캡처 시 바인딩된 메모리 주소를 사용하므로, 사전에 `graph_vars`에 데이터를 복사해 두면 그래프 내부에서 자연스럽게 최신 입력을 읽습니다.
4. **출력 획득**: `outputs[:bs]`는 replay 완료 후 자동으로 최신 모델 출력을 포함합니다. 이 버퍼 역시 캡처 시 바인딩된 출력 버퍼이기 때문입니다.

### 3.5 `compute_logits`는 Graph 외부에

`compute_logits`는 `graph.replay()` **이후**에 호출되며, 그래프에 녹화되지 않았음에 주목하세요.

```python
return self.model.compute_logits(graph_vars["outputs"][:bs])
```

그 이유는 `compute_logits`가 보통 **lm_head 가중치 행렬 곱셈**을 포함하고, 출력 logits의 모양이 어휘 크기에 의존하며 샘플링 등 후속 작업과 연계될 수 있기 때문입니다. Graph 바깥에 두는 것이 더 유연합니다.

### 3.6 전역 `Context`(`context.py`)와 Graph의 데이터 계약

Decode 경로에서 `run_model`은 `graph.replay()` 전에 **`context.slot_mapping`, `context.context_lens`, `context.block_tables`**를 `graph_vars`의 해당 버퍼로 복사합니다. Attention 및 KV 기록 커널은 Python의 `context` 객체를 직접 읽지 않고, **`set_context`가 캡처 시 동일한 메모리에 바인딩한 의미 체계**에 의존합니다. 녹화 시 `set_context(False, slot_mapping=slot_mapping[:bs], ...)`는 사전 할당된 텐서를 사용했으며, replay 시 **같은 주소**에 실제 스케줄 데이터를 기록하면 그래프 안 커널이 현재 스텝의 KV 레이아웃을 읽게 됩니다.

`utils/context.py`의 핵심은 다음과 같습니다(18강과 일치, 여기서는 **Graph의 버퍼 재사용**을 강조).

```python
@dataclass
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None

_CONTEXT = Context()
def get_context(): return _CONTEXT
def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0,
              slot_mapping=None, context_lens=None, block_tables=None):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                       slot_mapping, context_lens, block_tables)
def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
```

**Prefill은 Graph를 타지 않음**: `set_context(True, cu_seqlens_q=..., ...)` 하에서의 가변 길이 FlashAttention은 **고정 shape의 decode 그래프**와 동일한 캡처 경로를 공유하지 않으므로, `run_model`의 첫 분기에서 직접 `self.model(...)` eager로 실행됩니다.

---

## 4. `enforce_eager` 파라미터 상세

### 4.1 설정 정의

`Config`에서 기본값은 `False`(Graph 활성화)입니다.

```python
enforce_eager: bool = False
```

### 4.2 `True`로 설정 시 영향

- **`capture_cudagraph()` 생략**: 시작 시 어떤 그래프도 녹화하지 않아 초기화 시간 절약.
- **`run_model`이 항상 eager 경로**: 매 스텝 표준 PyTorch 순전파.
- **종료 시 graph 관련 리소스 정리하지 않음**.

### 4.3 적용 시나리오

| 시나리오 | 설명 |
|--------|------|
| 수치 오류 디버깅 | 연산별 중간 결과 확인, Graph 내부에는 브레이크포인트 삽입 불가 |
| Graph 미지원 환경 | 일부 CUDA 버전 또는 디바이스에서 Graph 미지원 |
| 프로파일링 | 커널별 소요 시간 분석 필요 시, Graph는 모든 커널을 하나의 호출로 병합 |
| 신기능 개발 | 빠른 반복 시 빈번한 재캡처 방지 |

---

## 5. 설계 결정 심층 분석

### 5.1 Decode에만 CUDA Graph를 사용하는 이유

- **Prefill 단계**: 시퀀스 길이 편차가 크고 `cu_seqlens`, 총 토큰 수 등이 가변적이며 프리픽스 캐싱 등 특수 경로를 탈 수 있어 고정 shape 그래프로 모든 경우를 커버하기 어렵습니다.
- **Decode 단계**: 스텝마다 시퀀스당 1토큰만 계산하며, 배치 차원과 레이어 구조가 상대적으로 안정적이어서 bs별 단계 캡처가 적합합니다.

### 5.2 업계 비교

| 프레임워크 | CUDA Graph 사용 전략 |
|-----------|---------------------|
| nano-vllm | decode 전용, 배치 단계별, 최대 512 |
| vLLM | decode + 일부 chunked prefill, 더 세밀한 단계 관리 |
| TensorRT-LLM | 컴파일 시 정적 그래프, decode 기본 활성화 |
| DeepSpeed-FastGen | decode 전용, nano-vllm과 유사 |

### 5.3 CUDA Graph의 메모리 오버헤드

CUDA Graph 캡처 중 할당된 모든 GPU 메모리는 **고정(pin)**되어 replay 후에도 해제되지 않습니다. 따라서:

- 단계가 많을수록 → 메모리 오버헤드 증가 (`graph_pool` 공유 최적화에도 불구하고).
- `graph_vars`는 `max_bs` 기준 할당 → 런타임 bs가 작아도 이 메모리는 계속 점유.
- 공학적으로 **그래프 개수**, **캡처 소요 시간**, **GPU 메모리 점유** 사이의 균형이 필요합니다.

---

## 6. 실제 성능 영향

### 6.1 대표적 가속 효과

A100에서 Qwen2.5-7B의 decode 단계 기준 예시 (대표값):

| 지표 | Graph 없음(eager) | Graph 있음 | 가속비 |
|------|-------------------|-----------|--------|
| 단일 스텝 decode 지연 (bs=1) | ~15 ms | ~8 ms | ~1.9x |
| 단일 스텝 decode 지연 (bs=32) | ~18 ms | ~12 ms | ~1.5x |
| decode 처리량 | ~1200 tok/s | ~2000 tok/s | ~1.7x |

규칙: **배치가 작을수록 Graph 가속비가 커집니다**. 작은 배치일수록 launch 오버헤드 비중이 높기 때문입니다.

### 6.2 초기화 비용

Graph 캡처는 시작 시 별도의 웜업 + 캡처를 필요로 하며, 일반적인 소요 시간은 다음과 같습니다.

- 7B 모델, 20개 단계: 약 30–60초
- 일회성 비용이며, 시작 후 매 decode 스텝에서 이득을 봅니다.

---

## 7. 요약

- **CUDA Graph**는 반복되는 안정적인 GPU 연산을 그래프로 녹화해 `replay()`로 **CPU launch 오버헤드**를 낮춥니다. 특히 **shape가 안정적인 decode**에 적합합니다.
- nano-vllm은 **`graph_bs`** 단계로 나누어 녹화(작은 배치 촘촘, 큰 배치 16 간격)하고, **`graph_pool`**로 메모리 공유, **`graph_vars`**로 replay 전 실제 입력과 컨텍스트를 채워 넣습니다.
- **Prefill, enforce_eager, bs>512**는 eager 경로를 사용하여 정확성과 디버깅 가능성을 보장합니다.
- **`slot_mapping`을 `-1`로 먼저 채우는 것**은 KV 기록 커널과 맞물린 공학적 디테일입니다.
- 큰 것부터 작은 것 순으로 캡처하여 메모리 풀이 처음에 충분한 공간을 확보하도록 합니다.
- Graph의 핵심 이득은 GPU 연산 효율 자체가 아니라 **CPU-GPU 상호작용 감소**에서 옵니다.

---

## 8. 면접 예상 문제 (모범 답안 포함)

**1. CUDA Graph가 해결하려는 주요 성능 문제는 무엇인가요?**  
**답변**: 주로 **CPU 측 커널 런치 및 드라이버 스케줄링 오버헤드**입니다. 특히 decode처럼 작은 스텝이 고빈도로 실행될 때 두드러집니다. Graph는 여러 스텝을 **한 번의 그래프 제출**로 합쳐 CPU 병목을 줄이고 GPU 활용률을 높입니다. 핵심 원리는 기존처럼 매번 "CPU 파라미터 검사 → 드라이버 대기 → GPU 실행"을 반복하는 대신 "단 한 번의 cudaGraphLaunch → GPU가 DAG에 따라 전체 커널 실행"으로 단순화하는 것입니다.

**2. CUDA Graph가 decode와 주로 함께 사용되고 prefill과는 결합되지 않는 이유는?**  
**답변**: Prefill은 **가변 길이**로 attention의 `cu_seqlens`, 총 토큰 수 등 변화가 커서 계산 그래프 shape를 고정하기 어렵습니다. Decode는 **스텝마다 시퀀스당 1토큰**으로 배치 단위만 고려하면 그래프를 고정할 수 있습니다. 또한 decode 단계는 계산량은 적지만 커널 수가 많아 launch 오버헤드 비중이 높아 Graph의 이점이 가장 큽니다.

**3. nano-vllm은 서로 다른 배치 크기에 대해 어떤 그래프를 선택하나요?**  
**답변**: `graph_bs`에 `[1,2,4,8,16,32,...]` 등 여러 단계를 미리 정의하고, 런타임에 `bs = input_ids.size(0)`을 구해 `next(x for x in self.graph_bs if x >= bs)`로 **bs보다 작지 않은 최소 단계**를 선택합니다. 해당 길이의 `graph_vars` 앞부분에 실제 데이터를 복사 후 `replay()` 합니다. 나머지 부분은 `fill_(-1)` / `zero_()` 등으로 무효화합니다.

**4. `graph_pool`의 역할은?**  
**답변**: 여러 CUDAGraph가 **PyTorch가 반환한 메모리 풀을 공유**하게 하여 단편화를 줄이고 할당을 재사용합니다. 구체적으로 첫 번째 그래프 캡처 후 `graph.pool()`로 풀 핸들을 얻고, 이후 그래프들의 `torch.cuda.graph(graph, pool)`에 같은 핸들을 전달하여 모든 그래프가 동일한 GPU 메모리 영역을 사용하도록 합니다.

**5. 캡처 전에 웜업을 해야 하는 이유는?**  
**답변**: 첫 실행 시 **CUDA 컨텍스트 초기화, cuBLAS auto-tuning, torch.compile JIT 컴파일** 등 불안정한 동작이 발생합니다. 웜업을 먼저 하여 이런 동작을 완료한 후 캡처하면, 불안정한 동작이 그래프 안에 녹화되는 것을 막아 replay 동작을 예측 가능하게 만듭니다.

**6. `enforce_eager=True`로 설정하면 어떤 영향이 있나요?**  
**답변**: `capture_cudagraph()`를 실행하지 않고, `run_model`이 항상 즉시 순전파를 수행하여 디버깅은 편리하지만 **Graph의 launch 최적화 효과를 잃습니다**. 수치 오류 디버깅, 프로파일링, 신기능 개발 등에 적합합니다.

**7. `slot_mapping`을 먼저 `fill_(-1)`하는 이유는?**  
**답변**: 버퍼 길이는 `max_bs`인데 실제 사용은 앞쪽 `bs`개이므로, 과거 replay의 잔류 데이터를 제거하기 위해 `-1`로 채웁니다. 이는 **Triton 커널이 `slot == -1`을 건너뛰는** 약속과 일치하여 KV Cache 오염을 방지합니다. CUDA Graph 사용 시 "버퍼 재사용"으로 인한 공학적 도전 과제 중 하나입니다.

**8. 배치가 512를 초과하면 Graph를 사용하지 않는 이유는?**  
**답변**: 캡처 상한과 일관성을 유지하고 그래프 개수와 GPU 메모리 사용을 제어하기 위해서입니다. 초대형 배치에서는 연산이 이미 포화 상태여서 launch 오버헤드 비중이 작아 Graph의 상대적 이득이 줄어들며, 구현을 단순하고 유지보수 가능하게 유지합니다.

**9. 큰 배치부터 작은 배치 순(reversed)으로 캡처하는 이유는?**  
**답변**: 가장 큰 배치의 그래프를 먼저 녹화하면 메모리 풀이 첫 할당 시 충분히 큰 공간을 확보합니다. 이후 작은 배치의 그래프는 이 이미 확보된 공간 내에서 동작할 수 있어 메모리 확장과 단편화 문제를 피할 수 있습니다.

**10. CUDA Graph를 학습에 사용할 수 있나요?**  
**답변**: 이론적으로 가능하지만, 학습은 역전파, 그래디언트 누적 등 동적 연산이 포함되고 일반적으로 배치 크기가 고정되어 shape이 크므로 launch 오버헤드 비중이 낮아 Graph의 이득이 제한적이며 구현 복잡도가 높습니다. 추론(특히 decode)이 CUDA Graph의 주요 활용처입니다.

---

*참고 자료: PyTorch 문서의 `torch.cuda.graph`, NVIDIA CUDA Graphs 프로그래밍 가이드.*