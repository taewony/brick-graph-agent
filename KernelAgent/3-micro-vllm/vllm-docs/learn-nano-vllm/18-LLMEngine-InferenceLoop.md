# 제18강: LLMEngine 추론 루프

초보자를 위한 설명: 본 강의는 nano-vllm의 **핵심 엔진 `LLMEngine`**을 상세히 다루며, 초기화, 다중 프로세스 기동, 요청 추가, 추론 루프(step), 일괄 생성(generate), 그리고 가중치 로딩과 샘플러의 구현 원리를 포함합니다. 각 절마다 소스 코드 줄 단위 해설과 면접 핵심 포인트가 제공됩니다.

---

## 1. 개념 설명: LLMEngine의 역할

### 1.1 아키텍처에서 LLMEngine의 위치

nano-vllm의 전체 아키텍처는 세 계층으로 추상화할 수 있습니다.

```
사용자 인터페이스 계층:  LLMEngine.generate() / add_request()
         ↓
스케줄링 실행 계층:      Scheduler → ModelRunner → Model
         ↓
하위 자원 계층:           KV Cache (BlockManager) / CUDA Graph / Attention Kernel
```

`LLMEngine`은 **사용자와 추론 시스템 사이의 가교**로서 다음을 담당합니다.

1. **초기화**: 모델 가중치 로드, ModelRunner 생성, Scheduler 생성, Tokenizer 로드.
2. **요청 관리**: 사용자의 텍스트/토큰 입력을 받아 `Sequence` 객체로 포장.
3. **추론 루프**: `step()`을 반복 호출하여 Scheduler → ModelRunner → 후처리를 구동.
4. **결과 수집**: 완료된 시퀀스를 감지하여 생성 결과를 반환.

### 1.2 vLLM과의 비교

| 특성 | nano-vllm LLMEngine | vLLM LLMEngine |
|------|---------------------|----------------|
| 코드 분량 | ~200줄 | ~5000줄 |
| 비동기 지원 | 없음 (동기 루프) | 있음 (AsyncLLMEngine) |
| API 서비스 | 없음 | OpenAI API 서버 통합 |
| 스트리밍 출력 | 간단 구현 | 완전한 SSE 스트리밍 |
| 멀티모달 | 미지원 | 이미지/비디오 입력 지원 |
| LoRA | 미지원 | 동적 LoRA 지원 |

nano-vllm의 간결한 설계는 LLM 추론 엔진을 배우기에 최적의 입문 교재입니다.

---

## 2. 소스 코드 해설: `LLMEngine.__init__` 초기화 과정

아래는 저장소의 `nanovllm/engine/llm_engine.py`와 일치합니다(`Config` 필드 필터링, `atexit`으로 자식 프로세스 정리 포함). 파일 상단에는 `from dataclasses import fields`, `import atexit` 등 임포트가 있습니다.

```python
class LLMEngine:
    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)
```

### 2.1 Config 설정 객체

`Config`는 모든 추론 파라미터를 집약하며, 핵심 필드는 다음과 같습니다.

| 필드 | 기본값 | 의미 |
|------|--------|------|
| `model` | — | 모델 경로 또는 HuggingFace 모델 이름 |
| `max_num_seqs` | 256 | 최대 동시 시퀀스 수 |
| `max_model_len` | — | 모델이 지원하는 최대 시퀀스 길이 |
| `tensor_parallel_size` | 1 | 텐서 병렬 GPU 수 |
| `enforce_eager` | False | CUDA Graph 사용 안 함 여부 |
| `enable_prefix_caching` | False | 프리픽스 캐싱 사용 여부 |
| `block_size` | 16 | KV Cache 블록당 토큰 수 |
| `gpu_memory_utilization` | 0.9 | GPU 메모리 사용 비율 상한 |

### 2.2 다중 프로세스 기동 (Tensor Parallel)

```python
for i in range(1, config.tensor_parallel_size):
    event = ctx.Event()
    process = ctx.Process(target=ModelRunner, args=(config, i, event))
    process.start()
```

`tensor_parallel_size > 1`일 때(예: GPU 2장 이상 사용), 여러 워커 프로세스를 기동해야 합니다.

**실행 흐름 상세:**

1. `ctx = multiprocessing.get_context("spawn")`: `spawn` 방식을 사용하여 자식 프로세스 생성(`fork`보다 안전하며 CUDA 컨텍스트 상속 문제를 피함).
2. 루프는 `i=1`에서 시작 (메인 프로세스 자신이 rank 0).
3. 각 자식 프로세스마다 `Event` 객체를 생성하여 **메인-자식 프로세스 간 동기화**에 사용.
4. `ctx.Process(target=ModelRunner, ...)`로 자식 프로세스 생성, 진입 함수는 `ModelRunner.__init__`.
5. `process.start()`로 자식 프로세스 시작.

**동기화 메커니즘:**

- 메인 프로세스(rank=0)가 `model_runner.call("run", ...)`를 호출하면, `Event`를 통해 각 자식 프로세스에게 동일한 순전파 연산을 수행하도록 알림.
- 각 자식 프로세스가 실행을 마치면, 다시 `Event`를 통해 메인 프로세스에게 결과를 수집할 수 있음을 알림.
- 이 설계는 텐서 병렬에서 **모든 GPU가 동일한 순전파 단계를 동기적으로 실행**하도록 보장합니다.

**왜 멀티스레드가 아닌 멀티프로세스를 사용하는가?**

- Python의 **GIL(전역 해석기 잠금)**이 멀티스레드의 병렬도를 제한함.
- 각 GPU는 독립적인 **CUDA 컨텍스트**를 필요로 하며, 멀티프로세스는 이를 자연스럽게 격리.
- `torch.distributed`의 NCCL 백엔드는 멀티프로세스 아키텍처에서 가장 잘 동작함.

### 2.3 `atexit`과 자식 프로세스 정리

```python
atexit.register(self.exit)
```

프로세스 종료 시 `exit()`가 호출되며, `model_runner.call("exit")`를 통해 각 랭크에 자원 해제를 알리고 자식 프로세스를 `join`하여 좀비 프로세스나 GPU 핸들 누수를 방지합니다. 면접에서는 "**spawn 자식 프로세스는 반드시 join으로 정리**"라고 한 문장으로 답하면 됩니다.

### 2.4 메인 프로세스의 ModelRunner

```python
self.model_runner = ModelRunner(config, 0, self.events)
```

메인 프로세스(rank=0)도 자신만의 `ModelRunner`를 생성하여 다음을 담당합니다.

- 0번 GPU에 모델 가중치 로드.
- KV Cache 할당.
- CUDA Graph 캡처 (활성화된 경우).
- 순전파 계산의 **조정자**로서 연산 시작 및 결과 수집.

### 2.5 Tokenizer와 `eos`

```python
self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
config.eos = self.tokenizer.eos_token_id
```

HuggingFace `transformers`의 `AutoTokenizer` 사용:

- 모델 유형을 자동으로 인식하고 해당 토크나이저 로드.
- `encode`(텍스트 → 토큰 ID 리스트) 및 `decode`(토큰 ID → 텍스트) 지원.
- 주의: 토크나이저는 **메인 프로세스**에서만 로드되며, 자식 프로세스에는 필요 없음.

### 2.6 Scheduler 생성

```python
self.scheduler = Scheduler(config)
```

스케줄러는 요청의 생명주기를 관리합니다.

- **대기 큐(waiting)**: 새로 추가된 요청.
- **실행 큐(running)**: 현재 추론 중인 요청.
- **스케줄링 결정**: 각 스텝에서 어떤 시퀀스가 prefill에 참여하고 어떤 시퀀스가 decode에 참여할지 결정.
- **KV Cache 관리**: BlockManager를 통한 KV 블록 할당 및 해제.

(스케줄러의 자세한 구현은 이전 스케줄링 강의 참조)

### 2.7 전역 `Context` (`context.py`): Attention의 '암시적 매개변수'

`ModelRunner`가 prefill/decode 텐서를 준비할 때, `set_context(...)`를 호출하여 **cu_seqlens, slot_mapping, context_lens, block_tables** 등을 프로세스 수준 전역 변수에 기록합니다. 모델 순전파 내의 Attention은 `get_context()`를 통해 이를 읽습니다. 이렇게 하면 **모든 레이어 함수에 이 텐서들을 인자로 계속 전달하지 않아도 되어** 코드가 짧아지지만, 대신 **단일 프로세스 내에서 동시에 하나의 스케줄링 경로만 실행할 수 있다**는 제약이 따릅니다(교육용 프로젝트로서는 허용 가능).

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

CUDA Graph와의 관계(16강): `capture_cudagraph`에서 `set_context(False, slot_mapping=..., ...)`와 `run_model`이 replay 전에 `graph_vars`에 기록하는 내용은 **동일한 의미 구조**를 갖습니다. 즉, 그래프 안에 녹화된 커널이 replay 시에도 올바른 KV 메타데이터를 읽을 수 있도록 보장합니다.

---

## 3. 소스 코드 해설: `add_request` 메서드

```python
def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
    if isinstance(prompt, str):
        prompt = self.tokenizer.encode(prompt)
    seq = Sequence(prompt, sampling_params)
    self.scheduler.add(seq)
```

### 3.1 입력 형식 유연성

`add_request`는 두 가지 입력을 동시에 지원합니다.

| 입력 유형 | 예시 | 처리 방식 |
|----------|------|-----------|
| `str` | `"Hello, world"` | `tokenizer.encode()`로 토큰 ID 리스트 변환 |
| `List[int]` | `[1, 2, 3, 4]` | 직접 사용 |

`List[int]` 지원의 장점: 벤치마크 테스트 시 랜덤 토큰 ID를 직접 사용하여 토크나이즈 오버헤드를 건너뛸 수 있음.

### 3.2 Sequence 객체

`Sequence`는 nano-vllm에서 추론 요청 하나를 나타내는 핵심 데이터 구조입니다.

```python
class Sequence:
    def __init__(self, prompt_token_ids, sampling_params):
        self.seq_id = next_id()              # 전역 고유 ID
        self.prompt_token_ids = prompt_token_ids  # 원본 프롬프트 토큰 ID
        self.completion_token_ids = []        # 생성된 토큰 ID (점차 추가됨)
        self.sampling_params = sampling_params # 샘플링 파라미터
        self.logical_blocks = []             # 할당된 KV 블록 번호들
        self.is_finished = False             # 완료 여부
```

각 Sequence는 생애 주기 동안 **대기 → prefill → decode (반복) → 완료**를 거칩니다.

### 3.3 SamplingParams 샘플링 파라미터

```python
@dataclass
class SamplingParams:
    temperature: float = 1.0     # 온도, 무작위성 조절
    max_tokens: int = 256        # 최대 생성 토큰 수
    ignore_eos: bool = False     # EOS 토큰 무시 여부
```

- `temperature > 0`: 확률 샘플링을 사용하며, 값이 클수록 무작위적.
- `temperature = 0`: greedy decoding으로 전환 (가장 높은 확률의 토큰 선택).
- `ignore_eos = True`: EOS 토큰이 생성되어도 계속 진행, 벤치마크 등에 사용.

---

## 4. 소스 코드 해설: `step` 메서드 — 추론 루프의 핵심

소스 코드와 일치하는 완전한 구현은 다음과 같습니다 (참고: `num_tokens`의 부호 규약은 `generate`에서 prefill/decode 처리량을 집계하기 위함).

```python
def step(self):
    seqs, is_prefill = self.scheduler.schedule()
    token_ids = self.model_runner.call("run", seqs, is_prefill)
    self.scheduler.postprocess(seqs, token_ids)
    outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
    num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
    return outputs, num_tokens
```

### 4.1 첫 번째 단계: 스케줄링 `scheduler.schedule()`

스케줄러는 현재 상태에 따라 이번 스텝의 실행 전략을 결정합니다.

**반환값**:
- `seqs`: 이번 스텝 계산에 참여하는 시퀀스 리스트.
- `is_prefill`: 불 값, 이번 스텝이 prefill인지 decode인지 표시.

**스케줄링 전략**:
1. **prefill 우선**: 대기 큐에 새 요청이 있고 자원이 허락되면 먼저 prefill 수행.
2. **decode 후행**: 모든 활성 시퀀스가 prefill을 마쳤다면 decode 수행.
3. **자원 제약**: `max_num_seqs`, 사용 가능한 KV 블록 수의 제한을 받음.

**Prefill vs Decode의 차이**:

| 차원 | Prefill | Decode |
|------|---------|--------|
| 시퀀스당 토큰 수 | 프롬프트 길이 (수천까지 가능) | 1 |
| KV Cache 연산 | 일괄 기록 | 1개 슬롯 추가 |
| Attention 유형 | Flash Attention (가변 길이) | Flash Attention with KV Cache |
| CUDA Graph | 사용 안 함 | 사용 (가능한 경우) |

### 4.2 두 번째 단계: 실행 `model_runner.call("run", seqs, is_prefill)`

`call` 메서드는 다중 프로세스 통신을 캡슐화합니다.

1. 메인 프로세스가 `seqs`와 `is_prefill`을 직렬화하여 공유 메모리를 통해 각 자식 프로세스로 전달.
2. `Event`를 통해 모든 자식 프로세스에 실행 시작을 알림.
3. 각 프로세스(메인 포함)가 `ModelRunner.run(seqs, is_prefill)` 호출.
4. `run` 메서드 내부:
   - `seqs`에서 `input_ids`, `positions`, `slot_mapping` 등 추출.
   - `run_model(input_ids, positions, is_prefill)`로 순전파 수행.
   - logits에 대해 샘플링하여 다음 토큰 ID 획득.
5. 메인 프로세스가 모든 자식 프로세스 완료를 기다렸다가 결과 수집.

### 4.3 세 번째 단계: 후처리 `scheduler.postprocess(seqs, token_ids)`

후처리의 핵심 로직:

```python
def postprocess(self, seqs, token_ids):
    for seq, token_id in zip(seqs, token_ids):
        # 새 토큰을 시퀀스에 추가
        seq.completion_token_ids.append(token_id)

        # 정지 조건 확인
        if self._should_stop(seq, token_id):
            seq.is_finished = True
            self._free_blocks(seq)  # KV Cache 블록 해제
```

**정지 조건**:
- EOS 토큰을 생성했으며 `ignore_eos=False`일 때.
- `max_tokens` 상한 도달.
- 모델의 `max_model_len` 도달.

**자원 해제**: 시퀀스가 완료되면 `BlockManager.free()`를 통해 점유하던 KV 블록을 해제하여 다른 요청이 사용할 수 있게 합니다.

### 4.4 네 번째 단계: 출력 수집

```python
outputs = [(seq.seq_id, seq.completion_token_ids)
           for seq in seqs if seq.is_finished]
```

**이번 스텝에 새로 완료된** 시퀀스만 반환합니다. 이전 스텝에서 이미 완료된 시퀀스는 다시 나타나지 않습니다.

### 4.5 `num_tokens`: 처리량 통계의 '부호화'

```python
num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
```

| `is_prefill` | 의미 | `num_tokens` |
|--------------|------|----------------|
| `True` | 이번 스텝에서 프롬프트 처리 (여러 토큰 가능) | **양수** = 이번 배치의 모든 시퀀스 현재 길이 합 (prefill 처리 토큰 규모와 연관) |
| `False` | 이번 스텝 decode (시퀀스당 1 토큰) | **음수** = `-batch_size`, 절댓값이 이번 스텝에 새로 생성된 토큰 수 |

`generate`에서 `perf_counter()`와 함께 사용: **prefill 스텝**에서는 `num_tokens / Δt`로 **Prefill tok/s**를 얻고, **decode 스텝**에서는 `-num_tokens / Δt` (음수×음수=양수)로 **Decode tok/s**를 얻습니다. 이는 **하나의 스칼라 값**으로 단계를 구분하면서 통계 정보를 함께 전달하는 간결한 표기법입니다.

---

## 5. 소스 코드 해설: `generate` 메서드 — 일괄 추론 진입점

```python
def generate(self, prompts, sampling_params, use_tqdm=True):
    if use_tqdm:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
    if not isinstance(sampling_params, list):
        sampling_params = [sampling_params] * len(prompts)
    for prompt, sp in zip(prompts, sampling_params):
        self.add_request(prompt, sp)
    outputs = {}
    prefill_throughput = decode_throughput = 0.
    while not self.is_finished():
        t = perf_counter()
        output, num_tokens = self.step()
        if use_tqdm:
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
        for seq_id, token_ids in output:
            outputs[seq_id] = token_ids
            if use_tqdm:
                pbar.update(1)
    outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
    outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
    if use_tqdm:
        pbar.close()
    return outputs
```

### 5.1 완전한 생애 주기

```
generate() 호출
    ├── add_request(prompt_1, sp_1)  →  Sequence_1이 대기 큐로 진입
    ├── add_request(prompt_2, sp_2)  →  Sequence_2가 대기 큐로 진입
    └── ...
    │
    ├── step() #1:  schedule → prefill(Seq_1) → postprocess
    ├── step() #2:  schedule → prefill(Seq_2) → postprocess
    ├── step() #3:  schedule → decode([Seq_1, Seq_2]) → postprocess
    ├── step() #4:  schedule → decode([Seq_1, Seq_2]) → postprocess
    │   ...         (Seq_1 완료, KV 블록 해제)
    ├── step() #N:  schedule → decode([Seq_2]) → postprocess
    │   ...         (Seq_2 완료)
    └── is_finished() == True → 모든 결과 반환
```

### 5.2 `is_finished` 판단

```python
def is_finished(self):
    return self.scheduler.is_finished()
```

스케줄러가 **처리 대기 중인 요청이 없다**고 판단하면 `True`를 반환하고 루프가 종료됩니다. (구체적인 조건은 `scheduler.py`의 `is_finished` 구현 참조)

### 5.3 처리량 통계 (소스 코드와 동일)

`use_tqdm=True`이면, **매 스텝** `step()` 후에 `num_tokens`의 부호에 따라 진행 막대에 **Prefill / Decode** 문자열을 업데이트합니다(이전 절의 공식 참조). 이는 슬라이딩 평균이 아니라 **현재 스텝의 순간 처리량**으로, prefill 중심과 decode 중심 사이에서 스케줄링이 전환될 때의 변동을 관찰하기에 좋습니다.

| 지표 | 소스 코드에서의 구현 핵심 |
|------|------------------------|
| Prefill | `num_tokens > 0`일 때, `prefill_throughput = num_tokens / (perf_counter() - t)` |
| Decode | 그 외에는 `decode_throughput = -num_tokens / (perf_counter() - t)` (`num_tokens`가 음수이므로 -를 곱해 양수로) |

**Event 동기화**: `tensor_parallel_size > 1`일 때, `ModelRunner.call` 내부에서 `multiprocessing.Event`를 사용하여 rank0와 각 워커가 **같은 스텝의 `run`을 동시에 실행**하도록 보장합니다. 이는 분산 텐서 병렬 시 각 카드의 순전파가 정렬되도록 합니다. 면접 답변: **Event는 크로스 프로세스 배리어(barrier)로, CUDA 디바이스 동기화는 아님**.

---

## 6. 소스 코드 해설: 가중치 로딩 (loader.py)

### 6.1 커스텀 가중치 로딩이 필요한 이유

HuggingFace 모델의 가중치 명명 규칙과 nano-vllm 내부 파라미터 명칭은 다를 수 있습니다. 특히 **packed module** (예: QKV가 하나의 Linear 레이어로 병합된 경우)은 특별한 처리가 필요합니다.

### 6.2 `load_model` 완전 분석

```python
def load_model(model, path):
    # 모델이 정의한 packed 매핑 관계 가져오기
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})

    # 모든 safetensors 가중치 파일 순회
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                # packed module에 속하는지 확인
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    # 일반 가중치: 직접 로드
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
```

### 6.3 핵심 개념 해설

#### Safetensors 포맷

`safetensors`는 HuggingFace에서 내놓은 안전한 가중치 파일 포맷입니다.

| 특성 | safetensors | pickle (torch.save) |
|------|------------|---------------------|
| 안전성 | 임의 코드 실행 불가 | 악성 코드 주입 가능 |
| 로딩 속도 | mmap 지원하여 매우 빠름 | 역직렬화 필요 |
| 임의 접근 | 키로 단일 텐서 읽기 지원 | 전체 파일 로드 필요 |

#### Packed Modules 매핑

Qwen2 모델을 예로 들면, HuggingFace 가중치에서 Q, K, V는 세 개의 분리된 행렬입니다.

```
model.layers.0.self_attn.q_proj.weight  → shape [hidden_size, hidden_size]
model.layers.0.self_attn.k_proj.weight  → shape [kv_hidden_size, hidden_size]
model.layers.0.self_attn.v_proj.weight  → shape [kv_hidden_size, hidden_size]
```

하지만 nano-vllm은 효율을 위해 이들을 하나의 `qkv_proj`로 병합합니다.

```
model.layers.0.self_attn.qkv_proj.weight  → shape [hidden_size + 2*kv_hidden_size, hidden_size]
```

`packed_modules_mapping`은 이러한 병합 관계를 정의합니다.

```python
packed_modules_mapping = {
    "q_proj": ("qkv_proj", 0),   # Q는 shard 0에
    "k_proj": ("qkv_proj", 1),   # K는 shard 1에
    "v_proj": ("qkv_proj", 2),   # V는 shard 2에
    "gate_proj": ("gate_up_proj", 0),
    "up_proj": ("gate_up_proj", 1),
}
```

#### `weight_loader` 콜백

각 파라미터는 가중치의 슬라이싱과 배치를 처리하기 위해 커스텀 `weight_loader` 함수를 가질 수 있습니다.

```python
def weight_loader(param, loaded_weight, shard_id):
    # shard_id에 따라 병합 파라미터 내에서 위치 결정
    # loaded_weight를 param의 해당 슬라이스에 복사
    shard_size = loaded_weight.shape[0]
    param.data[shard_id * shard_size : (shard_id + 1) * shard_size] = loaded_weight
```

#### `default_weight_loader`

packed가 아닌 일반 파라미터를 위해 전체를 그대로 복사합니다.

```python
def default_weight_loader(param, loaded_weight):
    param.data.copy_(loaded_weight)
```

### 6.4 텐서 병렬 시의 가중치 로딩

`tensor_parallel_size > 1`일 때는 가중치 로딩 시 **분할**도 고려해야 합니다.

- **열 병렬(Column Parallel)**: QKV 프로젝션, FFN의 gate/up 프로젝션은 **출력 차원** 기준 분할.
- **행 병렬(Row Parallel)**: Output 프로젝션, FFN의 down 프로젝션은 **입력 차원** 기준 분할.
- 각 랭크는 자신에게 속한 가중치 조각만 로드합니다.

---

## 7. 소스 코드 해설: Sampler 샘플러

### 7.1 추론에서 샘플링의 역할

모델 순전파의 출력은 **logits**(정규화되지 않은 로그 확률)이며, **샘플링**을 통해 다음 토큰을 얻어야 합니다.

### 7.2 Sampler 소스 코드 분석

```python
class Sampler(nn.Module):
    @torch.compile
    def forward(self, logits, temperatures):
        # ① 온도 스케일링
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))

        # ② 확률 분포 계산
        probs = torch.softmax(logits, dim=-1)

        # ③ Gumbel-max 샘플링
        sample_tokens = probs.div_(
            torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        ).argmax(dim=-1)

        return sample_tokens
```

### 7.3 줄 단위 해설

#### 온도 스케일링

```python
logits = logits.float().div_(temperatures.unsqueeze(dim=1))
```

- `.float()`: FP32로 변환하여 FP16에서의 softmax 오버플로 방지.
- `temperatures.unsqueeze(dim=1)`: `[batch]`를 `[batch, 1]`로 확장하여 브로드캐스팅 지원.
- `.div_()`: in-place 나눗셈으로 메모리 절약.
- 온도 \(T\)의 효과: logits를 \(T\)로 나눈 후 softmax. \(T \to 0\)이면 greedy에 가까워지고, \(T \to \infty\)이면 균등 분포에 가까워짐.

#### Gumbel-max 트릭

```python
sample_tokens = probs.div_(
    torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
).argmax(dim=-1)
```

이것은 **범주 분포 샘플링의 효율적인 방법**으로, `torch.multinomial`과 동등하지만 GPU 병렬 처리에 더 적합합니다.

**수학적 원리**:

Gumbel-max 정리에 따르면, \(G_i \sim \text{Gumbel}(0, 1)\)이 독립 동일 분포의 Gumbel 확률 변수일 때:

\[
\arg\max_i (\log p_i + G_i) \sim \text{Categorical}(p_1, p_2, \ldots, p_V)
\]

여기서 \(p_i\)는 각 카테고리의 확률입니다.

**코드에서의 동등 변환**:

1. `exponential_(1)`은 지수 분포 확률 변수 \(E_i \sim \text{Exp}(1)\)를 생성.
2. Gumbel 확률 변수는 \(G_i = -\log(E_i)\)로 생성 가능.
3. \(\arg\max(\log p_i - \log E_i) = \arg\max(p_i / E_i)\).
4. 따라서 `probs / exponential` 후 `argmax`를 취하면 `probs`로부터 확률적으로 샘플링하는 것과 동등.
5. `.clamp_min_(1e-10)`은 0으로 나누기를 방지.

**왜 `torch.multinomial`을 직접 사용하지 않는가?**

- `torch.multinomial`은 내부적으로 CDF(누적 분포 함수)를 계산해야 하며, 큰 어휘(150K+ 등)에 대해 \(O(V)\)의 직렬 연산임.
- Gumbel-max는 **요소별** 나눗셈과 argmax만 필요하므로 GPU의 SIMD 병렬 처리에 자연스럽게 적합함.
- `@torch.compile`을 통해 이 연산들을 더욱 하나의 커널로 융합할 수 있음.

### 7.4 `@torch.compile` 가속

```python
@torch.compile
def forward(self, logits, temperatures):
    ...
```

`torch.compile`의 역할:

1. **연산자 융합(Operator Fusion)**: `.float()` → `.div_()` → `softmax` → `.div_()` → `.argmax()`를 더 적은 커널 호출로 융합.
2. **메모리 최적화**: 중간 텐서의 할당과 해제를 줄임.
3. **백엔드 최적화**: PyTorch의 Inductor 백엔드가 최적화된 Triton 커널을 생성.

일반적인 가속 효과: 샘플링 단계가 2-3배 빨라질 수 있음.

---

## 8. 완전한 호출 체인 정리

사용자가 `llm.generate(["Hello"], [SamplingParams()])`를 호출했을 때:

```
generate()
  ├── add_request("Hello", SamplingParams())
  │     ├── tokenizer.encode("Hello") → [15496]
  │     ├── Sequence([15496], SamplingParams())
  │     └── scheduler.add(seq)
  │
  └── while not is_finished():
        └── step()
              ├── scheduler.schedule()
              │     └── 반환 (seqs, is_prefill=True)  ← 첫 스텝은 prefill
              │
              ├── model_runner.call("run", seqs, True)
              │     ├── input_ids=[15496], positions=[0] 준비
              │     ├── slot_mapping 할당
              │     ├── run_model(input_ids, positions, is_prefill=True)
              │     │     └── model(input_ids, positions) → hidden_states
              │     │         └── compute_logits(hidden_states) → logits
              │     └── sampler(logits, temperatures) → token_id
              │
              ├── scheduler.postprocess(seqs, [token_id])
              │     ├── seq.completion_token_ids.append(token_id)
              │     └── 정지 조건 확인
              │
              └── 이후 step: is_prefill=False, decode 경로로 진행
                    └── CUDA Graph replay (가능한 경우)
```

---

## 9. 요약

- **LLMEngine**은 nano-vllm의 핵심 엔진으로 Tokenizer, Scheduler, ModelRunner를 연결합니다.
- **초기화**는 "설정 → 다중 프로세스 → ModelRunner → Tokenizer → Scheduler" 순서로 이루어집니다.
- **다중 프로세스**는 `spawn` + `Event`를 사용하여 텐서 병렬의 프로세스 동기화를 구현합니다.
- **`add_request`**는 문자열과 토큰 ID 리스트 두 가지 입력을 지원하며, Sequence로 포장하여 스케줄러에 추가합니다.
- **`step`**은 추론 루프의 최소 단위입니다: 스케줄링 → 실행 → 후처리 → 출력 수집.
- **`generate`**는 일괄 추론의 진입점으로, 모든 요청을 추가한 후 step을 반복 호출하여 전부 완료될 때까지 진행합니다.
- **가중치 로딩**은 packed module의 매핑과 슬라이싱을 처리하며 safetensors 포맷을 지원합니다.
- **Sampler**는 Gumbel-max 트릭으로 `torch.multinomial`을 대체하고 `@torch.compile`과 결합하여 효율적으로 샘플링합니다.

---

## 10. 면접 핵심 포인트 (모범 답안 포함)

**1. LLMEngine의 `step()` 메서드는 무엇을 하나요?**  
**답변**: 매 `step`은 한 번의 추론 루프를 완료합니다. ① `scheduler.schedule()`이 계산에 참여할 시퀀스와 prefill/decode 여부를 결정; ② `model_runner.call("run", ...)`이 순전파를 실행하여 토큰 ID 획득; ③ `scheduler.postprocess()`가 새 토큰을 시퀀스에 추가하고 정지 조건 확인; ④ 완료된 시퀀스 출력을 수집합니다.

**2. 다중 프로세스에 `fork` 대신 `spawn`을 사용하는 이유는?**  
**답변**: `fork`는 부모 프로세스의 CUDA 컨텍스트를 복사하여 자식 프로세스가 자신의 GPU를 올바르게 초기화할 수 없게 합니다. `spawn`은 완전히 새로운 프로세스를 만들어 각 프로세스가 독립적으로 CUDA 컨텍스트를 초기화하므로, NCCL 등 다중 GPU 통신 라이브러리의 요구 사항과 일치합니다.

**3. `generate`와 `step`의 관계는?**  
**답변**: `generate`는 사용자 대상 고수준 인터페이스로, 내부에서 모든 요청이 완료될 때까지 `step`을 반복 호출합니다. `step`은 추론 루프의 최소 실행 단위로 매번 하나의 스케줄 배치를 처리합니다. 비유하자면 `generate`가 "전체 시험"이고 `step`은 "한 문제를 푸는 것"입니다.

**4. Packed modules mapping은 어떤 문제를 해결하나요?**  
**답변**: HuggingFace 모델은 Q, K, V를 세 개의 독립된 가중치 행렬로 저장하지만, nano-vllm은 계산 효율을 위해 이들을 하나의 `qkv_proj`로 병합합니다. `packed_modules_mapping`이 매핑 관계를 정의하여, 가중치 로딩 시 분리된 행렬을 병합 파라미터의 올바른 슬라이스에 조립합니다. FFN의 `gate_proj` + `up_proj` → `gate_up_proj`도 유사합니다.

**5. Gumbel-max 샘플링이 `torch.multinomial`보다 나은 점은?**  
**답변**: `torch.multinomial`은 CDF 계산을 위해 순차 스캔이 필요하여 복잡도가 \(O(V)\)입니다. Gumbel-max는 요소별 나눗셈 + argmax로, GPU의 SIMD 병렬 처리에 자연스럽게 적합합니다. `@torch.compile`과 결합하면 하나의 커널로 융합되어 큰 어휘(150K+) 환경에서 훨씬 빠릅니다.

**6. Sampler에서 `temperatures.unsqueeze(dim=1)`의 역할은?**  
**답변**: `logits` 형태는 `[batch, vocab_size]`, `temperatures` 형태는 `[batch]`입니다. `unsqueeze(dim=1)`로 `[batch, 1]`로 만들어 **브로드캐스팅**을 통해 각 행(row)마다 자신에게 맞는 온도 값으로 나눗셈을 수행합니다. 서로 다른 시퀀스가 서로 다른 온도 파라미터를 사용할 수 있게 됩니다.

**7. 토크나이저는 왜 메인 프로세스에서만 로드하나요?**  
**답변**: 토크나이저의 역할은 텍스트와 토큰 ID 간 변환으로, CPU 측의 전처리 및 후처리입니다. 텐서 병렬의 자식 프로세스는 GPU에서의 모델 순전파 계산만 담당하며 텍스트 처리에 관여하지 않으므로, 메모리 낭비를 막기 위해 로드하지 않습니다.

**8. Sampler에서 `@torch.compile`의 역할은?**  
**답변**: 여러 PyTorch 연산자(타입 변환, 나눗셈, softmax, 지수 분포 샘플링, argmax)를 더 적은 GPU 커널로 융합하여 중간 텐서 할당과 커널 런치 오버헤드를 줄입니다. PyTorch의 Inductor 백엔드가 최적화된 Triton 코드를 생성하여 일반적으로 2~3배 가속됩니다.

**9. 시퀀스의 정지 조건에는 어떤 것들이 있나요?**  
**답변**: 세 가지 조건 중 하나라도 만족하면 정지합니다. ① EOS 토큰을 생성했으며 `ignore_eos=False`인 경우; ② 생성된 토큰 수가 `SamplingParams.max_tokens`에 도달; ③ 전체 시퀀스 길이(프롬프트 + 생성)가 `Config.max_model_len`에 도달. 정지 후 Scheduler는 해당 KV Cache 블록을 해제합니다.

**10. safetensors 포맷이 PyTorch 네이티브 `.pt` 포맷보다 어떤 장점이 있나요?**  
**답변**: ① **안전성**: pickle을 사용하지 않아 악성 코드 주입 불가; ② **로딩 속도**: mmap(메모리 매핑) 지원, 전체 파일 역직렬화 없이 단일 텐서를 필요할 때만 로드 가능; ③ **임의 접근**: 키를 통해 특정 가중치만 읽을 수 있어, 텐서 병렬처럼 자신의 랭크에 해당하는 조각만 로드할 때 유리합니다.

---

*참고 자료: HuggingFace safetensors 문서, PyTorch torch.compile 튜토리얼.*