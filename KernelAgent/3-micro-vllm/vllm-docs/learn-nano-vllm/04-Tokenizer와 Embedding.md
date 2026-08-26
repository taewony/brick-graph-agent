# 강의 04: Tokenizer와 Embedding

> 먼저 문자열을 토큰 id로 변환하고(Tokenizer 워크플로), **어휘 병렬**을 살펴봅니다: 각 GPU 카드는 `V/tp_size`개의 행만 임베딩으로 저장하고, mask + all_reduce로 전체 벡터를 구성합니다; 마지막으로 **ParallelLMHead**가 Prefill에서 시퀀스당 마지막 위치만 취하는 이유—자기회귀의 '다음 토큰 예측'과 일치함을 이해합니다.

## 본 강의 목표

1. **Tokenizer 워크플로** 설명: 정규화 → 어휘 매핑 → 특수 토큰 → (선택) chat 템플릿.
2. **VocabParallelEmbedding**의 **분할 방식**, **mask 의미**, **all_reduce 필요성**을 이해합니다.
3. **ParallelLMHead.forward**에서 `cu_seqlens_q`와 **마지막 토큰만 취하는** 이유를 설명합니다.
4. 면접에서 **Embedding 순전파**와 **LMHead 순전파**의 TP 하에서의 차이(reduce vs gather)를 비교할 수 있습니다.

## 핵심 개념

### Tokenizer 워크플로 (HuggingFace 기준)

nano-vllm의 **`example.py`**는 엔진 외부에서 **`AutoTokenizer`**를 사용하며, 전형적인 단계는 다음과 같습니다:

1. **로드**: `from_pretrained(model_dir)`로 `tokenizer.json` / `vocab` 등을 읽습니다.
2. **대화 형식**: `apply_chat_template(messages, ...)`로 **역할 마크가 포함된** 문자열을 생성하여, 명령어 모델이 **user/assistant** 경계를 이해하기 쉽게 만듭니다.
3. **인코딩**: 엔진 측 또는 스크립트 측에서 `encode`하여 **토큰 id 시퀀스**(정수 텐서)를 얻고, 이를 **`embedding` 레이어 입력**으로 사용합니다.

**면접 빈출 포인트**: Tokenizer는 **`nanovllm` 핵심 패키지에 속하지 않지만**, **어휘 크기 V, embed 가중치 shape**과 강하게 연관됩니다; **어휘 병렬**은 바로 **V 차원**을 따라 분할됩니다.

### 어휘 병렬 (Vocab Parallelism)

**임베딩 행렬**이 너무 클 때(큰 어휘 × 은닉 차원), **어휘 차원**에서 분할할 수 있습니다:

- `r`번째 GPU 카드는 행 인덱스 구간 `[v_start, v_end)`에 해당하는 **`V/tp_size`개의 행**만 저장합니다.
- 순전파 시: 어떤 토큰 id가 **이 카드의 구간에 속하면**, **로컬 행**으로 테이블 조회; 그렇지 않으면 **로컬 기여도 0**.
- 멀티 카드 시: 각 카드는 **부분 벡터**를 얻고, **`all_reduce`(합산)**로 병합해야 합니다(본 카드가 아닌 행의 가중치는 0이므로, 합산이 곧 연결 후 투영하는 선형 중첩 개념과 동일).

**직관**: 각 GPU 카드는 **자기 몫의 어휘에 대한 embedding**을 계산하고, 다른 위치는 0으로 하여, **더하면** 전체 embedding이 됩니다.

### LMHead와 Embedding 가중치 공유

많은 모델에서 **출력 레이어와 입력 임베딩은 가중치를 공유**합니다(weight tying). `ParallelLMHead`는 `VocabParallelEmbedding`을 상속받아 **분할된 가중치**를 재사용합니다; 하지만 **순전파**는 다릅니다:

- **Prefill**: 시퀀스의 각 위치에 hidden이 있지만, **학습/추론 목표**는 보통 **마지막 위치에서 다음 토큰 예측**입니다; nano-vllm은 **`cu_seqlens_q`**를 사용하여 **각 시퀀스의 마지막 query 위치**를 가져옵니다.
- **Decode**: 보통 시퀀스당 **1개의 토큰**, shape과 컨텍스트 `is_prefill`은 `model_runner`가 설정합니다.

### `cu_seqlens_q`란 무엇인가 (본 강의 관련)

**누적 시퀀스 길이(Cumulative sequence lengths)**: 길이 `batch+1`, **펼쳐진 토큰 시퀀스**에서 각 하위 시퀀스의 시작·끝 인덱스를 기록합니다.  
`cu_seqlens_q[1:] - 1`은 곧 **각 시퀀스의 마지막 토큰이 펼쳐진 배열에서의 인덱스** — **마지막 hidden만을 취하는 데** 사용됩니다.

## 소스 코드 분석 (전체 소스 코드 및 줄별 주석 포함)

다음 코드는 저장소 `nanovllm/layers/embed_head.py`와 일치합니다(가중치 로딩 이해를 위해 `weight_loader` 포함).

```python
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y
            dist.all_reduce(y)
        return y


class ParallelLMHead(VocabParallelEmbedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        context = get_context()
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        logits = F.linear(x, self.weight)
        if self.tp_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0)
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
```

### VocabParallelEmbedding 구간별 주석

| 코드 조각 | 설명 |
|----------|------|
| `dist.get_rank()` / `get_world_size()` | 현재 **TP 그룹** 내 rank와 **병렬도 tp_size** |
| `num_embeddings % self.tp_size == 0` | 어휘 행 수는 반드시 **나누어 떨어져야** 함, 그렇지 않으면 균등 분할 불가 |
| `num_embeddings_per_partition` | 카드당 **로컬 어휘 행 수** `V/tp_size` |
| `vocab_start_idx` / `vocab_end_idx` | 이 카드가 담당하는 **전역 토큰 id 구간** |
| `self.weight` shape `(V/tp, D)` | 오직 **본 분할**의 임베딩 테이블만 저장 |
| `weight_loader` | **전체 HF 가중치**에서 행 기준으로 **`narrow`** 잘라 `copy_`, TP rank에 정렬 |
| `mask = (x >= ...) & (x < ...)` | **어떤 위치가 이 카드 어휘에 속하는지** 표시 |
| `x = mask * (x - self.vocab_start_idx)` | **전역 id**를 **로컬 행 번호**로 변환; 이 카드에 속하지 않는 id는 0으로 설정 (mask와 함께 동작) |
| `F.embedding(x, self.weight)` | 표준 테이블 조회; 범위 밖 id 동작은 mask와 이후 곱셈에 의존 |
| `mask.unsqueeze(1) * y` | 이 카드 어휘가 아닌 위치의 **임베딩을 0으로**, 잘못된 값이 리덕션에 들어가는 것 방지 |
| `dist.all_reduce(y)` | **합산**으로 각 카드 기여도를 병합, **완전한 D 차원 벡터**를 얻음 |

### ParallelLMHead 구간별 주석

| 코드 조각 | 설명 |
|----------|------|
| `assert not bias` | 출력 레이어는 **bias 없음**, Qwen 계열 구현과 일관, 병렬화 단순화 |
| `get_context()` | **전역 추론 컨텍스트** 가져오기 (prefill/decode, cu_seqlens 등) |
| `if context.is_prefill` | **Prefill**: 시퀀스가 펼쳐져, `hidden` shape이 **모든 위치**에 대응 |
| `last_indices = context.cu_seqlens_q[1:] - 1` | 각 시퀀스의 **마지막 토큰**이 펼쳐진 `hidden` 내에서의 인덱스 |
| `x = x[last_indices].contiguous()` | **마지막 hidden**만 유지, shape `(batch, D)`, **다음 토큰 logits** 계산 준비 |
| `F.linear(x, self.weight)` | embedding과 동일 가중치의 **선형 레이어**: `logits = x @ W^T` (shape 세부사항은 레이아웃 기준) |
| `tp_size > 1`일 때 `gather` + `cat` | **어휘 차원 분할** 하에서, 각 카드는 **일부 vocab 열**만 보유; **logits 마지막 차원**에서 전체 어휘 logits로 연결 필요 (**rank0로 gather**가 일반적 패턴) |

**주의**: 멀티 카드 시 **rank 비0**은 `None`을 반환할 수 있으며, 필요할 때만 logits를 소비하도록 엔진이 보장합니다; 읽고 있는 `sampler`/`engine`을 기준으로 삼으세요.

## 그림 설명 (텍스트/ASCII 설명)

**어휘 병렬 Embedding (tp=2 예시)**:

```
전역 token id:  0 ... V/2-1  |  V/2 ... V-1
                 ----카드0----    ----카드1----

token이 카드0 구간에 속함 -> 카드0이 벡터 연산, 카드1은 0으로 -> all_reduce 합산 -> 완전한 벡터
```

**Prefill 시 LMHead가 마지막을 취함**:

```
배치 내 3개 시퀀스, 펼친 후 hidden 인덱스:
  seq0: [0,1,2]
  seq1: [3,4]
  seq2: [5,6,7,8]

cu_seqlens_q 유사값 [0,3,5,9]
last_indices = [2,4,8]  -> 이 세 군데의 hidden만 취해 logits 계산
```

## 면접 출제 포인트

- **어휘 병렬 vs 행 병렬/열 병렬**: 여기서 병렬화되는 것은 **임베딩 행렬의 행(vocab 차원)**입니다.
- **mask + all_reduce를 사용하는 이유**: 각 카드는 일부 id만 담당, **나머지는 반드시 0**으로 만들어 리덕션해야 합니다.
- **LMHead가 prefill에서 마지막만 취하는 이유**: **인과 LM의 예측 위치**에 정렬 (바로 **다음** 토큰 예측).
- **TP>1 시 logits에 gather/cat 필요**: 각 카드는 **부분 vocab logits**, 전체 어휘로 연결해야 샘플링 가능.

## 자주 나오는 면접 질문

1. **만약 `mask * y` 없이 바로 all_reduce 하면 어떻게 되나요?**  
   답변: 본 분할이 아닌 id에서 **잘못된 비영 임베딩**이 발생하여, 리덕션 후 **결과가 오염**됩니다.

2. **Decode 단계에서도 LMHead에 `cu_seqlens_q`가 필요할까요?**  
   답변: 보통 **시퀀스당 한 단계**이므로, `is_prefill`이 False일 때는 **해당 경로를 타지 않습니다**; `context` 기준입니다.

3. **weight tying 시 가중치는 어떻게 로드하나요?**  
   답변: `weight_loader`가 **동일한 체크포인트의 행을 분할**하여 각 카드로 보내고, **embedding과 lm_head가 Parameter를 공유**합니다(모델 구현이 그런 경우).

4. **Tokenizer 어휘 크기와 `num_embeddings`가 일치하지 않으면 어떻게 되나요?**  
   답변: 설정/가중치 불일치로 **로드 실패** 또는 **범위 이탈**이 발생합니다; HF `config.vocab_size`와 정렬되어야 합니다.

## 요약

- **Tokenizer**는 엔진 밖에서 텍스트를 **id**로 변환; **어휘 크기**가 임베딩 shape을 결정합니다.
- **VocabParallelEmbedding**은 **구간 mask + 로컬 행 번호 + all_reduce**로 **전체 테이블 중복 저장 없는** 임베딩을 구현합니다.
- **ParallelLMHead**는 **prefill**에서 **`cu_seqlens_q`**를 사용해 **시퀀스별 마지막 위치**를 찾아, **자기회귀 목표**와 일치시킵니다; 멀티 카드 시 **logits 차원에 대해 gather/연결**합니다.

## 다음 강의 예고

다음 강의 **《05-강의05-Attention 메커니즘과 FlashAttention》** 에서는 **`store_kvcache` Triton 커널**, `flash_attn_varlen_func`과 `flash_attn_with_kvcache` 두 분기, 그리고 **prefix cache(`block_tables`)** 하에서 어떻게 **KV cache**로부터 K, V를 읽는지 분석합니다.