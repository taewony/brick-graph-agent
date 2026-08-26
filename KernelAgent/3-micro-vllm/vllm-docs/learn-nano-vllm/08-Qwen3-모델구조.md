# 강의 08: Qwen3 모델 아키텍처

> "디코더-only + GQA + 텐서 병렬 + 융합 선형 레이어"로 조직된 인과 언어 모델: embedding 진입, 레이어별 Self-Attn+MLP, 최종 RMSNorm, logits은 `lm_head`가 따로 계산합니다.

## 본 강의 목표

- 데이터 흐름에 따라 **Qwen3Model / DecoderLayer / Attention / MLP** 각 모듈의 역할을 구술할 수 있습니다.
- **GQA(Grouped-Query Attention)** 를 이해합니다: `num_heads`와 `num_kv_heads`의 관계 및 KV Cache를 절약하는 이유.
- **`packed_modules_mapping`** 을 이해합니다: HuggingFace에서 분할된 `q_proj/k_proj/...`를 nano-vllm의 융합 가중치로 매핑하는 방법.
- 본 저장소에서의 **Pre-Norm 잔차** 작성법을 숙달합니다(`RMSNorm` 이중 분기와 연계).
- **`tie_word_embeddings`** 와 **`ParallelLMHead`** 가 추론 시스템에서 차지하는 위치를 설명합니다.

## 핵심 개념

### 1. 인과 디코더(Decoder-only) 총람

입력 `input_ids` → `embed_tokens`로 은닉 상태 획득 → \(N\)개의 `Qwen3DecoderLayer` 반복 → 최종 `RMSNorm` → (엔진의 다른 곳 또는 `compute_logits`에서) `lm_head`가 어휘 logits로 매핑.

nano-vllm에서 `Qwen3ForCausalLM.forward`는 **마지막 레이어 은닉 상태**만 반환하며, 일부 프레임워크의 'forward가 직접 logits 반환' 방식과 다르므로 **logits는 `compute_logits`가 별도 호출**하는 설계임을 유의해야 합니다.

### 2. 텐서 병렬(TP) 하의 헤드 수

```text
tp_size = dist.get_world_size()
num_heads = total_num_heads // tp_size
num_kv_heads = total_num_kv_heads // tp_size
```

각 rank는 **자기 분할**의 Q 헤드와 KV 헤드만 보유합니다; `assert total % tp_size == 0`로 나누어 떨어짐을 보장합니다. 면접에서 "왜 나누어 떨어질 수 있는가"라는 질문에는: 모델 설정과 TP 차수가 미리 매칭되어야 합니다.

### 3. GQA: 여러 Query 헤드가 소수의 KV 헤드를 공유

**전체** attention 헤드 수를 \(H\), KV 헤드 수를 \(H_{kv}\)라 하고, \(H_{kv} \mid H\) (보통 \(H_{kv} \le H\)).

- **MHA**: \(H_{kv} = H\), 각 헤드에 독립적인 K, V.
- **GQA**: \(H_{kv} < H\), 여러 Q 헤드가 같은 K/V 세트를 재사용(구현상 주로 브로드캐스트나 중복 인덱싱으로 수행).

**이점**: KV Cache와 K/V 투영 파라미터 수가 \(H_{kv}/H\) 비율로 감소, 장문 컨텍스트 추론 시 메모리와 대역폭을 더 절약.

본 저장소의 `Attention` 모듈은 `num_heads`와 `num_kv_heads`를 받아, 내부적으로 브로드캐스트/그룹핑 로직을 구현합니다(`attention.py` 참조, 본 강의에서는 전개하지 않음).

### 4. QKV 융합 선형 레이어 `QKVParallelLinear`

단일 행렬 곱셈으로 `hidden_size`에서 다음으로 매핑:

\[
\text{out\_dim} = (H \cdot d) + 2 \cdot (H_{kv} \cdot d)
\]

즉 **Q 총 길이 + K 총 길이 + V 총 길이**, 그런 다음 `split`으로 세 조각으로 나눕니다. 이점: 한 번의 GEMM, 더 나은 Tensor Core 활용, 가중치 파편화 감소.

### 5. Q/K에 대한 RMSNorm (qkv_bias가 없을 때)

```text
if not self.qkv_bias:
    self.q_norm = RMSNorm(self.head_dim, ...)
    self.k_norm = RMSNorm(self.head_dim, ...)
```

이는 **Qwen3 시리즈가 'bias 없음' 설정에서 Q, K에 per-head 정규화**를 적용하는 일반적인 변형입니다(구체적인 체크포인트와 정렬; 면접에서는 "학습 안정성 향상/공식 구현과 정렬"이라고 언급하면 충분합니다).

### 6. RoPE와 스케일링

```text
self.scaling = head_dim ** -0.5
self.rotary_emb = get_rope(head_dim, ..., base=rope_theta, ...)
```

- `scaling`: \(\frac{1}{\sqrt{d_h}}\)로서 attention logits 스케일링에 사용.
- `rope_theta`: config에서 비롯됨(소스 코드 기본값은 `1000000` 등 규모로, HuggingFace `Qwen3Config` 기준).

### 7. MLP: SwiGLU + 열 병렬 + 행 병렬

- `MergedColumnParallelLinear(hidden, [intermediate]*2)`: **게이트와 업 프로젝션**을 하나의 넓은 행렬 레이어로 병합.
- `SiluAndMul`: SiLU 게이트 곱셈.
- `RowParallelLinear`: 다운 프로젝션으로 각 TP rank 결과를 취합.

### 8. `packed_modules_mapping`과 가중치 로딩

HuggingFace는 관습적으로 **나누는** `q_proj, k_proj, v_proj`와 `gate_proj, up_proj`를 사용합니다. nano-vllm은 성능을 위해 **융합**하여 `qkv_proj`와 `gate_up_proj`로 만듭니다. 로딩 시 **이름 매핑**을 제공하여 loader가 HF 텐서를 어떻게 **연결/분할**하여 융합 레이어로 보낼지 알려줘야 합니다.

매핑 테이블 의미 예시:

- `"q_proj": ("qkv_proj", "q")`: HF의 `q_proj`가 융합 레이어 안에서 Q에 해당하는 부분임을 의미.
- `"gate_proj": ("gate_up_proj", 0)`: 첫 번째 구간 gate.
- `"up_proj": ("gate_up_proj", 1)`: 두 번째 구간 up.

구체적인 분할 규칙은 `loader` 구현에 있습니다(본 강의에서는 **설계 동기**만 기억하면 됩니다).

### 9. `tie_word_embeddings`

`config.tie_word_embeddings`가 참이면:

```text
self.lm_head.weight.data = self.model.embed_tokens.weight.data
```

**입력 임베딩**과 **출력 logits 투영**이 같은 가중치 행렬을 공유하여, 파라미터 수를 줄입니다. GPT 계열의 일반적인 설정입니다. 주의: 병렬 분할 하에서 `VocabParallelEmbedding`과 `ParallelLMHead`가 공유 저장소 의미론을 지원해야 합니다.

---

## 소스 코드 분석

### `Qwen3Attention.forward`

1. `qkv = self.qkv_proj(hidden_states)`: 한 번의 선형 변환으로 QKV 획득.
2. `split`으로 `q_size, kv_size, kv_size` 세 조각으로 나눔.
3. `view`로 `[..., num_heads, head_dim]` 및 KV 헤드 shape 구성.
4. 선택적 `q_norm` / `k_norm`.
5. `rotary_emb(positions, q, k)`: RoPE.
6. `self.attn(q, k, v)`: 스케일링 점곱 attention + KV Cache (엔진 측).
7. `o_proj`: 멀티헤드 병합하여 `hidden_size`로.

### `Qwen3DecoderLayer.forward`

- `input_layernorm`: 첫 번째 레이어와 이후 레이어 분기는 강의 07 참조.
- `self_attn` → `post_attention_layernorm` → `mlp`.
- `(hidden_states, residual)`을 반환하여 다음 레이어가 계속 진행.

### `Qwen3Model.forward`

- `residual = None` 초기화; 각 레이어마다 업데이트.
- 마지막으로 `norm(hidden_states, residual)`로 잔차 체인 종료.

### `Qwen3ForCausalLM`

- `packed_modules_mapping`: 클래스 속성, 로더가 사용.
- `compute_logits`: **추론 디코딩 루프**에서 마지막 은닉 상태를 얻은 후 어휘 점수 계산.

---

## 그림 설명

### 단일 Decoder 레이어 (논리)

```text
[hidden_in] → input_layernorm + residual → Qwen3Attention → post_attention_layernorm + residual → Qwen3MLP → [hidden_out]
```

### GQA 개요 (개념)

```text
Q:  head0 head1 head2 head3   (num_heads = 4)
K,V:  kv0        kv1           (num_kv_heads = 2)

매핑: Q0,Q1 -> kv0의 K,V
      Q2,Q3 -> kv1의 K,V
(구체적인 브로드캐스트 방식은 구현 기준)
```

### 가중치 매핑 (구술)

```text
HF:  q_proj | k_proj | v_proj     -->  nano: qkv_proj [ Q | K | V ]
HF:  gate_proj | up_proj          -->  nano: gate_up_proj [ gate | up ]
```

---

## 면접 출제 포인트

### Pre-Norm vs Post-Norm

본 구현은 **Pre-Norm**(정규화가 서브레이어 앞에 있고, 잔차가 우회). 학습 안정성이 보통 Post-Norm보다 우수(특히 극심층 네트워크에서).

### 왜 `head_dim`을 명시적으로 설정하는가

일부 모델은 `hidden_size / num_heads`가 나누어 떨어지지 않거나 설정과 병합 헤드 차원이 특수한 경우; `getattr(config, 'head_dim', None)`으로 기본 유도값 재정의를 허용.

### `qkv_bias`와 Q/K Norm의 상호 배타적 표현

소스 코드: `if not self.qkv_bias`여야 `q_norm/k_norm` 생성. 구체적인 모델 레시피와 일관되며, 면접에서는 "공식 체크포인트 동작을 따른다"고 답변.

### 추론 시스템에서 `forward`가 logits를 반환하지 않는 이점

- 디코딩 루프에서 **마지막 토큰**의 은닉 상태만 필요하여 logits 계산;
- 또는 배치 시 스케줄러가 `compute_logits`를 통일적으로 스케줄링하여, 샘플러, CUDA Graph 등 모듈과 디커플링에 유리.

---

## 자주 나오는 면접 질문

1. **MHA 대비 GQA의 단점은?**  
   표현력이 약간 떨어질 수 있지만(KV 공유), 대부분의 대형 모델은 너비/깊이 증가로 보완; 추론 이득이 현저.

2. **왜 QKV 선형 레이어를 융합하는가?**  
   GEMM 횟수 감소, 병렬도 향상, 메모리 지역성 개선.

3. **`o_proj`가 왜 RowParallelLinear인가?**  
   Attention 출력이 각 헤드 차원에서 often **행 병렬**로 완전한 hidden으로 리덕션되어야 함(텐서 병렬 설계와 일관, 자세한 내용은 linear 구현 참조).

4. **`packed_modules_mapping`이 해결하는 문제는?**  
   가중치 형식(HF)과 추론 엔진 형식(융합 레이어)이 불일치할 때의 **자동 재배열**.

5. **RoPE의 `max_position`은 어디서 오는가?**  
   `config.max_position_embeddings`, 학습 컨텍스트 길이와 관련.

6. **SiLU가 왜 MLP에 고정되어 있는가?**  
   `assert hidden_act == "silu"`: 본 구현은 이 경로만 지원하며, Qwen3 SwiGLU 레시피와 정렬.

---

## 요약

nano-vllm에서 Qwen3의 아키텍처는 표준 **Decoder-only**입니다: GQA로 KV 비용 절감, TP로 Q/KV 헤드 분할, 융합 선형 레이어와 `packed_modules_mapping`으로 가중치 매핑 협업, Pre-Norm 잔차와 RMSNorm의 깊은 결합, `tie_word_embeddings`로 선택적 입출력 임베딩 공유. 면접에서 **데이터 흐름 + GQA 동기 + 매핑 테이블 역할**을 명확히 설명하면, 일반적으로 대부분의 출제 포인트를 커버합니다.

## 다음 강의 예고

다음 강의 **KV Cache 원리와 구현**: attention 중복 계산에서 출발하여, 메모리 공식을 유도하고, `ModelRunner.allocate_kv_cache`의 블록 할당과 텐서 shape에 깊이 들어갑니다.