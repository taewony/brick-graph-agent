
## 어휘 병렬 (Vocab Parallelism)

"내가 가진 사전(Vocab)에 있는 단어면 내가 찾아서 알려주고, 내 사전에 없는 단어면 모른다고(0) 대답한 뒤, 흩어진 대답들을 다 같이 합치는(All-Reduce)" 과정입니다.

이를 통해 트랜스포머는 수학적인 결과값의 손실이나 변경 없이, 거대한 임베딩 레이어의 메모리 부담을 여러 GPU로 완벽하게 분산시킬 수 있게 됩니다.

### 트랜스포머 임베딩의 기초와 문제점
트랜스포머의 첫 단계는 입력된 정수 형태의 토큰 ID를 연속적인 실수 벡터로 바꾸는 것입니다. 이를 담당하는 것이 임베딩 행렬(Embedding Matrix)입니다.
- 행렬의 크기: $E \in \mathbb{R}^{V \times H}$$V$: 
- 어휘 사전 크기 (Vocabulary Size). 예: Llama 3의 경우 128,256.$H$: 은닉 차원 (Hidden Dimension). 
예: 4096, 8192 등.작동 방식: 입력 토큰 ID가 $x$일 때, 임베딩 행렬의 $x$번째 행(Row)을 그대로 가져옵니다(Lookup).

문제점 ("임베딩 행렬이 너무 클 때"):어휘 사전과 은닉 차원이 커지면, 임베딩 행렬 하나만으로도 수 GB의 GPU 메모리를 차지합니다. 따라서 여러 개의 GPU(카드)로 모델을 나눌 때, 이 거대한 행렬도 나누어 저장할 필요가 생깁니다.


이 부분은 단순 구현 테크닉이 아니라, **Transformer embedding layer를 “선형대수 관점”에서 어떻게 분산시키는가**의 문제다.

우선 핵심부터 말하면:

> Vocab Parallelism은
> “Embedding lookup도 사실은 gigantic one-hot × matrix multiplication”이라는 관점에서 나온다.

이걸 이해하면 왜 `all_reduce(sum)`가 가능한지 자연스럽게 보인다.

---

# 1) Transformer의 Embedding을 다시 보자

Transformer 입력 토큰:

```text
token_id = 12345
```

이걸 embedding vector로 변환:

[
E \in \mathbb{R}^{V \times H}
]

* (V): vocab size
* (H): hidden dimension

예:

* vocab = 128K
* hidden = 8192

그러면 embedding matrix 크기:

[
128000 \times 8192
]

엄청 크다.

---

# 2) 사실 Embedding Lookup은 Matrix Multiply다

보통은:

```python
x = E[token_id]
```

라고 생각하지만, 수학적으로는:

[
x = onehot(token_id) \cdot E
]

예:

[
onehot(12345)
=============

[0,0,0,\dots,1,\dots,0]
]

그러면:

[
x = e_{12345}
]

즉:

* one-hot vector가
* embedding matrix의 특정 행만 선택

하는 것이다.

---

# 3) 이제 vocab dimension으로 분할해보자

GPU 4장이라고 해보자.

Embedding matrix:

[
E \in \mathbb{R}^{V \times H}
]

를 vocab 방향으로 쪼갠다:

[
E =
\begin{bmatrix}
E_0 \
E_1 \
E_2 \
E_3
\end{bmatrix}
]

각 GPU는:

[
E_r \in \mathbb{R}^{V/4 \times H}
]

만 저장.

예:

| GPU  | vocab range  |
| ---- | ------------ |
| GPU0 | 0~31999      |
| GPU1 | 32000~63999  |
| GPU2 | 64000~95999  |
| GPU3 | 96000~127999 |

---

# 4) token lookup은 어떻게 되나?

토큰:

```text
token_id = 50000
```

이면:

* GPU1만 실제 embedding row를 가짐
* 나머지는 없음

---

# 5) 각 GPU의 local output

각 GPU는:

[
x_r
]

를 계산.

---

## GPU1

토큰이 자기 범위 안:

[
x_1 = E_1[50000]
]

실제 embedding vector 생성.

---

## GPU0/GPU2/GPU3

토큰 없음:

[
x_0 = x_2 = x_3 = 0
]

---

# 6) 왜 all_reduce(sum)가 가능한가?

이제:

[
x = x_0 + x_1 + x_2 + x_3
]

인데 실제로는:

[
x = 0 + real + 0 + 0
]

이므로:

[
x = real
]

즉:

* 합산(sum)이
* 실제 embedding recovery가 된다.

---

# 7) 여기서 중요한 선형대수 관점

Embedding lookup을:

[
x = onehot \cdot E
]

로 보면,

[
E = E_0 + E_1 + E_2 + E_3
]

(정확히는 padding된 block matrix 개념)

이므로:

[
x
=

onehot \cdot (E_0 + E_1 + E_2 + E_3)
]

분배법칙:

# [

onehotE_0 + onehotE_1 + onehotE_2 + onehotE_3
]

즉:

[
x = x_0 + x_1 + x_2 + x_3
]

따라서:
👉 all_reduce(sum)이 mathematically exact.

---

# 8) “연결(concat)”이 아니라 “합(sum)”인 이유

많이 헷갈리는 부분이다.

Tensor Parallel에는 두 종류가 있다:

| 방식                     | 결과 병합     |
| ---------------------- | --------- |
| hidden dimension split | concat 필요 |
| vocab dimension split  | sum 가능    |

---

## Hidden split 예시

[
W = [W_0 ; W_1]
]

출력:

[
y = [y_0 ; y_1]
]

이건 feature가 서로 다르므로 concat 필요.

---

## Vocab split 예시

각 GPU가:

[
same\ hidden\ vector\ space
]

에 대한 partial contribution을 계산.

즉:

[
x = x_0 + x_1
]

형태.

그래서 sum 가능.

---

# 9) 왜 “없는 행은 0”이 중요한가?

핵심이다.

GPU0은 token row를 모르므로:

[
x_0 = 0
]

이어야 선형성이 유지된다.

즉:

[
x = \sum_r x_r
]

가 성립.

---

# 10) 실제 CUDA/GPU 관점

vLLM/PyTorch 내부에서는 대략:

```python
if vocab_start <= token_id < vocab_end:
    local_idx = token_id - vocab_start
    out = embedding_table[local_idx]
else:
    out = zeros(hidden_dim)
```

그 후:

```python
dist.all_reduce(out, op=SUM)
```

---

# 11) 그런데 왜 굳이 vocab parallelism을 쓰나?

대형 모델에서 embedding/output projection이 매우 큼.

예:

[
V=256K,\quad H=12288
]

FP16만 해도:

[
256000 \times 12288 \times 2
\approx 6GB
]

embedding 하나만 6GB.

LM head까지 tied weight면 더 중요.

---

# 12) Input embedding과 Output projection은 사실 같은 구조

LLM 마지막:

[
logits = hW^T
]

여기서:

[
W \in \mathbb{R}^{V \times H}
]

즉 embedding matrix transpose.

그래서 vocab parallelism은:

* input embedding
* output logits

둘 다에 매우 중요.

---

# 13) 왜 vLLM에서 중요해지나?

vLLM은:

* KV cache
* continuous batching
* memory fragmentation

등으로 VRAM 압박이 심함.

그래서:

* embedding/lm_head도 shard 필요
* vocab parallelism이 필수적.

---

# 14) 핵심 직관 한 줄

> Vocab Parallelism은
> “Embedding lookup을 sparse linear algebra로 해석했기 때문에 가능한 분산 방식”이다.

그리고:

> all_reduce(sum)이 가능한 이유는
> 각 GPU의 출력이 동일 벡터 공간에서의 “부분 기여(partial contribution)”이기 때문이다.

---

## 'LM head(Language Modeling head)'

- 언어모델의 가장 마지막 단계에 위치하여, 추상적인 수치 데이터(벡터)를 우리가 읽을 수 있는 단어(토큰) 확률로 변환해주는 '최종 출력층'을 말합니다.
- 쉽게 비유하자면, 트랜스포머의 본체(Blocks)가 문장의 의미를 깊게 고민하는 '뇌'라면, LM head는 그 고민의 결과를 입 밖으로 내뱉는 '입'과 같습니다.

1. LM head의 주요 역할트랜스포머 블록을 통과한 결과물은 모델의 차원(Model Dimension) 크기를 가진 복잡한 벡터 형태입니다. LM head는 이를 다음과 같은 과정을 통해 단어로 바꿉니다.
- 차원 맞추기 (Linear Projection): 모델 내부의 벡터(예: 768차원)를 사전(Vocabulary)에 등록된 전체 단어 수(예: 50,000개)와 동일한 크기의 벡터로 확장합니다.
- 확률 변환 (Softmax): 확장된 수치들을 Softmax 함수에 통과시켜, "이 자리에 올 단어가 '사과'일 확률은 80%, '바나나'일 확률은 10%..."와 같이 0~1 사이의 확률값으로 만듭니다.

2. 왜 'Head'라고 부르나요?트랜스포머 본체는 공통적으로 사용하면서, 그 위에 어떤 '머리(Head)'를 붙이느냐에 따라 모델의 용도가 달라지기 때문입니다.
- LM Head: 다음 단어를 예측하는 언어 생성용 모델 (GPT 등)
- Classification Head: 문장이 긍정인지 부정인지 분류하는 감성 분석용 모델
- QA Head: 질문에 대한 답의 시작과 끝 위치를 찾는 질의응답용 모델

3. 기술적 특징: 가중치 공유 (Weight Tying)
- 많은 모델(예: BERT, GPT)에서는 메모리 효율을 위해 모델의 가장 처음에 단어를 벡터로 바꾸는 '임베딩 레이어(Embedding Layer)'의 가중치를 LM head에서도 그대로 재사용하곤 합니다. 단어를 벡터로 바꿀 때 쓴 기준을, 벡터를 다시 단어로 바꿀 때도 똑같이 적용하는 셈입니다.


왜냐하면 직관적으로는 아래 2개가 서로 완전히 다른 역할처럼 보이고, 공유할 것이 없는 것처럼 생각된다.

* 입력 embedding
* 출력 LM head

그런데 실제로는 많은 LLM이:

[
W_{embed} = W_{lmhead}
]

즉 **weight tying**을 사용한다.

이걸 Transformer 관점에서 단계적으로 보자.

---

# 1) 입력 embedding부터 다시 보자

Vocabulary 크기:

[
V
]

Hidden dimension:

[
H
]

Embedding matrix:

[
E \in \mathbb{R}^{V \times H}
]

---

예:

[
E =
\begin{bmatrix}
e_0 \
e_1 \
e_2 \
...
\end{bmatrix}
]

각 row:

* 하나의 token vector

---

토큰 id:

[
t
]

입력 시:

[
x = E[t]
]

즉:

* embedding lookup
* 특정 row 선택

---

# 2) Transformer 내부 계산

Transformer block들을 지나면:

[
h \in \mathbb{R}^{H}
]

를 얻는다.

이 (h)는:

> “다음 토큰이 무엇이어야 하는가?”

를 담은 hidden representation.

---

# 3) 이제 LM Head 등장

다음 token logits 계산:

[
logits = hW^T
]

여기서:

[
W \in \mathbb{R}^{V \times H}
]

---

결과:

[
logits \in \mathbb{R}^{V}
]

즉:

* vocab의 모든 token에 대한 score

---

# 4) 이걸 자세히 보면 엄청 흥미롭다

각 token score:

[
logit_i = h \cdot W_i
]

즉:

* hidden state (h)
* token embedding-like vector (W_i)

의 dot product.

---

# 5) 이제 핵심 직관

입력 embedding이 의미하는 것:

[
E_i
]

=
“token i의 semantic vector”

---

출력에서도 사실 필요한 것:

> “현재 hidden state가 어떤 token semantic과 가장 잘 맞는가?”

즉:

[
score_i = similarity(h, token_i)
]

---

그러면 자연스럽게:

> “입력 때 사용한 token semantic vector를
> 출력 scoring에도 쓰면 되지 않나?”

라는 생각이 나온다.

---

# 6) 그래서 weight tying

즉:

[
W = E
]

로 둔다.

출력:

[
logits = hE^T
]

---

의미:

[
logit_i = h \cdot e_i
]

즉:

* hidden state와
* token embedding 사이의 similarity

를 계산.

---

# 7) 왜 이게 자연스러운가?

언어 모델 목표:

[
P(token | context)
]

이다.

Transformer hidden state (h)는:

> “지금 어떤 의미 공간 위치에 있는가”

를 표현.

Embedding vector (e_i)는:

> “각 token의 의미 위치”

를 표현.

따라서:

* 가장 가까운 token을 선택하면
* 다음 단어 prediction.

---

# 8) 기하학적(geometry) 관점

Embedding space를 생각해보자.

예:

```text id="h9yqei"
king
queen
prince
```

는 가까이 위치.

Transformer hidden state도:

* “다음 단어가 queen일 듯”
  한 방향으로 이동.

그러면:

[
h \cdot e_{queen}
]

이 커짐.

---

즉 LM head는 사실상:

> hidden state를 vocabulary embedding space에 projection해서
> nearest token을 찾는 과정

이다.

---

# 9) weight tying 안 하면?

원래는:

* 입력 embedding:
  [
  E
  ]

* 출력 projection:
  [
  W
  ]

를 따로 둘 수 있다.

즉:

[
E \neq W
]

---

그러면:

* 입력 semantic space
* 출력 semantic space

를 따로 학습.

가능은 하다.

---

# 10) 그런데 tying의 장점이 엄청 큼

## (1) 파라미터 절약

매우 중요.

예:

[
V=128K,\quad H=8192
]

이면:

[
128000 \times 8192
\approx 1B
]

거의 10억 parameter.

---

embedding + lm_head 따로면:

* 20억 파라미터 가까움.

shared면 절반.

---

# 11) 일반화 성능도 좋아짐

연구적으로도 알려져 있음.

왜냐하면:

* 입력 의미 공간
* 출력 의미 공간

이 일관되게 유지됨.

---

즉:

* token representation이 unified됨.

---

# 12) 아주 중요한 직관

Transformer는 사실:

> “의미 공간 위를 이동하는 dynamical system”

처럼 볼 수 있다.

* 입력 token → embedding space 진입
* Transformer layers → 의미 변환
* LM head → 다시 token space로 projection

그런데:

* 들어올 때 쓰는 좌표계
* 나갈 때 쓰는 좌표계

를 같게 쓰는 것이 weight tying.

---

# 13) 선형대수 관점으로 다시 정리

입력:

[
x = onehot(t)E
]

출력:

[
logits = hE^T
]

즉:

* 같은 matrix
* forward / transpose 형태로 사용.

---

이건 autoencoder 느낌과도 비슷하다.

* embedding = encode
* lm head = decode

---

# 14) 왜 transpose인가?

Embedding:

[
E: V \rightarrow H
]

즉:

* vocab basis
* → hidden space

---

LM head:

[
E^T: H \rightarrow V
]

즉:

* hidden state
* → vocab score

---

완전히 dual relationship.

---

# 15) Transformer 관점에서 가장 중요한 이해

많은 사람들이:

```text id="g67js9"
embedding = 단순 lookup table
```

라고 생각하는데 실제로는:

> embedding matrix는
> “token semantic basis”다.

그리고:

> LM head는
> hidden state를 그 basis에 대해 측정(measurement)하는 과정이다.

---

# 16) 최종 핵심 요약

Weight tying의 의미:

[
W_{lmhead} = W_{embed}
]

즉:

* 입력 시:

  * token → semantic vector
* 출력 시:

  * hidden semantic → token score

를 같은 의미 공간에서 수행.

---

따라서:

> 입력 embedding과 LM head는
> 서로 unrelated한 두 레이어가 아니라,
>
> “동일 semantic space의 encoder/decoder 쌍”
>
> 이다.
