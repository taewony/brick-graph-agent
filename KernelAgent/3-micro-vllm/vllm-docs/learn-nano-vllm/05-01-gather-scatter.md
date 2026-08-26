좋다.
Scatter/Gather는 GPU 프로그래밍, CUDA, Transformer, KV cache, Embedding lookup 등을 이해할 때 핵심 개념이다.

그리고 중요한 점은:

> Scatter/Gather는 단순 API 이름이 아니라,
> “메모리 접근 패턴(memory access pattern)” 자체를 의미한다.

---

# 1) 가장 직관적인 정의

# Gather

> 여러 위치에서 데이터를 “읽어와서 모은다”

즉:

```text id="9rq3wu"
many locations → one packed output
```

---

# Scatter

> 하나의 연속된 데이터를 여러 위치에 “흩뿌린다”

즉:

```text id="1h7i55"
packed input → many locations
```

---

# 2) CPU 배열 예시로 먼저 보자

배열:

```python id="9abqjl"
A = [10,20,30,40,50]
```

인덱스:

```python id="g2n7mx"
idx = [3,1,4]
```

---

# Gather

```python id="0g7prw"
out[i] = A[idx[i]]
```

결과:

```python id="i5q5h6"
out = [40,20,50]
```

즉:

* 여러 위치에서 읽어서
* 연속 output 생성.

---

# Scatter

반대 방향.

```python id="9grqia"
A[idx[i]] = values[i]
```

예:

```python id="jwcjlwm"
values = [100,200,300]
```

결과:

```python id="93l00j"
A = [10,200,30,100,300]
```

즉:

* packed input을
* 여러 위치에 분산 저장.

---

# 3) 왜 GPU에서 중요하나?

GPU 성능 핵심은:

[
memory\ bandwidth
]

이다.

특히:

* CUDA core 계산보다
* memory access pattern이 더 중요할 때 많음.

---

GPU는:

```text id="ptm0cd"
contiguous memory access
```

를 매우 좋아한다.

즉:

```text id="w8v10z"
thread0 -> A[0]
thread1 -> A[1]
thread2 -> A[2]
```

같은 access.

이를:

> coalesced memory access

라고 한다.

---

# 4) Scatter/Gather는 비연속 접근

예:

```text id="9z0mnj"
thread0 -> A[100]
thread1 -> A[7]
thread2 -> A[9999]
```

처럼 랜덤 접근.

즉:

* cache locality 낮음
* memory transaction 증가
* bandwidth utilization 감소

---

그래서 GPU에서 scatter/gather는:

> 강력하지만 비싼 연산

이다.

---

# 5) Embedding lookup은 사실 Gather

아주 중요.

Embedding:

[
E \in \mathbb{R}^{V \times H}
]

토큰 ids:

```python id="6x63qk"
[100, 57, 8123]
```

이면:

```python id="5ifowr"
out[0] = E[100]
out[1] = E[57]
out[2] = E[8123]
```

즉:

* 거대한 table에서
* 필요한 row들을 읽어옴.

완전한 gather operation.

---

# 6) KV Cache도 Gather 많음

Attention 시:

* 과거 token들
* page/block
* batch마다 다른 sequence

를 읽어야 함.

즉:

```text id="d4egk5"
random block reads
```

---

vLLM의 PagedAttention 핵심 난제도 사실:

> KV cache gather를 얼마나 효율적으로 하느냐

이다.

---

# 7) Attention 자체도 Gather 성격

Attention:

[
Attention(Q,K,V)
]

에서:

[
softmax(QK^T)V
]

는 결국:

* relevant V rows를 weighted gather하는 것과 유사.

---

즉:

* attention = differentiable gather

라고 볼 수도 있다.

---

# 8) Scatter는 어디서 많이 쓰이나?

## (1) Gradient accumulation

학습 시:

```python id="r1xkjt"
grad[token_id] += ...
```

여러 thread가:

* 서로 다른 위치에 gradient write.

---

## (2) KV cache write

새 token 생성 시:

```text id="yyg5p7"
new K,V → cache slots
```

scatter write 발생.

---

## (3) MoE routing

Mixture-of-Experts에서:

* token들을
* expert별 buffer로 분배.

완전한 scatter.

---

# 9) CUDA kernel 수준에서 보면

---

# Gather kernel

```cuda id="fqvruw"
out[i] = input[index[i]];
```

* index 배열 기반 read.

---

# Scatter kernel

```cuda id="zjlwm6"
output[index[i]] = input[i];
```

* index 배열 기반 write.

---

# 10) 왜 Scatter가 더 어려운가?

중요.

Gather:

* 여러 thread가 read만 함.
* 충돌 적음.

Scatter:

* 여러 thread가 같은 위치 write 가능.

예:

```python id="x5v49q"
A[5] += ...
```

동시에 여러 thread 접근.

---

이 경우:

* race condition
* atomic operation 필요.

GPU에서 atomic은 비쌈.

---

그래서 일반적으로:

| 연산      | 난이도      |
| ------- | -------- |
| Gather  | 상대적으로 쉬움 |
| Scatter | 더 어려움    |

---

# 11) Transformer에서 실제 예시

# Input Embedding

```python id="gm9e6u"
hidden = embedding[token_ids]
```

→ Gather

---

# Output logits sampling

Top-k token 선택 후:

* 결과 buffer에 write

→ Scatter 성격 일부 존재.

---

# KV Cache Append

```python id="hhx2ze"
cache[slot] = new_kv
```

→ Scatter

---

# 12) vLLM과 Scatter/Gather

vLLM은 사실상:

> “초고속 KV gather/scatter 시스템”

이라고 봐도 된다.

왜냐하면:

* batch마다 sequence 길이 다름
* memory paging 존재
* token 생성 비동기적

이라서:

```text id="h6gboq"
contiguous tensor
```

가 거의 유지되지 않는다.

---

따라서:

* efficient gather
* page table lookup
* cache indirection

이 핵심.

---

# 13) Deep Learning Framework 수준 의미

PyTorch:

```python id="8m0axq"
torch.gather()
torch.scatter()
```

존재.

---

예:

```python id="ifjlwm"
torch.gather(x, dim, idx)
```

=
index 기반 read.

---

```python id="3c8efk"
torch.scatter(...)
```

=
index 기반 write.

---

# 14) 시스템 관점 핵심 직관

| 연산      | 본질       |
| ------- | -------- |
| Gather  | 랜덤 read  |
| Scatter | 랜덤 write |

---

그리고 GPU는:

> “순차적(contiguous) access”를 좋아하고,
> scatter/gather는 이를 깨뜨린다.

---

# 15) 마지막으로 LLM 관점 핵심

LLM inference의 상당 부분은 사실:

```text id="n9t7oe"
dense GEMM
```

보다도,

```text id="8gqpkc"
irregular memory movement
```

문제로 바뀌고 있다.

특히:

* KV cache
* MoE routing
* PagedAttention
* speculative decoding

등에서는 scatter/gather 최적화가 매우 중요하다.

---

# 최종 핵심 요약

| 연산      | 의미            |
| ------- | ------------- |
| Gather  | 여러 위치에서 읽어 모음 |
| Scatter | 여러 위치로 흩뿌려 기록 |

그리고:

* Embedding lookup = gather
* KV cache append = scatter
* Attention = weighted gather
* MoE routing = scatter/gather

이다.

GPU kernel 관점에서는 결국:

> “비연속 메모리 접근을 얼마나 효율적으로 처리하느냐”

의 문제다.
