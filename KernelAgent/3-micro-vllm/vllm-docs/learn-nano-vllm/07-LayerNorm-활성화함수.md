# 강의 07: RMSNorm과 활성화 함수

> "제곱 평균 제곱근 스케일링"으로 LayerNorm의 평균 중심화를 대체하고, SwiGLU 게이팅과 결합하여 현대 LLM 디코더 레이어에서 안정적이면서도 효율적인 표준 공식을 구성합니다.

## 본 강의 목표

- **RMSNorm**과 **LayerNorm**의 공식 차이 및 공학적 선택을 비교합니다.
- `rsqrt`, 혼합 정밀도 경로(먼저 float으로 변환 후 다시 dtype 기록)를 사용하는 이유를 이해합니다.
- **`add_rms_forward`**를 숙달합니다: 잔차와 정규화의 융합 순서 및 대규모 텐서 읽기/쓰기를 한 번 줄일 수 있는 이유.
- **SwiGLU**를 이해합니다: `SiluAndMul`과 `gate_up_proj`가 어떻게 게이트 피드포워드를 구성하는지.
- **`@torch.compile`**이 이 모듈에서 주는 이점과 면접에서 자주 묻는 후속 질문을 명확히 설명할 수 있습니다.

## 핵심 개념

### 1. LayerNorm 복습 (비교용)

마지막 차원의 벡터 \(x \in \mathbb{R}^d\)에 대해:

\[
\mathrm{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
\]

여기서 \(\mu\)는 평균, \(\sigma^2\)는 분산, \(\gamma,\beta\)는 학습 가능한 아핀 변환입니다(일부 구현에서는 bias를 끌 수 있음).

특징: **평균 빼기**로 각 벡터가 0을 중심으로 하도록 한 후, 표준 편차로 스케일링합니다.

### 2. RMSNorm 공식

RMSNorm은 **평균을 빼지 않고**, 제곱 평균 제곱근(Root Mean Square)만으로 스케일 정규화를 수행합니다:

\[
\mathrm{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}
\]

\[
\mathrm{RMSNorm}(x) = \gamma \odot \frac{x}{\mathrm{RMS}(x)}
\]

비교:

| 차원 | LayerNorm | RMSNorm |
|------|-----------|---------|
| 평균 빼기 여부 | 예 | 아니오 |
| 스케일링 기준 | 분산 (중심화 후) | 2차 모멘트의 제곱근 |
| 파라미터 수 (일반적) | \(\gamma,\beta\) | 보통 \(\gamma\)만 (본 구현은 `weight`만) |
| 계산량 | 약간 높음 (mean 한 번 더) | 약간 낮음 |

직관: 심층 Transformer에서는 **엄격한 영평균**보다 **스케일 안정성**이 더 중요한 경우가 많습니다. RMSNorm은 여러 LLM에서 LN과 비슷하거나 더 나은 효과를 보이며, 계산 비용도 더 저렴합니다.

### 3. `1/sqrt` 대신 `rsqrt`를 사용하는 이유

\[
\frac{x}{\sqrt{v + \epsilon}} = x \cdot \mathrm{rsqrt}(v + \epsilon)
\]

GPU에서 **`torch.rsqrt`**는 단일 명령어이거나 더 잘 융합된 경로인 경우가 많아, 수치적으로 `sqrt`를 먼저 구한 후 나누는 방식에서 발생하는 추가적인 반올림 단계를 피합니다(구현 세부사항은 하드웨어마다 다름). 면접에서는: **융합, 관용적인 작성법, 프레임워크 연산자와의 정렬**이라고 답할 수 있습니다.

### 4. `rms_forward`의 dtype 로직

```text
orig_dtype = x.dtype
x = x.float()
... float32 상에서 var와 rsqrt 계산 ...
x = x.to(orig_dtype).mul_(self.weight)
```

**간단한 이유**: fp16/bf16에서 직접 제곱, 평균, `rsqrt`를 구하면 오버플로나 언더플로가 발생하기 쉽습니다. 먼저 fp32에서 통계량을 계산한 후 다시 기록하는 것이 혼합 정밀도 학습/추론의 일반적인 패턴입니다.

### 5. `add_rms_forward`: 잔차를 융합한 Pre-Norm 변형

`residual`이 존재할 때:

1. `x = x.float().add_(residual.float())`: **고정밀도**에서 **\(x \leftarrow x + \text{residual}\)**을 수행합니다(여기서 명명법상 `x`는 하위 레이어 입력이고, `residual`은 이전 분기의 정규화되지 않은 흐름입니다).
2. `residual = x.to(orig_dtype)`: **더해진 주 경로**를 다음 하위 레이어의 '잔차 분기'로 저장합니다(전형적인 Pre-Norm: **norm은 한 쪽에만 적용되고, 잔차는 다른 쪽으로 우회**합니다).
3. 더해진 `x`에 대해 RMSNorm을 수행합니다(`rms_forward`의 후반부와 동일).

이것이 "먼저 norm을 하고, 하위 레이어를 거친 후 잔차를 더하는" 방식과 수학적으로 일치하려면 전체 `forward` 편성과 일관되어야 합니다. 본 저장소에서는 `forward`에서 `residual is None` 분기를 통해 첫 번째 레이어와 이후 레이어의 차이를 구현합니다(다음 절 소스 코드 참조).

### 6. SwiGLU와 `SiluAndMul`

표준 FFN은 \(\mathrm{down}(\sigma(\mathrm{gate}(x)) \odot \mathrm{up}(x))\)로 쓸 수 있습니다. \(\sigma\)가 **SiLU/Swish** \(z \cdot \sigma(z)\)일 때, 보통 **SwiGLU** 계열이라고 부릅니다.

본 프로젝트에서 `MergedColumnParallelLinear`는 한 번에 **gate와 up 두 부분이 연결된** 결과를 생성하며, `SiluAndMul`은:

```text
x, y = chunk(2, -1)
out = silu(x) * y
```

를 수행합니다. 즉 **게이트 지로**와 **값 지로**가 각각 절반의 채널을 가지고, `silu(x)`가 게이트 역할을 합니다.

---

## 소스 코드 분석

### `RMSNorm` 구조와 `forward` 분기

소스 코드에서 `forward`는 `residual`의 유무에 따라 호출됩니다:

- `rms_forward(x)`: 정규화만 수행.
- `add_rms_forward(x, residual)`: 먼저 잔차를 더한 후 정규화하고, 업데이트된 `residual` 텐서를 반환.

`Qwen3DecoderLayer`와 연동될 때(반드시 `qwen3.py`와 대조할 것):

```text
if residual is None:
    hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
else:
    hidden_states, residual = self.input_layernorm(hidden_states, residual)
```

- **첫 번째 레이어**: `residual is None` → 먼저 `input_layernorm(hidden_states)`을 실행(`rms_forward` 경로), 그리고 **정규화되지 않은 원본** `hidden_states`를 `residual` 변수에 저장하여 이후 레이어가 지름길 분기로 사용할 수 있게 합니다.
- **이후 레이어**: `residual` 있음 → `add_rms_forward` 경로, "먼저 잔차를 더한 후 RMSNorm"을 구현하고 `residual`을 업데이트합니다.

독자들이 `RMSNorm`만 읽으면 혼란스러울 수 있으니, 반드시 **DecoderLayer**와 함께 봐야 합니다(면접에서 이를 능동적으로 설명하면 전체 경로를 읽은 것으로 보입니다).

### `rms_forward` 줄별 분석

- `var = x.pow(2).mean(dim=-1, keepdim=True)`: 마지막 차원에 대한 **제곱의 평균**, 즉 \(\frac{1}{d}\|x\|_2^2\) (다른 라이브러리에서는 \(1/d\)를 쓰지 않는 구현도 있지만, 본 구현은 mean으로 위 공식과 일관).
- `x.mul_(torch.rsqrt(var + self.eps))`: 제자리 스케일링, 메모리 절약.
- `mul_(self.weight)`: 학습 가능한 스케일링 \(\gamma\).

### `add_rms_forward` 줄별 분석

- 먼저 더한 후 norm, 출력은 `(normalized, new_residual)`, `new_residual`은 더한 결과(다음 레이어가 "정규화되지 않은 지름길"로 사용).

### `SiluAndMul`

- `chunk(2, -1)`: 마지막 차원 길이가 짝수여야 하며, 앞뒤 절반이 각각 gate와 value에 대응.
- `F.silu(x) * y`: 요소별 연산; 추가 bias 없이 간결.

---

## 그림 설명

### RMSNorm 데이터 흐름 (잔차 없음)

```text
  x [..., d]
      |
      +-----> pow(2) -> mean(last) -> +eps -> rsqrt ----+
      |                                                    |
      +------------------------ mul <----------------------+
      |
      * weight
      |
  output
```

### SwiGLU와 본 저장소 선형 레이어

```text
  hidden
     |
  gate_up_proj (MergedColumn: 2 * intermediate 열)
     |
  +------+------+
  gate     up
  |        |
 silu      |
  \       /
   * (요소별)
      |
   down_proj
```

### DecoderLayer와의 잔차 (개념)

```text
  ---- residual (지름길, norm 안 됨) ------------------------+
  |                                                        |
  v                                                        |
  x ----> RMSNorm(add) ----> Attention ----> ...           |
```

(구체적인 텐서 명명은 `qwen3.py` 기준, 위 그림은 **norm과 잔차의 역할 분담**을 강조합니다.)

---

## 면접 출제 포인트

### RMSNorm vs LayerNorm (구술 템플릿)

"LayerNorm은 평균을 제거한 후 분산으로 스케일링합니다. RMSNorm은 평균을 제거하지 않고 RMS로만 스케일링하여, 평균 계산이 한 번 빠지고 파라미터도 보통 더 적습니다. 대형 모델에서 실무적으로 RMSNorm이 매우 흔히 사용되며, Pre-Norm과 함께 쓰면 안정적입니다."

### Pre-Norm이 더 잘 학습되는 이유 (간략히)

그래디언트가 잔차 지름길을 통해 더 쉽게 전달되어, 심층에서 그래디언트 소실/폭발이 덜 발생합니다(자세한 내용은 ResNet/Transformer 논문의 설명을 전개할 수 있음).

### `torch.compile`이 여기서 최적화하는 것

- `rms_forward` / `add_rms_forward` / `SiluAndMul`의 작은 연산자 그래프를 **융합**하여 커널 런칭 횟수를 줄입니다.
- 고정된 rank shape에 대해 반복 호출 시 이점이 더욱 두드러집니다. 주의할 점: **첫 컴파일에 오버헤드**가 있고, 동적 shape은 재컴파일을 유발할 수 있습니다.

### 수치 안정성

- `eps`로 0 나누기 방지.
- fp32에서 통계량을 계산한 후 저정밀도로 다시 캐스트.

---

## 자주 나오는 면접 질문

1. **RMSNorm은 평균 빼기가 빠졌는데, 표현력을 해치지 않을까요?**  
   후속 선형 레이어와 attention이 여전히 임의의 아핀 오프셋을 학습할 수 있습니다. 실무에서 RMSNorm은 LLM에서 우수한 성능을 보입니다.

2. **`rsqrt`와 `1/sqrt`를 면접에서 어떻게 답변할까요?**  
   등가의 수학적 관계입니다. `rsqrt`는 일반적인 융합 연산자로, 성능과 수치적 습관에 유리합니다.

3. **ReLU FFN 대비 SwiGLU의 장점은?**  
   게이트 메커니즘의 표현력이 더 강력합니다. SiLU는 매끄러워 최적화 성질이 더 좋습니다(논문 및 공학적 경험과 결합하여 답변).

4. **왜 gate와 up을 하나의 `MergedColumnParallelLinear`로 합치나요?**  
   한 번의 행렬 곱셈으로 `hidden`에서 `2 * intermediate`로 매핑하여, 커널 횟수를 줄이고 텐서 병렬 분할과 메모리 병합 접근에 유리합니다.

5. **`add_rms_forward`에서 왜 `add_`를 먼저 하고 norm을 하나요?**  
   Pre-Norm 잔차 편성과 일관됩니다: 하위 레이어 입력과 지름길을 더한 후, 해당 흐름에 RMSNorm을 적용하여 다음 하위 레이어 입력으로 만듭니다.

6. **LayerNorm의 \(\beta\)는 어디로 갔나요?**  
   본 구현의 RMSNorm은 `weight`만 있고 학습 가능한 bias는 없으며, 일반적인 LLaMA 계열 설정에 부합합니다.

---

## 요약

RMSNorm은 RMS 스케일링으로 LN의 중심화 제거+분산 스케일링을 대체하여 계산이 더 가볍습니다. `rsqrt`와 fp32 통계는 안정적인 혼합 정밀도의 관용적인 작성법입니다. `add_rms_forward`는 "잔차 더하기 + 정규화"를 하나로 묶어 중간 텐서를 줄입니다. `SiluAndMul`은 SwiGLU의 게이트 곱셈을 완성합니다. `torch.compile`은 이러한 고빈도 소형 모듈에 적합하지만, 공학적으로 컴파일 시간과 shape 안정성을 실측해야 합니다.

## 다음 강의 예고

다음 강의 **Qwen3 모델 아키텍처**: `Qwen3Attention`의 GQA, 텐서 병렬 분할부터 `packed_modules_mapping`과 가중치 로딩 매핑까지, 전체 디코더 스택을 연결합니다.