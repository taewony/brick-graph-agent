# 강의 06: RoPE 회전 위치 인코딩

> 2차원 회전을 이용해 '위치'를 attention의 Q/K 벡터에 기록하여, 모델이 상대적 거리를 이해하면서도 추론 시 위치에 따라 테이블을 조회하여 효율적으로 적용할 수 있게 합니다.

## 본 강의 목표

- RoPE(Rotary Position Embedding)의 수학적 동기를 이해합니다: 회전을 사용하는 이유, 절대적/상대적 위치 인코딩과의 관계.
- 복소수 관점과 회전 행렬 관점의 동등한 유도를 숙달하고, `apply_rotary_emb` 구현과 대응시킬 수 있습니다.
- `inv_freq`, cos/sin 사전 계산 캐시, `torch.compile` 및 `lru_cache`가 공학적으로 어떤 역할을 하는지 파악합니다.
- 면접 멘트를 구축합니다: RoPE 외삽, NTK-aware, YaRN 등 자주 나오는 후속 질문들.

## 핵심 개념

### 1. attention에 위치 정보가 필요한 이유

자기 attention은 토큰 순서 변경에 불변합니다: 위치 정보를 바꾸지 않고 순서를 뒤섞으면 출력도 변하지 않습니다. 언어는 순서가 있으므로, 반드시 Q, K(때로는 V에도)에 위치를 주입해야 합니다. 초기에는 절대적 위치 임베딩(입력에 더함)이 사용되었고, RoPE는 위치를 Q, K의 **기하학적 변환**으로 인코딩하며, 일정한 **상대성**도 유지합니다.

### 2. RoPE의 핵심 아이디어 (직관)

각 위치 \(m\)에 대해, head 차원의 벡터를 여러 2차원 부분 공간 상의 벡터로 간주하고, 각 부분 공간에 **위치 \(m\)에 의존하는 회전**을 적용합니다. 이렇게 하면 내적 \(\langle R_m q, R_n k\rangle\)이 자연스럽게 상대 위치 \(m-n\)에 의존하게 됩니다.

### 3. 2차원 회전과 복소수

2차원 벡터 \((x_1, x_2)\)를 \(\theta\) 각도로 회전:

\[
\begin{pmatrix} y_1 \\ y_2 \end{pmatrix}
=
\begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}
\begin{pmatrix} x_1 \\ x_2 \end{pmatrix}
\]

동등하게, \((x_1, x_2)\)를 복소수 \(x_1 + i x_2\)로 보면, 회전은 \(e^{i\theta}\)를 곱하는 것입니다.

RoPE에서, 서로 다른 주파수는 head 내의 각기 다른 '쌍'에 대응합니다: \(j\)번째 쌍은 주파수 \(\theta_j\)를 사용하고, 위치 \(m\)에서의 위상은 \(m\theta_j\)입니다.

### 4. inv_freq의 의미

`rotary_dim = d`라고 하면, 보통 짝수 차원을 취해 쌍으로 처리합니다. \(j\)번째 주파수(코드에서 `arange(0, rotary_dim, 2)`가 \(j=0,1,\ldots\)을 제공):

\[
\theta_j = \mathrm{base}^{-2j/d}
\]

즉 `inv_freq[j] = 1 / base^(2j/d)`입니다. 위치 \(t\)에서 해당 주파수의 위상은 \(t \cdot \theta_j\)입니다(코드에서 `einsum("i,j->ij", t, inv_freq)`로 모든 위치, 모든 주파수를 한 번에 계산).

`base`(예: 10000)는 주파수 스펙트럼을 제어합니다: 고주파 성분은 세밀한 상대 위치를 인코딩하고, 저주파는 장거리 패턴에 대응합니다.

### 5. cos/sin을 캐시하는 이유

각 `(position, freq)`마다 \(\cos(t\theta_j)\), \(\sin(t\theta_j)\)가 필요합니다. 추론 시 같은 위치가 대량으로 반복 조회됩니다. 미리 `[max_position, 1, rotary_dim]` 크기의 캐시(cos과 sin을 마지막 차원에 연결)를 계산해 두면, 순전파 시 `cos_sin_cache[positions]`만 인덱싱하면 되므로, 반복적인 삼각함수 계산과 브로드캐스트 비용을 피할 수 있습니다.

### 6. torch.compile과 lru_cache (공학 관점)

- **`@torch.compile` (`RotaryEmbedding.forward`에 적용)**: forward를 더 효율적인 융합 커널 경로로 컴파일하여 파이썬 스케줄링 오버헤드를 줄입니다. 정적 shape의 RoPE 테이블 조회+chunk와 잘 어울립니다.
- **`@lru_cache(1)`의 `get_rope`**: 동일한 `(head_size, rotary_dim, max_position, base)`에 대해 단 **하나의** `RotaryEmbedding` 모듈 인스턴스만 생성하여, 레이어마다 중복 생성과 중복 버퍼를 방지합니다. 전형적인 **싱글톤 팩토리**입니다(참고: `rope_scaling`은 `None`으로 단언되어 있으며, 본 구현은 스케일링 계열 RoPE를 구현하지 않았습니다).

---

## 소스 코드 분석

다음은 nano-vllm의 `rotary_embedding.py` 핵심 로직에 해당합니다.

### `apply_rotary_emb` 줄별 분석

```python
def apply_rotary_emb(x, cos, sin):
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)
```

- **`chunk(..., 2, dim=-1)`**: 마지막 차원을 절반으로 분할, 2차원 회전의 두 성분(실수부/허수부 또는 \(x_1,x_2\))에 대응합니다.
- **`y1, y2`**: 즉 회전 행렬 곱셈  
  \(\begin{smallmatrix}\cos&-\sin\\ \sin&\cos\end{smallmatrix}\)을 \((x_1,x_2)\)에 적용한 것입니다.
- **`.float()`**: 삼각함수와 회전은 float32에서 더 안정적이며, 마지막에 `.to(x.dtype)`로 fp16/bf16으로 돌아가 혼합 정밀도 학습/추론과 일관됩니다.

### `RotaryEmbedding.__init__`

- `assert rotary_dim == head_size`: 본 구현은 head 차원 전체가 RoPE에 참여해야 합니다(일부 차원이 회전하지 않는 변종 없음).
- `inv_freq`: 위 문단의 공식 참조.
- `freqs = einsum(t, inv_freq)`: shape `[max_position, rotary_dim/2]`, 각 위치가 한 행, 각 주파수가 한 열.
- `cos`, `sin` 뒤에 `torch.cat((cos, sin), dim=-1)`: 마지막 차원 길이가 `rotary_dim`, 앞부분 cos, 뒷부분 sin; `unsqueeze_(1)`는 head 차원과 브로드캐스트하기 위해 차원을 예약.
- `register_buffer(..., persistent=False)`: 모델 디바이스를 따라 이동하며, 기본적으로 체크포인트에 기록하지 않음(완전히 하이퍼파라미터로 재구성 가능하므로).

### `forward`

- `cos_sin = self.cos_sin_cache[positions]`: `positions`는 현재 배치의 각 토큰 위치 id, 비연속적인 위치도 지원(예: 디코딩 단계).
- `cos, sin = cos_sin.chunk(2, dim=-1)`: 초기화 시 연결한 순서와 대응.
- `query`, `key` 각각에 `apply_rotary_emb` 호출, **Q/K에만 회전 적용**, 값 벡터 `v`는 보통 회전하지 않음(표준 RoPE 관행).

### `get_rope`

- `lru_cache(1)`: 가장 최근 파라미터 조합에 대한 생성 결과를 캐시; 실제 공학에서 head 설정이 고정되어 있으면 전역 싱글톤과 동등.
- `assert rope_scaling is None`: 동적 NTK/YaRN 등은 inv_freq 변경이나 보간 로직이 필요하며, 여기서는 구현되지 않음.

---

## 그림 설명

### RoPE 정보 흐름 (간략화)

```text
[초기화: 1회만]
inv_freq → freqs 테이블 → cos/sin 캐시

[순전파: 각 attention 레이어마다]
positions → 테이블 조회 cos_sin → apply_rotary_emb ← query, key
                                              ↓
                                        회전된 Q, 회전된 K
```

### 2차원 블록 회전 (마지막 차원의 한 쌍)

```text
  [..., x1 | x2, ...]  --chunk-->  x1, x2
         |                            |
         v                            v
    cos, sin (위치별 브로드캐스트)   y1 = x1*cos - x2*sin
                                     y2 = x2*cos + x1*sin
         |                            |
         +-------- cat ----------->  [..., y1 | y2, ...]
```

---

## 면접 출제 포인트

### RoPE와 상대 위치

RoPE는 \(\langle R_m q, R_n k\rangle\)가 \(m-n\) 형태의 상대 구조에 의존하게 합니다(표준 설정 하에서). 이것이 LLaMA/Qwen 등에 널리 사용되는 이유입니다.

### 외삽 (Extrapolation)

학습 최대 길이가 \(L_{\mathrm{train}}\)인데, 테스트 시 \(L_{\mathrm{test}} > L_{\mathrm{train}}\)이면, 한 번도 본 적 없는 높은 위치 인덱스에서는, 고정 base의 위상 분포가 학습 분포와 불일치하여 attention 점수 분포가 이동 → **외삽 성능이 저하**됩니다.

### NTK-aware (사고방식)

**더 긴 컨텍스트에서 base 또는 주파수를 재조정**하여, 고주파 성분의 변화를 더 느리게 만들어, 외삽 시 '주파수 스펙트럼과 학습 불일치'를 완화합니다. 구현 시 주로 `inv_freq` 계산을 변경하거나 스케일링 인자를 도입하며, 학습 설정과 구별해야 합니다.

### YaRN (사고방식)

**NTK식 보간**과 **attention 온도/절단** 등의 전략을 결합하여, 컨텍스트 확장 시 근거리와 원거리 동작을 모두 고려합니다; 공학적으로 흔히 사용하는 장문 컨텍스트 솔루션 계열에 속합니다.

### 본 소스 코드 관련 단답형

- 왜 `rotary_dim == head_size`인가? 본 파일은 전체 head 회전을 가정하여 구현을 단순화했습니다.
- 왜 cos/sin을 매 배치마다 새로 계산하지 않는가? 추론 핫 경로에서 테이블 조회가 더 경제적입니다.
- `torch.compile`이 야기할 수 있는 문제는? 동적 shape이 극단적으로 변할 때 재컴파일이 발생; 실측이 필요합니다.

---

## 자주 나오는 면접 질문

1. **RoPE와 절대 위치 임베딩의 장단점 비교?**  
   RoPE는 Q/K 내적의 기하학적 구조에 녹아들어 상대 위치 성질이 더 명확합니다; 절대 임베딩은 구현이 단순하지만 더 긴 시퀀스로 일반화할 때 튜닝된 RoPE보다 못한 경우가 많습니다.

2. **왜 일반적으로 Q, K에만 RoPE를 적용하나요?**  
   표준 유도에서 회전은 query/key에 작용하여 상대 위치 관계를 인코딩합니다; value는 회전하지 않는 것이 일반적이며 효과적인 설정입니다.

3. **base가 10000이라는 것은 무엇을 의미하나요?**  
   각 차원의 회전 속도의 지수 분포를 제어하며; 값이 클수록 보통 더 낮은 주파수로 치우쳐, 모델링 가능한 파장과 장거리 의존성에 영향을 줍니다.

4. **면접관에게 inv_freq 라인을 어떻게 설명할까요?**  
   각 차원 쌍에 대해 등비 수열의 각속도를 지정하여, 다중 스케일의 위치 패턴이 중첩되어 표현될 수 있도록 합니다.

5. **장문 컨텍스트 확장을 위해 모델 교체 외에 무엇을 말할 수 있나요?**  
   위치 보간(PI), NTK-aware, YaRN, 윈도우 attention 등을 언급하고, 본 저장소 구현은 'rope_scaling이 없는 기본 RoPE'임을 설명합니다.

---

## 요약

RoPE는 위치 정보를 head 차원 상의 블록별 2차원 회전으로 구현합니다. `inv_freq`는 다중 스케일 각속도를 정의하고, cos/sin 테이블은 중복 계산을 피하며, `torch.compile`과 `lru_cache`는 각각 순전파와 모듈 싱글톤을 최적화합니다. 면접에서 장문 컨텍스트를 추궁받을 때, **외삽 실패 원인**과 **NTK / YaRN의 사고방식**을 명확히 설명하면 산업계 실무와 연결할 수 있습니다.

### 필기용 의사 코드 요약 (면접 화이트보드)

면접관이 RoPE 핵심을 작성해보라고 하면, 이 정도만 쓰면 됩니다:

1. `inv_freq`와 각 위치 `t`의 `freqs[t, j] = t * inv_freq[j]` 사전 계산;
2. `cos, sin` 후 쌍 `(x1,x2)`에 대해 회전 수행;
3. Q, K 각각 한 번씩 회전 후 attention 진입.

### FlashAttention과의 관계

RoPE는 보통 FlashAttention 커널에 진입하기 **전에** Q/K 텐서에 적용됩니다. 커널 내부는 효율적인 softmax(QKᵀ)V 계산만 담당합니다. 만약 어떤 구현이 RoPE를 CUDA 커널에 융합했다면, 이는 추가 최적화이며 수학적 정의와 일치하기만 하면 됩니다.

(끝)

---

## 다음 강의 예고

다음 강의는 **RMSNorm과 활성화 함수(SwiGLU / SiLU 포함)**로 들어갑니다: 정규화가 어떻게 심층 네트워크를 안정화하는지, 잔차가 RMSNorm과 어떻게 융합되는지, 그리고 게이트 피드포워드가 Qwen 계열 모델에서 표준 구현되는 방식을 살펴봅니다.