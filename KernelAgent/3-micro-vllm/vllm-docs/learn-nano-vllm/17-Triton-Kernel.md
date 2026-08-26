# 제17강: Triton 커널 작성

초보자를 위한 설명: 본 강의는 'Triton이란 무엇인가'에서 시작하여 CUDA 프로그래밍과 비교하고, nano-vllm의 **KV Cache 기록 커널**의 모든 코드를 상세히 해석하여 Triton 프로그래밍 모델과 GPU 병렬 사고를 이해하도록 돕습니다. 각 절에는 소스 코드 해설과 면접 핵심 포인트가 포함되어 있습니다.

아래 코드는 `nanovllm/layers/attention.py`의 구현과 일치합니다(변수명 `N`은 배치 내 토큰 수와 동일).

```python
@triton.jit
def store_kvcache_kernel(key_ptr, key_stride, value_ptr, value_stride, k_cache_ptr, v_cache_ptr, slot_mapping_ptr, D: tl.constexpr):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)

def store_kvcache(key, value, k_cache, v_cache, slot_mapping):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)
```

---

## 1. 개념 설명

### 1.1 Triton이란 무엇인가

**Triton**은 OpenAI가 개발한 **GPU 프로그래밍 언어이자 컴파일러**로, 개발자가 CUDA C++ 같은 저수준 세부 사항을 몰라도 **Python에 가까운 문법**으로 고성능 GPU 커널(kernel)을 작성할 수 있게 하는 것을 목표로 합니다.

Triton의 포지셔닝:

```
추상화 수준:  Python (PyTorch)  >  Triton  >  CUDA C++  >  PTX 어셈블리
성능 상한:    Python (PyTorch)  <  Triton  ≈  CUDA C++  ≈  PTX 어셈블리
개발 효율:    Python (PyTorch)  >  Triton  >  CUDA C++  >  PTX 어셈블리
```

Triton은 **CUDA에 버금가는 성능**을 유지하면서도 **개발 진입 장벽을 크게 낮추어** 딥러닝 시스템에서 커스텀 커널을 작성할 때 가장 선호되는 도구가 되었습니다.

### 1.2 Triton vs CUDA 프로그래밍 비교

| 비교 차원 | CUDA C++ | Triton |
|----------|----------|--------|
| **프로그래밍 언어** | C++ 확장 | Python (`@triton.jit` 데코레이터 사용) |
| **병렬 처리 단위** | **스레드 레벨(Thread-level)** | **블록 레벨(Block-level)** |
| **메모리 관리** | 공유 메모리, 레지스터 수동 관리 | 컴파일러가 자동 관리 |
| **인덱스 계산** | `threadIdx.x`, `blockIdx.x` 수동 작성 | `tl.program_id()`, `tl.arange()` |
| **메모리 접근 최적화** | 병합 접근(coalescing) 수동 처리 | 컴파일러가 자동 최적화 |
| **동기화** | `__syncthreads()` 수동 동기화 | 컴파일러가 자동 삽입 |
| **컴파일** | nvcc 컴파일 | JIT(적시 컴파일), 첫 호출 시 컴파일 |
| **디버깅 난이도** | 높음 | 중간 |

### 1.3 Triton의 블록 레벨 병렬 모델

이것이 Triton을 이해하는 가장 핵심적인 개념입니다.

CUDA에서는 '**각 스레드가 무엇을 할지**'를 고민해야 합니다.

```c++
// CUDA: 각 스레드가 하나의 요소 처리
__global__ void add_kernel(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
```

Triton에서는 '**각 프로그램(블록)이 데이터 한 덩어리를 처리하는 방식**'을 고민합니다.

```python
# Triton: 각 프로그램이 BLOCK_SIZE개의 요소 처리
@triton.jit
def add_kernel(a_ptr, b_ptr, c_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    tl.store(c_ptr + offsets, a + b, mask=mask)
```

차이점 요약:
- CUDA: **각 스레드**의 동작을 관리하며 **공유 메모리**, **스레드 동기화**, **메모리 병합** 등을 수동 처리.
- Triton: **각 블록이 어느 데이터를 처리할지**만 정의하면, 컴파일러가 스레드 할당, 공유 메모리, 병합 접근을 자동 처리.

### 1.4 Triton 핵심 API 요약

| API | 기능 | CUDA 대응 |
|-----|------|----------|
| `@triton.jit` | 함수를 GPU 커널로 표시 | `__global__` |
| `tl.program_id(axis)` | 현재 프로그램의 ID 획득 | `blockIdx.x` |
| `tl.arange(start, end)` | 연속된 정수 시퀀스 생성 | 직접 대응 없음, 수동 작성 필요 |
| `tl.load(ptr + offsets)` | 전역 메모리에서 데이터 로드 | 수동 `ptr[idx]` 읽기 |
| `tl.store(ptr + offsets, val)` | 전역 메모리에 데이터 기록 | 수동 `ptr[idx] = val` 대입 |
| `tl.constexpr` | 컴파일 타임 상수 | `constexpr` 또는 템플릿 매개변수 |
| `tl.zeros(shape, dtype)` | 0으로 초기화된 블록 생성 | CUDA 공유 메모리 수동 초기화 |
| `tl.dot(a, b)` | 블록 레벨 행렬 곱셈 | `wmma` 호출 또는 수동 작성 |

### 1.5 `@triton.jit` 데코레이터

```python
@triton.jit
def my_kernel(arg1, arg2, CONST: tl.constexpr):
    ...
```

이 데코레이터의 역할:

1. **JIT 컴파일 표시**: 함수 본문은 즉시 실행되지 않고, 첫 호출 시 Triton 컴파일러가 GPU 머신 코드(PTX → SASS)로 컴파일합니다.
2. **타입 추론**: 호출 시 전달된 인자 타입에 따라 내부 변수 타입을 자동 추론합니다.
3. **특수화(Specialization)**: `tl.constexpr` 매개변수는 컴파일 시점에 고정되어, 컴파일러가 상수 폴딩, 루프 전개 등의 최적화를 수행할 수 있습니다.
4. **캐싱**: 컴파일 결과는 캐시되어, 동일한 매개변수 타입의 후속 호출 시 다시 컴파일하지 않습니다.

주의 사항:
- 커널 함수는 **반환값을 가질 수 없으며**, 모든 출력은 `tl.store`를 통해 메모리에 기록됩니다.
- 커널 내에서 표준 Python 함수(`print`, `len` 등)를 호출할 수 없지만, `tl.` 계열 API는 호출 가능합니다.
- Python의 동적 특성(리스트 컴프리헨션, 딕셔너리 연산 등)을 사용할 수 없습니다.

---

## 2. 소스 코드 해설: `store_kvcache_kernel` 줄 단위 분석

nano-vllm의 KV Cache 기록 커널은 `nanovllm/model/attention.py`에 위치하며 Triton으로 구현되었습니다. Attention 레이어에서 계산된 Key와 Value를 KV Cache의 올바른 위치에 기록하는 역할을 합니다.

### 2.1 KV Cache 기록에 커스텀 커널이 필요한 이유

nano-vllm의 PagedAttention 아키텍처(9강 복습)에서 KV Cache는 **슬롯(slot)** 단위로 구성됩니다.

- 각 슬롯은 토큰 하나의 K/V 벡터에 해당합니다.
- `slot_mapping[i]`은 i번째 토큰이 어느 슬롯에 기록되어야 하는지 알려줍니다.
- 슬롯 할당은 비연속적이며(BlockManager가 관리), 단순 연속 메모리 복사로는 처리할 수 없습니다.

만약 PyTorch로 구현한다면:

```python
for i in range(N):
    slot = slot_mapping[i]
    if slot != -1:
        k_cache[slot] = key[i].reshape(-1)
        v_cache[slot] = value[i].reshape(-1)
```

이러한 Python 루프 + 요소별 연산은 매우 비효율적입니다. Triton을 사용하면 이를 **병렬화**하여 모든 토큰이 동시에 각자의 슬롯에 기록되도록 할 수 있습니다.

### 2.2 커널 소스 코드 전체 분석

```python
@triton.jit
def store_kvcache_kernel(
    key_ptr,           # Key 텐서의 시작 포인터
    key_stride,        # Key 텐서 0번 차원의 보폭(stride)
    value_ptr,         # Value 텐서의 시작 포인터
    value_stride,      # Value 텐서 0번 차원의 보폭
    k_cache_ptr,       # K Cache의 시작 포인터
    v_cache_ptr,       # V Cache의 시작 포인터
    slot_mapping_ptr,  # 슬롯 매핑 테이블의 시작 포인터
    D: tl.constexpr    # 토큰당 K/V 벡터 총 차원 (num_heads * head_dim)
):
    # 현재 프로그램의 ID 획득, 각 프로그램은 하나의 토큰 처리
    idx = tl.program_id(0)

    # 이 토큰이 기록될 슬롯 번호 읽기
    slot = tl.load(slot_mapping_ptr + idx)

    # 센티널 값: slot == -1은 무효 토큰(padding)을 의미, 바로 건너뜀
    if slot == -1:
        return

    # 소스 텐서에서 Key와 Value의 오프셋 계산
    # key 형태: [N, num_heads, head_dim], 평탄화 시 key[idx]는 idx * key_stride부터 시작
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)

    # 소스 텐서에서 K, V 벡터 로드
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)

    # 캐시 내 목표 오프셋 계산
    # cache 형태: [total_slots, D], slot에 해당하는 시작 위치는 slot * D
    cache_offsets = slot * D + tl.arange(0, D)

    # K Cache와 V Cache에 기록
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)
```

### 2.3 줄 단위 심층 해설

#### `idx = tl.program_id(0)`

- `tl.program_id(axis)`는 지정된 축에서 현재 프로그램의 인덱스를 반환합니다.
- `axis=0`은 첫 번째 차원을 의미합니다. Triton은 최대 3축의 프로그램 그리드를 지원합니다(CUDA의 3D 그리드와 유사).
- 이 커널에서 그리드 크기는 `(N,)`이며, 각 프로그램은 `idx`번째 토큰을 처리합니다.

#### `slot = tl.load(slot_mapping_ptr + idx)`

- `slot_mapping` 배열에서 `idx`번째 토큰이 대응하는 목표 슬롯 번호를 읽습니다.
- `slot_mapping`은 1D 텐서이며, `slot_mapping_ptr + idx`는 포인터 연산입니다.

#### `if slot == -1: return`

- `-1`은 센티널 값(sentinel value)으로, 이 토큰이 **패딩(padding)**임을 나타내며 KV Cache에 기록할 필요가 없습니다.
- CUDA Graph의 `run_model`에서 `graph_vars["slot_mapping"].fill_(-1)`이 남는 자리를 `-1`로 표시하는 것과 일치하는 약속입니다.
- Triton은 간단한 `if` 조건 분기를 지원하지만, 컴파일러가 이를 전통적인 분기 점프가 아닌 **조건부 실행(predicated execution)**으로 변환합니다.

#### `key_offsets = idx * key_stride + tl.arange(0, D)`

이것은 Triton에서 매우 전형적인 **블록 레벨 인덱스 계산**입니다.

- `key_stride`는 Key 텐서의 토큰 차원 보폭입니다. `[N, num_heads, head_dim]` 형태의 연속 텐서인 경우 `key_stride = num_heads * head_dim = D`입니다.
- `tl.arange(0, D)`는 `[0, 1, 2, ..., D-1]` 벡터를 생성합니다.
- `idx * key_stride + tl.arange(0, D)`는 `idx`번째 토큰의 D개 요소 전체에 대한 오프셋을 만듭니다.

그림 설명(D=4 가정):

```
key 메모리 레이아웃:
[token0_k0, token0_k1, token0_k2, token0_k3, token1_k0, token1_k1, ...]
 ↑ idx=0의 오프셋: [0, 1, 2, 3]      ↑ idx=1의 오프셋: [4, 5, 6, 7]
```

#### `key = tl.load(key_ptr + key_offsets)`

- **벡터화 로드**: 전역 메모리에서 D개의 요소를 한 번에 레지스터로 로드합니다.
- Triton 컴파일러는 자동으로 이러한 접근들을 **병합(coalesce)**하여 최소 횟수의 메모리 트랜잭션으로 만듭니다.
- CUDA C++에서 동등한 기능을 구현하려면 `__shared__` 메모리와 `__syncthreads()`를 수동으로 처리해야 합니다.

#### `cache_offsets = slot * D + tl.arange(0, D)`

- KV Cache 메모리 레이아웃: 각 슬롯은 연속된 D개의 요소를 차지합니다.
- `slot * D`는 `slot`번째 슬롯의 시작 위치입니다.
- `+ tl.arange(0, D)`는 해당 슬롯 내 모든 요소를 커버합니다.

#### `tl.store(k_cache_ptr + cache_offsets, key)`

- **벡터화 기록**: D개의 요소를 KV Cache의 목표 위치에 한 번에 기록합니다.
- 서로 다른 프로그램이 서로 다른 슬롯에 기록하므로 쓰기 충돌이 없으며 동기화도 필요 없습니다.

### 2.4 핵심 설계 요약

| 설계 포인트 | 설명 |
|------------|------|
| 프로그램당 토큰 하나 처리 | 토큰 수 N에 대응하여 병렬성을 극대화 |
| `D`를 `tl.constexpr`로 | 컴파일 시점에 고정하여 루프 전개 등 최적화 가능 |
| `slot == -1` 건너뛰기 | 패딩 및 CUDA Graph의 `fill_(-1)`과 협력 |
| 공유 메모리 없음, 동기화 없음 | 각 프로그램이 완전 독립적이므로 동기화 오버헤드 회피 |
| `stride` 사용으로 연속성 가정 배제 | 비연속 메모리 레이아웃도 지원하여 범용성 향상 |

---

## 3. 소스 코드 해설: `store_kvcache` 래퍼 함수

```python
def store_kvcache(key, value, k_cache, v_cache, slot_mapping):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim

    store_kvcache_kernel[(N,)](
        key, key.stride(0),
        value, value.stride(0),
        k_cache, v_cache,
        slot_mapping,
        D
    )
```

### 3.1 매개변수 분석

| 매개변수 | 형태 | 의미 |
|--------|------|------|
| `key` | `[N, num_heads, head_dim]` | 현재 스텝에서 Attention으로 계산된 Key |
| `value` | `[N, num_heads, head_dim]` | 현재 스텝에서 Attention으로 계산된 Value |
| `k_cache` | `[total_slots, D]` | 전역 K Cache (모든 시퀀스 공유) |
| `v_cache` | `[total_slots, D]` | 전역 V Cache |
| `slot_mapping` | `[N]` | token → slot 매핑 |

### 3.2 그리드 설정

```python
store_kvcache_kernel[(N,)](...)
```

`[(N,)]`은 Triton 커널의 **그리드 크기**를 정의하며, CUDA의 `gridDim`에 해당합니다. 여기서는 1차원 그리드만 사용하며, N개의 프로그램이 N개의 토큰을 각각 처리합니다.

CUDA에서의 동등한 표기법은 대략:

```c++
store_kvcache_kernel<<<N, 1>>>(key, key_stride, value, value_stride, ...);
```

그러나 Triton에서 각 "프로그램"은 내부에서 `tl.arange`를 통해 여러 요소를 처리할 수 있으므로 CUDA처럼 명시적으로 `blockDim`을 지정할 필요가 없습니다.

### 3.3 `key.stride(0)`의 의미

`key.stride(0)`은 `key` 텐서의 0번 차원에 대한 **보폭**을 반환합니다(단위는 요소 수, 바이트가 아님).

`[N, num_heads, head_dim]` 형태의 연속 텐서의 경우:
- `stride(0) = num_heads * head_dim = D`
- `stride(1) = head_dim`
- `stride(2) = 1`

D를 하드코딩하지 않고 stride를 전달하면, 커널이 비연속 메모리 레이아웃(예: `key.permute(...)` 후의 텐서)도 처리할 수 있습니다.

### 3.4 암묵적 매개변수 전달

Triton은 PyTorch 텐서를 자동으로 GPU 포인터로 변환합니다.

- `key` → `key_ptr` (텐서의 `.data_ptr()`)
- `key.stride(0)` → `key_stride` (Python int, 커널 매개변수로 전달)
- `D` → 컴파일 타임 상수 (`tl.constexpr`)

---

## 4. Triton 프로그래밍 심화 개념

### 4.1 컴파일 및 캐싱 메커니즘

```
첫 호출 → Triton 컴파일러가 Python 커널 → LLVM IR → PTX → SASS 로 변환
           ↓
      컴파일 결과 캐싱 (~/.triton/cache/)
           ↓
후속 호출 → 캐시된 바이너리를 직접 로드
```

컴파일 트리거 조건(매개변수 특수화):
- `tl.constexpr` 매개변수 값 변경
- 입력 텐서의 dtype 변경
- 새로운 `num_warps`, `num_stages` 설정

### 4.2 메모리 접근 최적화

Triton 컴파일러는 다음을 자동으로 수행합니다.

1. **접근 병합(Coalescing)**: 여러 스레드가 연속된 주소에 접근할 때 하나의 메모리 트랜잭션으로 병합.
2. **벡터화(Vectorization)**: 여러 스칼라 load/store를 128-bit 너비의 벡터 연산으로 병합.
3. **레지스터 할당**: 어떤 중간값을 레지스터에 둘지 자동 결정.

개발자가 유의할 점:
- `tl.arange`로 생성되는 오프셋이 **연속적**이도록 하여 병합 접근을 용이하게 합니다.
- `D`를 2의 거듭제곱으로 정렬하면 컴파일러 최적화에 유리합니다.

### 4.3 CUDA Graph와의 협업

Triton 커널은 CUDA Graph와 완벽하게 호환됩니다. `capture_cudagraph` 진행 중:

1. 커널 첫 실행 시 JIT 컴파일 완료(warmup 단계).
2. capture 단계의 `torch.cuda.graph(...)`가 Triton 커널 호출을 녹화합니다.
3. replay 시 Triton 커널은 일반 CUDA 커널처럼 재생됩니다.

따라서 `store_kvcache_kernel`은 Graph capture 과정에서 일반 CUDA 커널과 동일하게 동작하며 특별한 처리가 필요하지 않습니다.

### 4.4 LLM 추론에서 Triton의 기타 응용

nano-vllm의 `store_kvcache_kernel`은 가장 단순한 Triton 커널에 불과합니다. 업계에서 Triton으로 구현하는 일반적인 LLM 관련 커널은 다음과 같습니다.

| 커널 | 기능 | 복잡도 |
|------|------|--------|
| Flash Attention | 융합된 QKV 어텐션 계산 | 높음 |
| RMSNorm | Root Mean Square 정규화 | 낮음 |
| Rotary Embedding | RoPE 위치 인코딩 | 중간 |
| SiLU + Mul | 게이트 활성화 함수 | 낮음 |
| FP8 행렬 곱셈 | 저정밀도 행렬 곱셈 | 높음 |

---

## 5. 실습: Triton 커널 실행 이해하기

### 5.1 실행 과정 시각화

3개의 토큰을 KV Cache에 기록해야 하고 `D=4`인 경우를 가정합니다.

```
입력:
  key = [[k00, k01, k02, k03],    # token 0
         [k10, k11, k12, k13],    # token 1
         [k20, k21, k22, k23]]    # token 2

  slot_mapping = [5, -1, 2]       # token 1은 패딩

Grid = (3,)  →  3개의 프로그램 기동

Program 0 (idx=0):
  slot = slot_mapping[0] = 5
  slot != -1 → 계속
  key_offsets = [0, 1, 2, 3]
  key = [k00, k01, k02, k03]
  cache_offsets = [20, 21, 22, 23]   # slot=5, 5*4=20
  k_cache[20:24]에 [k00, k01, k02, k03] 기록

Program 1 (idx=1):
  slot = slot_mapping[1] = -1
  slot == -1 → return (건너뜀)

Program 2 (idx=2):
  slot = slot_mapping[2] = 2
  slot != -1 → 계속
  key_offsets = [8, 9, 10, 11]
  key = [k20, k21, k22, k23]
  cache_offsets = [8, 9, 10, 11]     # slot=2, 2*4=8
  k_cache[8:12]에 [k20, k21, k22, k23] 기록
```

### 5.2 일반적인 디버깅 방법

1. **`triton.testing.do_bench`로 성능 측정**:
   ```python
   ms = triton.testing.do_bench(lambda: store_kvcache(key, value, k_cache, v_cache, slot_mapping))
   ```

2. **CPU 참조 구현으로 정확성 검증**:
   ```python
   for i in range(N):
       slot = slot_mapping[i].item()
       if slot != -1:
           assert torch.allclose(k_cache[slot], key[i].reshape(-1))
   ```

3. **컴파일된 PTX/SASS 확인**:
   ```python
   print(store_kvcache_kernel.cache[0].asm["ptx"])
   ```

---

## 6. 요약

- **Triton**은 Python 문법으로 GPU 커널을 작성하여 CUDA C++보다 개발 효율이 높고 성능은 비슷합니다.
- 핵심 사고 전환: CUDA의 **스레드 레벨 병렬**에서 Triton의 **블록 레벨 병렬**로.
- `store_kvcache_kernel`은 전형적인 **element-wise scatter 커널**입니다. 각 프로그램이 한 토큰의 K/V를 읽어 `slot_mapping`에 따라 KV Cache의 목표 위치에 기록합니다.
- `tl.constexpr`로 `D`를 컴파일 시점에 고정하면 컴파일러가 루프 전개를 할 수 있습니다.
- `slot == -1` 센티널 값 설계는 CUDA Graph의 `fill_(-1)`과 협력하여 패딩 토큰이 KV Cache를 오염시키지 않도록 합니다.
- `key.stride(0)`로 stride를 전달하여 커널이 비연속 메모리에도 대응할 수 있습니다.

---

## 7. 면접 예상 문제 (모범 답안 포함)

**1. Triton과 CUDA 프로그래밍의 핵심적인 차이는 무엇인가요?**  
**답변**: 가장 핵심적인 차이는 **병렬 처리 단위**입니다. CUDA는 스레드 레벨(thread-level) 프로그래밍으로 개발자가 각 스레드의 동작, 공유 메모리, 스레드 동기화 등을 수동으로 관리해야 합니다. Triton은 블록 레벨(block-level) 프로그래밍으로, 개발자는 각 프로그램이 어떤 데이터 블록을 처리할지만 정의하면 스레드 할당, 공유 메모리 관리, 메모리 병합을 컴파일러가 자동으로 수행합니다. Triton은 Python 문법을 사용하여 CUDA C++보다 개발 효율이 크게 높습니다.

**2. `@triton.jit`의 역할은 무엇인가요?**  
**답변**: 함수를 JIT 컴파일 GPU 커널로 표시합니다. 첫 호출 시 Triton 컴파일러가 Python 코드를 GPU 머신 코드로 컴파일(LLVM IR → PTX → SASS 파이프라인)하며, 컴파일 결과는 캐시됩니다. CUDA의 `__global__` 키워드와 유사하지만 타입 추론과 컴파일 타임 특수화를 지원합니다.

**3. `tl.program_id(0)`은 CUDA의 무엇에 해당하나요?**  
**답변**: CUDA의 `blockIdx.x`에 해당하며, 그리드의 0번 차원에서 현재 실행 유닛의 인덱스를 반환합니다. 하지만 CUDA와 달리 Triton의 "프로그램"은 내부에서 `tl.arange` 등의 연산으로 여러 요소를 처리하므로, 하나의 Triton 프로그램은 기능적으로 하나의 CUDA 블록에 더 가깝습니다.

**4. `store_kvcache_kernel`에서 `D`를 `tl.constexpr`로 선언한 이유는 무엇인가요?**  
**답변**: `tl.constexpr`은 `D`를 **컴파일 시점에 고정**시켜 컴파일러가 **루프 전개, 벡터화 명령어 선택** 등 최적화를 수행할 수 있게 합니다. D가 런타임 변수라면 컴파일러가 루프 횟수를 확정할 수 없어 생성 코드의 효율이 떨어집니다. 그 대가로 D가 변경될 때는 재컴파일이 필요합니다.

**5. `slot_mapping`에서 `-1`의 역할은 무엇인가요?**  
**답변**: `-1`은 센티널 값으로, 해당 위치가 패딩 토큰(무효)임을 나타냅니다. 커널은 `slot == -1`을 감지하면 즉시 `return`하여 KV Cache에 기록하지 않습니다. 이는 CUDA Graph에서 `graph_vars["slot_mapping"].fill_(-1)`로 남는 자리를 `-1`로 표시하여 KV Cache에 쓰레기 데이터가 들어가는 것을 방지하는 설계와 일치합니다.

**6. `store_kvcache` 래퍼 함수에서 `[(N,)]`의 의미는 무엇인가요?**  
**답변**: Triton 커널의 **그리드 크기**를 정의하며, CUDA의 `gridDim`에 해당합니다. `(N,)`은 1차원 그리드로, N개의 프로그램을 시작하여 각 프로그램이 토큰 하나씩 처리하도록 합니다. Triton에서는 그리드를 대괄호 `[]` 안에 커널 함수 이름 뒤에 붙여 표기합니다.

**7. `D`를 직접 사용하지 않고 `key.stride(0)`을 전달하는 이유는 무엇인가요?**  
**답변**: `stride`를 사용하면 커널이 **비연속(non-contiguous)** 텐서도 처리할 수 있기 때문입니다. 연속 텐서에서는 `stride(0) == D`이지만, `transpose`, `permute` 등을 거친 텐서는 stride와 D가 다를 수 있습니다. stride를 전달하는 것이 더 안전하고 범용적인 방법입니다.

**8. Triton 커널의 JIT 컴파일은 어떤 장단점이 있나요?**  
**답변**:
- **장점**: 구체적인 `constexpr` 매개변수, 데이터 타입에 맞춰 특화된 코드를 생성하여 성능이 더 우수하며, Python처럼 작성하여 개발이 편리합니다.
- **단점**: 첫 호출 시 컴파일 오버헤드(보통 수백 ms~수 초)가 있고, 매개변수 조합별로 각각 컴파일이 필요하며, PyTorch 네이티브 코드보다 디버깅이 어렵습니다.
- 추론 시나리오에서는 컴파일 오버헤드가 일회성이며 캐싱되므로 JIT를 용인할 수 있습니다.

**9. 이 커널의 성능 병목은 어디에 있나요?**  
**답변**: 주된 병목은 **전역 메모리 대역폭(Global Memory Bandwidth)**입니다. 커널의 계산량은 극히 적고(포인터 연산과 비교만), 데이터량은 K와 V 각각에 대해 `2 * N * D`개 요소의 읽기/쓰기입니다. 최적화 방향으로는 메모리 접근 병합 보장, D를 적절한 단위로 정렬하는 것 등이 있습니다. LLM 추론 전체로 보면 이 커널은 보통 병목이 되지 않습니다. Attention과 FFN의 행렬 곱셈이 훨씬 더 많은 시간을 소모하기 때문입니다.

**10. LLM 추론에서 Triton 생태계의 위상은 어떤가요?**  
**답변**: Triton은 LLM 추론 시스템에서 커스텀 커널을 작성하는 주류 도구가 되었습니다. Flash Attention 2/3에 Triton 구현이 있고, vLLM의 여러 핵심 커널(예: PagedAttention)이 Triton을 사용하며, PyTorch의 `torch.compile` 백엔드인 Inductor도 Triton 코드를 생성합니다. CUDA C++에 비해 Triton은 개발 반복 속도가 빠르고 PyTorch 생태계와의 통합이 긴밀합니다.

---

*참고 자료: Triton 공식 튜토리얼(triton-lang.org), OpenAI Triton 논문(MAPL 2019).*