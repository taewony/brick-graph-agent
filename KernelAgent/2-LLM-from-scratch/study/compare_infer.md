## `compare_infer.py` 설계 가이드: PyTorch vs cuTile 추론 성능 비교

### 1. 비교 목표 및 측정 지표
**목표**: 동일한 언어 모델 아키텍처에 대해, **PyTorch 네이티브 Attention**을 사용한 추론과 **cuTile 기반 Attention**(`AttentionFMHA.py` 활용)을 사용한 추론의 **엔드-투-엔드 성능**을 공정하게 비교한다.

**측정 지표**:
- **TTFT (Time To First Token)**: 프롬프트 입력 후 첫 번째 토큰이 생성되기까지의 시간 (Prefill 시간 포함)
- **응답 시간 (Total Response Time)**: 프롬프트 입력부터 마지막 토큰 출력까지 전체 소요 시간
- **토큰당 평균 생성 속도 (tokens/sec)**: (총 생성 토큰 수) / (응답 시간 - TTFT)

**공정성 조건**:
- 동일한 모델 가중치 (한쪽에서 학습 후 복사)
- 동일한 입력 프롬프트, 동일한 생성 파라미터 (greedy decoding)
- KV Cache 적용 (양쪽 모두)
- Warm-up 후 측정, GPU 동기화 포함

---

### 2. 모델 아키텍처 가정
비교를 위해 간단한 **GPT-2 스타일의 디코더 전용 Transformer**를 가정한다.
- `n_layer`: 6
- `n_head`: 12
- `d_model`: 768
- `head_dim`: 64
- `max_seq_len`: 1024 (비교 테스트용)
- 어휘 크기: 50257 (GPT-2 tokenizer 기준)

모델은 `nn.Module`로 구현하며, Attention 모듈만 교체 가능한 구조로 설계한다.

---

### 3. PyTorch 구현 (Baseline)
- Attention에 `torch.nn.functional.scaled_dot_product_attention` 사용 (내부적으로 FlashAttention 커널 활용)
- KV Cache를 `past_key_values` 리스트 형태로 관리 (HuggingFace 스타일)
- Prefill 시 프롬프트 전체를 한 번에 처리, 이후 디코딩 루프에서 새 토큰만 입력
- 나머지 레이어 (LayerNorm, Linear, FFN)는 PyTorch 그대로 사용

**장점**: PyTorch의 최적화된 백엔드를 그대로 이용, 구현 간결

---

### 4. cuTile 구현 (실험군)
- Attention 부분만 cuTile 커널(`fused_mha_kernel`)로 대체
- `AttentionFMHA.py`의 커널을 KV Cache를 지원하도록 확장:
  - Prefill: Q, K, V가 모두 긴 시퀀스 → FMHA 타일링으로 처리하고 K, V를 캐시에 저장
  - Decode: Q 길이 1 → K, V 캐시와 함께 **GEMV 스타일의 FlashDecoding**으로 처리 (또는 단순 FMHA로 전체 캐시를 재계산할 수도 있으나, 이 경우 공정하지 않음)
- 나머지 연산(Linear, LayerNorm 등)은 PyTorch 그대로 사용 → **오직 Attention 커널만 교체**
- `ct.launch` 호출로 인한 Python 오버헤드를 피하기 위해, 선택적으로 **CUDA Graph**를 디코딩 루프에 적용할 수 있지만, 기본 비교에는 적용하지 않고 raw 호출로 측정 (나중에 CUDA Graph 버전과도 비교 가능)

**주의**: cuTile 커널은 `AttentionFMHA.py`에서 배치 차원과 헤드 차원을 평탄화하여 호출하므로, 헤드 차원을 분리하지 않고 `[batch*heads, seq, dim]` 형태로 전달한다. 따라서 KV Cache도 같은 형태로 관리해야 한다.

---

### 5. `compare_infer.py` Top-Down 설계

#### 5.1 스크립트 구조
```
compare_infer.py
├── 공통 모델 클래스 정의 (Attention 모듈만 갈아끼우도록)
│   ├── PyTorchAttention (nn.Module)
│   └── CuTileAttention (nn.Module)  # ct.launch 호출
├── 모델 생성 및 가중치 복사 함수
│   ├── build_model(config) → PyTorch 모델
│   └── build_cutile_model(pytorch_model) → CuTile 모델 (가중치 복사)
├── 생성 루프 함수 (공통 인터페이스)
│   └── generate(model, input_ids, max_new_tokens, ...)
├── 성능 측정 루틴
│   └── benchmark(model, prompt, num_runs)
└── 메인: 두 모델 로드 → 측정 → 결과 출력
```

#### 5.2 Attention 모듈 설계 상세

**PyTorchAttention**:
```python
class PyTorchAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.qkv = nn.Linear(d_model, 3*d_model)
        self.out = nn.Linear(d_model, d_model)
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

    def forward(self, x, attn_mask=None, past_kv=None, use_cache=False):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, nh, T, hd]
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        present_kv = (k, v) if use_cache else None
        # FlashAttention 활용
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1,2).contiguous().view(B, T, C)
        return self.out(out), present_kv
```

**CuTileAttention**:
- Prefill/Decode 분기
- Prefill: `(B*nh, T, head_dim)` 형태로 FMHA 커널 호출, K/V 캐시에 저장
- Decode: Q는 `[B*nh, 1, head_dim]`, K/V 캐시는 `[B*nh, cache_len, head_dim]`. FlashDecoding 방식으로 커널을 작성하거나, 간단히 FMHA 커널을 전체 캐시에 대해 호출 (비효율적이지만 구현 단순). **공정성을 위해 Prefill 시 FMHA, Decode 시 GEMV 커널을 별도 구현하는 것이 좋다**. 여기서는 cuTile로 GEMV 커널을 미리 준비했다고 가정한다.
- `ct.launch` 호출: 커널 이름, 인자 (포인터, shape, strides 등), 스트림 등 전달. 인자는 `ct.Array`로 래핑.
- 예시:
```python
class CuTileAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.qkv = nn.Linear(d_model, 3*d_model)  # 동일
        self.out = nn.Linear(d_model, d_model)
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        # cuTile 커널 모듈 로드 (미리 컴파일된 것을 사용)
        self.prefill_kernel = ct.Kernel("fused_mha_kernel")
        self.decode_kernel = ct.Kernel("fused_mha_decode_kernel")  # 별도 GEMV 커널

    def forward(self, x, attn_mask=None, past_kv=None, use_cache=False):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, nh, T, hd]
        # cuTile 커널은 [B*nh, seq, hd] 형태를 요구
        q_ = q.reshape(B*self.n_heads, T, self.head_dim).contiguous()
        k_ = k.reshape(B*self.n_heads, T, self.head_dim).contiguous()
        v_ = v.reshape(B*self.n_heads, T, self.head_dim).contiguous()
        if past_kv is not None:
            # 캐시와 결합
            k_ = torch.cat([past_kv[0], k_], dim=1)
            v_ = torch.cat([past_kv[1], v_], dim=1)
        present_kv = (k_, v_) if use_cache else None
        # 출력 텐서 할당
        out = torch.empty_like(q_)
        if T == 1:  # Decode
            self.decode_kernel.launch(q_, k_, v_, out, ...)  # 실제 인자
        else:       # Prefill
            self.prefill_kernel.launch(q_, k_, v_, out, ...)
        out = out.reshape(B, self.n_heads, T, self.head_dim).transpose(1,2).contiguous().view(B, T, C)
        return self.out(out), present_kv
```
- 실제 `ct.launch` 호출은 매우 저수준이며, 커널 파라미터를 정확히 맞춰야 한다. `AttentionFMHA.py`의 `run_fmha` 함수를 참고한다.

#### 5.3 가중치 복사
PyTorch 모델을 먼저 생성하고, `.state_dict()`를 얻은 후 CuTile 모델에 `load_state_dict`로 복사한다. QKV 가중치, 출력 가중치 등 모든 파라미터가 동일하게 설정된다.

#### 5.4 생성 루프 (Greedy Decoding)
```python
@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=50):
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([input_ids], dtype=torch.long, device='cuda')
    past_key_values = None
    generated = []
    # Prefill
    logits, past_key_values = model(input_ids, past_key_values=past_key_values, use_cache=True)
    next_token = logits[:, -1, :].argmax(dim=-1)
    generated.append(next_token.item())
    # Decode loop
    for _ in range(max_new_tokens-1):
        logits, past_key_values = model(next_token.unsqueeze(1), past_key_values=past_key_values, use_cache=True)
        next_token = logits[:, -1, :].argmax(dim=-1)
        generated.append(next_token.item())
    return generated
```
두 모델에서 동일한 토큰 시퀀스가 생성되는지 확인한다.

#### 5.5 성능 측정
```python
def measure(model, tokenizer, prompt, max_new_tokens=50, warmup=5, repeats=20):
    # 웜업
    for _ in range(warmup):
        generate(model, tokenizer, prompt, max_new_tokens)
    torch.cuda.synchronize()
    times_ttft = []
    times_total = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        # 측정용 generate 수정: TTFT 기록을 위해 첫 토큰 생성 시점에 중간 시간 기록
        # 여기서는 별도 함수로 구현
        ttft, total_time = timed_generate(model, tokenizer, prompt, max_new_tokens)
        times_ttft.append(ttft)
        times_total.append(total_time)
    return np.mean(times_ttft), np.mean(times_total)
```

`timed_generate` 함수:
- 입력 전달 전 동기화 후 시작 시간 기록
- 첫 토큰을 얻는 즉시 중간 시간 기록 (`ttft = time.perf_counter() - start`)
- 전체 루프 완료 후 최종 시간 기록
- GPU 동기화는 각 시간 측정 직전에 수행

#### 5.6 결과 출력
- TTFT 평균 ± 표준편차
- 총 응답 시간 평균 ± 표준편차
- 생성된 토큰 일치 여부
- 필요 시 프로파일링 (torch.profiler) 추가

---

### 6. 공정성 확보를 위한 주의사항

1. **커널 런치 오버헤드**: cuTile의 `ct.launch`는 Python 호출로 인한 지연이 존재한다. PyTorch의 `sdpa`는 C++ 수준에서 디스패치되므로 유리하다. 이 차이는 실제 사용 시나리오의 일부이므로 그대로 측정하되, CUDA Graph 적용 시의 결과도 추가로 비교할 수 있다.
2. **워크로드 크기**: 시퀀스 길이가 짧을 경우 Python 오버헤드가 지배적일 수 있으므로, 512, 1024 등 다양한 프롬프트 길이로 실험한다.
3. **GPU 점유율**: 작은 배치에서는 cuTile 커널이 SM을 충분히 활용하지 못할 수 있다. 이는 커널 설계 문제이므로, 추후 FlashDecoding의 split-K 등을 도입할 수 있다.
4. **메모리 할당**: 측정 중 텐서 할당이 발생하면 편차가 생기므로, KV Cache 등 필요한 버퍼를 미리 할당해둔다.
5. **데이터 정확도**: `torch.allclose`로 최종 로짓이 허용 오차 내에서 일치하는지 확인한다. float16 연산에서는 약간의 차이가 발생할 수 있다.

---

### 7. 예상 코드 스켈레톤

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from transformers import GPT2Tokenizer
import cutile as ct  # 가상의 import

# ================= 모델 정의 =================
class GPTBlock(nn.Module):
    def __init__(self, config, attn_type='pytorch'):
        ...
        if attn_type == 'pytorch':
            self.attn = PyTorchAttention(config.d_model, config.n_head)
        else:
            self.attn = CuTileAttention(config.d_model, config.n_head)
        ...

    def forward(self, x, past_kv=None, use_cache=False):
        normed = self.ln1(x)
        attn_out, present_kv = self.attn(normed, past_kv=past_kv, use_cache=use_cache)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, present_kv

class GPTModel(nn.Module):
    def __init__(self, config, attn_type='pytorch'):
        ...
        self.blocks = nn.ModuleList([GPTBlock(config, attn_type) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, input_ids, past_key_values=None, use_cache=False):
        x = self.embed(input_ids)
        new_past = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            past_kv = past_key_values[i] if past_key_values is not None else None
            x, present_kv = block(x, past_kv, use_cache)
            if use_cache:
                new_past.append(present_kv)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, new_past

# ================= 가중치 복사 =================
def copy_weights(src_model, dst_model):
    dst_model.load_state_dict(src_model.state_dict())

# ================= 생성 및 측정 =================
def timed_generate(model, tokenizer, prompt, max_new=50):
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([input_ids], device='cuda')
    past_kv = None
    generated = []
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    # Prefill
    logits, past_kv = model(input_ids, past_kv, use_cache=True)
    next_token = logits[:, -1, :].argmax(dim=-1)
    generated.append(next_token.item())
    t_first = time.perf_counter()
    # Decode
    for _ in range(max_new-1):
        logits, past_kv = model(next_token.unsqueeze(1), past_kv, use_cache=True)
        next_token = logits[:, -1, :].argmax(dim=-1)
        generated.append(next_token.item())
    torch.cuda.synchronize()
    t_end = time.perf_counter()
    return t_first - t_start, t_end - t_start, generated

def benchmark():
    config = GPTConfig()
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    prompt = "Once upon a time"  # 짧은 프롬프트, 혹은 긴 프롬프트로 실험

    # PyTorch 모델
    model_pt = GPTModel(config, attn_type='pytorch').cuda().half()
    # cuTile 모델
    model_ct = GPTModel(config, attn_type='cutile').cuda().half()
    copy_weights(model_pt, model_ct)

    # 웜업
    for _ in range(5):
        generate(model_pt, tokenizer, prompt, 50)
        generate(model_ct, tokenizer, prompt, 50)

    # 측정
    ttft_pt, total_pt, _ = timed_generate(model_pt, tokenizer, prompt, 50)
    ttft_ct, total_ct, _ = timed_generate(model_ct, tokenizer, prompt, 50)

    print(f"PyTorch - TTFT: {ttft_pt*1000:.2f} ms, Total: {total_pt*1000:.2f} ms")
    print(f"cuTile  - TTFT: {ttft_ct*1000:.2f} ms, Total: {total_ct*1000:.2f} ms")
```

---

### 8. 확장 고려사항
- 더 정밀한 측정을 위해 여러 반복 평균
- `torch.cuda.Event`를 사용한 정밀 타이밍
- cuTile 커널에 CUDA Graph 적용 후 비교 (추가 실험)
- 프롬프트 길이별 성능 곡선 그래프 출력

---

### 9. 결론
이 설계를 따르면, 두 구현의 성능을 **동일한 조건에서 엄밀하게 비교**할 수 있다. `compare_infer.py`는 모델 가중치를 공유하고, 동일한 생성 로직을 사용하므로, 오직 Attention 커널의 차이만 성능에 반영된다. 이를 통해 cuTile 이식의 효과와 추가 최적화 필요 영역을 정량적으로 평가할 수 있다.