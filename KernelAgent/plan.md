# micro-vLLM Tier 1 & Tier 2 성능 개선 실행 계획 (plan.md)

> 대상: `KernelAgent/3-micro-vllm/` (Windows-native cuTile 기반 micro-vLLM)
> 목적: 호스트 오버헤드(Tier 1)와 커널 퓨전/개선(Tier 2)을 적용해 **eager-hybrid 디코드 처리량(현재 557.66 tok/s)** 을 끌어올리고, 그 결과를 벤치마크로 정량화한다.
> 참고: 진단 근거는 `KernelAgent/perf_improve_plan.md` (병목 ①~⑦)에 정리되어 있다.

---

## 0. 사전 준비 & 기준선(BaseLine) 측정

### 0-1. 환경 확인
- GPU: NVIDIA GeForce RTX 5070 (Blackwell), Windows 11
- Python 환경에 다음이 설치되어 있어야 한다 (README 참고):
  ```powershell
  pip install torch            # CUDA 지원 PyTorch (논문 기준 PyTorch 2.13.0+cu130 / CUDA 13.0)
  pip install tilegym[tileiras]   # cuTile (cuda.tile)
  pip install transformers huggingface_hub
  ```
- 모델 경로: `~/huggingface/Qwen3-0.6B/`
  - ⚠️ `README.md` 138~143행에 따르면 이 폴더는 `Qwen2.5-3B-Instruct`를 **이름만 변경**해 둔 것일 수 있음. 논문 표기("Qwen3-0.6B")와 실제 폴더 내용을 먼저 확인:
    ```powershell
    python src/inspect_model.py $env:USERPROFILE\huggingface\Qwen3-0.6B
    ```
    → 출력의 `Architecture`/`Hidden Size`/`Layers`가 논문의 주장과 일치하는지 확인 후 진행.

### 0-2. 기준선 측정 (변경 전 성능 기록)
작업 디렉토리를 `KernelAgent/3-micro-vllm`으로 두고, 아래 3개 벤치를 **개선 전에 1회 이상** 돌려 기준 수치를 남긴다.

```powershell
cd KernelAgent/3-micro-vllm

# (a) 주 처리량 벤치 (eager-hybrid cuTile)  →  기준: 약 557.66 tok/s
$env:NANO_VLLM_USE_CUTILE="1"
$env:NANO_VLLM_CUTILE_PREFILL_STRATEGY="hybrid"
python bench.py --use-cutile --cutile-prefill-strategy hybrid

# (b) 프리픽스 캐시 TTFT 벤치  →  기준: TTFT 감소 41.1/66.1/79.7%
python bench_prefix_cache.py --use-cutile --requests 8 --static-prefix-tokens 2048 --dynamic-suffix-tokens 64 --max-tokens 64 --out-jsonl baseline_prefix.jsonl

# (c) 그린 컨텍스트 스트레스 벤치 (선택)  →  기준: decode gap P99 등
python bench_green_stress.py --repeats 10 --green-api cuda_core --prefill-sms 32 --decode-sms 16
```

결과는 `test-result-*.md` 패턴으로 별도 기록해 두면 논문 재현성에 좋다.

---

## 1. Tier 1 — 순수 호스트 최적화 (커널/구조 불변, 교육성 100% 보존)

**핵심**: 매 스텝 반복되는 텐서 할당 + `pin_memory` + H2D 전송을 **영속 버퍼 재사용**으로 제거한다.

### 1-1. 수정 파일

| 파일 | 수정 대상 함수 | 변경 내용 |
|---|---|---|
| `nanovllm/engine/model_runner.py` | `__init__` | `allocate_kv_cache()` 뒤에 영속 CUDA 버퍼 + pinned CPU 스테이징 버퍼를 preallocate |
| `nanovllm/engine/model_runner.py` | `prepare_prefill` | `torch.tensor(...).pin_memory().cuda(non_blocking=True)` → CPU 스테이징 채우고 CUDA 버퍼에 `copy_(non_blocking=True)` |
| `nanovllm/engine/model_runner.py` | `prepare_decode` | 동일하게 영속 버퍼 재사용 |
| `nanovllm/engine/model_runner.py` | `prepare_sample` | temperatures를 영속 버퍼로 |
| `nanovllm/engine/model_runner.py` | `prepare_block_tables` | 매 스텝 새 텐서 생성 → 재사용 버퍼에 `fill_(-1)` 후 직접 기록, 슬라이스 반환 |

### 1-2. 구체 변경 명세

1. **`__init__`** 에서 (모델/캐시 할당 이후) 다음 버퍼 추가:
   ```python
   self._buf_input_ids   = torch.empty(config.max_num_batched_tokens, dtype=torch.int64, device="cuda")
   self._buf_positions   = torch.empty(config.max_num_batched_tokens, dtype=torch.int64, device="cuda")
   self._buf_slot_mapping= torch.empty(config.max_num_batched_tokens, dtype=torch.int32, device="cuda")
   self._buf_context_lens= torch.empty(config.max_num_seqs,          dtype=torch.int32, device="cuda")
   self._buf_block_tables= torch.empty(config.max_num_seqs, self._max_num_blocks, dtype=torch.int32, device="cuda")
   self._buf_temperatures= torch.empty(config.max_num_seqs,          dtype=torch.float32, device="cuda")
   # pinned CPU 스테이징 (H2D 비동기 복사용)
   self._cpu_input_ids    = torch.empty(config.max_num_batched_tokens, dtype=torch.int64, pin_memory=True)
   self._cpu_positions    = torch.empty(config.max_num_batched_tokens, dtype=torch.int64, pin_memory=True)
   self._cpu_slot_mapping = torch.empty(config.max_num_batched_tokens, dtype=torch.int32, pin_memory=True)
   self._cpu_context_lens = torch.empty(config.max_num_seqs,          dtype=torch.int32, pin_memory=True)
   self._cpu_temperatures = torch.empty(config.max_num_seqs,          dtype=torch.float32, pin_memory=True)
   ```
   (`_max_num_blocks = (max_model_len + block_size - 1) // block_size`)

2. **`prepare_decode`** 는 Python 리스트를 CPU 스테이징 버퍼로 옮긴 뒤 CUDA 버퍼로 복사:
   ```python
   n = len(seqs)
   self._cpu_input_ids[:n] = torch.tensor(input_ids, dtype=torch.int64)
   self._cpu_positions[:n] = torch.tensor(positions, dtype=torch.int64)
   self._cpu_slot_mapping[:n] = torch.tensor(slot_mapping, dtype=torch.int32)
   self._cpu_context_lens[:n] = torch.tensor(context_lens, dtype=torch.int32)
   self._buf_input_ids[:n].copy_(self._cpu_input_ids[:n], non_blocking=True)
   self._buf_positions[:n].copy_(self._cpu_positions[:n], non_blocking=True)
   self._buf_slot_mapping[:n].copy_(self._cpu_slot_mapping[:n], non_blocking=True)
   self._buf_context_lens[:n].copy_(self._cpu_context_lens[:n], non_blocking=True)
   block_tables = self.prepare_block_tables(seqs)   # 재사용 버퍼 반환
   set_context(False, slot_mapping=self._buf_slot_mapping[:n], context_lens=self._buf_context_lens[:n],
               block_tables=block_tables, use_cutile=self.use_cutile)
   return self._buf_input_ids[:n], self._buf_positions[:n]
   ```
   `prepare_prefill` 도 같은 패턴 (최대 `num_batched_tokens` 슬라이스 사용).

3. **`prepare_block_tables`**:
   ```python
   max_len = max(len(seq.block_table) for seq in seqs)
   self._buf_block_tables.fill_(-1)          # 전체 -1로 초기화
   for i, seq in enumerate(seqs):
       self._buf_block_tables[i, :len(seq.block_table)] = torch.tensor(seq.block_table, dtype=torch.int32)
   return self._buf_block_tables[:len(seqs), :max_len]
   ```
   (블록 테이블 쓰기는 CPU 텐서로 만들어 한 번에 기록하거나, numpy 경유로 더 빠르게 가능)

4. **`prepare_sample`**:
   ```python
   n = len(seqs)
   self._cpu_temperatures[:n] = torch.tensor([s.temperature for s in seqs], dtype=torch.float32)
   self._buf_temperatures[:n].copy_(self._cpu_temperatures[:n], non_blocking=True)
   return self._buf_temperatures[:n]
   ```

### 1-3. 검증 & 측정

```powershell
# 정합성: 레이어/마이그레이션 테스트
python src/tests/test_migration.py
python src/tests/test_layers.py

# 텍스트 정상 생성 확인 (출력 품질 회귀 체크)
$env:NANO_VLLM_USE_CUTILE="1"
python example.py

# 성능 측정 (기준선과 동일 조건)
python bench.py --use-cutile --cutile-prefill-strategy hybrid
```

**기대 효과**: 디코드 스텝당 호스트 준비 시간(할당+H2D)이 크게 감소 → 디코드 처리량 상승. 목표: 557.66 → **700 tok/s 이상**.

---

## 2. Tier 2 — 경량 커널 퓨전/개선 (모듈러 유지, 치명 구간만)

### 2-1. `store_kvcache` 프리필 어텐션 융합

**수정 파일**: `nanovllm/layers/attention.py`, `nanovllm/layers/cutile_attention.py`

- **현재**: `Attention.forward`(146~147행)가 어텐션 **앞에서** 별도 커널 `store_kvcache_cutile_kernel`(attention.py 26~41)을 호출. K/V를 캐시에 쓰고 → 어텐션 커널이 다시 읽음(메모리 트래픽 2배 + 커널 1개 추가). 게다가 `grid=(N,1,1)`(토큰당 블록 1개, 행 폭 D 전체 복사)로 점유율·타일 효율이 낮음.
- **변경 (2단계)**:
  1. **즉효**: `store_kvcache_cutile_kernel`을 2D 그리드(`(num_heads, N)` 또는 토큰 청크 분할) + 벡터화 로드로 개선. (커널 1개 유지, 단일 커널 효율 개선)
  2. **완전 융합**: non-paged 프리필 경로(`cutile_fmha_prefill` → `fmha_prefill_kernel`)에서 K/V 타일을 로드할 때 **동시에 `k_cache/v_cache`에 `ct.store`**. 이러면 store 커널이 사라지고 K/V 읽기가 1회로 줄어든다.
- **주의**: prefix-cache(paged) 경로는 기존 캐시 K/V를 읽어야 하므로 융합 대상은 **새로 계산되는 접미사(suffix) 토큰**에 한정. 기존 `fmha_prefill_paged_kernel`의 로직과 병행 설계 필요.

### 2-2. RMSNorm(q/k) + RoPE 융합

**수정 파일**: `nanovllm/layers/rotary_embedding.py`(+ `nanovllm/models/qwen3.py`, 필요 시 `cutile_attention.py`에 커널 추가)

- **현재**: `Qwen3Attention.forward`(qwen3.py 81~84)에서 `q_norm`/`k_norm`(RMSNorm) → `rotary_emb`(RoPE)가 각각 별도 PyTorch op. RoPE(`apply_rotary_emb`, rotary_embedding.py 6~14)는 `x.float()` 업캐스트 + chunk + 곱셈 4회 + cat + downcast로 op 수가 많음.
- **변경**: `q/k`에 대해 **RMSNorm+RoPE를 하나의 cuTile 커널**로 융합. 입력 `q,k`(view 전 텐서)를 로드 → RMSNorm → 회전(rotary) → 원본 버퍼에 store. `Qwen3Attention.forward`에서 `q_norm/k_norm + rotary_emb` 호출을 단일 호출로 교체.
- **효과**: 레이어당 커널 4~6개 감소, WDDM 발사 지연 직접 감소. (성능분석.md의 "RMSNorm+RoPE 부분 융합" 방안과 동일)

### 2-3. 디코드 어텐션 split-KV / GQA K·V 재사용

**수정 파일**: `nanovllm/layers/cutile_attention.py` (`paged_decode_kernel`, 202~268)

- **현재**: (seq, head)당 블록 1개가 KV 블록을 순회하며 `q(1,D) × K(D,256)` GEMV 수행. 단일 쿼리 토큰이라 텐서코어 활용이 낮고 HBM 대역폭에 묶임. GQA에서 **동일 K/V를 여러 쿼리 헤드가 중복 로드**.
- **변경**:
  1. **split-KV**: 한 (seq, head)의 KV 블록들을 여러 스레드블록에 나눠 부분 합산 후 reduce(atomic 또는 별도 reduce 커널). 대역폭 활용 개선.
  2. **GQA 재사용**: 같은 KV 헤드를 공유하는 Q 헤드들을 한 블록에서 묶어 K/V를 1회 로드 → 여러 Q 헤드에 재사용.
- **주의**: 이 변경은 디코드가 **메모리 바운드**라는 전제 하에 효과가 큼. 우선 순위는 2-1/2-2보다 낮을 수 있음.

### 2-4. padded 프리필 제거 → varlen 단일 커널

**수정 파일**: `nanovllm/layers/cutile_attention.py` (`_cutile_fmha_prefill_padded`, 279~345 / `cutile_fmha_prefill`, 347~423)

- **현재**: `strategy="hybrid"` + batch>1 이면 `_cutile_fmha_prefill_padded`가 **배치마다 `.item()` 호출 + 64배수 패딩 버퍼로 호스트 루프 복사**(307~326) 후 커널 발사. 패딩 연산 낭비 + 배치 직렬화.
- **변경**: `cu_seqlens_q/k`를 직접 읽는 **varlen 단일 커널**(모든 배치를 한 번의 launch로)로 대체. `fmha_prefill_kernel`의 batch 인덱싱(`bid_y`)을 이미 활용 중이므로, `_bhtd_view` 방식의 per-batch Python 루프(392~421)도 제거하고 단일 launch로 통합.
- **효과**: 프리필 경로의 호스트 직렬화 제거, 패딩 제거로 연산량 감소.

### 2-5. 검증 & 측정

```powershell
# 정합성 (각 변경 후 반드시)
python src/tests/test_migration.py
python src/tests/test_layers.py
$env:NANO_VLLM_USE_CUTILE="1"
python example.py

# 프리필 전략별 성능 비교 (Tier 2-1/2-4 검증)
python bench.py --use-cutile --cutile-prefill-strategy direct
python bench.py --use-cutile --cutile-prefill-strategy hybrid
python bench.py --use-cutile --cutile-prefill-strategy padded

# 프리픽스 캐시 TTFT (Tier 2가 프리필에 미치는 영향 확인)
python bench_prefix_cache.py --use-cutile --requests 8 --static-prefix-tokens 2048 --dynamic-suffix-tokens 64 --max-tokens 64 --out-jsonl tier2_prefix.jsonl
```

---

## 3. 벤치마크 실행 요약 (측정 방법)

| 벤치 스크립트 | 측정 대상 | 핵심 인자 | 출력 지표 |
|---|---|---|---|
| `bench.py` | 다중 사용자 처리량(디코드) | `--use-cutile` `--cutile-prefill-strategy {hybrid,direct,padded}` `--cutile-cudagraph` `--enforce-eager` | `Throughput: XXX tok/s` |
| `bench_prefix_cache.py` | 프리픽스 캐시 TTFT/히트율 | `--use-cutile` `--requests` `--static-prefix-tokens` `--dynamic-suffix-tokens` `--max-tokens` `--out-jsonl` | `mean_ttft_ms`, `cache_hit_ratio`, `throughput_tok_s` |
| `bench_green_stress.py` | 그린 컨텍스트 지연 평활화 | `--repeats` `--green-api {auto,pytorch,cuda_core}` `--prefill-sms` `--decode-sms` | `decode_gap_p99_ms`, `throughput`, `improved_runs` |

### 공통 환경변수
```powershell
$env:NANO_VLLM_USE_CUTILE="1"                    # cuTile 백엔드 활성화
$env:NANO_VLLM_CUTILE_PREFILL_STRATEGY="hybrid"  # 프리필 전략
$env:NANO_VLLM_USE_GREEN_CONTEXTS="1"            # (그린 벤치 전용)
$env:NANO_VLLM_GREEN_CONTEXT_API="cuda_core"     # (그린 벤치 전용)
```

### 측정 시 준수 사항
1. **동일 조건 반복**: 각 수치는 최소 2~3회 반복해 중앙값/평균 기록 (벤치 내부 warmup 존재).
2. **변경 전/후 동일 파라미터**: `--cutile-prefill-strategy` 등 동일하게 유지.
3. **결과 기록**: 각 실행 결과를 `test-result-<단계>.md` 로 저장해 논문 재현성 확보.
4. **GPU 상태 확인**: 측정 전 `nvidia-smi`로 GPU 점유/온도 안정 확인.

---

## 4. 판정 기준 & 목표

| 단계 | 지표 | 기준선 | 목표 |
|---|---|---|---|
| Tier 1 | `bench.py` 디코드 처리량 | 557.66 tok/s | ≥ 700 tok/s |
| Tier 2 | `bench.py` 디코드 처리량 | (Tier 1 결과) | 추가 상승 (커널 수 감소분) |
| Tier 2 | 프리픽스 TTFT | 감소율 41.1/66.1/79.7% | 유지 이상 (회귀 없음) |
| 정합성 | 출력 텍스트 품질 | 기준 | 회귀 없음 (`test_migration.py`/`example.py` 통과) |

---

## 5. 진행 순서 (체크리스트)

- [ ] 0-1. 모델 폴더 실제 아키텍처 확인 (`inspect_model.py`)
- [ ] 0-2. 기준선 벤치 3종 기록
- [ ] 1. Tier 1 구현 (`model_runner.py` 영속 버퍼)
- [ ] 2. Tier 1 정합성 테스트 + `bench.py` 측정 → 결과 기록
- [ ] 3. Tier 2-1 `store_kvcache` 개선(즉효) → 측정
- [ ] 4. Tier 2-2 RMSNorm+RoPE 융합 → 측정
- [ ] 5. Tier 2-4 padded→varlen → 측정
- [ ] 6. Tier 2-3 split-KV decode (선택) → 측정
- [ ] 7. Tier 2-1 완전 융합(store in attention) → 측정
- [ ] 8. 최종 결과 정리 → 논문 수치 갱신

> 각 항목은 **구현 → 정합성 테스트 → 성능 측정** 순서로 진행하고, 하나씩 커밋 단위로 나눠 회귀 원인을 추적하기 쉽게 한다.
