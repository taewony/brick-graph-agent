# micro-vLLM 논문 실험 실행조건 (Execution Conditions)

이 문서는 논문에 기재되는 모든 실험의 **고정 실행조건**을 정의한다.
모든 벤치마크는 아래 조건을 지키며, 실행 시 `run_experiment.py`가 이 조건을 자동 기록한다.
조건을 바꾸면 반드시 기준선(baseline)을 다시 측정해야 한다.

---

## 1. 하드웨어 (고정)

| 항목 | 값 |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 (Blackwell, SM 100/101) |
| SM 수 | 48 |
| VRAM | ~12 GB |
| L2 캐시 | 48 MB |
| 호스트 OS | Windows 11 |

> 확인: `nvidia-smi` 로 측정 시점의 클럭/온도/전력을 함께 기록한다(스로틀링 감지용).

## 2. 소프트웨어 (고정)

| 항목 | 값 |
|---|---|
| Python | (측정 시점 기록) |
| PyTorch | 2.13.0+cu130 (논문 표기) |
| CUDA | 13.0 |
| cuTile / TileGym | `cuda.tile` (tilegym[tileiras]) |
| FlashAttention | WSL2 기준선 전용 (Linux) |

> `run_experiment.py`가 `torch.__version__`, `torch.version.cuda`, git 커밋 해시를 자동 기록한다.

## 3. 모델 (⚠️ 확인 필요)

| 항목 | 값 |
|---|---|
| 경로 | `~/huggingface/Qwen3-0.6B/` |
| **실제 모델** | ⚠️ 폴더 내용 확인 필수 (`inspect_model.py`) |

> 현재 해당 폴더는 `Qwen2.5-3B-Instruct`(Qwen2ForCausalLM, hidden 2048, 36 layers, 16 heads, 2 KV heads GQA 8:1)로 확인됨.
> **논문에 "Qwen3-0.6B"로 기재하려면 진짜 Qwen3-0.6B를 다시 받아야 하고, 아니면 논문을 "Qwen2.5-3B"로 정정해야 한다.**
> 이 모순이 해결되기 전에는 모든 수치에 "모델 불일치" 주석을 붙인다.

## 4. 처리량 벤치마크 (`bench.py`) — 논문 §5.1, §5.2

| 파라미터 | 값 | flag |
|---|---|---|
| 요청 수 | 256 | `--num-seqs 256` |
| 입력 토큰 | 100 ~ 1024 (균등) | `--max-input-len 1024` |
| 출력 토큰 | 100 ~ 1024 (균등) | `--max-output-len 1024` |
| 시드 | 0 | `--seed 0` |
| max_model_len | 4096 | `--max-model-len 4096` |
| temperature | 0.6 | (고정) |
| ignore_eos | True | (고정) |

**비교축(독립변수)**:

| 축 | 값 | flag |
|---|---|---|
| 백엔드 | cuTile / FlashAttention | `--use-cutile` 유무 |
| 프리필 전략 | hybrid / direct / padded | `--cutile-prefill-strategy` |
| 실행 모드 | eager / CUDA Graph | `--cutile-cudagraph` 유무 |
| **CUDA Graph 버퍼 모드** | persistent(신, Tier 3a) / copy(구) | `--graph-mode` |

**측정 지표**: `Throughput` (tok/s), `Time` (s). (run 중 `Decode Throughput`은 보조)

## 5. 프리픽스 캐시 벤치마크 (`bench_prefix_cache.py`) — 논문 §5.3

| 파라미터 | 값 |
|---|---|
| 조건 | no_cache / warm_cache / prefix_changed |
| 정적 프리픽스 | 1024 / 2048 / 3072 토큰 |
| 동적 접미사 | 64 토큰 |
| 출력 | 64 토큰 |
| requests | 8 |

**측정 지표**: `mean_ttft_ms`, `cache_hit_ratio`, `throughput_tok_s`.

## 6. 그린 컨텍스트 벤치마크 (`bench_green_stress.py`) — 논문 §5.6

| 파라미터 | 값 |
|---|---|
| green API | cuda_core |
| 프리필 SM / 디코드 SM | 32 / 16 |
| repeats | 10 |
| 보호 디코드 | 프롬프트 32, 출력 256 |
| 간섭 프리필 | 프롬프트 3072, 출력 1, 주입 12회 |

**측정 지표**: `decode_gap_p99_ms`, `decode_step_p99_ms`, `throughput`, `improved_runs`.

---

## 7. 기준선 추적 규칙 (Baseline Tracking)

1. 모든 실행은 `run_experiment.py` 로 한다 — 실행조건(env/모델/flags/결과)이 JSONL로 누적된다.
2. **코드를 바꾸기 전** 기준선을 1회 측정해 JSONL에 기록한다.
3. **코드를 바꾼 후** 동일 flags로 다시 측정해, 이전 기준선과 비교한다.
4. **기준선이 ±5% 이상 흔들리면** 코드 변경 원인을 의심하기 전에 먼저 `nvidia-smi`(클럭/온도/전력/점유 프로세스)를 확인한다.
5. 기준선 JSONL의 `env.nvidia_smi` 를 보면 스로틀링 여부를 바로 판별할 수 있다.

---

## 8. 실험실 PC 확인 순서 (Verification Order)

코드 변경 전/후를 `--graph-mode` 로 A/B 비교한다. 순서는 아래와 같다.

### 8-1. 1회 실행으로 Tier 3a A/B 비교 (한 번에)

```powershell
cd KernelAgent\3-micro-vllm

# (권장) graph 모드 A/B를 한 번에: copy(구) vs persistent(신) 자동 비교
python run_experiment.py --use-cutile --cutile-cudagraph --compare-graph-modes --out-jsonl baseline.jsonl
```

- 내부에서 `--graph-mode copy` 와 `--graph-mode persistent` 를 **각각 실행**하고,
  마지막에 `copy vs persistent` 처리량과 delta(%)를 출력한다.
- 각 실행의 환경(git 해시, nvidia-smi, 모델 config)은 동일하게 `baseline.jsonl`에 기록된다.

### 8-2. 단일 모드 실행 (개별 확인)

```powershell
# 구버전(복사)만
python run_experiment.py --use-cutile --cutile-cudagraph --graph-mode copy --out-jsonl baseline.jsonl

# 신버전(영속 버퍼)만
python run_experiment.py --use-cutile --cutile-cudagraph --graph-mode persistent --out-jsonl baseline.jsonl

# eager 기준선 (그래프 없음)
python run_experiment.py --use-cutile --cutile-prefill-strategy hybrid --out-jsonl baseline.jsonl
```

### 8-3. 결과 판정

| 상황 | 판정 |
|---|---|
| copy ≈ persistent (또는 persistent ≥ copy) | Tier 3a 재적용 정상. 이전 4배 저하는 **GPU 상태(열/점유)** 때문이었음 |
| persistent가 copy보다 현저히 느림 | Tier 3a 코드에 실제 문제. `--graph-mode copy`로 유지하고 원인 조사 |
| **둘 다 이전 기준(398/372 tok/s)보다 크게 낮음** | GPU 스로틀링/점유 의심 → `baseline.jsonl`의 `env.nvidia_smi` 확인 |

### 8-4. 재현 기준선 (지난 정상 수치)

| 설정 | 기준선 |
|---|---|
| eager (hybrid) | 371.88 tok/s |
| graph (copy 모드) | 397.93 tok/s |

> 이 수치보다 ±5% 이상 벗어나면 위 8-3 표에 따라 원인을 판정한다.

---

## 9. Tier 2b 적용 내역 (RMSNorm 퓨전)

**변경 파일**: `nanovllm/layers/layernorm.py`

- `rms_forward`(q_norm/k_norm): 수동 RMSNorm(`float→pow→mean→rsqrt→mul→cast→mul`, 약 8커널) → **`F.rms_norm`(단일 fused 커널)**.
- `add_rms_forward`(input/post layernorm): **float32 residual add는 유지**(정밀도 보존), RMSNorm 본체만 `F.rms_norm`으로 치환.
- 기대: 레이어당 RMSNorm 관련 커널 수 감소 → WDDM 발사 지연 감소.

> ⚠️ 수치적 동등성: `F.rms_norm`은 bf16에서 내부적으로 안정적 알고리즘 사용. 출력이 수동 구현과 미세하게 다를 수 있으나,
> 처리량/TTFT/지연 같은 **성능 지표에는 무관**. (정확도 검증이 필요하면 `example.py` 출력 텍스트 품질 확인)

---

## 10. 논문 수치 재추출 프로토콜 (Windows)

Tier 2b 적용 후, 아래 순서로 **모든 Windows 수치를 재측정**한다.
각 실행은 `run_experiment.py`(처리량) 또는 해당 벤치 스크립트를 사용하며, `nvidia_smi`가 2900 MHz 근처(스로틀링 없음)인지 확인한다.

### 10-1. 처리량 (§5.1, §5.2)

```powershell
cd KernelAgent\3-micro-vllm

# ① eager-hybrid (기준)
python run_experiment.py --use-cutile --cutile-prefill-strategy hybrid --out-jsonl baseline.jsonl

# ② CUDA Graph (graph)
python run_experiment.py --use-cutile --cutile-cudagraph --out-jsonl baseline.jsonl

# ③ 프리필 전략 비교
python run_experiment.py --use-cutile --cutile-prefill-strategy direct --out-jsonl baseline.jsonl
python run_experiment.py --use-cutile --cutile-prefill-strategy padded --out-jsonl baseline.jsonl
```

### 10-2. 프리픽스 캐시 TTFT (§5.3)

```powershell
# 정적 프리픽스 1024 / 2048 / 3072 토큰
python bench_prefix_cache.py --use-cutile --requests 8 --static-prefix-tokens 1024 --dynamic-suffix-tokens 64 --max-tokens 64 --out-jsonl prefix_1024.jsonl
python bench_prefix_cache.py --use-cutile --requests 8 --static-prefix-tokens 2048 --dynamic-suffix-tokens 64 --max-tokens 64 --out-jsonl prefix_2048.jsonl
python bench_prefix_cache.py --use-cutile --requests 8 --static-prefix-tokens 3072 --dynamic-suffix-tokens 64 --max-tokens 64 --out-jsonl prefix_3072.jsonl
```

각 `*_jsonl`의 `SUMMARY_JSON`에서 `mean_ttft_ms`, `cache_hit_ratio` 를 읽는다.

### 10-3. 그린 컨텍스트 스트레스 (§5.6)

```powershell
python bench_green_stress.py --repeats 10 --green-api cuda_core --prefill-sms 32 --decode-sms 16
```

### 10-4. WSL2 FlashAttention 기준선 (§5.1, ⚠️ Linux 전용)

Windows에서 재측정 불가. **WSL2 환경에서 별도 실행**해야 한다:

```bash
cd micro-vllm
python bench.py   # flash_attn 백엔드 (NANO_VLLM_USE_CUTILE 미설정)
```

---

## 11. 논문 수치 재현 상태 체크리스트

| 항목 | Windows 재측정 가능 | 명령 |
|---|---|---|
| eager-hybrid 처리량 | ✅ | 10-1 ① |
| CUDA Graph 처리량 | ✅ | 10-1 ② |
| direct/padded 프리필 | ✅ | 10-1 ③ |
| 프리픽스 캐시 TTFT | ✅ | 10-2 |
| 그린 컨텍스트 | ✅ | 10-3 |
| WSL2 FlashAttention | ❌ (Linux) | 10-4 |
