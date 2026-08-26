코드까지 다 읽었습니다. `micro_vllm_성능분석.md`의 진단(WDDM + Eager 호스트 오버헤드)은 방향이 맞고, 실제 구현을 보면 **그보다 더 구체적이고 즉효성 있는 병목들**이 보입니다. 제 생각을 우선 정리합니다.

---

## 결론 (한 줄)

개선 여지는 충분히 있습니다. 그리고 병목은 GPU 커널 연산이 아니라 **호스트 측 — (1) 매 스텝 텐서 준비, (2) 커널 개수, (3) Windows WDDM 발사 지연**입니다. 교육적 모듈성을 해치지 않는 순서로 접근하면 `557 tok/s`에서 의미 있는 폭으로 올릴 수 있습니다.

## 코드에서 확인한 병목 (근거 포함)

**① 매 스텝마다 텐서를 새로 만들고 H2D 복사** — `model_runner.py`
- `prepare_decode`(240~256), `prepare_prefill`(202~238)가 **매 스텝** Python 리스트를 만들고 `torch.tensor(...).pin_memory().cuda(non_blocking=True)`를 호출.
- `prepare_block_tables`(196~200)도 매 스텝 새 int32 텐서를 할당.
- `prepare_sample`(258~263)도 매 스텝 temperatures를 할당.
- → 디코드 한 스텝에 **고정 비용의 메모리 할당 + 4~5번의 H2D 전송**이 매번 들어갑니다. 이게 디코드 처리량을 가장 크게 깎는 원인입니다.

**② `store_kvcache`가 별도 커널 + 비효율 구조** — `attention.py`
- `store_kvcache_cutile_kernel`(26~41)은 `grid=(N,1,1)`, 즉 **토큰당 스레드블록 1개**가 `D=num_heads×head_dim`짜리 한 줄을 복사합니다. 작은 N에서는 점유율이 낮고, 행 폭 D가 커서 타일 효율도 나쁩니다.
- 더 큰 문제는 **프리필에서 K/V를 캐시에 쓴 뒤 attention 커널이 다시 읽는다**는 점입니다. 별도 커널 1개 + K/V 메모리 트래픽 2배. vLLM 식이면 프리필 커널 안에서 store를 끝냅니다.

**③ padded 프리필의 호스트 파이썬 루프** — `cutile_attention.py`
- `_cutile_fmha_prefill_padded`(279~345)가 배치마다 `.item()`으로 시퀀스 길이를 읽고, q/k/v를 64의 배수로 **패딩한 버퍼에 호스트 루프로 복사**(307~326) 후 커널을 쏩니다. 패딩 연산 낭비 + 배치 직렬화 오버헤드.

**④ 디코드 어텐션이 사실상 GEMV (메모리 바운드)** — `cutile_attention.py`
- `paged_decode_kernel`(202~268)은 (seq, head)당 블록 1개가 KV 블록을 순회하며 `q(1,D) × K(D,256)`를 `ct.mma`로 계산합니다. 단일 쿼리 토큰이라 텐서코어 활용이 낮고, HBM 대역폭에 묶입니다.
- GQA(0.6B는 16Q/8KV 헤드)에서 **동일한 K/V를 여러 쿼리 헤드가 중복 로드**합니다. split-KV(flash-decode)나 그룹 내 K/V 공유가 없어 대역폭을 못 채웁니다.

**⑤ CUDA Graph가 정적 버퍼 복사 때문에 역효과** — `model_runner.py`
- `run_model`(266~282)이 매 스텝 `graph_vars["input_ids"][:bs] = ...`, `slot_mapping`, `context_lens`, `block_tables`를 **전부 복사**합니다. 성능분석.md의 "입력 텐서 카피 비용 역전"이 코드로 그대로 확인됩니다. 복사 대상을 최소화하지 않으면 그래프 재생 이득이 항상 상쇄됩니다.

**⑥ 레이어별 PyTorch eager 커널 폭증** — `qwen3.py` / `rotary_embedding.py`
- `Qwen3DecoderLayer.forward`(145~158) 한 층에 RMSNorm → QKV GEMM → q/k norm → RoPE → attention → store → O GEMM → RMSNorm → gate/up GEMM → SiLU → down GEMM. **레이어당 ~10개 커널, 28레이어 = 스텝당 수백 개 커널**이 각각 WDDM 발사 지연을 겪습니다.
- RoPE(`apply_rotary_emb`, 6~14)는 `x.float()` 업캐스트 + chunk + 곱셈 4회 + cat + downcast로 **쿼리/키 각각 여러 op**가 됩니다.

**⑦ 구조적 한계: prefill/decode 비중첩** — `scheduler.py`
- `schedule`(24~58)은 prefill과 decode를 **상호배타적 단계**로 처리합니다(chunked prefill 없음). 그래서 그린 컨텍스트의 SM 분할이 "순차 프리필 중단"을 못 없애는 겁니다 — 이건 커널 문제가 아니라 스케줄러 문제입니다.

## 개선 방안 (우선순위, 교육성 보존 고려)

**Tier 1 — 커널 안 건드리는 순수 호스트 최적화 (교육성 100% 보존, 최우선)**
1. `input_ids/positions/slot_mapping/context_lens/block_tables/temperatures`를 **영속 버퍼로 preallocate**하고 매 스텝 `copy_`만. 매 스텝 alloc + pin + H2D 오버헤드를 거의 제거. 구현 난이도 낮고 효과 가장 큼.
2. `prepare_block_tables`의 리스트→텐서 변환 제거, 재사용 버퍼에 `-1` 채우고 직접 기록.

**Tier 2 — 경량 커널 퓨전/개선 (모듈러 유지 + 치명 구간만)**
3. `store_kvcache`를 **프리필 attention 커널에 융합** (K/V를 캐시에 쓰면서 어텐션 수행) → 커널 1개 감소 + K/V 트래픽 절반.
4. `q_norm/k_norm + RoPE` 융합 (성능분석.md의 "RMSNorm+RoPE" 제안과 일치) → 레이어당 커널 4~6개 감소.
5. 디코드 어텐션에 **split-KV / GQA K·V 재사용** 도입 → 대역폭 활용 개선.
6. padded 프리필 제거 → cu_seqlens 기반 varlen 단일 커널(배치 루프 제거).

**Tier 3 — 시스템/구조 (효과 최대, 난이도 높음)**
7. **CUDA Graph 재설계**: 복사 대상을 `input_ids`·`positions`(그리고 바뀔 때만 block_table)으로 최소화하고 graph pool 활용. 복사 비용이 사라지면 Windows에서도 그래프 재생 이득이 살아납니다. 필요하면 성능분석.md의 **Breakable CUDA Graph**까지.
8. **Chunked prefill** (긴 프리필을 잘라 디코드와 인터리브) → 그린 컨텍스트가 못 푼 "순차 프리필 일시정지"를 구조적으로 해결. 논문 §5.6의 한계를 미래연구로 자연스럽게 연결.
9. 환경: **HAGS(하드웨어 가속 GPU 스케줄링) 활성화** 또는 TCC 모드 — 코드 변경 없이 발사 지연 감소.

## 교육성 트레이드오프에 대한 제 생각

- Tier 1은 커널/구조를 전혀 안 건드리는 **호스트 리팩터링**이라 교육적 정체성과 충돌이 없습니다. 이걸 먼저 해서 "순수 오버헤드 제거로 얼마나 오르는지"를 측정하면, 논문의 "Eager 오버헤드" 주장을 정량적으로 강화하는 근거가 됩니다.
- Tier 2의 퓨전은 성능분석.md의 **방안 B(모듈러 유지 + 부분 퓨전)**에 정확히 해당합니다. `modular_engine.py` vs `fused_engine.py`를 분리 제공하면 교육 비교 실험 자체가 논문 기여가 됩니다.
- Tier 3(그래프/청크드 프리필)은 교육 대상 범위를 넓히는 방향이라 오히려 가치가 큽니다.

## 기대 효과 (방향성 추정, 측정 필요)

- Tier 1 + 2만으로도 디코드 **557 → 700~900 tok/s** 수준은 충분히 노려볼 만합니다(커널 수와 호스트 비용이 크게 줄어드는 만큼). Linux FlashAttention(2138 tok/s)과의 격차 대부분은 WDDM 발사 지연이라, Tier 3의 그래프 재설계가 성공하면 격차가 급격히 줄 가능성이 있습니다.
- 정확한 수치는 실제 측정이 필요하므로, 먼저 **Tier 1을 구현하고 step별 커널 수/발사 시간을 프로파일**해 보는 걸 추천합니다.

---

한 가지 짚어둘 점: `test-result-cuTile.md`에는 모델이 **Qwen2.5-3B**로 찍혀 있는데, `bench.py`(34행)와 논문은 **Qwen3-0.6B**를 씁니다. 벤치마크 산출물의 모델 표기가 구버전 경로(`3-nano-vllm`) 기준이라, 논문 수치의 재현 문서와 이 결과 파일을 맞춰두는 게 좋겠습니다.

원하시면 **Tier 1부터 바로 구현**해 드릴 수 있고, 아니면 특정 항목(예: CUDA Graph 재설계, chunked prefill)의 상세 설계를 먼저 잡아볼까요?