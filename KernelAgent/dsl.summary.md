# dsl.summary.md - Kernel Engineering DSL Optimization Summary
**MatMul ➔ FMHA ➔ LLM-from-scratch ➔ micro-vllm 전 주기 최적화 지식 명세 요약**

본 문서는 `v7.dsl` 지식 모델 규격("하드웨어 메타데이터 ➔ 설계/튜닝 공간 ➔ 물리 제약 ➔ 아키텍처 지식 ➔ 코드 구현 프리미티브 ➔ 자동 검증 수치 조건")에 따라 구조화된 4단계 최적화 마이그레이션 DSL 문서들의 핵심 설계, 튜닝 파라미터, 실증 성능 성과 및 아키텍처 통찰을 집대성한 요약서입니다.

---

## 1. [matmul.dsl.html (0-MatMul)](file:///D:/code/metaprogramming/KernelAgent/0-MatMul/matmul.dsl.html)
* **핵심 설계**: Persistent CTA 스케줄링 및 L2 Cache Swizzling (M-fast, N-slow 레이아웃 그룹 스위즐링).
* **튜닝 파라미터**: 
  - `num_ctas = 192` (RTX 5070의 48개 SM에 SM당 4개 블록을 배정하여 글로벌 메모리 로드 지연을 은닉).
  - `group_size_m = 8` (48.0 MB 대용량 L2 캐시 지역성을 최대한 활용하여 B 행렬 타일 재사용률 극대화).
* **물리적 제약 조치**: swizzled persistent 스케줄러의 비가분 차원(Non-divisible shape) 메모리 경계 오정렬 문제를 방어하기 위한 그룹 패딩 수식 (`total_tiles = num_groups * GROUP_SIZE_M * NUM_BID_N`) 통합.
* **실증 성과**: 
  - $256 \times 256$ 소형 크기에서 PyTorch 대비 **1.79배 성능 우위** (4.07 TFLOPS).
  - $4096 \times 4096$ 대형 크기에서 cuBLAS/CUTLASS 대비 **91% 수준 도달** (61.78 TFLOPS).

---

## 2. [FMHA.dsl.html (1-FMHA)](file:///D:/code/metaprogramming/1-FMHA/FMHA.dsl.html)
* **핵심 설계**: Fused Multi-Head Attention, Pipelined Global Load Latency Hints, Causal Loop Bounding.
* **튜닝 파라미터**: 
  - QK 로드 시 `latency=2` (레지스터/SRAM 사전 프리페치), V 로드 시 `latency=4` (DRAM 지연을 $QK^T$ Tensor Core 연산과 오버랩).
  - Causal masking 시 이터레이션 상한 `Tc = ct.cdiv(min(m_end, k_seqlen), TILE_N)`으로 dynamic bounding.
* **아키텍처 통찰**: MatMul(연산 제한적)과 달리 Attention은 Softmax 등의 $O(S^2)$ 메모리 퓨전 한계가 지배하는 메모리 바운드 연산이므로, cuTile의 온라인 소프트맥스 및 퓨즈드 구현이 **전체 시퀀스 길이 범위에서 PyTorch 대비 지속적으로 2배 수준의 우위**를 제공할 수 있음을 규명.
* **실증 성과**: 
  - 모든 시퀀스 영역에서 PyTorch SDPA 대비 **~2.0배 가속 효과** 지속 유지.
  - Causal $4096 \times 4096$ 어텐션 구동 시 최고 **125.89 TFLOPS** 달성.

### 💡 FMHA.dsl.html 프레임워크 반영 및 요약
1. **지식 및 물리 제약 조건 매핑 (`<hardware_constraints>` & `<knowledge_rules>`)**:
   * **하드웨어 포화(Grid Saturation)**: RTX 5070 GPU의 48개 SM을 가득 채우기 위해 1024 시퀀스 기준 1536개의 블록을 기동하여 Latency Hiding을 달성하도록 가이드라인 정의.
   * **지연 은닉 파이프라이닝(Pipelined Latency Hiding)**: Query와 Key 로드 시 `latency=2`로 레지스터/SRAM에 prefetch하고, Value 로드 시 `latency=4`를 명시하여 $QK^T$ Tensor Core 연산(`ct.mma`) 도중에 DRAM 로드를 병렬 오버랩하는 규칙 삽입.
   * **루프 dynamic bounding 최적화**: Causal masking 시 불필요한 상삼각 행렬 연산을 방지하고자 이터레이션 상한을 `Tc = ct.cdiv(min(m_end, k_seqlen), TILE_N)`으로 dynamic 조율하여 50%의 컴퓨팅 효율을 확보한 지식 규칙화.
2. **마일스톤 및 코드 Primitives (`<milestones>`)**:
   * **Stage 1 (Non-Causal FMHA)**: DRAM 글로벌 메모리 write-back을 제거하고 온칩 연산을 보증하기 위해 `latency` prefetch 지시어가 주입된 cuTile Fused 어텐션 핵심 코드 primitive를 이식.
   * **Stage 2 (Causal FMHA)**: dynamic loop bounding 및 하부 삼각 mask를 퓨즈드 적용하는 cuTile 코드 스니펫 제공.
3. **MatMul과의 시스템 설계적 대조 규명**:
   * MatMul(Compute-bound)과 달리 Attention(Memory-bound)은 중간 softmax 등의 메모리 I/O 비용이 크기 때문에, PyTorch의 복잡한 런타임 디스패치를 우회하는 cuTile의 온라인 소프트맥스 및 퓨즈드 레이아웃이 시퀀스 길이 전체 범위에서 상시 2.0배에 가까운 강력한 성능 우위를 제공할 수 있음을 아키텍처 분석 규칙으로 포함.
4. **검증 평가 루프 (`<agent_eval_loop>`)**:
   * FP16 16-bit 오차 정합성 규격(`atol <= 1e-2, rtol <= 5e-2`) 통과 여부 및 Causal $4096 \times 4096$에서 실측된 125.89 TFLOPS 달성 목표를 성공 기준(`acceptance_criteria`)으로 명시.
      
---

## 3. [LLM-from-scratch.dsl.html (2-LLM-from-scratch)](file:///D:/code/metaprogramming/KernelAgent/2-LLM-from-scratch/LLM-from-scratch.dsl.html)
* **핵심 설계**: Causal Padding (SDPA 폴백 방지), 정적 KV 캐시 (`scatter_` 인플레이스), E2E CUDAGraph Capture.
* **물리적 제약 조치**:
  - $T < 64$ 경계 오정렬로 인한 PyTorch SDPA 강제 폴백을 방지하고자 $64$ 크기로 텐서를 causal padding 후 복원하는 기법 탑재.
  - dynamic `torch.cat` KV 캐시를 배제하고 static contiguous 캐시 버퍼에 graph-compatible `scatter_` 인플레이스 업데이트를 연결하여 VRAM 파편화 및 Caching Allocator 지연 제거.
* **FFN 블록 결정**: FFN(Linear)은 cuBLASLt 기반 `nn.Linear`를 유지하고 CUDA Graph에 묶어 실행하는 것이 성능면에서 극상임을 정량화 (CUDA Graph가 호스트 디스패치를 소거하므로 FFN 퓨전 커널은 불필요).
* **TTFT 단축 최적화**: 미래 슬롯 자동 마스킹 성질을 이용해 캐시 리셋 `fill_(0.0)` eager 호출을 제거하고, prefill 시 캐시 다이렉트 쓰기를 구현하여 TTFT를 **2.50 ms**로 단축 (0.83ms 절감).
* **실증 성과**: PyTorch SDPA 대비 **5.28배 Decoding Throughput 향상 (3,701.09 tokens/sec)**.

### 💡 LLM-from-scratch.dsl.html 설계 및 핵심 요약
1. **E2E 최적화 지식 명세 (`<knowledge_rules>`)**:
   * **지연 차단 CUDA Graphs**: 매 토큰 생성 시 누적되는 PyTorch/cuTile 연산들의 호스트-디바이스 간 론칭 레이턴시(2~3 ms)를 CUDAGraph Capture를 통해 통째로 바이너리로 캡처하여 소거하는 룰 명시.
   * **정적 KV Cache (`scatter_` 인플레이스)**: `torch.cat`을 배제하고 contiguous static 텐서를 사전 할당한 뒤, Graph와 호환되는 `scatter_` 연산으로 VRAM 재배정 및 메모리 파편화를 차단하는 구조 명세.
   * **Causal Padding (Zero SDPA Fallback)**: 프롬프트 $T < 64$일 때 64 크기로 Q, K, V를 `F.pad` 처리 후 어텐션을 실행하여 PyTorch SDPA 폴백을 원천 차단하고 원래 크기 $T$로 슬라이싱하는 기법 규칙화.
   * **FFN/MLP 최적 설계 판단**: FFN은 cuTile 융합 커널 대신 고도로 어셈블리 최적화된 cuBLASLt 기반 `nn.Linear`를 유지하고 CUDA Graph에 묶어 실행하는 것이 실효성 측면에서 극상임을 정량적으로 입증.
   * **TTFT 캐시 초기화 오버헤드 단축**: 캐시 리셋용 `fill_(0.0)` eager 커널 기동을 제거하고(미래 슬롯은 causal masking에 의해 자동 무효화됨), prefill 포워드 시 캐시에 direct-in-place 쓰기를 적용해 Prefill Latency를 3.33ms에서 2.50ms로 단축한 지식화.
2. **개발 마일스톤 및 코드 Primitives (`<milestones>`)**:
   * **Stage 1 (cuTile Attention Raw)**: Causal padding 및 dynamic concat 방식을 이식한 cuTile Eager 론칭 코드.
   * **Stage 2 (cuTile CUDA Graphs)**: Static KV Cache `scatter_` 인플레이스 업데이트, `torch.cuda.CUDAGraph()` capture & replay 예제 코드, 그리고 TTFT 최적화용 direct-writing 코드 primitive 탑재.
3. **검증 평가 루프 (`<agent_eval_loop>`)**:
   * PyTorch Baseline 대비 출력 텍스트의 100% token-by-token 정합성(Accuracy Parity) 유지 규격 검증.
   * TTFT Prefill 지연시간 3.0 ms 미만(실측 2.50 ms), 디코딩 생성 속도 3500 tokens/sec 초과 (실측 3701.09 t/s, PyTorch 대비 5.28배 Speedup) 성공 기준 명문화.
      
---

## 4. [micro-vllm.dsl.html (micro-vllm)](file:///D:/code/metaprogramming/KernelAgent/micro-vllm/micro-vllm.dsl.html)
* **핵심 설계**: Paged KV Cache (GPU vectorized block mapping), CUDA Green Contexts SM 물리 분할.
* **튜닝 파라미터**: Prefill Context (32 SMs), Decode Context (16 SMs) 분할.
* **물리적 제약 조치**:
  - dynamic shape padding이 매 이터레이션 임시 텐서 생성/GC를 유발해 Throughput을 67.04% 폭락시키는 PyTorch Allocator 스래싱을 실증 규명하고, 무패딩 Eager Serving 모드 적용.
  - prefix-cached prefill 연산의 형상 불일치를 해결하기 위해 GPU 상에서 arange 인덱스 연산으로 주소를 direct mapping하는 벡터화 레이어 통합.
* **아키텍처 통찰 (L2 Cache Residency)**: Decode 전용 SM을 16개로 물리 분할 구획함으로써, 대형 prefill 연산이 GPU의 48.0 MB L2 캐시 내부의 decode 가중치 데이터를 방출(eviction)하는 현상을 원천 방어.
* **실증 성과**:
  - Eager serving Throughput **470.82 tok/s** 확보.
  - Green Contexts 활성화 시 **Decode P50 ITL 14.0% 단축** 및 전체 Throughput 10.6% 향상.
  - 32 SM 조건 하에서도 batch-1 prefill occupancy 포화 임계치 도달에 힘입어 TTFT latency 평탄화 방어 완료 (243.42ms로 -0.5% 유지).

### 🎨 micro-vllm.dsl.html 설계 및 핵심 요약

1. **최종 서빙 튜닝 지식 명세 (`<knowledge_rules>`)**:
   - **메모리 할당자 스래싱 차단**: dynamic shape padding 적용 시 텐서 재배치 및 가비지 컬렉션 부하로 성능이 67.04% 폭락함을 규명하고, 무패딩 Eager Serving 방식을 명시해 **470.82 tok/s**의 높은 처리량을 보존하는 아키텍처 룰 정의.
   - **Green Contexts SM 물리 분할**: 무거운 prefill의 연산 독점과 디코드 L2 캐시 방출(Eviction)로 인한 tail latency 스파이크를 방어하기 위해 SM 분할(Prefill: 32 SM / Decode: 16 SM)을 적용하여 중간값 Decode ITL 14.0% 단축 및 전체 처리량 10.6% 향상을 확보하는 규칙 수립.
   - **Prefill SM 포화점 제약**: batch-1 prefill의 하드웨어 occupancy 포화 제한점인 32 SM을 efficiency sweet-spot으로 지정하여 자원의 효율적 활용을 지시하는 규칙 명시.
   - **GPU 벡터화 캐시 재배치**: dynamic continuous batching 하에서 prefix-cached prefill 연산의 형상 불일치 해결을 위해 GPU direct indexing을 활용하는 룰 정의.
2. **개발 마일스톤 및 코드 Primitives (`<milestones>`)**:
   - **Stage 1 (Paged Cache Vectorized Mapping)**: CPU 루프를 배제하고 GPU 상에서 high-speed index 매핑 연산으로 블록 캐시를 재구성하는 PyTorch direct mapping 코드 primitive.
   - **Stage 2 (Green Contexts SM Partitioning)**: CUDA Python 1.0 API를 직접 연동해 RTX 5070의 48개 SM을 물리 파티션(32, 16)으로 분할하고, 모델 런너 포워드 루프 전후에 Context를 Push/Pop 결합하는 핵심 구동 코드 primitive 탑재.
3. **검증 평가 루프 (`<agent_eval_loop>`)**:
   - 256명 동시 사용자 continuous batching 서빙 부하 및 Green Context 벤치마크 검증 명령어 (`bench.py`, `bench_green.py`) 연동.
   - 수치 정확도 검증, Eager serving Throughput `> 450 tokens/sec`, P50 ITL 지연시간 단축 비율 `>= 10.0%`, Peak VRAM `< 11.5 GB` 성공 판단 기준 (`acceptance_criteria`) 명문화.
---

# 📋 전체 최적화 여정 요약

## 0. MatMul 최적화
- **핵심 설계**
  - Persistent CTA Scheduling
  - Swizzled `GROUP_SIZE_M` 그룹핑

- **튜닝 파라미터**
  - `num_ctas = 192` (SM Occupancy 극대화)
  - `GROUP_SIZE_M = 8` (48MB L2 Cache 적극 활용)

- **물리 제약 대응**
  - Non-divisible Shape 메모리 경계 침범 버그 방지 수식 통합

- **정량 성과**
  - $256 \times 256$: PyTorch 대비 **1.79배 가속**
  - 대형 행렬: cuBLAS 성능의 **91% 수준 도달**

---

## 1. FMHA 최적화
- **핵심 설계**
  - Fused Attention
  - Pipelined Global Load Latency Hints
  - Causal Loop Bounding

- **튜닝 파라미터**
  - Q/K Load: `latency=2`
  - V Load: `latency=4`
  - `Tc` Dynamic Causal Bounding

- **정량 성과**
  - PyTorch SDPA 대비 **약 2배 가속**
  - Causal Attention 최고 **125.89 TFLOPS**

---

## 2. LLM-from-scratch 최적화
- **핵심 설계**
  - Causal Padding (SDPA Fallback 방지)
  - Static KV Cache (`scatter_` In-place)
  - End-to-End CUDA Graph Capture

- **물리 제약 대응**
  - `T < 64` 구간 Padding 처리
  - Static Contiguous Cache 적용
  - Allocator 지연 제거

- **FFN 최적화 결론**
  - CUDA Graph와 결합된 cuBLASLt 기반 `nn.Linear` 유지가 최적

- **TTFT 개선**
  - `fill_(0.0)` 제거
  - Direct Prefill Writing 적용
  - Prefill Latency: **2.50 ms**

- **정량 성과**
  - Decoding Throughput: **3,701.09 tokens/sec**
  - PyTorch 대비 **5.28배 향상**

---

## 3. micro-vLLM 최적화
- **핵심 설계**
  - Paged KV Cache
  - GPU Vectorized Block Mapping
  - CUDA Green Contexts

- **튜닝 파라미터**
  - Prefill Context: 32 SM
  - Decode Context: 16 SM

- **물리 제약 대응**
  - Dynamic Padding에 의한 Allocator Thrashing 규명
  - No-Padding Eager Serving 채택

- **캐시 지역성 최적화**
  - Decode용 SM 분리
  - L2 Cache Eviction 방지

- **정량 성과**
  - Decode P50 ITL **14.0% 감소**
  - 전체 Throughput **10.6% 향상**

