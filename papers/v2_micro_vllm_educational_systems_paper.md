# cuTile 기반 Windows 네이티브 교육용 LLM 추론 엔진 설계 및 성능 평가

## Design and Performance Evaluation of a cuTile-Based Windows-Native LLM Inference Engine for Educational Purposes

### 국문 요약

고성능 LLM 서빙은 FlashAttention·Triton·NCCL 등 리눅스 중심 스택이라 학습자가 프리필·디코드·KV 캐시·스케줄러·메모리 할당 상호작용을 관찰하기 어렵다. 본 논문은 Windows 네이티브 CUDA Python/cuTile 경로의 소형 교육 산출물 마이크로-vLLM 구현·성능 분석 사례연구다. WSL2 FlashAttention 경로는 2159.85 tok/s로 초기 Windows cuTile 경로(463.35 tok/s) 대비 4.66배 빠르다. cuTile KV-스토어 수리, 하이브리드 프리필 디스패치, 호스트 측 영속 버퍼 재사용, RMSNorm 퓨전으로 eager-hybrid 경로는 600.66 tok/s(29.6% 향상)에 도달했고, CUDA 그래프 재생은 성능이 하락하는 부정적 결과를 보였다. 고정 컨텍스트 에이전트 워크로드에서 웜 프리픽스 KV 캐시 재사용은 연산 프리필 토큰을 64로 줄이고 TTFT를 63.2%·80.6%·89.3% 감소시켰다(프리픽스 변경 음성대조 0% 히트). CUDA 그린 컨텍스트는 활성화 검증 후 보호 디코드 꼬리 지연을 완화(디코드 스텝 P99 10/10 개선, 부호검정 p≈0.002)하나 순차 프리필 중단은 제거하지 못한다. 이는 구현·측정·긍정·부정 결과를 모두 가르치는 서빙 교육의 근거다.

### 국문 키워드

LLM 서빙, CUDA Python, cuTile, 하이브리드 프리필 디스패치, 프리픽스 KV 캐시, CUDA 그린 컨텍스트

### 영문 요약 (Abstract)

High-performance LLM serving relies on Linux-centric stacks (FlashAttention, Triton, NCCL, vLLM-style schedulers), making it hard for students and Windows-based local researchers to inspect how prefill, decode, KV-cache management, scheduling, and allocation interact inside an engine. We present micro-vLLM as a Windows-native CUDA Python/cuTile educational artifact. The WSL2 FlashAttention reference reaches 2159.85 tok/s versus 463.35 tok/s for the initial Windows cuTile path (4.66x). After cuTile KV-store repair, hybrid prefill dispatch, host-side persistent-buffer reuse, and RMSNorm fusion, the eager-hybrid path reaches 600.66 tok/s (+29.6%); CUDA Graph replay is functionally repaired but performance-negative. For fixed-context agent workloads, warm prefix KV-cache reuse reduces computed prefill tokens to 64 and TTFT by 63.2%, 80.6%, and 89.3% for 1024/2048/3072-token prefixes, validated by a changed-prefix negative control (0% hit). CUDA Green Contexts, after activation validation, significantly smooth protected decode tail latency (decode-step P99 improved in 10/10 runs, sign test p≈0.002) but do not eliminate sequential prefill pauses. These results support teaching implementation, measurement, positive optimizations, and negative results in the full serving loop.

### 영문 키워드

LLM serving, CUDA Python, cuTile, hybrid prefill dispatch, prefix KV cache, CUDA Green Contexts

---

## 1. 서론

로컬 LLM 추론은 강의실, 연구실, 사적 데이터 분석 환경, 오프라인 개발 환경에서 점점 중요해지고 있다. 이러한 환경에서 학습자가 필요로 하는 것은 단순히 LLM API를 호출하는 법이 아니라, 프롬프트가 어떻게 프리필(prefill) 연산이 되는지, KV 캐시 블록이 어떻게 할당되는지, 디코드(decode)가 왜 지연에 민감한지, 스케줄러 결정이 체감 응답 시간에 어떻게 반영되는지, 그리고 어떤 최적화가 전체 서빙 루프 안에서는 왜 실패하는지를 이해하는 능력이다.

vLLM, TensorRT-LLM, FlashAttention 기반 스택, Triton 커널 등 생산 수준의 서빙 시스템은 높은 성능을 제공하지만 첫 교육용 산출물로는 적합하지 않다. 이들은 규모가 크고, 리눅스 중심이며, 최적화된 의존성과 긴밀하게 결합되어 있다[1,2,3]. WSL2는 Windows 사용자에게 실용적인 경로이지만, 학습자와 네이티브 런타임 사이에 또 하나의 경계를 추가한다. 많은 학생과 로컬 연구자가 NVIDIA GPU를 탑재한 소비자용 Windows 머신을 사용한다는 점에서, Windows 네이티브 GPU 개발은 여전히 중요하다.

본 논문은 이와 다른 기여를 목표로 한다. 바로 **교육용 시스템 산출물로서의 Windows 네이티브 LLM 서빙 구현 및 성능 분석 사례연구**이다. 마이크로-vLLM은 vLLM의 더 빠른 대체재로 위치하지 않는다. 학습자와 연구자가 검사해야 할 서빙 메커니즘, 즉 프리필, 디코드, 페이징된 KV 캐시, 프리픽스 재사용, CUDA 컨텍스트 활성화, 스케줄러 동작, 할당 오버헤드, 꼬리 지연의 트레이드오프를 노출하도록 설계된다. 이 산출물는 서빙 동작을 생산 스택 뒤에 숨기지 않고 **재현 가능한 근거**로 바꿀 때 가치를 가진다.

본 논문의 중심 워크로드는 **고정 컨텍스트 에이전트 워크로드**이다. 많은 에이전트 시스템은 짧은 사용자 요청 앞에 안정적인 컨텍스트를 반복해서 덧붙인다. 시스템 프롬프트, 데이터베이스 스키마, 도구 계약, `SKILL.md` 파일, 정책 텍스트, 예시, 루브릭, 강의 모듈이 그 예이다. Text-to-SQL 에이전트는 데이터베이스 스키마와 규칙 컨텍스트를 반복 포함하고, CUDA 튜터링 에이전트는 강의 자료와 커널 템플릿을 반복 포함하며, 교육용 지식 에이전트는 선택된 지식 번들 발췌를 반복 포함한다. 이러한 워크로드는 비용이 큰 정적 프리픽스를 요청 간에 재사용할 수 있으므로, 자연스럽게 프리픽스 KV 캐시 재사용에 적합하다.

**코드 가용성.** 마이크로-vLLM 산출물는 `0-MatMul`, `1-FMHA`, `2-LLM-from-scratch`, `3-micro-vllm`의 단계적 마이그레이션 경로로 구성된다. 본 논문의 벤치마크 구동기는 `bench_prefix_cache.py`와 `bench_green_stress.py`이다. 저장소는 공개되어 있다: `[이중 익명 심사를 위해 저장소 URL 생략 — 게재 확정본에서 제공]`. 실험은 Windows 11, NVIDIA GeForce RTX 5070(48 SM, 약 12GB VRAM, 48MB L2), PyTorch 2.12.0+cu130, CUDA 13.0, Qwen2.5-3B-Instruct에서 수행되었다.

## 2. 이론적 배경

### 2.1 LLM 서빙 루프

자기회귀(autoregressive) LLM 서빙은 **프리필** 단계와 **디코드** 단계로 구성된다. 프리필은 프롬프트를 처리하여 모든 프롬프트 토큰에 대한 KV 캐시 엔트리를 기록한다. 디코드는 이전에 캐시된 키·밸류를 참조하면서 한 번에 하나의 토큰을 생성한다. 프리필은 프롬프트 길이와 병렬 연산 처리량에 민감하고, 디코드는 토큰당 지연, 메모리 대역폭, 스케줄러 오버헤드, KV 캐시 접근, 커널 실행 동작에 민감하다.

이 구분은 최적화가 한 단계만 개선하면서도 종단 간 서비스 지표는 그대로 두는 이유를 설명한다. 예컨대 프리픽스 캐시는 프리필 연산과 TTFT(Time-To-First-Token)를 크게 줄일 수 있지만, 벤치마크가 많은 출력 토큰을 생성하여 디코드가 총 실행 시간을 지배하면 처리량 개선은 제한적이다.

### 2.2 페이징된 KV 캐시와 프리픽스 재사용

페이징된 KV 캐시는 KV 메모리를 할당·매핑·재사용·축출 가능한 블록으로 나눈다[1]. 프리픽스 캐싱은 이 개념을 토큰-동일 프리픽스 재사용으로 확장한다. 새 요청이 이전 요청과 동일한 토큰 프리픽스를 공유하면, 런타임은 기존 KV 블록을 재사용하고 접미사만 새로 계산할 수 있다[6].

이 메커니즘은 정확한 토큰-프리픽스 안정성에 의존한다. 타임스탬프, 런 ID, 무작위 예시, 사용자별 텍스트처럼 동적인 메타데이터가 프롬프트 앞부분에 놓이면 재사용이 깨진다. 따라서 프롬프트 배치는 모델링 문제일 뿐 아니라 런타임 효율 문제이기도 하다. 프리픽스 재사용을 원한다면 정적 컨텍스트가 동적 사용자 콘텐츠보다 앞에 와야 한다.

### 2.3 CUDA Python, cuTile, nano-vLLM

CUDA Python은 CUDA 런타임·드라이버 API에 대한 Python 수준 접근을 제공하여, 호스트 측 GPU 제어를 가르치기에 유용하다[4]. cuTile 스타일 워크플로는 타일드 GPU 커널을 Python 안에서 표현함으로써 C++ CUDA보다 진입 장벽을 낮추면서도, 고수준 PyTorch 연산자보다 많은 하드웨어 동작을 노출한다. nano-VLLM은 vLLM 스타일 서빙을 위한 소형 참조 구현이다[8]. 마이크로-vLLM은 이러한 교육적 관점을 Windows 네이티브 CUDA Python 실험 경로에 적용한다.

### 2.4 CUDA 그린 컨텍스트

CUDA 그린 컨텍스트는 SM을 비롯한 GPU 자원을 컨텍스트 간에 분할할 수 있게 한다[5,7]. 본 논문에서 그린 컨텍스트는 일반적인 가속 메커니즘이 아니라 **자원 격리 메커니즘**으로 다룬다. 관심 문제는 프리필 간섭 하에서 지연에 민감한 디코드 경로를 보호할 수 있는가이다. 그린 컨텍스트 결과는 요청한 자원 분할 경로가 실제로 사용되었음을 활성화 메타데이터가 확인할 때만 유효하다.

## 3. 시스템 설계

### 3.1 설계 목표

마이크로-vLLM은 네 가지 목표를 중심으로 설계된다.

- **검사 가능성(inspectability)**: 학습자가 프롬프트 토큰이 어떻게 프리필 연산, KV 블록, 디코드 스텝이 되는지 추적할 수 있다.
- **수정 가능성(modifiability)**: 커널, 스케줄러 로직, 런타임 계측은 실험 중에 변경할 수 있을 만큼 작다.
- **Windows 네이티브 실행**: 산출물는 WSL2를 유일한 경로로 삼지 않고 Windows CUDA 환경에서 실행된다.
- **에이전트 워크로드 연관성**: 벤치마크는 교육용 에이전트, Text-to-SQL 에이전트, 로컬 지식 에이전트에서 반복되는 정적 컨텍스트를 반영한다.

### 3.2 마이그레이션 단계

| 단계 | 폴더 | 교육적 역할 |
| :--- | :--- | :--- |
| 0 | `0-MatMul` | 타일링, 공유 메모리, 스위즐링, GEMM 기초 학습 |
| 1 | `1-FMHA` | 온라인 소프트맥스, 인과 마스킹, 융합 어텐션 구현 |
| 2 | `2-LLM-from-scratch` | 최소 자기회귀 루프, KV 캐시, CUDA 그래프 실험 |
| 3 | `3-micro-vllm` | 페이징된 KV 캐시, 프리픽스 캐시, 연속 배칭, 그린 컨텍스트 |

이 구조는 추론 엔진을 하나의 모놀리식 서빙 블랙박스가 아니라, 검사 가능한 학습 단계의 연속으로 만든다.

### 3.3 마이크로-vLLM 서빙 아키텍처

마이크로-vLLM은 경량 Qwen 계열 서빙 루프를 사용한다. 스케줄러는 대기 중인 시퀀스와 실행 중인 시퀀스를 관리한다. 블록 관리자는 KV 캐시 블록을 할당하고 해시 기반 프리픽스 재사용을 수행한다. 각 시퀀스는 프롬프트 토큰, 생성 토큰, 블록 테이블, 캐시된 토큰 수를 추적한다. 모델 러너는 입력 ID, 위치, 슬롯 매핑, 시퀀스 길이, 블록 테이블을 포함하여 프리필과 디코드 준비를 분리한다.

프리픽스 캐시의 경우, 스케줄러는 시퀀스 할당 시 완전한 블록 해시를 확인한다. 동일한 프리픽스 블록이 존재하면 `seq.num_cached_tokens`가 증가하고, 프리필 경로는 캐시되지 않은 접미사만 모델에 전달한다. 어텐션 컨텍스트는 여전히 전체 키 길이를 반영하므로, 모델은 재사용된 프리픽스 KV 블록과 새로 계산된 접미사 KV 블록 모두를 참조할 수 있다.

그린 컨텍스트의 경우, 런타임은 요청한 경로가 실제로 활성화되었는지를 기록한다. PyTorch `GreenContext`는 대상 환경에서 임포트는 가능했으나 컨텍스트 생성을 거부했다. 동작하는 경로는 `cuda.core`와 `cuda.bindings.driver`를 사용하여 `Device.set_current(ctx)`와 `Device.set_current()`(복원)를 호출하는 방식이었다. 벤치마크 JSON은 `green_enabled`, `green_api_type`, `green_split_layout_width`, `green_prefill_resource_source`를 기록하여, 활성화가 폴백(fallback)으로 떨어졌을 때 허위 성능 주장을 방지한다.

### 3.4 고정 컨텍스트 프롬프트 레이아웃

벤치마크는 다음 프롬프트 레이아웃을 사용한다.

```text
[정적 시스템 프롬프트]
[정적 정책 / 도구 계약]
[정적 DB 스키마 또는 강의 컨텍스트]
[정적 예시 또는 루브릭]
[동적 사용자 질문]
```

정적 프리픽스는 요청 간에 의도적으로 동일하고, 동적 접미사는 사용자 질문마다 달라진다. 이는 고정 SQLite 스키마 위의 Text-to-SQL, 고정 모듈 위의 CUDA 튜터링, 고정 소스 발췌 위의 nano-VLLM 튜터링, 안정적인 강의·지식 번들 컨텍스트 위의 검색 에이전트를 대표한다.

## 4. 실험 방법

### 4.1 하드웨어 및 소프트웨어

실험은 Windows 11 대상 PC에서 NVIDIA GeForce RTX 5070 GPU로 수행되었다. 로컬 산출물 메타데이터는 48 SM, 약 12GB VRAM, 48MB L2 캐시를 보고한다. 최근 그린 컨텍스트 프리플라이트 출력은 PyTorch 2.12.0+cu130과 CUDA 13.0을 보고한다. 마이크로-vLLM은 Qwen 계열 로컬 모델을 사용하며, 스트레스 벤치마크 로그는 대상 PC에서 `Qwen2.5-3B-Instruct`(Qwen2ForCausalLM, hidden 2048, 36 레이어, 16 헤드, 2 KV 헤드 GQA 8:1, head_dim 128, vocab 151936, bf16)를 식별한다.

### 4.2 메트릭 정의

본 논문의 성능 지표는 다음과 같이 정의한다. 처리량(tok/s)은 총 생성 토큰 수를 벽시계 시간(wall-clock time)으로 나눈 값이다. TTFT는 요청 도착부터 첫 생성 토큰까지의 지연이다. ITL은 토큰 간 디코드 지연(inter-token latency)이다. "연산 프리필 토큰"은 프리픽스 재사용 이후 새로 계산된 토큰만을 센다.

### 4.3 기준 처리량 벤치마크

기준 처리량 벤치마크는 동일한 동적 다중 사용자 벤치마크에서 Windows 네이티브 cuTile 경로를 WSL2 FlashAttention 기준 경로와 비교한다. 이 비교는 Windows cuTile의 우월성을 보이려는 것이 아니라, 성숙한 최적화 스택에 대비해 산출물를 고정(anchor)하기 위한 것이다.

### 4.4 프리픽스 캐시 벤치마크

프리픽스 캐시 벤치마크는 `bench_prefix_cache.py`를 사용한다. 각 정적 프리픽스 길이에 대해 세 조건을 실행한다.

| 조건 | 설명 |
| :--- | :--- |
| `no_cache` | 각 요청 전에 영속 해시 테이블을 비우고 전체 프리필 수행 |
| `warm_cache` | 캐시를 프라이밍한 뒤 동일 정적 프리픽스 재사용 |
| `prefix_changed` | 프리픽스를 변경하여 정확-프리픽스 캐시 히트를 붕괴 |

정적 프리픽스 길이는 1024, 2048, 3072 토큰이고, 동적 접미사는 64 토큰, 생성 길이는 64 토큰이다. 벤치마크는 캐시 히트율, 캐시된 토큰 수, 연산 프리필 토큰, TTFT, 종단 간 지연, 디코드 ITL, 처리량을 기록한다.

### 4.5 그린 컨텍스트 스트레스 벤치마크

그린 컨텍스트 스트레스 벤치마크는 `bench_green_stress.py`를 사용한다. 하나의 보호 디코드 요청을 활성 상태로 유지하면서 큰 프리필 요청을 반복 주입한다. 주요 지표는 **보호 디코드 완료 간격(completion gap)**이며, 이는 현재 순차 엔진 루프에서 가시적인 디코드 토큰 사이에 삽입되는 프리필 유발 일시정지를 포함한다.

기본 스트레스 워크로드는 다음과 같다.

| 매개변수 | 값 |
| :--- | ---: |
| 보호 디코드 프롬프트 | 32 토큰 |
| 보호 디코드 출력 | 256 토큰 |
| 간섭 프리필 프롬프트 | 3072 토큰 |
| 간섭 프리필 출력 | 1 토큰 |
| 프리필 주입 횟수 | 12 |
| 주입 주기 | 4 디코드 스텝 이후, 매 8 디코드 스텝마다 |
| 그린 분할 | 프리필 32 SM, 디코드 16 SM |

그린 측 서브프로세스는 유효한 개입으로 인정받기 위해 `green_enabled=true`와 `green_api_type="cuda_core"`를 보고해야 한다.

## 5. 결과

### 5.1 WSL2 FlashAttention 기준 대 Windows cuTile

동적 다중 사용자 벤치마크는 133,966개의 생성 토큰을 처리했다.

| 백엔드 | 실행 수 | 평균 시간 | 평균 처리량 | 상대 처리량 |
| :--- | ---: | ---: | ---: | ---: |
| Windows cuTile | 3 | 289.16 s | 463.35 tok/s | 1.00x |
| WSL2 FlashAttention | 3 | 62.06 s | 2159.85 tok/s | 4.66x |

이 결과는 초기 Windows 네이티브 cuTile 백엔드가 성숙한 WSL2 FlashAttention 경로보다 느림을 보여준다. 이는 논문 구성의 약점이 아니라, 기여를 생산 서빙 대체가 아닌 Windows 네이티브 구현 및 성능 분석 사례연구로 위치시키는 근거이다.

### 5.2 하이브리드 프리필 디스패치와 CUDA 그래프 재생

초기 cuTile 기준 이후 다음 최적화를 적용했다. 첫째, PyTorch 고급 인덱싱 기반 KV 캐시 갱신을 cuTile KV-스토어 커널로 교체하여 디코드 그래프 캡처 경로를 수리했다. 둘째, **하이브리드 프리필 디스패치**를 도입했다. 다중 시퀀스 비-프리픽스 프리필은 배치 패딩 래퍼를 사용해 호스트/커널 실행 횟수를 줄이고, 단일 시퀀스와 프리픽스 캐시 프리필은 불필요한 텐서 실체화를 피하기 위해 직접(direct) 또는 페이징(paged) 래퍼를 사용한다. 셋째, 호스트 측 영속 버퍼 재사용과 RMSNorm 퓨전(`F.rms_norm`)으로 호스트 발사 지연과 커널 수를 줄였다.

| cuTile 모드 | 그래프 | 프리필 전략 | 총 시간 | 처리량 | 해석 |
| :--- | :--- | :--- | ---: | ---: | :--- |
| 초기 cuTile 평균 | off | — | 289.16 s | 463.35 tok/s | 초기 Windows cuTile 기준 |
| Hybrid eager | off | hybrid | 223.03 s | 600.66 tok/s | 현재 논문의 긍정적 cuTile 결과 |
| Direct eager | off | direct | 225.70 s | 593.57 tok/s | 단일 시퀀스 프리필 전략 |
| Padded eager | off | padded | 226.45 s | 591.58 tok/s | 다중 시퀀스 배치 패딩 전략 |
| Hybrid graph | on | hybrid | 354.92 s | 377.45 tok/s | 캡처 성공, 처리량 하락 |

eager-hybrid cuTile 경로는 초기 cuTile 평균 대비 처리량을 29.6% 향상시킨다. 반면 CUDA 그래프 재생은 기능적으로는 수리되었지만 현재 성능은 하락한다. 이는 중요한 시스템 교훈이다. 표준 최적화 메커니즘도 그래프 버킷 재생 동작, 커널 세분성, 페이징된 디코드 메모리 접근 패턴과 정렬되어야 종단 간 서빙을 개선할 수 있다.

#### CUDA 그래프 실패 원인 분석

CUDA 그래프 결과는 두 가지 별개의 실패로 해석해야 한다. 첫 번째는 **활성화 실패**이다. 초기 cuTile 디코드 경로는 `slot_mapping[mask]`, `k_cache_flat[valid_slots] = ...`와 같은 PyTorch 부울 마스킹 및 고급 인덱싱으로 KV 캐시 엔트리를 갱신했다. 그 경로는 캡처 중 호스트 런타임과 동적 인덱싱 동작을 호출하여 `cudaErrorStreamCaptureInvalidated` 실패를 일으켰다. 이를 고정 cuTile `store_kvcache_cutile_kernel`로 교체하여 그래프 캡처를 수리했다.

두 번째는 **성능 실패**이다. 캡처가 성공한 뒤에도 현재 cuTile 디코드 그래프는 동적 서빙 워크로드와 아직 정합하지 않는다. 그래프 버킷은 활성 토큰 집합이 요구하는 것보다 큰 고정 형태를 재생하여 과잉 계산을 일으킬 수 있다. 디코드 경로는 작은 커널이 많아 실행 제거의 이점이 커널 세분성에 의해 제한된다. 페이징된 KV 캐시 디코드는 불규칙한 메모리 접근을 보존하므로, 그래프 재생만으로는 그 메모리 병목이 제거되지 않는다. 직접 그래프-프리필 모드는 다중 요청 프리필에서 시퀀스당 실행 압력을 추가로 높여 256-요청 벤치마크에서 부적합해졌다. 따라서 본 논문에서 CUDA 그래프는 **기능적으로 사용 가능해졌으나 아직 가속으로 검증되지 않은 최적화**로 보고한다.

### 5.3 고정 컨텍스트 워크로드를 위한 프리픽스 KV 캐시

| 정적 프리픽스 | 프롬프트 토큰 | 웜 히트율 | 프리필 no-cache | 프리필 warm | TTFT no-cache | TTFT warm | TTFT 감소율 | 프리픽스 변경 TTFT |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 1088 | 94.1% | 1088 | 64 | 119.45 ms | 43.96 ms | 63.2% | 119.66 ms |
| 2048 | 2112 | 97.0% | 2112 | 64 | 226.35 ms | 43.94 ms | 80.6% | 225.73 ms |
| 3072 | 3136 | 98.0% | 3136 | 64 | 347.34 ms | 37.24 ms | 89.3% | 345.98 ms |

웜 프리픽스 캐시는 모든 정적 프리픽스 설정에서 연산 프리필 토큰을 64로 줄인다. 이는 정적 프리픽스가 완전한 KV 블록으로 재사용되고, 64토큰 동적 접미사만 새로 계산됨을 뜻한다. TTFT 감소율은 프리픽스 길이에 따라 커진다. 1024 토큰에서 63.2%, 2048 토큰에서 80.6%, 3072 토큰에서 89.3%이다.

프리픽스 변경 음성대조는 0.0% 캐시 히트와 no-cache 조건에 가까운 TTFT(119.66 / 225.73 / 345.98 ms)를 반환한다. 이는 개선이 측정 노이즈가 아니라 정확한 토큰-프리픽스 재사용에서 비롯됨을 검증한다. 또한 본 절의 수치는 페이징 프리필 어텐션 커널이 골든 레퍼런스와 수치적으로 일치함을 TDD 검증(표준 경로와 프리픽스 캐시 경로 모두)으로 확인한 이후에 얻은 것이다.

이 64토큰 생성 벤치마크에서는 종단 간 처리량도 개선된다(1024: 57.5→61.5, 2048: 45.0→51.8, 3072: 36.8→44.9 tok/s). 이는 프리필이 짧은 생성에서 차지하는 비중이 크기 때문이다. 긴 생성 워크로드에서는 디코드 루프가 총 실행 시간을 지배하므로 처리량 개선은 제한된다. 따라서 정확한 주장은 **고정 컨텍스트 워크로드의 프리필 연산 및 TTFT 감소**이지, 보편적 처리량 향상이 아니다.

### 5.4 그린 컨텍스트 스트레스 결과

반복되는 3072토큰 프리필 간섭 하에서, cuda-core 그린 컨텍스트는 보호 디코드 지연의 완만한 평활화를 보인다(10회 페어 실행). 모든 그린 측 실행은 `green_enabled=true`, `green_api_type="cuda_core"`, 32/16 SM 분할, `green_prefill_resource_source="device_sm_fallback"`을 기록했다.

| 지표 | 평균 델타 | 중앙값 델타 | 개선 run |
| :--- | ---: | ---: | ---: |
| 디코드 스텝 P50 | −1.96% | +1.42% | 4/10 |
| 디코드 스텝 P95 | −5.32% | −6.94% | 9/10 |
| 디코드 스텝 P99 | −6.00% | −6.03% | 10/10 |
| 디코드 간격 P50 | −2.12% | +1.05% | 4/10 |
| 디코드 간격 P95 | −5.60% | −6.05% | 9/10 |
| 디코드 간격 P99 | −0.57% | −0.38% | 6/10 |
| 디코드 간격 최대 | −0.93% | −1.04% | 8/10 |
| 처리량 | +2.40% | +0.69% | 5/10 |

음수 델타는 지연 감소(개선)를 의미한다. 10회 페어 실행에 대한 양측 이항 부호검정(two-sided binomial sign test)은 두 가지 주요 평활화 효과를 확인한다. 디코드 스텝 P99는 10/10 실행에서 개선(p≈0.002)하고, 디코드 스텝 P95와 디코드 간격 P95는 각각 9/10 실행에서 개선한다(p≈0.022). 반면 디코드 간격 P99(6/10, p≈0.75), 디코드 간격 최대(8/10), 처리량(5/10)의 차이는 α=0.05에서 유의하지 않다.

가장 강한 그린 컨텍스트 결과는 TTFT가 아니라, 적대적 프리필 주입 하에서의 디코드 스텝 꼬리 평활화이다. 디코드 스텝 P99는 평균 6.00% 개선되고 10회 페어 실행 전부에서 개선된다. 그러나 디코드 간격 P99와 최대 간격은 기준선에 가깝게 남는다(−0.57%, −0.93%). 이는 현재 순차 Python 엔진이 SM 분할로는 완전히 숨길 수 없는 프리필 일시정지를 여전히 삽입함을 나타낸다. 따라서 그린 컨텍스트는 본 산출물에서 **제한된 자원 격리 메커니즘**이지, 서빙 지연에 대한 완전한 해법이 아니다.

## 6. 고찰

### 6.1 고정 컨텍스트 에이전트가 중요한 이유

에이전트 워크로드는 논리적으로 상수인 컨텍스트에 대해 프리필 비용을 반복 지불한다. Text-to-SQL 에이전트는 스키마와 규칙 컨텍스트를, CUDA 튜터는 강의 자료·코드 스니펫·루브릭을, 지식 에이전트는 선택된 지식 번들 콘텐츠를 반복 포함한다. 프리픽스 KV 캐시는 이 반복 구조를 런타임에 가시화한다.

프리픽스 캐시 결과는 따라서 프롬프트 구성을 서빙 효율과 연결한다. 안정적 컨텍스트는 프롬프트 앞에 배치하고, 휘발성 메타데이터는 뒤로 미루거나 캐시된 프리픽스에서 제외해야 한다. 교육용·데이터 질의 에이전트의 경우, 런타임을 인식한 프롬프트 배치는 모델 변경 없이 TTFT를 줄일 수 있다.

### 6.2 그린 컨텍스트 결과가 가르치는 교훈

그린 컨텍스트는 다른 교훈을 가르친다. 이는 일반 가속기가 아니라 자원 분할 기반(substrate)이다. 산출물는 세 가지 엔지니어링 요점을 보여준다.

첫째, 활성화 유효성을 측정해야 한다. 초기 그린 실행은 런타임이 조용히 폴백했기 때문에 무효였다. 둘째, 서빙 루프가 순차적이면 자원 분할만으로 지연 개선이 보장되지 않는다. 셋째, 반복 프리필 간섭이 있는 스트레스 워크로드에서 그린 컨텍스트는 보호 디코드 지연을 완만하게 평활화하지만 프리필 유발 일시정지를 제거하지는 못한다.

이는 GPU 기능에 지연 분산을 귀속시키기 전에, 그 기능이 실제로 활성화되었는지와 스케줄러가 올바른 간섭 패턴을 만드는지를 증명해야 한다는 흔한 실수를 예방하므로 유용한 교육적 결과다.

### 6.3 부정적 결과의 교육적 가치

본 논문은 부정적·제한적 결과를 의도적으로 포함한다. WSL2 FlashAttention은 초기 Windows cuTile 경로보다 훨씬 빠르다. 하이브리드 eager는 cuTile 경로를 개선하지만, CUDA 그래프 재생은 현재 그래프 버킷과 페이징 디코드 조건에서 처리량을 해친다. 그린 컨텍스트는 활성화 메타데이터를 요구하며 현재 엔진에서 제한적 이점만 보인다. 이러한 결과는 시스템 주장이 어떻게 형성되어야 하는지, 즉 메커니즘, 계측, 대조 조건, 측정, 보수적 해석의 순서를 보여줌으로써 산출물를 교육에 더 유용하게 만든다.

### 6.4 제출 범위

본 논문은 더 넓은 에이전트 프레임워크 기여를 포함하지 않는다. 그러한 주제는 별도의 교육용 지식 에이전트 논문에 적합하다. 본 논문은 추론 엔진 산출물로서의 마이크로-vLLM과 측정된 서빙 루프 동작에 초점을 유지한다.

## 7. 결론

본 논문은 micro-vLLM을 **Windows 네이티브 LLM 서빙 시스템 구현 및 성능 분석 사례연구**로 제시한다. 산출물는 성숙한 리눅스 서빙 스택을 능가하지 않는다. WSL2 FlashAttention은 초기 Windows cuTile 백엔드 대비 평균 4.66배 높은 처리량을 달성한다. 그러나 cuTile KV-스토어 수리, 하이브리드 프리필 디스패치, 호스트 측 영속 버퍼 재사용, RMSNorm 퓨전 이후, eager-hybrid cuTile 경로는 초기 cuTile 평균 대비 29.6% 향상된 600.66 tok/s에 도달한다. 따라서 마이크로-vLLM은 학생과 연구자가 완전한 서빙 루프 인과성을 관찰할 수 있는, 검사·수정 가능한 마이그레이션 및 실험 경로를 제공한다.

가장 강력하게 측정된 결과는 고정 컨텍스트 에이전트 워크로드를 위한 프리픽스 KV 캐시 재사용이다. 웜 캐시는 모든 정적 프리픽스 길이에서 연산 프리필 토큰을 64로 줄이고, 1024·2048·3072토큰 프리픽스에 대해 TTFT를 각각 63.2%, 80.6%, 89.3% 감소시킨다. 프리픽스 변경 음성대조는 정확-프리픽스 의존성을 검증한다.

그린 컨텍스트 실험은 제한된 시스템 통찰을 제공한다. cuda-core 그린 컨텍스트는 RTX 5070 대상 PC에서 성공적으로 활성화되고, 반복 프리필 간섭 하에서 보호 디코드 지연을 완만하게 평활화한다. 디코드 스텝 P99는 평균 6.0% 개선되고 10회 페어 스트레스 실행 전부에서 개선된다(부호검정 p≈0.002). 그러나 디코드 간격 P99와 최대 간격은 순차 프리필 삽입에 지배된 채 남아, 비동기 서빙 루프 없이는 SM 분할만으로 충분하지 않음을 보여준다.

종합하면, 이 결과는 본 논문의 교육적 주장을 뒷받침한다. 효과적인 추론 엔진 교육은 성공적 최적화뿐 아니라 구현 트레이드오프, 활성화 유효성, 부정적 결과, 워크로드 의존성, 그리고 고립된 커널 동작과 종단 간 서빙 동작의 구분까지 가르쳐야 한다.

## 참고문헌

1. W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. Gonzalez, H. Zhang, and I. Stoica, "Efficient Memory Management for Large Language Model Serving with PagedAttention," in Proceedings of the 29th ACM Symposium on Operating Systems Principles (SOSP '23), pp. 611-626, 2023.
2. T. Dao, D. Y. Fu, S. Ermon, A. Rudra, and C. Ré, "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness," in Advances in Neural Information Processing Systems (NeurIPS), vol. 35, pp. 16344-16359, 2022.
3. P. Tillet, H.-T. Kung, and D. Cox, "Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations," in Proceedings of the 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages (MAPL '19), pp. 10-19, 2019.
4. NVIDIA Corporation, "CUDA Python Documentation," NVIDIA Developer Documentation. [Online]. Available: https://nvidia.github.io/cuda-python/ (Accessed: 2025).
5. NVIDIA Corporation, "Green Contexts," in CUDA C++ Programming Guide, NVIDIA Developer Documentation. [Online]. Available: https://docs.nvidia.com/cuda/cuda-c-programming-guide/ (Accessed: 2025).
6. vLLM Project, "Automatic Prefix Caching," vLLM Documentation. [Online]. Available: https://docs.vllm.ai/en/latest/automatic_prefix_caching/apc.html (Accessed: 2025).
7. PyTorch Foundation, "torch.cuda.green_contexts: Granular Resource Partitioning for CUDA Kernels," PyTorch Documentation. [Online]. Available: https://pytorch.org/docs/stable/ (Accessed: 2025).
8. GeeeekExplorer, "nano-VLLM," GitHub repository. [Online]. Available: https://github.com/GeeeekExplorer/nano-vLLM (Accessed: 2025).
9. "KernelAgent," micro-vLLM artifact repository, GitHub repository. [이중 익명 심사를 위해 저장소 URL 생략].

## 부록

### 부록 A.1 프리픽스 캐시 벤치마크 조건

| 조건 | 설명 |
| :--- | :--- |
| `no_cache` | 각 요청 전에 영속 해시 테이블을 비우고 전체 프리필 수행 |
| `warm_cache` | 캐시를 프라이밍한 뒤 동일 정적 프리픽스 재사용 |
| `prefix_changed` | 프리픽스를 변경하여 정확-프리픽스 캐시 히트를 붕괴 |

### 부록 A.2 그린 컨텍스트 스트레스 워크로드

| 매개변수 | 값 |
| :--- | ---: |
| 보호 디코드 프롬프트 | 32 토큰 |
| 보호 디코드 출력 | 256 토큰 |
| 간섭 프리필 프롬프트 | 3072 토큰 |
| 간섭 프리필 출력 | 1 토큰 |
| 프리필 주입 횟수 | 12 |
| 주입 주기 | 4 디코드 스텝 이후, 매 8 디코드 스텝마다 |
| 그린 분할 | 프리필 32 SM, 디코드 16 SM |
