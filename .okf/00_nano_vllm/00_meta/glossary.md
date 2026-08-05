---
type: Document
id: meta.glossary
title: nano-vLLM 핵심 용어 사전 (Glossary)
description: LLM 추론 및 vLLM 서빙 엔진에서 사용되는 핵심 도메인 용어 정의
status: draft
generated:
  by: agent:builder/1.0
  at: 2026-08-05T13:06:00Z
verified:
  - by: human:curator
    at: 2026-08-05T13:06:00Z
---

# nano-vLLM 핵심 용어 사전 (Glossary)

---

## 🔤 주요 용어 정의 (A ~ Z)

### AtomicConcept (원자적 개념)
- 더 이상 하위 개념으로 분해되지 않는 가장 작은 지식 단위. (예: `Prefill Phase`, `KV Cache Token Slot`)

### CompositeConcept (복합 개념)
- 두 개 이상의 원자적 개념 및 하위 복합 개념이 결합되어 형성된 시스템 단위. (예: `Block Manager`, `Autoregressive Loop`)

### Continuous Batching (연속 배치 처리)
- 프롬프트(Prefill)와 생성(Decode) 단계의 요청들을 요청 단위가 아닌 토큰/Iteration 단위로 동적 스케줄링하여 GPU 활용률을 극대화하는 기법.

### Decode Phase (디코드 단계)
- 이전 단계에서 생성된 토큰들을 입력으로 받아 다음 토큰을 1개씩 순차적으로 생성하는 단계 (Latency-sensitive, Memory bandwidth bound).

### KV Cache (Key-Value 캐시)
- 자기회귀(Autoregressive) 생성 시 이전 토큰들의 Key, Value 행렬을 메모리에 저장하여 중복 계산을 방지하는 메커니즘.

### PagedAttention (페이지드 어텐션)
- 가상 메모리의 페이징(Paging) 기법을 어텐션 메커니즘에 도입하여, KV 캐시를 연속되지 않은 물리 메모리 블록(Block) 단위로 파편화 없이 관리하는 아키텍처.

### Prefill Phase (프리필 단계)
- 입력 프롬프트 전체 토큰을 병렬로 가공하여 첫 번째 토큰을 생성하고 최초 KV 캐시를 구축하는 단계 (Compute bound).

### Prefix Caching (프리픽스 캐싱)
- 공통된 프롬프트 프론트엔드(시스템 프롬프트 등)의 KV 캐시를 래직스 트리(Radix Tree) 등으로 재사용하여 프리필 시간을 단축하는 기술.
