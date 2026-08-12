## Module 3: Serving Many - Batching & Scheduling

### 🎯 이 모듈의 목표

Module 2까지는 **단일 시퀀스(single sequence)** 의 추론 메커니즘을 이해했다면, Module 3에서는 **수백 개의 동시 요청(concurrent requests)** 을 어떻게 효율적으로 처리할지에 대한 **시스템 아키텍처**로 시야를 확장합니다. 핵심은 **Continuous Batching(연속 배치)** 과 전용 **Scheduler(스케줄러)** 를 통해 개별 요청의 지연 시간을 **전체 시스템의 처리량(throughput)** 으로 전환하는 방법입니다.

---

### 🧱 Lab 3.1: From Static to Continuous Batching

#### 3.1.1 핵심 개념: 배치(Batching)의 중요성

GPU에서 하나의 토큰을 처리하는 것과 수천 개의 토큰을 한 번에 처리하는 것은 **효율성 측면에서 거의 차이가 없습니다**. 즉, GPU는 **한 번에 많은 연산을 몰아줄 때** 최고의 성능을 냅니다. 따라서 **"어떻게 배치를 구성하느냐"** 가 시스템 성능의 핵심입니다.

#### 3.1.2 Static Batching (정적 배치) — 전통적 방식의 한계

**Static Batching**은 고정된 요청 집합으로 배치를 구성하고, 해당 배치의 모든 요청이 **현재 단계(phase)를 완전히 마칠 때까지** 처리한 후 다음 배치로 넘어가는 방식입니다.

> **비유하자면**: 식당에서 10명이 같이 와서 각자 다른 요리를 주문했는데, **모든 요리가 다 나올 때까지** 다음 손님을 받지 않는 것과 같습니다. 한 손님의 요리가 늦어지면 다른 9명도 함께 기다려야 합니다.

**문제점**:
- 디코드 단계에서 **하나의 긴 요청**이 전체 배치를 **병목(block)** 시킵니다.
- GPU는 짧은 요청들이 기다리는 동안 **유휴 상태(idle)** 로 남습니다.
- 결과적으로 **GPU 활용률이 30~50%** 에 그치고, **긴 꼬리 지연 시간(high tail latency)** 이 발생합니다.

#### 3.1.3 Continuous Batching (연속 배치) — vLLM의 혁신

**Continuous Batching**(Iteration-Level Scheduling이라고도 함)은 **매 스케줄링 반복(iteration)마다 배치를 새롭게 구성**하는 방식입니다.

> **비유하자면**: 식당에서 **요리가 나오는 대로** 손님을 받고, 각 손님의 **다음 요리**가 준비되면 바로 바로 서빙합니다. 한 손님의 요리가 늦어져도 다른 손님들은 계속해서 서비스를 받습니다.

**작동 과정**:

| 단계 | 설명 |
|---|---|
| **1. Selection (선택)** | 스케줄러가 **실행 준비가 된**(처리할 토큰이 있는) 요청들을 선택합니다 |
| **2. Packing (패킹)** | 각 요청의 **다음 계산 단계**를 하나의 배치로 묶습니다 |
| | • **Prefill 단계**: 새 요청의 전체 프롬프트 처리 |
| | • **Decode 단계**: 진행 중인 요청의 다음 토큰 1개 처리 |
| **3. Execution (실행)** | GPU가 이 **혼합(heterogeneous) 배치**를 실행합니다 |
| **4. Update & Dissolve (갱신 및 해체)** | 결과를 반환하고, 요청 상태를 갱신하며, 배치는 **즉시 해체(dissolve)** 됩니다. 스케줄러는 **바로 다음 배치**를 구성합니다 |

**효과**:
- GPU가 **지속적으로 포화(saturated)** 상태를 유지합니다.
- 짧은 요청은 **차단 없이** 완료되고 리소스를 해제합니다.
- **GPU 활용률 90% 이상** 달성.

#### 3.1.4 다이어그램으로 보는 차이

**Diagram 3.1.a: GPU Utilization Comparison**

```
Static Batching (정적 배치):
[Batch 1 처리] → [Decode A만 실행] → [유휴(idle)] → [Batch 2 처리] → ...
▉▉▉▉▉▉▉▉▉▉ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ ████████ ░░░░░░░░░░░░░░░░
          ↑ GPU가 놀고 있음 (30~50% 활용률)

Continuous Batching (연속 배치):
[Prefill A,B,C] → [Decode A,B,C] → [Decode A,C + Prefill D] → [Decode A,C,D + Prefill E] → ...
▉▉▉▉▉▉▉▉▉▉ ▉▉▉▉▉ ▉▉▉▉▉▉▉▉▉▉▉ ▉▉▉▉▉▉▉▉▉▉▉▉▉▉ ▉▉▉▉▉▉
          ↑ GPU가 쉬지 않고 일함 (90%+ 활용률)
```

**Diagram 3.1.b: Batch Composition Evolution**

```
Iteration 1: [Req A: Prefill] [Req B: Prefill]     ← 모두 프리필
Iteration 2: [Req A: Decode] [Req B: Decode] [Req C: Prefill]  ← 디코드 + 새 프리필 혼합
Iteration 3: [Req A: Decode] [Req C: Decode] [Req D: Prefill]  ← 계속 혼합
```

---

### ⚙️ Scheduler (스케줄러)의 역할

Continuous Batching을 가능하게 하는 핵심 컴포넌트는 **Scheduler**입니다. Scheduler는 다음을 담당합니다:

1. **대기 큐(Waiting Queue) 관리**: 새로 들어온 요청들을 대기시킵니다.
2. **우선순위 결정**: Prefill 작업을 Decode 작업보다 **우선 처리**하여 전체 처리량을 최대화합니다. (Prefill은 Compute-Bound, Decode는 Memory-Bound이므로 우선순위 전략이 중요)
3. **2단계 스케줄링(Two-Phase Scheduling)** 구현:
   - **Phase 1 (Prefill 우선)**: 대기 큐에서 Prefill 가능한 시퀀스를 우선 스케줄링
   - **Phase 2 (Decode 처리)**: Prefill이 완료된 후 Decode 단계의 시퀀스들을 배치로 구성

---

## Module 4: PagedAttention - The Logical Abstraction

> 💡 **참고**: HackMD 문서에서 Module 4는 Module 3의 **직접적인 후속 주제**로, "메모리 관리의 혁신"을 다룹니다.

### 🎯 이 모듈의 목표

Module 3에서 **"계산(compute) 효율"** 을 최적화했다면, Module 4에서는 **"메모리(memory) 효율"** 을 최적화합니다. 구체적으로, Module 2에서 확인한 **KV Cache의 메모리 문제**(연속 할당 시 발생하는 **외부 단편화/external fragmentation**)를 해결하는 것이 목표입니다.

### 🧱 PagedAttention의 핵심 개념

#### 문제: 기존 KV Cache 할당 방식의 한계

기존 방식은 각 요청의 KV Cache를 **연속된(contiguous) 메모리 공간**에 할당했습니다. 이 방식은:

- **외부 단편화(External Fragmentation)** 를 심각하게 유발합니다.
- GPU 메모리가 충분히 남아 있어도 **할당 가능한 연속 공간이 없어** OOM(Out Of Memory) 오류가 발생합니다.
- 결과적으로 **시스템 처리량이 심각하게 제한**됩니다.

> **비유하자면**: 도서관에서 각 사람이 **연속된 책장**을 통째로 배정받는다고 생각해보세요. 사람들이 책을 다 읽고 나가도, 그 공간이 **조각조각** 나뉘어 있어서 새로 온 사람은 **자신의 책을 모두 꽂을 수 있는 연속된 공간**을 찾기 어렵게 됩니다.

#### 해결책: PagedAttention의 논리적 추상화

PagedAttention은 **운영체제의 가상 메모리 페이징(paging) 개념**을 차용합니다.

**핵심 아이디어**: KV Cache를 **고정 크기 블록(fixed-size blocks)** 단위로 관리합니다.

> **비유하자면**: 이제는 각 사람이 **연속된 책장**이 아니라, **곳곳에 흩어진 책장의 빈 칸(블록)** 을 할당받습니다. 필요한 만큼의 빈 칸을 모아서 사용하고, 반납하면 그 칸은 다시 다른 사람이 사용할 수 있습니다. **메모리 낭비가 거의 없어집니다**.

#### 3가지 핵심 추상화

**1. Logical Block (논리적 블록) vs Physical Block (물리적 블록)** 

| 개념 | 설명 |
|---|---|
| **Logical Block** | 각 요청이 **논리적으로** 자신의 KV Cache를 구성하는 블록 번호 (연속적, 0, 1, 2, ...) |
| **Physical Block** | GPU 메모리에 **실제로** 할당된 블록의 물리적 주소 (불연속적, 7, 3, 15, ...) |

**2. Block Table (블록 테이블)** 

Logical Block 번호 → Physical Block 번호로 변환해주는 **매핑 테이블**입니다.

```
요청 A의 Block Table:
Logical Block 0 → Physical Block 7
Logical Block 1 → Physical Block 3
Logical Block 2 → Physical Block 15
Logical Block 3 → Physical Block 22
```

**3. Slot Mapping (슬롯 매핑)** 

특정 토큰의 KV Cache가 **어느 물리적 블록의 몇 번째 슬롯(slot)** 에 저장되어 있는지 계산합니다.

```
계산 공식:
- logical_block = token_position // BLOCK_SIZE  (어느 논리 블록?)
- physical_block = block_table[logical_block]   (물리 블록은?)
- slot = token_position % BLOCK_SIZE            (블록 내 몇 번째?)
```

### 📊 효과: Near-Zero Waste

| 항목 | 기존 방식 (Contiguous) | PagedAttention |
|---|---|---|
| **메모리 단편화** | 심각한 외부 단편화 | **거의 제로 (near-zero)**  |
| **메모리 활용률** | 50~70% | **~100%**  |
| **OOM 발생** | 빈번 | **거의 없음** |
| **처리량** | 제한적 | **10~23배 향상**  |

---

## 🔗 Module 3과 Module 4의 연결: 계산 효율 + 메모리 효율 = vLLM의 완성

이 두 모듈은 **vLLM의 양대 산맥**으로, 서로 **불가분의 관계**입니다.

### 1. 문제 정의의 연속성

- **Module 2**에서 "KV Cache의 메모리 문제"를 확인했습니다.
- **Module 3**은 이 문제를 해결하기 전에 **"계산 워크로드를 어떻게 효율적으로 조직할까?"** 를 먼저 다루었습니다.
- **Module 4**는 그 후 **"메모리 문제를 어떻게 해결할까?"** 를 다룹니다.

### 2. 상호 의존성 (Interdependence)

| 관점 | 설명 |
|---|---|
| **Module 3 → Module 4** | Continuous Batching이 **Prefill과 Decode가 혼합된 배치**를 지속적으로 생성합니다. 이 혼합 배치에서 각 요청의 KV Cache는 **동적으로 증가**합니다. PagedAttention은 이 **동적이고 불규칙한** KV Cache 성장을 **효율적으로 관리**하는 방법을 제공합니다. |
| **Module 4 → Module 3** | PagedAttention의 **블록 단위 할당**은 Scheduler가 **메모리 용량을 정확히 예측**하고 **배치 크기를 최적화**할 수 있게 합니다. 즉, 메모리 관리가 **스케줄링의 의사 결정**을 더 정교하게 만듭니다. |

### 3. 종합: vLLM의 3대 혁신

```
┌─────────────────────────────────────────────────────────────────────┐
│                        vLLM 3대 혁신                              │
├─────────────────────────────────────────────────────────────────────┤
│  1. Continuous Batching (Module 3)                                │
│     → GPU 활용률 30~50% → 90%+                                   │
│     → "계산(compute) 효율"의 혁신                                 │
├─────────────────────────────────────────────────────────────────────┤
│  2. Paged KV Cache (Module 4)                                     │
│     → 메모리 단편화 제거, near-zero waste                         │
│     → "메모리(memory) 효율"의 혁신                                │
├─────────────────────────────────────────────────────────────────────┤
│  3. Hybrid Scheduling (두 모듈의 결합)                            │
│     → Continuous Batching + PagedAttention이 함께 동작            │
│     → Prefill 우선 + 메모리 인지형 스케줄링          │
└─────────────────────────────────────────────────────────────────────┘
```

### 4. 하나의 시스템으로 통합

```
[요청 유입]
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│  Scheduler (Module 3)                                           │
│  - 대기 큐에서 Prefill 요청 우선 선택               │
│  - Block Manager(Module 4)에 메모리 할당 요청           │
└─────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│  Block Manager (Module 4)                                       │
│  - PagedAttention으로 KV Cache를 블록 단위로 할당   │
│  - Block Table로 논리→물리 주소 매핑                     │
│  - Slot Mapping으로 정확한 메모리 위치 계산            │
└─────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│  GPU Execution (두 모듈의 협업)                                 │
│  - Continuous Batching으로 혼합 배치 실행         │
│  - PagedAttention 커널로 불연속 메모리 접근         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📝 요약: Brick-by-Brick 관점

| 단계 | 학습한 개념 | 다음 개념으로의 연결 |
|---|---|---|
| **Module 1** | Autoregressive Loop + KV Cache | KV Cache가 메모리 병목임을 인지 |
| **Module 2** | KV Cache 메모리 문제 (단편화) | 메모리 문제 해결의 필요성 확인 |
| **Module 3** | Continuous Batching + Scheduler | 계산 효율 최적화, **동적 KV Cache 성장** 발생 |
| **Module 4** | PagedAttention (논리적 추상화) | **동적 KV Cache를 효율적으로 관리**하는 방법 제시 |
| **Module 5+** | 전체 시스템 통합 | Module 3 + 4가 결합된 vLLM 아키텍처 완성 |

> 💡 **핵심 인사이트**: Module 3는 **"GPU를 어떻게 쉬지 않게 할까?"** (계산 효율), Module 4는 **"GPU 메모리를 어떻게 낭비 없이 쓸까?"** (메모리 효율)라는 **상호 보완적인 질문**에 답합니다. 이 두 가지가 함께 작동할 때 vLLM은 **10~23배의 처리량 향상**을 달성할 수 있습니다.