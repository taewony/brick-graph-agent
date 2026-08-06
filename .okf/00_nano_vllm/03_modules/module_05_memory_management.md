---
type: Module
id: module.memory_management
title: Memory Management & Block Manager (메모리 관리 및 블록 관리자)
description: PagedAttention의 블록 단위 메모리 관리를 확장하여 GPU 메모리 풀 추상화, 블록 할당 전략, 그리고 CPU-GPU 스와핑까지 포함한 고급 메모리 관리 계층
status: draft
generated:
  by: agent:builder/1.0
  at: 2026-08-06T10:00:00Z
verified:
  - by: human:curator
    at: 2026-08-06T10:00:00Z
prerequisites:
  - module.paged_attention
  - atomic.paged_kv_cache
composes_into:
  - composite.distributed_serving
sources:
  - id: original_lab
    resource: https://hackmd.io/9Ivogn3dRwm3WgA4Kl3fJQ
    title: "nano-vLLM Module 5: Memory Management & Block Manager"
---

# Memory Management & Block Manager (메모리 관리 및 블록 관리자)

## 📌 개요

Module 4의 PagedAttention이 **논리-물리 주소 분리**라는 혁신적 아이디어를 제시했다면, Module 5는 이 아이디어를 **실제 시스템에서 운영 가능한 수준**으로 확장합니다.

본 모듈은 다음 세 가지 핵심 계층을 다룹니다:

1. **Memory Pool (메모리 풀)**: GPU 메모리를 사전 할당하고 블록 단위로 관리하는 추상화 계층
2. **Block Allocator (블록 할당자)**: First-Fit, Best-Fit 등 다양한 할당 전략을 구현
3. **Swap Manager (스와핑 관리자)**: GPU 메모리가 부족할 때 CPU 메모리로 블록을 스왑아웃(Swap-Out)하는 정책

이 세 계층이 함께 동작하여 **PagedAttention을 실제 프로덕션 환경에서 안정적으로 운영**할 수 있는 기반을 제공합니다.

---

## 🧩 포함 원자적 개념

- `atomic.memory_pool` — GPU 메모리를 고정 크기 블록들의 풀(Pool)로 추상화
- `atomic.block_allocator` — 블록 할당 및 해제를 담당하는 정책 (First-Fit, Best-Fit, Next-Fit)
- `atomic.swap_manager` — GPU 메모리 부족 시 블록을 CPU 메모리로 스왑아웃/스왑인하는 관리자
- `atomic.swap_policy` — 스와핑 대상 선정 정책 (LRU, LFU, FIFO 등)

## 🏗️ 복합 개념 (Composite Concept)

- `composite.block_manager` — Memory Pool + Block Allocator + Swap Manager를 통합한 최상위 블록 관리자

## 🔗 관련 관계

- **PREREQUISITES**: `paged_kv_cache`, `block_table`
- **PREREQUISITE_OF**: `distributed_serving` (Module 7)

---

## 🧱 1. Memory Pool: GPU 메모리의 추상화

### 1.1 개념 정의

**Memory Pool**은 GPU의 HBM(High Bandwidth Memory)을 **고정 크기 블록(Fixed-Size Blocks)** 들의 집합으로 추상화합니다.

- vLLM은 시스템 초기화 시점에 **전체 GPU 메모리의 일정 비율(예: 90%)** 을 Memory Pool로 예약(Reserve)합니다.
- 이 풀은 더 이상 개별 Sequence가 아닌 **블록 단위**로 할당/해제가 이루어지는 **공유 자원(Shared Resource)** 입니다.

### 1.2 메모리 단편화 제거 원리

기존 연속 할당 방식에서는 Sequence A(1024 tokens)가 종료되어도 해제된 공간이 **조각(Fragment)** 으로 남아 다른 Sequence가 사용하기 어려웠습니다.

Memory Pool + PagedAttention에서는:

1. 모든 블록의 크기가 **동일(예: 16 tokens)** 하므로, 해제된 블록은 **즉시 다른 Sequence가 재사용** 가능
2. 블록 단위 할당으로 **외부 단편화(External Fragmentation)가 사실상 제로(Zero)** 에 수렴
3. 메모리 활용률이 50~70%에서 **~100%** 로 향상

```
Memory Pool 상태 (블록 크기 = 16 tokens):
┌─────────────────────────────────────────────────────────────┐
│ Block 0 │ Block 1 │ Block 2 │ Block 3 │ Block 4 │ Block 5 │
│ (Seq A) │ (FREE)  │ (Seq B) │ (FREE)  │ (Seq C) │ (FREE)  │
└─────────────────────────────────────────────────────────────┘
                    ↑
            Seq D가 Block 1을 즉시 할당받음 (연속 공간 필요 없음!)
```

### 1.3 메모리 풀 크기 결정

Memory Pool의 크기는 다음 요소를 고려하여 결정됩니다:

| 요소 | 설명 |
|---|---|
| **모델 가중치 (Weights)** | 추론에 사용되는 모델 파라미터 크기 (예: 70B 모델 ≈ 140GB) |
| **워크로드 특성** | 평균 Sequence 길이, 동시 요청 수, 최대 배치 크기 |
| **안전 마진 (Safety Margin)** | 예기치 않은 메모리 요청을 대비한 여유 공간 (보통 5~10%) |

---

## 🧱 2. Block Allocator: 블록 할당 전략

### 2.1 개념 정의

**Block Allocator**는 Memory Pool 내에서 **어떤 블록을 어떤 Sequence에 할당할지** 결정하는 정책 엔진입니다.

### 2.2 할당 전략 비교

| 전략 | 설명 | 장점 | 단점 |
|---|---|---|---|
| **First-Fit** | 가장 먼저 발견된 충분한 크기의 빈 블록을 할당 | 구현 단순, 속도 빠름 | 작은 조각들이 앞쪽에 모일 수 있음 |
| **Best-Fit** | 요청 크기에 가장 근접한 빈 블록을 할당 | 메모리 활용률 최고 | 검색 오버헤드 큼 |
| **Next-Fit** | 마지막 할당 위치 이후부터 탐색 시작 | First-Fit보다 공평함 | 성능이 불안정할 수 있음 |

vLLM은 **블록 크기가 고정**되어 있으므로, 모든 블록이 동일한 크기(예: 16 tokens)를 가집니다. 따라서 **할당 전략의 차이가 거의 없으며**, 단순히 **사용 가능한 블록을 하나 반환**하면 됩니다.

### 2.3 블록 할당 및 해제 API

```python
# Block Allocator 인터페이스 (개념적)
class BlockAllocator:
    def allocate(self, num_blocks: int) -> List[int]:
        """지정된 개수만큼의 물리적 블록 번호를 할당"""
        pass

    def free(self, block_ids: List[int]) -> None:
        """할당된 블록들을 메모리 풀로 반환"""
        pass

    def get_num_free_blocks(self) -> int:
        """현재 사용 가능한 블록 개수 반환"""
        pass
```

---

## 🧱 3. Swap Manager: 메모리 오버플로우 대비

### 3.1 개념 정의

GPU 메모리가 부족한 상황(예: 동시 요청 급증)에서 **일부 Sequence의 KV Cache 블록을 CPU 메모리로 스왑아웃(Swap-Out)** 하고, 필요할 때 다시 스왑인(Swap-In)하는 메커니즘입니다.

### 3.2 스와핑이 필요한 이유

GPU 메모리는 비싸고 용량이 제한적입니다(예: A100 80GB). 반면 CPU 메모리는 상대적으로 저렴하고 대용량(수백 GB)입니다.

- **스왑아웃**: GPU 메모리가 부족할 때, 우선순위가 낮은 Sequence의 블록을 CPU 메모리로 이동
- **스왑인**: 해당 Sequence가 다시 처리될 때, CPU에서 GPU로 블록을 복원

### 3.3 스왑 정책 (Swap Policy)

어떤 블록을 스왑아웃할지 결정하는 정책은 시스템 성능에 큰 영향을 미칩니다.

| 정책 | 설명 | 적합한 상황 |
|---|---|---|
| **LRU (Least Recently Used)** | 가장 오래전에 사용된 블록부터 스왑 | 일반적 워크로드 |
| **LFU (Least Frequently Used)** | 사용 빈도가 가장 낮은 블록부터 스왑 | 빈도 차이가 큰 워크로드 |
| **FIFO (First In First Out)** | 가장 먼저 할당된 블록부터 스왑 | 구현 단순, 예측 가능 |
| **Priority-Based** | 우선순위가 낮은 요청의 블록부터 스왑 | SLA(Service Level Agreement)가 중요한 환경 |

### 3.4 스왑 성능 영향

스와핑은 **CPU-GPU 간 메모리 복사**를 수반하므로, 다음과 같은 트레이드오프가 존재합니다:

| 측면 | 영향 |
|---|---|
| **처리량 (Throughput)** | 스와핑이 발생하면 GPU가 대기해야 하므로 처리량이 일시적으로 감소 |
| **메모리 효율** | GPU 메모리 부족으로 인한 OOM을 방지하여 시스템 안정성 향상 |
| **지연 시간 (Latency)** | 스왑인에 추가 시간이 소요되어 개별 요청의 지연 시간이 증가할 수 있음 |

> 💡 **핵심**: 스와핑은 **"메모리가 부족할 때 시스템이 죽지 않도록"** 하는 **안전장치(Safety Net)** 입니다. 과도한 스와핑은 오히려 성능을 저하시키므로, Memory Pool 크기와 스왑 정책을 신중히 튜닝해야 합니다.

---

## 🧱 4. Composite: Block Manager (통합 블록 관리자)

### 4.1 개념 정의

**Block Manager**는 Memory Pool, Block Allocator, Swap Manager를 하나로 통합한 **최상위 메모리 관리 컴포넌트**입니다.

### 4.2 통합 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                      Block Manager                              │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │ Memory Pool │  │Block Allocator│  │    Swap Manager        │ │
│  │ (GPU 메모리 │  │ (할당 전략)   │  │  (CPU-GPU 스와핑)      │ │
│  │  추상화)    │  │             │  │                         │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │   Scheduler (Module 3)         │
                    │   - 메모리 가용량 조회         │
                    │   - 배치 크기 결정             │
                    └───────────────────────────────┘
```

### 4.3 Block Manager의 주요 책임

1. **메모리 할당/해제**: Sequence의 KV Cache 필요량에 따라 블록을 할당하고 해제
2. **메모리 사용량 모니터링**: 현재 사용 중인 블록 수, 가용 블록 수를 실시간 추적
3. **스와핑 트리거**: 메모리 부족 시 Swap Manager를 호출하여 블록을 CPU로 이동
4. **Block Table 관리**: 각 Sequence의 논리-물리 블록 매핑 테이블을 유지/갱신

---

## 📊 Module 5 성능 영향

| 지표 | PagedAttention 단독 (Module 4) | + Memory Pool (Module 5) | + Swap Manager (Module 5) |
|---|---|---|---|
| **메모리 단편화** | 제로 (Zero) | 제로 (Zero) | 제로 (Zero) |
| **메모리 활용률** | ~100% | ~100% | ~100% |
| **OOM 발생** | 거의 없음 | 거의 없음 | **완전히 제거** |
| **최대 동시 요청 수** | Memory Pool 크기에 제한 | Memory Pool 크기에 제한 | **Memory Pool + CPU 스왑 용량** |
| **시스템 안정성** | 높음 | 높음 | **매우 높음 (Graceful Degradation)** |

---

## 📝 요약: Brick-by-Brick 관점

| 단계 | 학습한 개념 | 다음 개념으로의 연결 |
|---|---|---|
| **Module 1** | Autoregressive Loop + KV Cache | KV Cache가 메모리 병목임을 인지 |
| **Module 2** | KV Cache 메모리 문제 (연속 할당 + 단편화) | 메모리 문제 해결의 필요성 확인 |
| **Module 3** | Continuous Batching + Scheduler | 계산 효율 최적화 |
| **Module 4** | PagedAttention (논리-물리 분리) | 메모리 단편화 제거 |
| **Module 5** | **Memory Pool + Block Allocator + Swap Manager** | **PagedAttention을 실제 시스템에서 운영 가능하게 하는 인프라** |
| **Module 6** | Prefix Caching | Memory Pool 기반 캐시 재사용 최적화 |
| **Module 7** | Distributed Serving | Block Manager를 여러 GPU/노드로 확장 |

> 💡 **핵심 인사이트**: Module 4의 PagedAttention이 **"어떻게 메모리를 조각 없이 관리할까?"** 라는 질문에 답했다면, Module 5는 **"그렇게 관리된 메모리를 어떻게 실제 시스템에서 안정적으로 운영할까?"** 라는 질문에 답합니다. Memory Pool은 **자원의 추상화**, Block Allocator는 **자원의 할당 전략**, Swap Manager는 **자원 부족 시의 안전장치**를 제공하여, PagedAttention을 **프로덕션 환경에 배포 가능한 수준**으로 끌어올립니다.