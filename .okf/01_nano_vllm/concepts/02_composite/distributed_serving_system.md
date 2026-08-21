---
type: CompositeConcept
id: composite.distributed_serving
title: Distributed Serving System (분산 서빙 시스템)
description: Tensor Parallelism 기반의 다중 GPU/다중 노드 분산 실행 엔진 위에, 로드 밸런서, 장애 허용(Fault
  Tolerance), 분산 KV 캐시를 통합하여 외부 클라이언트에게 고가용·고성능의 KV-API를 제공하는 엔드-투-엔드 서비스 스택
status: draft
generated:
  by: agent:builder/1.0
  at: 2026-08-06 17:00:00+00:00
verified:
- by: human:curator
  at: 2026-08-06 17:00:00+00:00
components:
- composite.distributed_executor
- atomic.distributed_kv
- atomic.load_balancer
- atomic.fault_tolerance
- atomic.observability
prerequisites:
- atomic.distributed_kv
- atomic.fault_tolerance
- atomic.master_worker
- atomic.shared_memory_ipc
- atomic.tensor_parallelism
- composite.block_manager
- composite.paged_attention_manager
- module.prefix_caching
sources:
- id: distributed_execution
  resource: https://deepwiki.com/liguodongiot/nano-vllm/5-distributed-execution
  title: nano-vLLM Distributed Execution
- id: vllm_parallelism
  resource: https://docs.vllm.ai/en/v0.21.0/serving/parallelism_scaling/
  title: vLLM Parallelism and Scaling
---

# Distributed Serving System (분산 서빙 시스템)

## 📌 개요

**Distributed Serving System**은 Module 7의 최상위 복합 개념으로, 지금까지 구축한 모든 단일 GPU 최적화(Continuous Batching, PagedAttention, Memory Pool, Prefix Caching)를 **다중 GPU 및 다중 노드 환경으로 확장**하고, 이를 **외부 클라이언트가 사용할 수 있는 안정적인 서비스**로 포장(Wrapping)합니다.

이 시스템은 크게 두 가지 계층으로 구성됩니다:

1. **분산 실행 엔진 (Distributed Execution Engine)** – Tensor Parallelism, NCCL 통신, Master-Worker 패턴을 통해 모델을 여러 GPU에 분할하고 동기화
2. **서비스 인프라 (Service Infrastructure)** – 로드 밸런서, 장애 허용, 분산 KV 캐시, 모니터링을 통해 고가용성과 확장성 제공

> 💡 **실제 배포 관점**: 이 시스템은 클라이언트가 `POST /kv/{key}` 요청을 보내면, 수천~수만 QPS를 처리하면서도 1~2ms의 지연 시간을 유지하고, GPU 장애 시 자동으로 복구되는 **운영 수준(Production-Grade)**의 서비스를 구현합니다.

---

## 🏗️ 전체 아키텍처 (End-to-End Architecture)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              클라이언트 (Client)                            │
│                    HTTP/gRPC/JSON-RPC 요청 (KV-API)                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        1. Load Balancer (로드 밸런서)                       │
│  - 라운드-로빈 / 가중 라운드-로빈 / 최소 연결 정책 적용                     │
│  - 헬스 체크(Health Check)로 가용 노드 풀 유지                              │
│  - 요청을 적절한 KV 노드로 라우팅                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                   2. Distributed KV (분산 키-값 저장소)                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Partitioner (파티셔너)                                            │   │
│  │  - 요청 키 → hash(key) % num_partitions 로 파티션 번호 계산        │   │
│  │  - 메타 서비스에서 해당 파티션의 현재 리더(Primary) 노드 조회      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Replica Manager (복제 관리자)                                      │   │
│  │  - 쓰기: 리더에 쓰고, 동기/비동기 방식으로 복제본에 전파           │   │
│  │  - 읽기: 일관성 요구에 따라 리더 또는 팔로워에서 읽기               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ (실제 추론 연산)
┌─────────────────────────────────────────────────────────────────────────────┐
│                   3. Distributed Executor (분산 실행 엔진)                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Tensor Parallelism (TP) Group                                     │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐              │   │
│  │  │  GPU 0   │ │  GPU 1   │ │  GPU 2   │ │  GPU 3   │              │   │
│  │  │ (Rank 0) │ │ (Rank 1) │ │ (Rank 2) │ │ (Rank 3) │              │   │
│  │  │ Master   │ │ Worker   │ │ Worker   │ │ Worker   │              │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘              │   │
│  │       │            │            │            │                     │   │
│  │       └────────────┴────────────┴────────────┘                     │   │
│  │                    NCCL All-Reduce (집단 통신)                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│  - 모델 가중치를 TP 크기만큼 분할(Shard)하여 각 GPU에 적재               │
│  - Prefill/Decode 단계에서 All-Reduce로 중간 결과 동기화                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                   4. Fault Tolerance (장애 허용)                           │
│  - Health Checker: 노드 응답 지연/실패 감지 → 장애 선언                   │
│  - Leader Election: 장애 노드가 담당하던 파티션의 리더를 다른 노드로 교체 │
│  - Replica Rebalancer: 새로운 복제본을 생성하고, 복제 지연 최소화          │
│  - Load Balancer 라우팅 테이블 자동 업데이트 → 트래픽 재라우팅             │
│  - 전체 복구 시간: 수초 이내 (클라이언트는 투명하게 서비스 이용 가능)      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                   5. Observability (모니터링, 선택)                        │
│  - 요청 수, 성공률, 레이턴시, 복제 지연 등 실시간 대시보드                │
│  - Prometheus + Grafana 또는 자체 메트릭 수집기 연동                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🧩 구성 요소 (Components)

| 구성 요소 | 설명 |
|---|---|
| `composite.distributed_executor` | Tensor Parallelism, NCCL, Master-Worker를 통합한 분산 실행 엔진 (Module 7 핵심) |
| [`atomic.distributed_kv`](../03_atomic/distributed_kv.md) | 각 GPU의 KV Cache 블록을 파티셔닝하고 복제하여 분산 저장소로 추상화 |
| [`atomic.load_balancer`](../03_atomic/load_balancer.md) | 클라이언트 요청을 가용 노드에 분산하고, 헬스 체크로 노드 상태 관리 |
| [`atomic.fault_tolerance`](../03_atomic/fault_tolerance.md) | 장애 감지, 리더 재선출, 복제 재배치를 통해 시스템 자가 회복 |
| `atomic.observability` | 메트릭, 로깅, 분산 트레이싱을 통한 운영 가시성 확보 (선택적) |

---

## 🔗 주요 동작 흐름 (End-to-End Flow)

### 1. 쓰기/읽기 요청 처리 (KV-API)

| 단계 | 구성 요소 | 상세 작업 |
|---|---|---|
| **1. 요청 수신** | Load Balancer | 클라이언트 HTTP/gRPC/JSON-RPC 요청 수신, 헬스 체크 결과에 따라 가용 인스턴스 풀 유지 |
| **2. 라우팅 결정** | Load Balancer (알고리즘) | 라운드-로빈, 가중 라운드-로빈, 최소 연결 등 정책 적용 → 선택된 KV 노드 ID 반환 |
| **3. 파티션 매핑** | Distributed KV – Partitioner | 요청 키 → `hash(key) % num_partitions`로 파티션 번호 계산, 메타 서비스에서 해당 파티션의 현재 리더(Primary) 노드 식별 |
| **4. 쓰기/읽기 실행** | Distributed KV – Replica Manager | 쓰기: 리더에 쓰고, 복제본에 동기/비동기 복제 <br> 읽기: 일관성 요구에 따라 리더 또는 팔로워에서 읽기 |
| **5. 응답 반환** | Load Balancer | KV 노드에서 받은 결과를 클라이언트에 그대로 전달 |

### 2. 장애 복구 시퀀스 (Fault Recovery)

1. Node A가 Heartbeat 실패 → Health Checker가 장애 감지
2. Fault Tolerance 모듈이 Node A가 담당하던 **파티션 12의 리더**를 Node B로 교체 (Leader Election)
3. Load Balancer가 라우팅 테이블을 즉시 업데이트하고, 향후 요청을 Node B로 전송
4. Replica Rebalancer가 Node C에 새로운 복제본을 생성하고, 복제 지연을 최소화
5. 전체 과정은 **수초 이내**에 완료되어 클라이언트는 투명하게 서비스 이용 가능

---

## 📐 일관성 및 가용성 옵션 (Consistency & Availability)

분산 KV 시스템은 사용 사례에 따라 다양한 일관성 수준을 제공합니다.

| 옵션 | 설명 | 사용 시점 |
|---|---|---|
| **Strong Consistency** (강한 일관성) | 쓰기 성공 시 모든 복제본이 최신 로그에 도달해야 함. 장애 시 쓰기 일시 중단. | 거래형 데이터, 정확성 필수 (예: 결제, 계정 잔액) |
| **Eventual Consistency** (최종 일관성) | 쓰기 직후 일부 복제본이 업데이트되지 않아도 OK. 백그라운드 동기화. | 캐시, 로그, 대용량 비정형 데이터 (추천, 로그 수집) |
| **Read-Only Fallback** (읽기 전용 대체) | 리더 장애 시 일시적으로 읽기 전용 모드로 전환, 쓰기 거부 후 복구 시 재개. | 높은 가용성이 필수인 서비스 (읽기 트래픽이 많은 경우) |

vLLM의 KV Cache는 **Eventual Consistency**를 기본으로 하며, 필요에 따라 Strong Consistency를 선택할 수 있습니다.

---

## 🔗 Module 6 (Prefix Caching)과의 통합

분산 환경에서 Prefix Caching은 **모든 GPU에서 동일한 해시 테이블을 공유**해야 합니다.

- 해시 테이블(`hash_to_block_id`)은 **모든 Rank에서 동기화(Synchronized)**되어야 함
- 새 블록이 할당되면 **NCCL Broadcast**를 통해 모든 Rank에 전파
- 캐시 히트 시에는 **NCCL 통신 없이** 각 GPU가 로컬 해시 테이블에서 바로 블록 ID를 조회 (Zero Overhead)

```python
# 분산 Prefix Caching 동기화 예시
def sync_hash_table_across_ranks():
    if rank == 0:  # Master
        serialized = pickle.dumps(hash_to_block_id)
        torch.distributed.broadcast(torch.tensor(len(serialized)), src=0)
        torch.distributed.broadcast(torch.tensor(serialized), src=0)
    else:  # Workers
        length = torch.distributed.recv(...)
        serialized = torch.distributed.recv(...)
        hash_to_block_id = pickle.loads(serialized)
```

---

## 📊 성능 및 확장성 (Performance & Scalability)

| 지표 | 단일 GPU | 분산 서빙 (TP=4) | 분산 서빙 + Prefix Caching |
|---|---|---|---|
| **최대 모델 크기** | 단일 GPU 메모리 한도 | **GPU 수 × 메모리** | 동일 |
| **KV Cache 용량** | 단일 GPU 여유분 | **GPU 수 × 여유분** | 동일 |
| **처리량 (Throughput)** | 1x | **~3.5x** (통신 오버헤드) | **~5-10x** (캐시 히트 시) |
| **지연 시간 (Latency)** | 기준 | 약간 증가 (All-Reduce) | **Prefix Hit 시 대폭 감소** |
| **가용성 (Availability)** | 단일 장애점 (SPOF) | **고가용 (자동 복구)** | 동일 |
| **최대 QPS** | 제한적 | **수천~수만 QPS** | 동일 |

---

## 📝 Brick-by-Brick 학습 관점

`distributed_serving`은 Module 7의 정점이자, **전체 vLLM 학습 여정의 완성**입니다.

```
[Module 1] KV Cache 존재 인지
     │
     ▼
[Module 2] 연속 할당의 한계(단편화) 인지
     │
     ▼
[Module 3] Continuous Batching (계산 효율)
     │
     ▼
[Module 4] PagedAttention (논리-물리 매핑)
     │
     ▼
[Module 5] Physical Block Manager (물리적 자원 운영)
     │
     ▼
[Module 6] Prefix Caching (Content-Addressable Memory)
     │
     ▼
[Module 7 - 분산 실행] Tensor Parallelism + NCCL + Master-Worker
     │
     ▼
[Module 7 - 서비스 인프라] Load Balancer + Fault Tolerance + Distributed KV
     │
     ▼
[★ 최종 완성] Distributed Serving System
     → 엔드-투-엔드 분산 서비스 스택
     → 클라이언트 요청 → 로드 밸런싱 → 분산 KV → TP 실행 → 장애 복구 → 응답
     → vLLM이 실제 프로덕션 환경에서 운영되는 최종 형태
```

> 💡 **최종 인사이트**: `distributed_serving`은 지금까지 학습한 모든 개념(Continuous Batching, PagedAttention, Memory Pool, Prefix Caching, Tensor Parallelism, NCCL, Master-Worker)을 **하나의 운영 가능한 서비스**로 통합한 결과물입니다. 이 시스템은 단일 GPU로는 불가능했던 대형 모델(70B+)을 다중 GPU로 서빙하고, 수천~수만 QPS를 처리하며, 장애 시 자동 복구되는 **진정한 프로덕션-레디(Production-Ready) LLM 서빙 엔진**을 완성합니다. 이것이 바로 vLLM이 업계에서 가장 널리 사용되는 오픈소스 LLM 서빙 엔진이 된 이유입니다.