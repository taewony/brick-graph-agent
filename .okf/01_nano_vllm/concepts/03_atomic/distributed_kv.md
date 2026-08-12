---
type: AtomicConcept
id: atomic.distributed_kv
title: Distributed KV (분산 키‑값 저장소)
description: 여러 노드에 KV 파티션을 저장·복제해 확장성·내결함성을 제공하고, 로드 밸런서를 통해 요청을 라우팅한다.
status: draft
generated:
  by: agent:builder/1.0
  at: 2026-08-06T08:11:19Z
verified: []
---

# Distributed KV (분산 키‑값 저장소)

## 📌 개념 정의
**Distributed KV**는 **키‑값** 데이터를 **여러 서버 인스턴스**에 **샤딩(파티셔닝)**하고 **복제**하여
- **수평 확장**(수평 스케일링)
- **고가용성**(HA) 및 **내결함성**(fault‑tolerance)

을 제공한다. 각 노드는 독립적인 KV 엔진(예: Redis, RocksDB) 을 실행하고, 메타데이터 서비스가 파티션 위치와 복제
상태를 관리한다.

## ⚙️ 핵심 구성 요소
| 컴포넌트 | 역할 |
|----------|------|
| **Partitioner** | 입력 키를 해시하고, `hash(key) % N` 로 파티션(샤드) 번호 결정 |
| **Replica Manager** | 각 파티션을 1+개의 복제본에 저장, 리더‑팔로워 동기화 |
| **Meta Service** | 파티션‑노드 매핑, 리더 선출, 헬스 체크 정보를 중앙 관리 |
| **Consistency Layer** | 읽기/쓰기 일관성 모델 제공 (Strong, Eventual, Read‑After‑Write 등) |
| **Load Balancer** | 클라이언트 요청을 적절한 KV 인스턴스로 라우팅 (다음 파일 `load_balancer.md` 참고) |
| **Fault Tolerance** | 노드 장애 감지 → 자동 리플리케이션 재배치 (다음 파일 `fault_tolerance.md` 참고) |

## 🚀 주요 동작 흐름
1. **쓰기**
   - 클라이언트 → Load Balancer → `Partitioner` 로 파티션 결정
   - 메타 서비스가 해당 파티션의 **리더** 노드(Primary) 식별
   - 리더에 쓰기 → 복제본(Secondary)에게 **동기식/비동기식** 복제 전파
2. **읽기**
   - 클라이언트 → Load Balancer → `Partitioner` 로 파티션 결정
   - 메타 서비스가 **리더** 혹은 **팔로워** 중 일관성 요구에 맞는 노드 선택
   - 선택된 노드에서 데이터 반환
3. **노드 장애**
   - 헬스 체크 실패 → `Fault Tolerance` 가 새로운 리더 선출 및 복제 재배치
   - 메타 서비스가 파티션‑노드 매핑을 업데이트하고 Load Balancer에 전파

## 🔗 관련 관계
- **PREREQUISITES**: `kv_cache` (키‑값 캐시와 연계)
- **COMPOSED_OF**: `load_balancer`, `fault_tolerance`, `distributed_serving` (전반적인 서비스 레이어)

---
type: AtomicConcept
id: atomic.distributed_kv_cache
title: Distributed KV Cache (분산 KV 캐시)
description: Tensor Parallelism 환경에서 KV Cache Heads를 GPU 수에 따라 분할(Shard)하고, 각 GPU가 자신이 담당하는 KV Heads에 대한 블록만 할당/관리하여 메모리 용량을 GPU 수에 비례하여 확장하는 분산 캐싱 기법
status: draft
generated:
  by: agent:builder/1.0
  at: 2026-08-07T08:40:00Z
verified:
  - by: human:curator
    at: 2026-08-07T08:40:00Z
prerequisites:
  - atomic.paged_kv_cache
  - atomic.nccl_communicator
composes_into:
  - composite.distributed_executor (Module 07)
---

# Distributed KV Cache (분산 KV 캐시)

## 📌 개념 정의

**Distributed KV Cache**는 Tensor Parallelism(TP) 환경에서 **KV Cache 또한 GPU 수에 따라 분할(Shard)** 하여 관리하는 기법입니다.

단일 GPU에서는 모든 KV Heads(예: 32개)를 하나의 GPU가 관리했지만, TP 환경에서는 각 GPU가 자신이 담당하는 KV Heads만 관리합니다.

## 🧩 KV Heads Sharding 예시

**가정**: `num_kv_heads = 8`, `tensor_parallel_size = 4`

| GPU (Rank) | 담당 KV Heads | KV Cache 블록 할당 |
|---|---|---|
| GPU 0 (Rank 0) | Head 0, 1 | 자신의 Memory Pool에서 블록 할당 |
| GPU 1 (Rank 1) | Head 2, 3 | 자신의 Memory Pool에서 블록 할당 |
| GPU 2 (Rank 2) | Head 4, 5 | 자신의 Memory Pool에서 블록 할당 |
| GPU 3 (Rank 3) | Head 6, 7 | 자신의 Memory Pool에서 블록 할당 |

## 🔄 동작 방식 (Decode Phase 기준)

1. **Prefill 단계**: 각 GPU가 자신에게 할당된 KV Heads에 대해서만 K, V 텐서를 계산하고, 자신의 Memory Pool에 저장합니다.
2. **Decode 단계 (Attention)**: 
   - 각 GPU는 자신의 KV Cache(일부 Heads)를 읽어 부분 어텐션(Partial Attention)을 계산합니다.
   - **All-Reduce**를 통해 모든 GPU의 부분 어텐션 결과를 합산하여 최종 Attention 출력을 생성합니다.

## 📊 메모리 용량 확장 효과

| 구성 | 단일 GPU KV Cache 용량 | 총 분산 용량 |
|---|---|---|
| TP=1 | 80GB × 1 = 80GB | 80GB |
| TP=4 | 80GB × 4 = 320GB (실제: 각 GPU가 80GB씩 담당) | **320GB** |
| TP=8 | 80GB × 8 = 640GB | **640GB** |

> 💡 **핵심 이점**: Distributed KV Cache는 **메모리 용량을 GPU 수에 비례하여 선형적으로 확장**합니다. 즉, TP=4를 사용하면 동시에 처리할 수 있는 최대 Sequence 길이 또는 배치 크기가 단일 GPU 대비 4배 증가합니다.

## ⚠️ 통신 오버헤드

- Decode 단계에서 Attention 결과를 동기화하기 위해 **매 Iteration마다 All-Reduce**가 발생합니다.
- KV Cache 자체는 NCCL 통신 없이 각 GPU가 독립적으로 관리하므로, 통신량은 Attention 결과 텐서 크기( `batch_size * seq_len * hidden_dim` )에 비례합니다.

## 🔗 관련 관계

- **PREREQUISITES**: `atomic.paged_kv_cache`, `atomic.nccl_communicator`
- **PREREQUISITE_OF**: `composite.distributed_executor`