---
type: Module
id: module.distributed_serving
title: Distributed Serving (분산 서빙)
description: 단일 GPU 메모리 한계를 극복하기 위해 Tensor Parallelism을 기반으로 모델을 여러 GPU에 분할하여 배치하고, NCCL을 통한 고속 GPU 간 통신과 Master-Worker 분산 실행 패턴을 구현하는 최상위 확장 계층
status: draft
generated:
  by: agent:builder/1.0
  at: 2026-08-06T15:00:00Z
verified:
  - by: human:curator
    at: 2026-08-06T15:00:00Z
prerequisites:
  - module.prefix_caching
  - composite.block_manager
  - composite.paged_attention_manager
composes_into:
  - composite.serving_system (최종 통합)
sources:
  - id: distributed_execution
    resource: https://deepwiki.com/liguodongiot/nano-vllm/5-distributed-execution
    title: "nano-vLLM Distributed Execution"
  - id: modelrunner_distributed
    resource: https://deepwiki.com/GeeeekExplorer/nano-vllm/2.2-modelrunner-distributed-execution-engine
    title: "ModelRunner - Distributed Execution Engine"
  - id: vllm_parallelism
    resource: https://docs.vllm.ai/en/v0.21.0/serving/parallelism_scaling/
    title: "vLLM Parallelism and Scaling"
---

# Distributed Serving (분산 서빙)

## 📌 개요

Module 6까지 우리는 **단일 GPU** 환경에서 최적의 KV Cache 관리(PagedAttention), 메모리 풀 운영(Block Manager), 그리고 프리픽스 캐싱(Prefix Caching)까지 구축했습니다.

Module 7은 이 모든 인프라를 **다중 GPU 및 다중 노드** 환경으로 확장합니다. **Tensor Parallelism(텐서 병렬성)** 을 통해 모델 가중치를 여러 GPU에 분할(Shard)하여 단일 GPU 메모리 한계를 극복하고, **NCCL**을 통한 고속 GPU 간 통신으로 분산 연산을 동기화합니다.

### Module 7의 핵심 가치

| 측면 | 설명 |
|---|---|
| **모델 확장성** | 단일 GPU에 적재할 수 없는 대형 모델(70B+)을 여러 GPU에 분산 배치 |
| **처리량 향상** | Tensor Parallelism으로 연산을 분산하여 Prefill/Decode 처리량 증가 |
| **메모리 용량 확장** | GPU 수에 비례하여 KV Cache 용량 증가 (단일 GPU 대비 N배) |
| **Multi-Node 지원** | 단일 노드(Single-Node)를 넘어 다중 노드(Multi-Node)로 확장 가능 |

> 💡 **실제 사례**: Llama 3 70B 모델(140GB)은 단일 A100 80GB에 적재할 수 없습니다. Tensor Parallelism(TP=4)로 4개의 A100에 모델을 분할하면, 각 GPU는 약 35GB의 가중치를 담당하게 되어 **KV Cache를 위한 여유 메모리**까지 확보할 수 있습니다.

---

## 🧩 포함 원자적 개념

- [`atomic.tensor_parallelism`](../03_atomic/tensor_parallelism.md) — 모델 가중치를 GPU 수에 따라 분할(Shard)하는 병렬화 전략
- [`atomic.master_worker`](../03_atomic/master_worker.md) — Rank 0(Master)가 워커들을 조정하는 Master-Worker 실행 패턴
- [`atomic.nccl_communicator`](../03_atomic/nccl_communicator.md) — NCCL 백엔드를 통한 GPU 간 All-Reduce 등 집단 통신(Collective Communication)
- [`atomic.shared_memory_ipc`](../03_atomic/shared_memory_ipc.md) — Master가 워커에게 명령과 데이터를 브로드캐스트하는 공유 메모리 IPC 채널
- `atomic.distributed_kv_cache` — 분산 환경에서 각 GPU가 자신이 담당하는 KV Cache 블록을 관리

## 🏗️ 복합 개념 (Composite Concept)

- `composite.distributed_executor` — ModelRunner, NCCL Process Group, Shared Memory를 통합한 분산 실행 엔진

## 🔗 관련 관계

- **PREREQUISITES**: [`block_manager`](../02_composite/block_manager.md), [`paged_attention_manager`](../02_composite/paged_attention_manager.md), [`composite.prefix_cache_manager`](../02_composite/caching_strategy.md)
- **COMPOSES_INTO**: [`serving_system`](../02_composite/serving_system.md) (전체 vLLM 시스템의 최상위 완성)

---

## 🧱 1. 분산 실행 아키텍처 (Distributed Execution Architecture)

### 1.1 Multi-Process 아키텍처

nano-vLLM은 **Multi-Process 분산 아키텍처**를 사용하여 모델 연산을 여러 GPU에 분할합니다.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LLMEngine (Main Process)                           │
│  - 요청 수신 및 토큰화                                                     │
│  - Scheduler를 통한 배치 구성                                               │
│  - ModelRunner에 실행 명령 전달                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ (spawn)
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Distributed Execution Layer                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Master Process (Rank 0)  │  Worker Process (Rank 1) │  Worker ... │   │
│  │  - GPU 0                  │  - GPU 1                 │  - GPU N-1   │   │
│  │  - 모델 가중치 Shard 0     │  - 모델 가중치 Shard 1    │  - Shard N-1 │   │
│  │  - KV Cache Shard 0       │  - KV Cache Shard 1      │  - Shard N-1 │   │
│  │  - 명령 직렬화 및 전송    │  - 명령 수신 및 실행      │              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                             │
│                              ▼ (NCCL All-Reduce)                          │
│                    GPU 간 텐서 동기화 (Collective Communication)           │
└─────────────────────────────────────────────────────────────────────────────┘
```

- 각 GPU는 독립적인 `ModelRunner` 프로세스로 실행됩니다.
- **Rank 0**이 Master Coordinator 역할을 수행하며, 나머지 Rank는 Worker로 동작합니다.
- 프로세스 간 통신은 **NCCL**(Linux) 또는 **Gloo**(Windows) 백엔드를 사용합니다.

### 1.2 프로세스 초기화 (Process Initialization)

각 `ModelRunner` 프로세스는 자신에게 할당된 GPU에서 독립적으로 초기화됩니다.

| 단계 | Rank 0 (Master) | Worker Ranks |
|---|---|---|
| 1. Process Group 초기화 | NCCL 백엔드로 `torch.distributed` 초기화 | 동일 |
| 2. CUDA Device 설정 | `torch.cuda.set_device(rank)` | 동일 |
| 3. 모델 로드 | Sharded Weights로 모델 인스턴스화 | 동일 |
| 4. Sampler 초기화 | 샘플링 설정 로드 | 동일 |
| 5. Warmup 실행 | 모델 워밍업 | 동일 |
| 6. KV Cache 할당 | `allocate_kv_cache()` 실행 | 동일 |
| 7. CUDA Graph 캡처 | Decode 배치용 CUDA Graph 캡처 | 동일 |
| 8. Shared Memory 생성 | **생성 (create)** | **연결 (attach)** |
| 9. Barrier 동기화 | 모든 Rank가 준비될 때까지 대기 | 동일 |
| 10. 실행 루프 진입 | LLMEngine으로 제어 반환 | **이벤트 루프 대기** |

> 💡 **중요**: Worker Rank는 초기화 완료 후 **이벤트 루프(event loop)** 에 진입하여 Master의 명령을 대기합니다. Master만이 LLMEngine으로 제어를 반환하고 요청을 처리합니다.

---

## 🧱 2. Master-Worker 실행 패턴 (Master-Worker Execution Pattern)

### 2.1 명령 디스패치 (Command Dispatch)

Master-Worker 패턴에서 Rank 0은 모든 분산 연산을 조정합니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Master (Rank 0)                              │
│  1. 실행할 명령 결정 ("run", "exit" 등)                        │
│  2. 명령 + 데이터를 Shared Memory에 직렬화(Serialization)       │
│  3. Worker에게 이벤트 신호(Event Signal) 전송                   │
│  4. 자신도 동일한 명령을 로컬에서 실행                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ (Shared Memory + Event)
┌─────────────────────────────────────────────────────────────────┐
│                    Workers (Rank 1, 2, ...)                     │
│  1. 이벤트 대기(Block)                                          │
│  2. 이벤트 수신 → Shared Memory에서 명령 읽기(Deserialize)      │
│  3. 명령 실행 (예: 모델 Forward Pass)                          │
│  4. 결과를 Shared Memory에 기록 (필요시)                        │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Shared Memory 프로토콜

Master와 Worker 간 통신은 **Shared Memory(SHM)** 를 통해 이루어집니다.

```
Shared Memory 메시지 포맷:
┌──────────────┬──────────────────────────────────────────┐
│ Bytes 0-3    │ Bytes 4 ~ (4 + message_length)          │
│ Message 길이 │ Pickle 직렬화된 데이터 [method, *args]   │
└──────────────┴──────────────────────────────────────────┘
```

| 동작 | 수행 주체 | 설명 |
|---|---|---|
| `write_shm()` | Rank 0 (Master) | 명령을 직렬화하여 Shared Memory에 쓰고, 모든 Worker에 신호 전송 |
| `read_shm()` | Rank 1+ (Workers) | 이벤트를 대기(Block)하다가, Shared Memory에서 명령을 읽고 역직렬화 |

Shared Memory 객체는 `"nanovllm"`이라는 이름으로 생성되며, 기본 용량은 **1 MB** (`2**20` bytes)입니다.

### 2.3 지원 명령어 (Supported Commands)

| 명령 | 설명 |
|---|---|
| `"run"` | 주어진 배치(Batch)에 대해 모델 추론(Inference) 실행 |
| `"exit"` | Worker 프로세스를 종료하고 리소스 정리 |

---

## 🧱 3. Tensor Parallelism (텐서 병렬성)

### 3.1 개념 정의

**Tensor Parallelism**은 모델의 **각 레이어 내부의 가중치 텐서(Weight Tensor)를 여러 GPU에 분할(Shard)** 하여, 단일 GPU 메모리로는 적재할 수 없는 대형 모델을 실행하는 기술입니다.

### 3.2 가중치 분할 (Weight Sharding)

nano-vLLM은 **Megatron-LM의 Tensor Parallelism 알고리즘**을 구현합니다.

```
[단일 GPU (TP=1)]
┌─────────────────────────────────────────────────────────────────┐
│  Linear Layer Weight: [hidden_size, intermediate_size]         │
│  (예: 4096, 11008) → 전체 가중치를 한 GPU가 보유               │
└─────────────────────────────────────────────────────────────────┘

[Tensor Parallelism (TP=4)]
┌─────────────────────────────────────────────────────────────────┐
│  Linear Layer Weight를 4개 GPU로 분할 (Column-wise Sharding)    │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌────────┐ │
│  │ GPU 0: Col 0 │ │ GPU 1: Col 1 │ │ GPU 2: Col 2 │ │GPU 3   │ │
│  │ ~ 2752       │ │ ~ 2752       │ │ ~ 2752       │ │~ 2752  │ │
│  └──────────────┘ └──────────────┘ └──────────────┘ └────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 분산 KV Cache (Distributed KV Cache)

Tensor Parallelism 환경에서 **KV Cache Heads도 GPU 수에 따라 분할(Shard)** 됩니다.

```
KV Heads Sharding (TP=4, Num KV Heads = 8):
┌─────────────────────────────────────────────────────────────────┐
│  GPU 0 │ GPU 1 │ GPU 2 │ GPU 3 │                              │
│ Head 0 │ Head 1 │ Head 2 │ Head 3 │ ← 각 GPU가 2개 Head 담당  │
│ Head 4 │ Head 5 │ Head 6 │ Head 7 │                          │
└─────────────────────────────────────────────────────────────────┘
```

각 GPU는 자신이 담당하는 KV Heads에 해당하는 **KV Cache 블록만 할당하고 관리**합니다.

### 3.4 Collective Communication (집단 통신)

Tensor Parallelism에서 GPU 간 **All-Reduce** 연산이 필수적입니다.

```
[Forward Pass에서의 All-Reduce]
1. 각 GPU가 자신의 Shard에 대해 부분 연산(Partial Computation) 수행
2. All-Reduce로 모든 GPU의 결과를 집계(Aggregate)하여 동기화
3. 다음 레이어로 진행

[사용 통신 백엔드]
- NCCL (Linux): NVIDIA GPU 간 최적화된 고속 통신
- Gloo (Windows): 크로스-플랫폼 통신 지원
```

---

## 🧱 4. 병렬화 전략 선택 가이드 (Parallelism Strategy Guide)

| 상황 | 권장 전략 | 설정 예시 |
|---|---|---|
| **모델이 단일 GPU에 적재 가능** | 분산 추론 불필요 | TP=1 |
| **모델이 단일 GPU보다 크지만, 단일 노드的多 GPU에 적재 가능** | Tensor Parallelism (단일 노드) | `tensor_parallel_size=4` (4-GPU 노드) |
| **모델이 단일 노드보다 큼** | Tensor Parallelism + Pipeline Parallelism | `tensor_parallel_size=8`, `pipeline_parallel_size=2` |
| **GPU 간 NVLINK 미존재 (예: L40S)** | Pipeline Parallelism 우선 | `tensor_parallel_size=1`, `pipeline_parallel_size=N` |
| **MoE (Mixture of Experts) 모델** | Data Parallel + Expert Parallel 결합 | `--data-parallel-size=4` |

### 4.1 Pipeline Parallelism (파이프라인 병렬성)

Tensor Parallelism이 **레이어 내부**를 분할하는 반면, **Pipeline Parallelism**은 **레이어 간(Layer-wise)** 으로 모델을 분할합니다.

```
[Pipeline Parallelism (PP=2)]
┌─────────────────────────────────────────────────────────────────┐
│  Node 1              │  Node 2                                │
│  ┌──────────────────┐ │  ┌──────────────────┐                 │
│  │ Layer 0 ~ 15     │ │  │ Layer 16 ~ 31    │                 │
│  │ (前半 레이어)    │ │  │ (後半 레이어)    │                 │
│  └──────────────────┘ │  └──────────────────┘                 │
└─────────────────────────────────────────────────────────────────┘
```

Pipeline Parallelism은 **GPU 개수가 모델 크기를 균등하게 나누지 못할 때** 유용합니다.

---

## 🔗 Module 6 (Prefix Caching)과의 통합

분산 환경에서 Prefix Caching은 **모든 GPU에서 동일한 해시 테이블을 공유**해야 합니다.

```
[분산 Prefix Caching]
┌─────────────────────────────────────────────────────────────────┐
│  Rank 0 (Master)          │  Rank 1 (Worker)                  │
│  ┌──────────────────────┐ │  ┌──────────────────────┐         │
│  │ hash_to_block_id     │ │  │ hash_to_block_id     │         │
│  │ (전체 해시 인덱스)   │ │  │ (전체 해시 인덱스)   │         │
│  └──────────────────────┘ │  └──────────────────────┘         │
│         │ (NCCL Broadcast)│          │                        │
│         └─────────────────┼──────────┘                        │
│                           ▼                                   │
│              모든 Rank가 동일한 해시 테이블 유지               │
└─────────────────────────────────────────────────────────────────┘
```

- 해시 테이블은 **모든 Rank에서 동기화(Synchronized)** 되어야 합니다.
- 새 블록이 할당되면 **Broadcast**를 통해 모든 Rank에 전파됩니다.

---

## 📊 성능 영향

| 지표 | 단일 GPU (TP=1) | Tensor Parallelism (TP=4) |
|---|---|---|
| **최대 모델 크기** | 단일 GPU 메모리 한도 (예: 80GB) | **GPU 수 × 메모리 (예: 4×80GB)** |
| **KV Cache 용량** | 단일 GPU 메모리 여유분 | **GPU 수 × 여유분** (선형 증가) |
| **Prefill 처리량** | 기준 (1x) | **~3.5x** (통신 오버헤드 고려) |
| **Decode 처리량** | 기준 (1x) | **~3.0x** (All-Reduce 오버헤드 고려) |
| **통신 오버헤드** | 없음 | All-Reduce per layer (NCCL 최적화) |

> ⚠️ **트레이드오프 인식**: Tensor Parallelism은 GPU 간 통신(All-Reduce) 오버헤드가 발생합니다. 따라서 **TP 크기를 무한정 늘리는 것은 비효율적**이며, 모델 크기와 GPU 메모리 용량에 맞춰 적절한 TP 값을 선택해야 합니다.

---

## 📝 Brick-by-Brick 학습 관점

Module 7은 지금까지 구축한 **모든 인프라를 다중 GPU로 확장**하는 최종 계층입니다.

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
[Module 5] Memory Pool + Block Allocator + Swap Manager (물리적 자원 운영)
     │
     ▼
[Module 6] Prefix Caching (Content-Addressable Memory)
     │
     ▼
[Module 7] Distributed Serving ★★★★★
     → Tensor Parallelism으로 모델 가중치 분할
     → Master-Worker 패턴으로 다중 GPU 프로세스 조정
     → NCCL로 GPU 간 All-Reduce 동기화
     → Shared Memory로 명령/데이터 전달
     → **단일 GPU의 한계를 넘어, 모델 크기와 처리량을 GPU 수에 비례하여 확장**
     │
     ▼
[Composite] serving_system (전체 vLLM 시스템 완성) ★
     → Module 1~7의 모든 개념이 통합된 최종 서빙 엔진
```

> 💡 **최종 인사이트**: Module 7은 vLLM을 **"단일 GPU 실험실 수준"**에서 **"실제 프로덕션 규모"**로 끌어올리는 마지막 퍼즐 조각입니다. Tensor Parallelism이 없었다면, vLLM은 70B 이상의 대형 모델을 서빙할 수 없었을 것입니다. Module 1~6이 **"어떻게 한 대의 GPU를 최대한 효율적으로 쓸까?"** 에 집중했다면, Module 7은 **"여러 대의 GPU를 어떻게 하나의 시스템처럼 묶을까?"** 에 답합니다. 이로써 vLLM은 **단일 GPU에서 수백 GPU 클러스터까지 확장 가능한** 통합 서빙 엔진으로 완성됩니다.