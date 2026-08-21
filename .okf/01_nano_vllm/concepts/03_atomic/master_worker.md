---
type: AtomicConcept
id: atomic.master_worker
title: Master-Worker Execution Pattern (마스터-워커 실행 패턴)
description: Rank 0(Master)가 전체 분산 시스템의 조정자(Coordinator) 역할을 수행하고, Rank 1..N(Workers)이
  Master의 명령에 따라 GPU 연산을 수행하는 동기식(Synchronous) 분산 실행 패턴
status: draft
generated:
  by: agent:builder/1.0
  at: 2026-08-07 08:20:00+00:00
verified:
- by: human:curator
  at: 2026-08-07 08:20:00+00:00
prerequisites:
- atomic.tensor_parallelism
composes_into:
- composite.distributed_executor (Module 07)
sources:
- id: vllm-paper
  resource: https://arxiv.org/abs/2209.06155
  title: vLLM - Easy, Fast and Cheap LLM Serving with PagedAttention
prerequisite_of:
- atomic.shared_memory_ipc
---

# Master-Worker Execution Pattern (마스터-워커 실행 패턴)

## 📌 개념 정의

**Master-Worker 패턴**은 분산 시스템에서 단일 **Master (Rank 0)**가 전체 워크플로우를 조정하고, 여러 **Worker (Rank 1..N)**들이 Master의 지시에 따라 실제 GPU 연산을 수행하는 동기식(Synchronous) 실행 모델입니다.

nano-vLLM/vLLM에서 이 패턴은 `torch.distributed`의 **Rank** 개념을 기반으로 동작합니다.

## 🏗️ 역할 분담 (Role Distribution)

| 역할 | Rank | 책임 |
|---|---|---|
| **Master** | 0 | - LLMEngine과 Scheduler 실행 <br> - 요청 수신 및 배치 구성 <br> - Workers에게 실행 명령(`run`, `exit`) 직렬화 및 전송 <br> - Shared Memory 생성 (초기화) |
| **Workers** | 1 ~ N-1 | - Master의 명령을 대기(Block) <br> - 명령 수신 시 자체 GPU에서 Forward Pass 실행 <br> - 결과를 Shared Memory 또는 NCCL을 통해 Master에 반환 |

## 🔄 초기화 및 실행 루프

```
[시스템 초기화]
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│ Master (Rank 0) & Workers (Rank 1..N) 병렬 실행               │
│  1. NCCL Process Group 초기화 (모든 Rank)                     │
│  2. 각 Rank가 자신의 CUDA Device 설정                         │
│  3. Master가 Shared Memory 생성 (create)                      │
│  4. Workers가 Shared Memory에 연결 (attach)                    │
└─────────────────────────────────────────────────────────────────┘
     │
     ▼
[실행 루프]
┌─────────────────────────────────────────────────────────────────┐
│ Master: 새 요청 처리 → Scheduler로 배치 구성                  │
│   │                                                           │
│   ├─► (직렬화) 명령 + 데이터 → Shared Memory에 기록            │
│   └─► Workers에 이벤트 신호 전송 (Event Signal)               │
└─────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│ Workers: 이벤트 대기(Block) → 해제 → Shared Memory 읽기       │
│   └─► 명령 실행 (Forward Pass) → NCCL로 결과 동기화           │
└─────────────────────────────────────────────────────────────────┘
     │
     ▼
[Master: 결과 수집 및 응답 생성 → 클라이언트 반환]
```

> 💡 **동기화 장벽(Synchronization Barrier)**: 모든 Rank는 `torch.distributed.barrier()`를 통해 초기화 완료 시점을 동기화하여, Master가 Workers보다 먼저 실행되는 레이스 컨디션(Race Condition)을 방지합니다.

## 🔗 관련 관계

- **PREREQUISITES**: [`atomic.tensor_parallelism`](tensor_parallelism.md)
- **PREREQUISITE_OF**: [`composite.distributed_serving`](../02_composite/distributed_serving_system.md)
- **SYNERGY WITH**: [`atomic.shared_memory_ipc`](shared_memory_ipc.md) (명령 전달 채널)