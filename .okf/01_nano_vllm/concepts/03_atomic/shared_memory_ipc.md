---
type: AtomicConcept
id: atomic.shared_memory_ipc
title: Shared Memory IPC (공유 메모리 프로세스 간 통신)
description: Master와 Workers 간 명령(Command) 및 메타데이터(Data)를 전달하기 위해 POSIX Shared Memory (/dev/shm)를 사용하는 고속 IPC(Inter-Process Communication) 채널. Pickle 직렬화를 통해 Python 객체를 주고받음
status: draft
generated:
  by: agent:builder/1.0
  at: 2026-08-07T08:30:00Z
verified:
  - by: human:curator
    at: 2026-08-07T08:30:00Z
prerequisites:
  - atomic.master_worker
composes_into:
  - composite.distributed_executor (Module 07)
---

# Shared Memory IPC (공유 메모리 프로세스 간 통신)

## 📌 개념 정의

**Shared Memory IPC**는 Master 프로세스(Rank 0)와 Worker 프로세스(Rank 1..N) 간에 **명령(Command)과 데이터를 주고받기 위한 고속 통신 채널**입니다.

Linux 환경에서는 **POSIX Shared Memory**(`/dev/shm`)를, Windows 환경에서는 **Named Memory Mapping**을 사용합니다. nano-vLLM은 기본적으로 Linux를 가정하며, `"nanovllm"`이라는 이름의 공유 메모리 객체를 생성합니다.

## 📐 프로토콜 (Protocol)

공유 메모리는 다음과 같은 간단한 **Length-Prefixed Serialization Protocol**을 사용합니다:

```
Shared Memory 레이아웃 (기본 크기: 1 MB = 2^20 bytes):
┌──────────────┬──────────────────────────────────────────┐
│ Bytes 0-3    │ Bytes 4 ~ (4 + message_length)          │
│ Message 길이 │ Pickle 직렬화된 데이터                  │
│ (4 bytes)    │ [method_name, *args, **kwargs]          │
└──────────────┴──────────────────────────────────────────┘
```

## 🔄 데이터 흐름 (Data Flow)

| 단계 | 수행 주체 | 동작 |
|---|---|---|
| **1. 쓰기 (Write)** | Master (Rank 0) | 1. `(method, args, kwargs)`를 `pickle.dumps()`로 직렬화 <br> 2. 길이(4 bytes)와 직렬화 데이터를 공유 메모리에 기록 <br> 3. Workers에 **Event Signal** (또는 Condition Variable) 전송 |
| **2. 읽기 (Read)** | Workers (Rank 1..N) | 1. Event Signal 대기(Block) <br> 2. 공유 메모리에서 길이 읽기 → 데이터 읽기 <br> 3. `pickle.loads()`로 역직렬화하여 명령어와 인자 복원 |

## 📦 지원 명령어 (Supported Commands)

| Command | 인자 | 설명 |
|---|---|---|
| `"run"` | `batch_data` (Token IDs, Metadata) | 주어진 배치에 대해 모델 Forward Pass 실행 |
| `"exit"` | 없음 | Worker 프로세스를 안전하게 종료하고 자원 정리 |

> 💡 **설계 이점**: TCP/IP 소켓 대신 공유 메모리를 사용하면 **마이크로초(μs) 수준의 지연 시간**으로 명령을 전달할 수 있어, 매 Iteration마다 발생하는 오버헤드를 최소화합니다.

## 🔗 관련 관계

- **PREREQUISITES**: `atomic.master_worker`
- **PREREQUISITE_OF**: `composite.distributed_executor`
- **SYNERGY WITH**: `atomic.nccl_communicator` (명령 전달은 IPC, 텐서 동기화는 NCCL)