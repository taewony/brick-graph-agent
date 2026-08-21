---
type: AtomicConcept
id: atomic.nccl_communicator
title: NCCL Communicator (NCCL 통신자)
description: NVIDIA Collective Communications Library (NCCL) 백엔드를 활용하여 Tensor Parallelism
  시 GPU 간 All-Reduce, Broadcast, All-Gather 등 집단 통신(Collective Communication)을 고속으로
  수행하는 통신 계층
status: draft
generated:
  by: agent:builder/1.0
  at: 2026-08-07 08:10:00+00:00
verified:
- by: human:curator
  at: 2026-08-07 08:10:00+00:00
prerequisites:
- atomic.tensor_parallelism
composes_into:
- composite.distributed_executor (Module 07)
sources:
- id: nccl_docs
  resource: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/overview.html
  title: NCCL User Guide
prerequisite_of:
- atomic.distributed_kv
---

# NCCL Communicator (NCCL 통신자)

## 📌 개념 정의

**NCCL (NVIDIA Collective Communications Library)** 은 NVIDIA GPU 간 고속 집단 통신(Collective Communication)을 제공하는 라이브러리입니다. 
Tensor Parallelism 환경에서 각 GPU가 계산한 **부분 결과(Partial Results)** 를 하나로 합치거나(All-Reduce), 데이터를 모든 GPU에 동시에 전송(Broadcast)하는 역할을 담당합니다.

## ⚙️ 주요 집단 통신(Primitives) 연산

| 연산 | 설명 | TP에서의 활용 |
|---|---|---|
| **All-Reduce** | 모든 GPU의 데이터를 합산(Sum/Avg)하여 모든 GPU에 동일한 결과를 반환 | Linear 레이어 Forward Pass의 출력 동기화 |
| **Broadcast** | 한 GPU(Rank 0)의 데이터를 모든 다른 GPU로 복사 | Master가 Scheduler 명령을 Workers에 전파 |
| **All-Gather** | 각 GPU의 데이터를 모아서 모든 GPU가 전체 데이터를 보유 | KV Cache 블록 공유 정보 동기화 |

## 🔧 PyTorch / vLLM에서의 구현

```python
# PyTorch 분산 패키지를 통한 NCCL 사용 예시
import torch.distributed as dist

# 1. Process Group 초기화 (NCCL 백엔드)
dist.init_process_group(backend='nccl', init_method='env://')

# 2. All-Reduce 실행 (텐서 동기화)
tensor = torch.randn(1024, device='cuda')
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
# 이제 tensor는 모든 GPU의 합계를 포함