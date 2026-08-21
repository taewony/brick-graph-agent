---
type: AtomicConcept
id: atomic.tensor_parallelism
title: Tensor Parallelism (텐서 병렬성)
description: Megatron-LM 스타일의 Column-wise / Row-wise 가중치 분할(Sharding)을 통해 하나의 대형 Linear
  레이어를 여러 GPU에 분산 배치하고, Forward Pass 시 All-Reduce로 부분 합을 동기화하는 모델 병렬화 전략
status: draft
generated:
  by: agent:builder/1.0
  at: 2026-08-07 08:00:00+00:00
verified:
- by: human:curator
  at: 2026-08-07 08:00:00+00:00
prerequisites:
- atomic.inference_only
composes_into:
- composite.distributed_executor (Module 07)
sources:
- id: megatron_tp
  resource: https://arxiv.org/abs/1909.08053
  title: 'Megatron-LM: Training Multi-Billion Parameter Language Models Using Model
    Parallelism'
prerequisite_of:
- atomic.master_worker
- atomic.nccl_communicator
---

# Tensor Parallelism (텐서 병렬성)

## 📌 개념 정의

**Tensor Parallelism (TP)** 은 단일 GPU 메모리에 적재할 수 없는 대형 LLM(예: 70B 파라미터)을 여러 GPU에 분할하여 실행하는 핵심 기술입니다. 
**Megatron-LM**에서 제안된 방식으로, Transformer 레이어 내부의 **Linear(MLP) 레이어와 Attention 레이어의 가중치 텐서를 Column-wise 또는 Row-wise로 분할(Shard)** 합니다.

## 🔄 작동 방식 (Column-wise 예시)

`Y = X * W` (Linear Layer) 연산에서, 가중치 텐서 `W`를 GPU 개수(`TP`)만큼 Column 방향으로 분할합니다.

| 구성 | 설명 |
|---|---|
| **입력 (X)** | 모든 GPU에 동일하게 브로드캐스트 (혹은 복제) |
| **가중치 (W)** | GPU 0: `W[:, 0 : d/tp]`, GPU 1: `W[:, d/tp : 2*d/tp]`, ... |
| **출력 (Y)** | 각 GPU가 자신의 Shard에 대해 부분 행렬 곱(Partial GEMM) 수행 |
| **동기화** | Forward Pass 종료 시 **All-Reduce**로 모든 GPU의 부분 합을 집계하여 최종 출력 생성 |

## 📊 메모리 이득

| TP 크기 | 가중치 당 GPU 메모리 사용량 | KV Cache 할당량 |
|---|---|---|
| 1 (단일 GPU) | 100% (기준) | 100% |
| 2 | 50% | 200% (KV Heads도 분할되므로 총 용량 증가) |
| 4 | 25% | 400% |
| 8 | 12.5% | 800% |

> 💡 **Trade-off**: TP를 늘리면 GPU 간 통신(All-Reduce) 오버헤드가 증가하므로, 모델 크기와 GPU 수에 맞는 최적의 TP 값을 선택해야 합니다.

## 🔗 관련 관계

- **PREREQUISITES**: [`atomic.inference_only`](inference_only.md)
- **PREREQUISITE_OF**: [`composite.distributed_serving`](../02_composite/distributed_serving_system.md) (Module 07)
- **SYNERGY WITH**: [`atomic.nccl_communicator`](nccl_communicator.md) (All-Reduce 구현), [`atomic.distributed_kv`](distributed_kv.md) (KV Heads 분할)