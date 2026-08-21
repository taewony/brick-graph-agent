---
type: AtomicConcept
id: atomic.memory_pool
title: Memory Pool (메모리 풀)
description: GPU HBM(High Bandwidth Memory)을 시스템 초기화 시점에 사전 할당(Pre-allocate)하여 고정
  크기 블록들의 연속된 풀로 관리함으로써, 런타임 동적 할당 오버헤드와 외부 단편화를 제거하는 추상화 계층
status: draft
generated:
  by: agent:builder/1.0
  at: 2026-08-06 12:00:00+00:00
verified:
- by: human:curator
  at: 2026-08-06 12:00:00+00:00
prerequisites:
- atomic.paged_kv_cache
composes_into:
- composite.block_manager (Module 5)
sources:
- id: original_lab
  resource: https://hackmd.io/9Ivogn3dRwm3WgA4Kl3fJQ
  title: 'nano-vLLM Module 5: Memory Management & Block Manager'
prerequisite_of:
- composite.block_manager
---

# Memory Pool (메모리 풀)

## 📌 개념 정의

**Memory Pool**은 GPU의 HBM(High Bandwidth Memory)을 시스템 부팅 시 **미리 할당(Pre-allocate)** 해두고, 이를 **동일한 크기의 블록(Block)들로 구성된 연속된 풀(Pool)**로 추상화하는 기법입니다.

- vLLM은 `torch.cuda` 또는 `cudaMalloc`을 통해 전체 GPU 메모리의 85~95%를 하나의 큰 텐서(또는 버퍼)로 예약합니다.
- 이 예약된 메모리 공간을 **고정 크기(예: 16 tokens)** 의 블록들로 나누어 관리합니다.
- 이후 모든 KV Cache 할당 요청은 이 풀에서 블록을 꺼내어 사용하고, 해제 시 다시 풀로 반환합니다.

### 왜 Memory Pool이 필요한가?

| 구분 | 기존 방식 (동적 할당) | Memory Pool 방식 |
|---|---|---|
| **할당 타이밍** | Sequence가 들어올 때마다 `cudaMalloc` 호출 | 시스템 시작 시 한 번만 `cudaMalloc` 호출 |
| **메모리 단편화** | 외부 단편화 심각 (조각 발생) | **외부 단편화 제로 (Zero)** (모든 블록 동일 크기) |
| **할당 속도** | 느림 (커널 호출 + 메모리 맵핑) | **매우 빠름** (포인터 연산만으로 할당) |
| **메모리 활용률** | 50~70% | **~100%** |

> 💡 **핵심 통찰**: Memory Pool을 사용하면 런타임에 `cudaMalloc`/`cudaFree`를 **단 한 번도 호출하지 않습니다**. 모든 메모리 관리는 사용자 공간(User Space)에서 포인터 산술(Pointer Arithmetic)과 Free List 관리로 처리됩니다. 이는 GPU 메모리 할당의 병목을 완전히 제거합니다.

---

## 🧱 Memory Pool의 내부 구조

### 1. 풀 초기화 (Pool Initialization)

```python
# 개념적 Python 코드 (실제 구현은 PyTorch/CUDA)
class MemoryPool:
    def __init__(self, total_memory_gb: float, block_size_tokens: int):
        self.block_size = block_size_tokens  # 예: 16
        self.num_blocks = total_memory_gb // block_size_tokens
        
        # GPU 메모리를 연속된 하나의 큰 텐서로 사전 할당
        self.pool_tensor = torch.empty(
            (self.num_blocks, block_size_tokens, hidden_dim), 
            dtype=torch.float16, device='cuda'
        )
        
        # Free List: 초기에는 모든 블록이 사용 가능
        self.free_list = list(range(self.num_blocks))
        self.allocated_count = 0
```

### 2. 풀의 물리적 레이아웃 (Physical Layout)

Memory Pool은 GPU 메모리 상에 **연속된(Contiguous) 가상 주소 공간**을 차지하지만, 논리적으로는 동일한 크기의 슬롯(Slot)들로 분할됩니다.

```
GPU HBM 메모리 주소 공간 (가상)
┌─────────────────────────────────────────────────────────────────────────┐
│  Block 0  │  Block 1  │  Block 2  │  Block 3  │ ... │  Block N-1  │
│ (16 tok)  │ (16 tok)  │ (16 tok)  │ (16 tok)  │     │  (16 tok)   │
└─────────────────────────────────────────────────────────────────────────┘
│<─────────────────────── Memory Pool (Pre-allocated) ─────────────────>│
```

### 3. 블록 할당 (Block Allocation)

Block Allocator가 블록을 요청하면, Memory Pool은 Free List에서 **하나의 블록 ID를 반환**하고 해당 ID를 Free List에서 제거합니다.

```python
def allocate_block(self) -> int:
    if len(self.free_list) == 0:
        raise OutOfMemoryError("Memory Pool is full!")
    
    block_id = self.free_list.pop()  # Free List에서 하나 꺼냄
    self.allocated_count += 1
    return block_id  # 물리적 블록 ID 반환
```

### 4. 블록 해제 (Block Deallocation)

Sequence가 종료되어 블록이 더 이상 필요 없어지면, Memory Pool은 해당 블록 ID를 Free List에 다시 추가합니다.

```python
def free_block(self, block_id: int) -> None:
    self.free_list.append(block_id)  # Free List에 다시 추가
    self.allocated_count -= 1
```

> 🚀 **성능 이점**: 이 모든 작업은 **CPU 연산만으로 처리**되며, GPU 커널 호출이나 메모리 복사가 전혀 발생하지 않습니다. 따라서 메모리 할당/해제가 **거의 제로 오버헤드(Zero Overhead)** 에 가깝습니다.

---

## 🔗 Module 5 내 다른 개념과의 관계

Memory Pool은 Module 5의 다른 원자적 개념들을 위한 **물리적 기반(Physical Foundation)** 입니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                      Memory Pool                                │
│  (GPU HBM을 고정 블록으로 추상화)                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ (할당/해제 요청)
┌─────────────────────────────────────────────────────────────────┐
│                    Block Allocator                              │
│  (First-Fit / Best-Fit 정책으로 블록 할당 전략 구현)             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ (메모리 부족 시)
┌─────────────────────────────────────────────────────────────────┐
│                     Swap Manager                                │
│  (Memory Pool이 가득 차면, 블록을 CPU 메모리로 스왑아웃)        │
└─────────────────────────────────────────────────────────────────┘
```

### 의존성 흐름

| 개념 | Memory Pool과의 관계 |
|---|---|
| **[`atomic.block_allocator`](block_allocator.md)** | Memory Pool의 Free List를 조작하여 블록을 할당/해제하는 **전략(Strategy)** |
| **[`atomic.swap_manager`](swap_manager.md)** | Memory Pool이 가득 찼을 때, 일부 블록을 CPU로 내보내고(Evict) Free List를 확보하는 **안전장치(Safety Net)** |
| **[`composite.block_manager`](../02_composite/block_manager.md)** | Memory Pool을 포함하여 위 모든 컴포넌트를 **통합 운영**하는 최상위 관리자 |

---

## 📊 Memory Pool 크기 결정 (Capacity Planning)

Memory Pool의 크기(`num_blocks`)는 다음 요소를 고려하여 결정됩니다.

| 요소 | 설명 | 권장 값 |
|---|---|---|
| **모델 가중치 (Weights)** | 모델 파라미터를 저장하는 데 필요한 메모리 (예: Llama 70B ≈ 140GB) | 고정값 |
| **KV Cache 최대 사용량** | 동시에 처리할 최대 Sequence 수 × 평균 길이 × 레이어 수 × head_dim | 워크로드 기반 예측 |
| **안전 마진 (Safety Margin)** | 예기치 않은 메모리 요청(Activation, 임시 버퍼)을 위한 여유분 | **전체 GPU 메모리의 5~10%** |
| **CUDA Context 오버헤드** | CUDA 드라이버 및 커널 실행에 필요한 예약 공간 | 보통 1~2GB |

**공식 (개념적)**:

```
pool_size = (total_gpu_memory - model_weights - safety_margin) // block_size
```

---

## ⚠️ Memory Pool의 한계 (Trade-offs)

| 장점 | 단점 / 고려사항 |
|---|---|
| 외부 단편화 제거 (Near-Zero) | **내부 단편화(Internal Fragmentation)** 발생 가능 (블록의 마지막 슬롯이 채워지지 않을 경우) |
| 할당/해제 속도 극대화 | 블록 크기(`block_size`)가 너무 작으면 Free List 관리 오버헤드 증가 |
| 메모리 활용률 100% 달성 | 블록 크기가 너무 크면 내부 단편화 심화 (예: 32 tokens 블록에 1 token만 사용) |
| 시스템 안정성 향상 | 풀 크기를 초과하는 요청 발생 시 Swap Manager에 의존해야 함 (CPU 스왑 지연 발생) |

> 💡 **최적의 블록 크기(Block Size)**: vLLM은 보통 **16 또는 32 tokens**를 기본 블록 크기로 사용합니다. 이는 내부 단편화를 최소화하면서도 Free List 관리를 효율적으로 유지할 수 있는 경험적(Heuristic) 값입니다.

---

## 📝 Brick-by-Brick 학습 관점

Memory Pool은 지금까지 학습한 개념 중 **"시스템을 실제로 운영하기 위한 첫 번째 실용적 인프라"**입니다.

```
[Module 1] KV Cache 존재 인지
     │
     ▼
[Module 2] 연속 할당의 한계(단편화) 인지 → "메모리를 어떻게 조각 없이 관리할까?"
     │
     ▼
[Module 3] Continuous Batching (계산 효율) → "GPU를 어떻게 쉬지 않게 할까?"
     │
     ▼
[Module 4] PagedAttention (논리-물리 분리) → "주소를 어떻게 변환할까?"
     │
     ▼
[Module 5 - Memory Pool] ★★★★★
     → "GPU 메모리를 어떻게 미리 준비해두고, 빠르게 할당/해제할까?"
     → 이 개념이 없으면 Block Allocator와 Swap Manager는 동작할 기반이 없음.
     → **실제 vLLM 시스템의 물리적 기반(Physical Foundation)**
     │
     ▼
[Module 5 - Block Allocator] 할당 전략 구현
     │
     ▼
[Module 5 - Swap Manager] 메모리 부족 시 대비
```

> 💡 **최종 인사이트**: Memory Pool은 vLLM의 **"창고(Warehouse)"**와 같습니다. 시스템이 켜지면 모든 GPU 메모리를 정리 정돈하여 동일한 크기의 상자(블록)들로 채워두고, 필요할 때마다 상자를 꺼내고 반납하는 방식입니다. 이 창고 덕분에 vLLM은 메모리 할당/해제에서 발생하는 병목을 완전히 제거하고, 최대 23배의 처리량 향상을 달성할 수 있습니다.
이 개념은 Module 5의 **가장 기초가 되는 원자적 개념**으로, GPU HBM을 고정 크기 블록 단위의 풀(Pool)로 추상화하여 **외부 단편화를 원천 차단**하고, PagedAttention이 블록을 할당/해제할 수 있는 **물리적 기반(Substrate)** 을 제공합니다.

## ⚙️ 핵심 동작
- **블록 프리 리스트**: 사용 가능한 블록을 리스트로 관리.
- **버퍼 재활용**: 사용이 끝난 블록을 즉시 풀에 반환.
- **동시 접근**: 락‑프리 알고리즘으로 다중 스레드 안전성을 보장.
1. **초기화** – 지정된 블록 수와 크기로 pool을 생성.
2. **할당** – `request(size)` → 가장 작은 충분한 블록 반환.
3. **해제** – `release(block)` → 블록을 pool에 반환, 재사용 대기.
    
## 🔗 관련 관계
- **PREREQUISITES**: [`continuous_batching`](continuous_batching.md), [`kv_cache`](kv_cache.md) (기본 KV 캐시 메모리)
- **PREREQUISITE_OF**: [`max_batch_size`](max_batch_size.md), [`block_allocator`](block_allocator.md) (블록 할당기), [`swap_manager`](swap_manager.md) (스와핑 관리자)