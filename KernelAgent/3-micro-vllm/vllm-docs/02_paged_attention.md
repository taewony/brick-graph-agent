# 02. PagedAttention: 메모리의 마법

vLLM의 핵심은 "어떻게 하면 파편화 없이 메모리를 효율적으로 사용할 것인가?"입니다. `nano-vllm`의 코드를 통해 이 마법이 어떻게 일어나는지 알아봅시다.

---

## 1. 핵심 개념: 물리적 블록 vs 논리적 블록

전통적인 방식에서는 문장이 길어질 것에 대비해 미리 큰 메모리를 할당하지만, PagedAttention은 이를 **블록(Block)** 단위로 잘라서 관리합니다.

*   **Logical Blocks (논리적 블록)**: 사용자 요청(Sequence)이 가진 연속적인 단어들의 묶음입니다.
*   **Physical Blocks (물리적 블록)**: GPU 메모리(KV Cache) 상의 실제 위치입니다.

`nano-vllm`은 이 둘 사이를 연결해주는 **`BlockManager`**를 가지고 있습니다.

## 2. BlockManager: 메모리 관리자 (`engine/block_manager.py`)

`BlockManager`는 GPU 메모리를 바둑판처럼 나누어 관리하는 역할을 합니다.

```python
class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks = [Block(i) for i in range(num_blocks)] # 물리 블록 생성
        self.free_block_ids = deque(range(num_blocks))     # 비어있는 블록 목록
```

### 주요 동작:
1.  **`allocate(seq)`**: 새로운 요청이 들어오면, 필요한 만큼 `free_block_ids`에서 블록을 꺼내 `seq.block_table`에 할당합니다.
2.  **Prefix Caching (해시 최적화)**: 똑같은 프롬프트가 들어오면, `hash_to_block_id`를 통해 이미 계산된 KV Cache 블록을 재사용합니다. (이것이 vLLM이 빠른 이유 중 하나입니다!)
3.  **`deallocate(seq)`**: 추론이 끝나면 사용하던 블록을 다시 `free_block_ids`로 돌려보내 다른 요청이 쓸 수 있게 합니다.

## 3. Slot Mapping: 데이터는 어디로 저장되나?

각 단어(Token)가 GPU 메모리의 정확히 어느 위치(Slot)에 저장되어야 하는지를 계산하는 것이 **Slot Mapping**입니다.

*   **계산 방식**: `블록 ID * 블록 크기 + 블록 내 오프셋`
*   이 매핑 정보는 추론 시점에 모델의 `Attention` 레이어로 전달됩니다.

## 4. Attention 레이어에서의 활용 (`layers/attention.py`)

`nano-vllm`은 성능을 위해 NVIDIA Triton으로 작성된 커널을 사용합니다.

```python
@triton.jit
def store_kvcache_kernel(...):
    # slot_mapping을 보고 정확한 GPU 메모리 위치(slot)에 
    # Key와 Value 데이터를 저장합니다.
    slot = tl.load(slot_mapping_ptr + idx)
    ...
    tl.store(k_cache_ptr + cache_offsets, key)
```

추론 시에는 `flash_attn_with_kvcache` 함수에 `block_table` 정보를 넘겨주어, 흩어져 있는 메모리 블록들을 마치 하나인 것처럼 읽어서 Attention 연산을 수행합니다.

---

## 📝 학생들을 위한 요약
1.  **유연함**: 메모리를 미리 크게 잡지 않고, 8개나 16개 단어 단위(Block)로 쪼개서 씁니다.
2.  **재사용**: 똑같은 질문은 다시 계산하지 않고 저장된 블록을 재사용합니다 (`Prefix Caching`).
3.  **정교함**: `BlockManager`가 예약하고, `Slot Mapping`이 위치를 알려주며, `Triton Kernel`이 데이터를 저장합니다.

다음 장에서는 이렇게 관리되는 데이터들이 **Scheduler**를 통해 어떻게 순서대로 처리되는지 알아보겠습니다.
