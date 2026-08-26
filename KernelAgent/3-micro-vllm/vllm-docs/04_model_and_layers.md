# 04. 모델과 레이어: 거대 모델을 나누는 기술

LLM은 너무 커서 하나의 GPU에 담기 어려운 경우가 많습니다. `nano-vllm`은 **텐서 병렬 처리(Tensor Parallelism)**라는 기술을 사용하여 모델을 여러 조각으로 나누어 실행합니다.

---

## 1. 텐서 병렬 처리 (Tensor Parallelism)

`layers/linear.py`를 보면 일반적인 `nn.Linear` 대신 특별한 레이어들이 등장합니다.

### A. ColumnParallelLinear (세로로 나누기)
*   **원리**: 행렬의 출력 차원(Column)을 GPU 개수만큼 나눕니다.
*   **역할**: 각 GPU가 결과값의 일부씩만 계산합니다. (예: Q, K, V 벡터의 일부 계산)

### B. RowParallelLinear (가로로 나누기)
*   **원리**: 행렬의 입력 차원(Row)을 GPU 개수만큼 나눕니다.
*   **역할**: 각 GPU가 계산한 부분 결과를 합쳐서 최종 결과를 만듭니다. 이때 **`all_reduce`**라는 통신을 통해 모든 GPU의 데이터를 하나로 합칩니다.

> **비유**: 아주 큰 벽화를 그릴 때, 왼쪽은 A가 오른쪽은 B가 그리는 것과 같습니다. 마지막에 둘이 그린 그림을 합치면(All-Reduce) 하나의 큰 벽화가 완성됩니다.

## 2. Rotary Embedding (RoPE)

최신 모델(Llama, Qwen 등)은 단어의 위치 정보를 파악하기 위해 **RoPE**라는 기술을 사용합니다. (`layers/rotary_embedding.py`)

*   **동작**: 단어 벡터를 특정 각도만큼 회전시켜 위치 정보를 주입합니다.
*   **장점**: 문장이 길어져도 위치 정보를 상대적으로 잘 파악하며, 학습 때보다 더 긴 문장도 어느 정도 처리할 수 있습니다.

```python
def apply_rotary_emb(x, cos, sin):
    # 단어 벡터를 회전시키는 마법의 공식
    x1, x2 = torch.chunk(x, 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1)
```

## 3. 모델 조립: Qwen3 사례 (`models/qwen3.py`)

`nano-vllm`의 `Qwen3ForCausalLM` 클래스는 위에서 배운 부품들을 조립하여 완성됩니다.

1.  **QKVParallelLinear**: Q, K, V 계산을 병렬로 한꺼번에 수행합니다.
2.  **RMSNorm**: 데이터를 정규화하여 학습과 추론을 안정적으로 만듭니다.
3.  **Attention**: 앞서 배운 PagedAttention(KV Cache)을 사용하여 효율적으로 연산합니다.

---

## 📝 학생들을 위한 요약
1.  **협동 학습**: `ColumnParallel`과 `RowParallel`을 통해 여러 GPU가 하나의 모델처럼 동작합니다.
2.  **회전하는 위치 정보**: `RoPE`를 통해 모델은 단어가 몇 번째 단어인지 인식합니다.
3.  **효율적 구조**: Qwen3와 같은 최신 아키텍처는 이 모든 병렬화 기술이 집약된 결정체입니다.

마지막 장에서는 이 모든 것이 합쳐져서 실제로 어떻게 실행되는지, **Engine**의 흐름을 정리하며 마무리하겠습니다.
