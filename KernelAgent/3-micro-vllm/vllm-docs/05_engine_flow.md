# 05. 엔진의 흐름: 모든 조각이 하나로

드디어 `nano-vllm`의 마지막 장입니다. 앞서 배운 부품들이 어떻게 하나로 뭉쳐서 실제 LLM 서비스를 만들어내는지 살펴봅시다.

---

## 1. LLM Engine의 초기화 과정

사용자가 `LLM(model_path)`를 호출하면 어떤 일이 벌어질까요? (`nanovllm/engine/llm_engine.py`)

1.  **설정 로드**: 모델 크기, 레이어 개수, GPU 개수(TP Size) 등을 확인합니다.
2.  **멀티 프로세스 생성**: 만약 GPU가 여러 개라면, 각 GPU마다 `ModelRunner` 프로세스를 띄웁니다.
3.  **모델 로드**: 각 GPU는 자신의 담당 파트(Shard)만 메모리에 올립니다.
4.  **스케줄러와 메모리 준비**: `BlockManager`가 사용할 KV Cache 공간을 GPU 메모리에 미리 할당해 둡니다.

## 2. 추론 루프: Generate의 비밀

`llm.generate()` 함수는 다음과 같은 반복문을 수행합니다.

```python
while not self.is_finished():
    # 1. 스케줄링: 어떤 요청을 처리할까? (Prefill 또는 Decode)
    seqs, is_prefill = self.scheduler.schedule()
    
    # 2. 실행: GPU에서 모델을 돌려 단어를 생성하자!
    token_ids = self.model_runner.call("run", seqs, is_prefill)
    
    # 3. 후처리: 생성된 단어를 각 요청(Sequence)에 추가하자!
    self.scheduler.postprocess(seqs, token_ids)
```

이 루프가 한 번 돌 때마다 모든 사용자의 답변에 단어가 하나씩 추가됩니다. 이 과정이 완료되면 비로소 `outputs`가 반환됩니다.

## 3. 실습: 나만의 nano-vllm 실행하기 (`example.py`)

실제로 `nano-vllm`을 사용하는 코드는 vLLM 라이브러리와 매우 흡사합니다.

```python
from nanovllm import LLM, SamplingParams

# 1. 모델 로드 (가벼운 모델 권장)
llm = LLM("~/huggingface/Qwen3-0.6B/", tensor_parallel_size=1)

# 2. 답변 생성 규칙 설정
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)

# 3. 질문 던지기 (Batch 처리 가능!)
prompts = ["안녕하세요, 자기소개 부탁드려요.", "1부터 100까지 소수를 나열해줘."]
outputs = llm.generate(prompts, sampling_params)

# 4. 결과 확인
for output in outputs:
    print(f"답변: {output['text']}")
```

---

## 🚀 시리즈 마무리: 더 넓은 세상으로

여러분은 `nano-vllm`을 통해 최신 LLM 서빙의 핵심을 마스터했습니다!

*   **PagedAttention**: 메모리를 효율적으로 쓰는 법.
*   **스케줄링**: 수많은 요청의 우선순위를 정하는 법.
*   **텐서 병렬화**: 거대한 모델을 여러 장비에 나누어 돌리는 법.

이 지식은 실제 **vLLM, TGI, TensorRT-LLM**과 같은 상용 프레임워크를 이해하고 기여하는 밑거름이 될 것입니다. 이제 더 크고 멋진 인공지능 세상을 탐험해 보세요!

---
> **최종 요약**: 작은 코드가 큰 원리를 담고 있습니다. `nano-vllm`은 여러분의 LLM 엔지니어링 여정의 든든한 시작점이 될 것입니다.
