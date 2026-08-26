# 밑바닥부터 LLM 훈련하기

GPT 훈련 파이프라인의 모든 부분을 직접 작성하여 각 구성 요소가 무엇을 하고 왜 필요한지 이해하는 실습 워크숍입니다.

Andrej Karpathy의 [nanoGPT](https://github.com/karpathy/nanoGPT)는 LLM과 트랜스포머에 대한 저의 첫 실제 경험이었습니다. 몇 백 줄의 PyTorch 코드만으로 작동하는 언어 모델을 구축할 수 있다는 것을 보면서 AI에 대한 저의 생각이 완전히 바뀌었고, 이 분야에 더 깊이 들어가도록 영감을 주었습니다.

이 워크숍은 다른 사람들에게도 같은 경험을 제공하려는 저의 시도입니다. nanoGPT는 GPT-2(124M 파라미터) 재현을 목표로 하며 많은 내용을 다룹니다. 이 프로젝트는 핵심만 남기고 규모를 줄여 노트북에서 한 시간 이내에 훈련할 수 있는 약 10M 파라미터 모델로 만듭니다. 이는 단일 워크숍 세션에서 완료하도록 설계되었습니다.

## 무엇을 만들게 될까요?

MacBook에서 처음부터 훈련되어 셰익스피어 스타일의 텍스트를 생성할 수 있는 작동하는 GPT 모델을 만들게 됩니다. 다음을 작성하게 됩니다:

-   **토크나이저** — 텍스트를 모델이 처리할 수 있는 숫자로 변환
-   **모델 아키텍처** — 트랜스포머: 임베딩, 어텐션, 피드포워드 레이어
-   **훈련 루프** — 순방향 패스, 손실, 역전파, 옵티마이저, 학습률 스케줄링
-   **텍스트 생성** — 훈련된 모델에서 샘플링

## 전제 조건

-   모든 종류의 랩톱 또는 데스크톱 (Mac, Linux 또는 Windows)
-   Python 3.12+
-   Python 코드 읽기에 대한 이해 (ML 경험은 필요하지 않습니다)

훈련은 Apple Silicon GPU (MPS), NVIDIA GPU (CUDA) 또는 CPU를 자동으로 사용합니다. [Google Colab](https://colab.research.com/)에서도 작동합니다. 파일을 업로드하고 `!python train.py`를 실행하세요.

## 시작하기

### 로컬 설정 (권장)

1.  **`uv` 설치**: `uv`가 설치되어 있지 않다면, [uv 문서](https://docs.astral.sh/uv/)의 지침을 따르세요.

2.  **프로젝트 설정**: 프로젝트 루트에서 다음을 실행하여 의존성을 설치하고 `scratchpad` 디렉토리를 생성하세요:

    ```bash
    uv sync
    mkdir scratchpad
    ```

3.  **생성된 코드**: 워크숍의 핵심 파일은 `scratchpad/` 디렉토리에 있습니다:
    -   `model.py`: GPT 모델 아키텍처를 정의합니다.
    -   `train.py`: 훈련 루프 및 데이터 로딩 로직을 포함합니다.
    -   `generate.py`: 훈련된 모델로부터 텍스트 생성을 처리합니다.

4.  **모델 훈련**: 훈련을 시작하려면 (기본 5000 스텝), 다음을 실행하세요:

    ```bash
    uv run python scratchpad/train.py data/shakespeare.txt
    ```
    훈련 중에는 100 스텝마다 검증 손실과 생성된 텍스트 샘플이 출력됩니다.

5.  **텍스트 생성**: 훈련 후 `checkpoint_final.pt` 파일이 저장됩니다. 이를 사용하여 텍스트를 생성하세요:

    ```bash
    uv run python scratchpad/generate.py scratchpad/checkpoint_final.pt --prompt "To be or not" --max_new_tokens 200
    ```
    다양한 생성 스타일을 위해 `--prompt`, `--max_new_tokens`, `--temperature`, `--top_k`를 조정할 수 있습니다.

---

문서를 순서대로 진행하세요. 각 파트는 파이프라인의 한 부분을 작성하는 방법을 안내하며, 각 구성 요소가 무엇을 하고 왜 필요한지 설명합니다. 끝까지 진행하면 직접 작성한 작동하는 `model.py`, `train.py`, `generate.py`를 갖게 될 것입니다.

| Part | 무엇을 작성할 것인가 | 개념 |
|------|-------------------|----------|
| [Part 1: Tokenization](docs/01-tokenization.md) | 문자 수준 토크나이저 | 문자 인코딩, 어휘 크기, 작은 데이터에서 BPE가 실패하는 이유 |
| [Part 2: The Transformer](docs/02-the-transformer.md) | 전체 GPT 모델 아키텍처 | 임베딩, 자기-어텐션, 레이어 정규화, MLP 블록 |
| [Part 3: The Training Loop](docs/03-training-loop.md) | 완전한 훈련 파이프라인 | 손실 함수, AdamW, 기울기 클리핑, 학습률 스케줄링 |
| [Part 4: Text Generation](docs/04-text-generation.md) | 추론 및 샘플링 | 온도, top-k, 자동 회귀 디코딩 |
| [Part 5: Putting It All Together](docs/05-putting-it-together.md) | 실제 데이터로 훈련, 실험 | 손실 곡선, 스케일링 실험, 다음 단계 |
| [Part 6: Competition](docs/06-competition.md) | 최고의 AI 시인 훈련 | 데이터셋 찾기, 스케일 업, 최고의 시 제출 |

## 아키텍처: GPT 한눈에 보기

```
입력 텍스트
    │
    ▼
┌─────────────────┐
│   토크나이저    │  "hello" → [20, 43, 50, 50, 53]  (문자 수준)
└────────┬────────┘
         ▼
┌─────────────────┐
│  토큰 임베딩 +  │  토큰 ID → 벡터 (n_embd 차원)
│  위치 임베딩    │  + 위치 정보
└────────┬────────┘
         ▼
┌─────────────────┐
│  트랜스포머     │  × n_layer
│  블록:          │
│  ┌────────────┐ │
│  │ 레이어 정규화│ │
│  │ 자기-어텐션  │ │  n_head 병렬 어텐션 헤드
│  │ + 잔차 연결  │ │
│  ├────────────┤ │
│  │ 레이어 정규화│ │
│  │ MLP (FFN)    │ │  4배 확장, GELU, 다시 투영
│  │ + 잔차 연결  │ │
│  └────────────┘ │
└────────┬────────┘
         ▼
┌─────────────────┐
│  레이어 정규화  │
│  선형 → 로짓     │  vocab_size 출력 (다음 토큰에 대한 확률)
└─────────────────┘
```

## 이 워크숍의 모델 구성

| 구성 | 파라미터 | n_layer | n_head | n_embd | 훈련 시간 (M3 Pro) |
|--------|--------|---------|--------|--------|---------------------|
| Tiny | ~0.5M | 2 | 2 | 128 | ~5분 |
| Small | ~4M | 4 | 4 | 256 | ~20분 |
| **Medium (기본값)** | **~10M** | **6** | **6** | **384** | **~45분** |

모든 구성은 문자 수준 토크나이징 (vocab_size=65) 및 block_size=256을 사용합니다.

## 토크나이징: 문자 vs BPE

이 워크숍은 셰익스피어에 대해 **문자 수준** 토크나이징을 사용합니다. BPE 토크나이징 (GPT-2의 50k 어휘)은 작은 데이터셋에서는 작동하지 않습니다. 대부분의 토큰 바이그램이 너무 드물어서 모델이 패턴을 학습할 수 없습니다.

| 토크나이저 | 어휘 크기 | 필요한 데이터셋 크기 |
|-----------|-----------|-------------------|
| **문자 수준** | ~65 | 작음 (셰익스피어, ~1MB) |
| **BPE (tiktoken)** | 50,257 | 큼 (TinyStories+, 100MB+) |

Part 5에서는 더 큰 데이터셋을 위해 BPE로 전환하는 방법을 다룹니다.

## 주요 참고 자료

- [nanoGPT](https://github.com/karpathy/nanoGPT) — 이 워크숍의 기반이 된 프로젝트. PyTorch로 약 300줄 만에 GPT 훈련 최소화
- [build-nanogpt 비디오 강의](https://github.com/karpathy/build-nanogpt) — 빈 파일에서 GPT-2를 구축하는 4시간 비디오
- [Karpathy의 microgpt](http://karpathy.github.io/2026/02/12/microgpt/) — 순수 Python, 의존성 없음으로 200줄 만에 전체 GPT 구현
- [nanochat](https://github.com/karpathy/nanochat) — 전체 ChatGPT 클론 훈련 파이프라인
- [Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762) — 오리지널 트랜스포머 논문
- [GPT-2 논문 (2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — 비지도 학습자로서의 언어 모델
- [TinyStories 논문](https://arxiv.org/abs/2305.07759) — 큐레이션된 데이터로 훈련된 작은 모델이 기대 이상의 성능을 내는 이유
