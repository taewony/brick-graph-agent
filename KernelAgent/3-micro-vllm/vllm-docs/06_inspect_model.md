# 06. 모델 분석: 메타데이터와 메모리 설계도

vLLM은 모델을 실행하기 전, 모델의 '설계도(Metadata)'를 먼저 읽어 들입니다. 이 설계도를 분석해야만 GPU 메모리를 얼마나 확보할지, 여러 개의 GPU로 나눌 수 있을지를 결정할 수 있습니다.

---

## 1. 왜 메타데이터를 분석해야 하나요?

우리가 집을 짓기 전 설계도를 보듯, vLLM은 모델의 `config.json`을 분석하여 다음을 결정합니다.

1.  **KV Cache 크기**: 한 단어(Token)를 저장할 때 몇 바이트가 필요한가?
2.  **병렬 처리 전략**: 이 모델의 Head 개수가 내 GPU 개수로 나누어지는가?
3.  **GQA(Grouped Query Attention) 적용 여부**: 메모리를 얼마나 아낄 수 있는 모델인가?

## 2. 분석 스크립트 사용법 (`src/inspect_model.py`)

우리가 만든 분석 스크립트는 모델의 '신체 검사' 결과지를 보여줍니다.

```bash
python src/inspect_model.py "모델_저장_경로"
```

### 핵심 출력 지표 설명
*   **KV Heads (GQA)**: 값이 1보다 크면(예: 32:8) 메모리 효율이 매우 좋은 모델입니다.
*   **Token당 KV Cache 크기**: 답변 한 단어당 들어가는 '메모리 비용'입니다. 레이어(Layer)가 많을수록 비싸집니다.
*   **Tensor Parallel 가능 여부**: GPU를 여러 개 쓸 때 모델을 '깔끔하게' 쪼갤 수 있는지 알려줍니다.

---

## 3. 실습을 위한 모델 다운로드 방법

분석을 위해서는 모델 파일이 로컬 PC에 있어야 합니다. DeepSeek와 같은 대형 모델을 다운로드하는 두 가지 방법을 소개합니다.

### 방법 A: PowerShell 이용하기 (권장)
`huggingface-cli`를 사용하면 가장 빠르고 안정적으로 다운로드할 수 있습니다.

```powershell
# 1. 도구 설치
pip install huggingface_hub

# 2. 모델 다운로드 (D:\models 폴더에 저장 예시)
huggingface-cli download deepseek-ai/deepseek-coder-7b-instruct-v1.5 `
    --local-dir "D:\models\deepseek-coder-7b-instruct-v1.5" `
    --local-dir-use-symlinks False
```

### 방법 B: Python 스크립트 이용하기 (`src/download_model.py`)
코드 안에서 경로를 지정하고 다운로드를 제어하고 싶을 때 유용합니다.

```python
# src/download_model.py 실행
python src/download_model.py --repo "deepseek-ai/deepseek-coder-7b-instruct-v1.5" --dest "D:/models"
```

---

## 📝 학생들을 위한 요약
1.  **지피지기**: 모델을 무작정 돌리기 전에 메타데이터(`config.json`)를 통해 모델의 성격을 파악해야 합니다.
2.  **계산된 추론**: vLLM은 이 메타데이터를 기반으로 **BlockManager**의 크기를 정하고 메모리 낭비를 줄입니다.
3.  **준비물**: 분석과 실행을 위해서는 모델을 로컬 폴더에 올바르게 다운로드하는 것이 첫걸음입니다.

---

  이제 학생들이 모델 다운로드 -> 메타데이터 분석 -> vLLM 실행의 전체 과정을 직접 실습할 수 있는 환경이 완벽하게
  갖춰졌습니다.


  💡 실습 시나리오 제안:
   1. 모델 준비: python src/download_model.py --repo "deepseek-ai/deepseek-coder-1.3b-instruct" --dest "./models"
      명령어로 가벼운 모델을 다운로드합니다.
   2. 구조 분석: python src/inspect_model.py "./models/deepseek-coder-1.3b-instruct" 명령어로 이 모델의 Hidden Size와 KV
      Cache 메모리 요구량을 확인합니다.
   3. 실제 실행: example.py를 수정하여 위 경로의 모델을 돌려봅니다.