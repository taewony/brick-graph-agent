첨부해주신 `infer.sh`는 Linux나 WSL 환경에서 사용되는 **Bash 쉘 스크립트**입니다. 이를 Windows PowerShell에서 실행하려면 PowerShell 문법에 맞게 변환한 `.ps1` 스크립트를 작성해야 합니다.

PowerShell에서는 변수 선언, 환경 변수(`PYTHONPATH`) 설정, 그리고 줄바꿈 문자(`\` 대신 ``` 사용)가 다르게 적용됩니다.

### 🚀 1. PowerShell 스크립트 작성 (`infer.ps1`)

작업 중인 `2-LLM-from-scratch` 폴더 안에 **`infer.ps1`** 이라는 이름으로 새 파일을 만들고, 아래의 코드를 복사해서 붙여넣으세요.

```powershell
# 설정 변수
$CHECKPOINT = "checkpoint_final.pt"
$PROMPT = "O Romeo, Romeo! wherefore art thou Romeo?"
$MAX_TOKENS = 200
$TEMPERATURE = 0.8

Write-Host "🚀 Starting LLM Inference Process..."
Write-Host "Checkpoint: $CHECKPOINT"
Write-Host "Prompt: '$PROMPT'"

# PYTHONPATH 설정 (현재 경로의 src 폴더 추가, Windows는 구분자로 ';' 사용)
$CurrentDir = Get-Location
$env:PYTHONPATH = "$CurrentDir\src;$env:PYTHONPATH"

# 추론 실행 (PowerShell의 줄바꿈 문자인 백틱(`) 사용)
python src\generate.py `
    $CHECKPOINT `
    --prompt "$PROMPT" `
    --max_new_tokens $MAX_TOKENS `
    --temperature $TEMPERATURE

Write-Host "✅ Inference Completed."

```

### 💻 2. PowerShell에서 실행하는 방법

1. PowerShell을 열고 해당 폴더(`2-LLM-from-scratch`)로 이동합니다.
2. 터미널에 아래 명령어를 입력하여 실행합니다.
```powershell
.\infer.ps1

```



---

### ⚠️ 만약 "이 시스템에서 스크립트를 실행할 수 없으므로..." 라는 에러가 뜬다면?

Windows는 기본적으로 보안을 위해 스크립트 실행을 막아두는 경우가 많습니다. 에러가 발생한다면 PowerShell을 **관리자 권한**으로 실행한 뒤, 아래 명령어를 한 번 입력하여 권한을 허용해 주시면 됩니다.

```powershell
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser

```

(위 명령어를 입력하고 `Y`를 누른 뒤, 다시 `.\infer.ps1`을 실행하시면 정상적으로 작동합니다.)