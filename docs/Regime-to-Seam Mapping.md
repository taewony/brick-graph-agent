좋은 질문입니다.  
**Regime-to-Seam Mapping**은 에이전트가 실패했을 때 “이 실패는 어느 수정 지점(seam)을 건드릴 것인가”를 정해둔 **정적 정책**입니다.  
우리가 논의한 Super-Agent 모델 기준으로 보면, 이 매핑은 다음 위치에서 정의되고 소비됩니다.

---

## 1. 어느 파일에 정의되는가?

### 현재 `regimes` 저장소 기준

| 역할 | 파일 | 설명 |
|------|------|------|
| **Regime 분류** | `loop/regimes.py` | 평가 결과와 구조 신호를 받아 실패 유형(regime)을 분류 |
| **Seam 정의** | `targets/sql/action_space.py` | 어떤 seam들이 존재하는지 정의 (프롬프트 변환, SQL 게이트 등) |
| **Regime→Seam 매핑** | `loop/regimes.py` 또는 `loop/hypothesize.py` | 실패 유형별로 시도할 seam을 연결한 테이블/정책 |

일반적으로 **`loop/regimes.py`에 정적 매핑 테이블**을 두는 것이 자연스럽습니다.

예를 들면:

```python
# loop/regimes.py (개념적 예시)
REGIME_SEAM_MAP = {
    "missing_join":      ["prompt_transform:add_join_hint"],
    "missing_group_by":  ["prompt_transform:add_group_by_hint"],
    "wrong_table":       ["column_scoring:threshold_tuning"],
    "excess_columns":    ["prompt_transform:filter_unused_columns"],
}
```

이 매핑은 `loop/hypothesize.py`가 개선 후보를 만들 때 참조합니다.  
즉, **regime을 분류하는 파일과 그 regime에 맞는 seam을 고르는 파일이 분리**되어 있을 수 있지만, 실질적인 정책은 `loop/regimes.py` 또는 그와 유사한 설정에 존재합니다.

---

### 우리가 제안한 YAML 기반 모델 기준

YAML로 선언한다면 이 매핑은 다음 위치에 두는 것이 좋습니다.

```text
config/agent_model/
├── regimes.yaml          # 실패 유형과 seam 매핑 정책
└── workflows/
    └── mentor_workflow.yaml
```

예시:

```yaml
regimes:
  missing_join:
    seams:
      - prompt_transform:add_join_hint
    priority: 1

  missing_group_by:
    seams:
      - prompt_transform:add_group_by_hint
    priority: 2

  wrong_table_selection:
    seams:
      - column_scoring:threshold_tuning
    priority: 3
```

이 YAML은 OKF 컴파일러가 `behaviors.yaml`에 반영하거나, 런타임 로더가 직접 읽어서 loop에 주입합니다.

---

## 2. 누가, 언제 정의해 두는가?

| 항목 | 설명 |
|------|------|
| **누가** | 시스템 설계자 또는 에이전트 모델러. 즉, 우리가 개발 시점에 수동으로 정의합니다. |
| **언제** | 에이전트를 실행하기 전, 타겟을 조립(`build_target`)하거나 mentor loop을 구성할 때 미리 정의해 둡니다. |
| **성격** | 초기에는 **고정된 정책**입니다. 런타임 중 스스로 배우지 않습니다. |
| **변경 주체** | 나중에 Mentor가 `loop/hypothesize.py`를 통해 새로운 seam을 제안하거나 매핑을 바꿀 수 있지만, 최초의 매핑은 사람이 설계합니다. |

즉, 런타임 에이전트가 실패할 때마다 모든 seam을 탐색하는 것이 아니라,  
**미리 사람이 정의한 regime→seam 표**를 보고 가장 가능성 높은 수정 지점부터 시도하게 됩니다.

---

## 3. 전체 흐름에서의 위치

```
평가 결과 (SqlOutcome / OkfOutcome)
        ↓
[loop/regimes.py]
   구조 신호 + 정오 여부를 보고 실패 유형(regime) 분류
        ↓
[Regime→Seam Mapping]
   예: missing_join → prompt_transform:add_join_hint
        ↓
[loop/hypothesize.py]
   해당 seam에 대한 개선 후보 생성
        ↓
[loop/gates.py + mock_eval]
   후보 검증
        ↓
적용 또는 폐기
```

따라서 **Regime-to-Seam Mapping**은  
- **`loop/regimes.py`(코드)** 또는  
- **`config/agent_model/regimes.yaml`(우리 YAML 모델)**  

에 정의되어, 실패가 발생한 시점에 mentor가 그 표를 읽고 사용하는 **사전 정의된 정책**입니다.