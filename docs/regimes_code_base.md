주어진 repository 구조에서 **Super-Agent**(text-to-SQL + OKF ingest/lint/ask + incremental-learning mentor) 개발에 재활용할 수 있는 코드 파일을 **재사용 수준별**로 분류해 드리겠습니다.

---

## 1. 완전 재사용 가능한 범용 코어 (그대로 가져다 사용)

이 파일들은 특정 도메인에 의존하지 않는 **ActiveGraph 기반 Self-Improving Agent의 공통 인프라**입니다.  
Super-Agent의 모든 역할(SQL, OKF, Mentor)이 공유해야 하는 핵심 계층입니다.

| 파일 | 재사용 이유 |
|------|-------------|
| `agent/agent.py` | 에이전트 런타임 초기화, behavior 스냅샷, 실행 진입점. Super-Agent 전체를 감싸는 기본 틀로 재사용 |
| `agent/behaviors.py` | behavior 등록/관리 패턴. 각 역할(SQL/OKF/Mentor)의 behavior를 동일한 방식으로 등록할 때 재사용 |
| `agent/events.py` | 이벤트 상수 정의 패턴. Super-Agent의 통합 이벤트 vocabulary를 정의할 때 베이스로 사용 |
| `agent/build.py` | 에이전트 조립(타겟, action space, taxonomy 연결) 로직. Super-Agent의 역할별 타겟을 조립할 때 재사용 |
| `agent/embedders.py` | 임베딩 모듈. SQL 컬럼 스코어링, OKF 개념 검색, 요청 라우팅 등 **모든 의미 검색에 공통 사용** |
| `agent/transforms.py` | 프롬프트 변환 파이프라인의 범용 구현. SQL prompt pipeline과 OKF ask prompt pipeline 모두에서 재사용 |
| `agent/reader_transforms.py` | Reader 호출 전후에 적용하는 변환. OKF/SQL에서 LLM 입출력을 정규화할 때 재사용 |
| `agent/signals.py` | 구조적 신호 추출(조인/WHERE/GROUP BY 등)의 추상화. OKF의 lint 신호나 SQL 구조 신호 정의에 재사용 가능 |
| `agent/tokenize.py` | 토크나이저 래퍼. 모든 LLM 기반 처리에서 공통 사용 |
| `agent/stoplist.py` | 불용어 처리. 임베딩 검색에서 노이즈 제거용으로 재사용 |

---

## 2. Self-Improving Loop 전용 모듈 (Mentor 역할에 직접 재사용)

이 파일들은 **Incremental-Learning Mentor**를 구현하는 데 필요한 핵심 로직을 이미 포함하고 있습니다.

| 파일 | 재사용 이유 |
|------|-------------|
| `loop/runner.py` | 개선 루프 실행기. Mentor가 관찰→분석→제안→검증→적용하는 전체 사이클을 구동하는 데 재사용 |
| `loop/behaviors.py` | 개선 루프 전용 behavior들. Mentor의 observe/analyze/propose/validate/apply 단계를 그대로 재사용 |
| `loop/events.py` | 개선 루프 이벤트 정의. Mentor 관련 이벤트(`MENTOR_OBSERVE` 등)를 정의할 때 재사용 |
| `loop/gates.py` | 개선 제안의 승격(promotion)을 위한 게이트 검증 로직. Mentor가 새 transform/rule을 적용하기 전에 통과시켜야 할 게이트로 재사용 |
| `loop/hypothesize.py` | 개선 후보 생성(가설 생성) 로직. Mentor가 실패 패턴으로부터 프롬프트 개선안을 도출할 때 재사용 |
| `loop/regimes.py` | 평가 결과를 기반으로 regime(상태)을 분류하는 로직. SQL/OKF 결과를 성공/실패 패턴으로 그룹화할 때 재사용 |
| `loop/mock_eval.py` | 개선안 검증을 위한 목 평가 환경. Mentor의 validate 단계에서 hold-out 평가를 시뮬레이션할 때 재사용 |
| `loop/attribute.py` | 행동 속성 평가/기여도 분석. 어떤 transform이 성능에 기여했는지 추적할 때 재사용 |

---

## 3. SQL Target 전용 모듈 (Text-to-SQL 역할에 직접 재사용)

`targets/sql/` 아래 파일들은 이미 **Text-to-SQL** 역할에 특화되어 있으므로, Super-Agent의 SQL 역할 구현에 **거의 그대로** 사용할 수 있습니다.

| 파일 | 재사용 이유 |
|------|-------------|
| `targets/sql/agent/agent.py` | SQL 에이전트의 실행 진입점. Super-Agent 내 SQL 서브에이전트로 포함 |
| `targets/sql/agent/behaviors.py` | SQL용 behavior 4종(encode_schema, score_columns, prompt_pipeline, draft_query). Super-Agent의 SQL 이벤트 체인에 그대로 재사용 |
| `targets/sql/agent/events.py` | SQL 이벤트 상수. Super-Agent의 통합 이벤트 vocabulary에 SQL 부분으로 통합 |
| `targets/sql/action_space.py` | SQL ActionSpace. 프롬프트 변환 파이프라인 + SQL 게이트를 캡슐화. SQL 역할의 액션 공간으로 재사용 |
| `targets/sql/target.py` | SqlTarget 조립 팩토리. SQL 역할의 타겟 구성에 재사용 |
| `targets/sql/taxonomy.py` | SQL 구조 결정적 탐지기(조인/WHERE/GROUP BY). SQL 평가 신호 생성에 재사용 |
| `targets/sql/outcome.py` | SQL Outcome 정의. SQL 평가 결과를 구조화하는 데 재사용 |
| `targets/sql/prompt_transforms.py` | SQL 프롬프트 변환 파이프라인 등록/실행. SQL 프롬프트 최적화에 재사용 |
| `targets/sql/eval.py` | SQL 평가 백엔드. SQL 정오 판정에 재사용 |
| `targets/sql/sql_parse.py` | SQL 파서. SQL 구조 분석(조인/WHERE 등)에 재사용 |
| `targets/sql/exec.py` | SQL 실행 유틸리티. SQLite 실행 및 에러 캡처에 재사용 |
| `targets/sql/hypothesize.py` | SQL 개선 가설 생성. Mentor가 SQL 실패 원인을 분석할 때 재사용 |

---

## 4. OKF 역할에 패턴으로 재사용할 파일 (구현 참조용)

OKF ingest/lint/ask는 SQL과 도메인이 다르기 때문에 직접 재사용은 어렵지만, 아래 파일들은 **구조적 패턴을 그대로 차용**하여 OKF용으로 변형할 수 있습니다.

| SQL 파일 | OKF에서 대응되는 역할 |
|----------|------------------------|
| `targets/sql/action_space.py` | OKF ActionSpace (개념/규칙/스키마 통합 프롬프트 변환 + 가드레일) |
| `targets/sql/taxonomy.py` | OKF Taxonomy (규칙 충돌, 개념 매핑 오류, 스키마 무결성 등 결정적 lint 신호) |
| `targets/sql/outcome.py` | OKF Outcome (answer, context_parts, applied_rules, lint_errors) |
| `targets/sql/prompt_transforms.py` | OKF prompt transforms (개념 트리 정리, 규칙 주입, 힌트 생성) |
| `targets/sql/eval.py` | OKF 평가기 (답변 정확성, 규칙 준수, 개념 근거) |
| `targets/longmemeval/action_space.py` | OKF ActionSpace 설계 시 참조 (특히 장기 메모리/컨텍스트 관리 패턴) |
| `targets/longmemeval/target.py` | OKF Target 조립 팩토리 패턴 참조 |
| `targets/longmemeval/taxonomy.py` | OKF Lint 신호 정의 시 참조 |
| `targets/longmemeval/outcome_summary.py` | OKF Outcome 요약 및 감사(audit) 투영 방식 참조 |

---

## 5. 재사용 우선순위 요약

Super-Agent 개발 시 **가장 먼저 가져와야 할 파일**은 다음과 같습니다.

1. **범용 코어 전체** — `agent/`, `eval/`, `loop/`  
   → Super-Agent의 뼈대를 구성  
2. **SQL Target 전체** — `targets/sql/`  
   → Text-to-SQL 역할을 즉시 통합  
3. **OKF Target 개발 시 참조** — `targets/longmemeval/` + `targets/sql/action_space.py`, `taxonomy.py`, `outcome.py`, `prompt_transforms.py`  
   → OKF 고유의 ActionSpace/Taxonomy/Outcome/Prompt pipeline을 구현할 때 패턴으로 활용  

---

## 6. Coding Agent에 전달할 재활용 지침 예시

```text
We have an existing repository with an ActiveGraph-based self-improving agent core.
The following directories are directly reusable for the Super-Agent:

- agent/ → use as shared runtime and services (reader, embedder, transforms, events, behaviors)
- eval/ → use as evaluation abstraction (types, real evaluator)
- loop/ → use as incremental-learning mentor loop (runner, behaviors, gates, hypothesize)
- targets/sql/ → use as the text-to-sql role implementation (agent, action_space, taxonomy, outcome, prompt_transforms, eval, sql_parse)

For the OKF role, do NOT copy SQL-specific code directly. Instead, use the following files as structural templates:
- targets/sql/action_space.py → OKF ActionSpace
- targets/sql/taxonomy.py → OKF deterministic lint detectors
- targets/sql/outcome.py → OKF Outcome
- targets/sql/prompt_transforms.py → OKF prompt pipeline
- targets/longmemeval/target.py → OKF Target assembly pattern

Implement the Super-Agent by composing these existing components, adding only OKF-specific parsers and graph schema definitions.
```

이렇게 정리하면 Coding Agent가 불필요한 재개발 없이 **기존 코드를 최대한 재활용**하면서 Super-Agent를 빠르게 구축할 수 있습니다.