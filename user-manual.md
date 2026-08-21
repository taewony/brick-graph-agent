# brick.agent User Manual

**brick.agent**는 두 가지 자연어 Q&A 서비스를 하나의 CLI로 제공하는 에이전트 시스템입니다.

1. **OKF Q&A** — 사내 문서를 OKF(Open Knowledge Format) wiki 지식베이스로 구축하고, 그 지식에 대해 질문에 답합니다.
2. **SQL Q&A** — 자연어 질문을 SQL로 변환해 SQLite 데이터베이스에서 실제로 조회하고 결과를 돌려줍니다.

모든 실행은 **이벤트 소싱 구조** 위에서 동작하며, LLM 호출은 `llm.requested` / `llm.responded` 이벤트로 로그에 남습니다(`--trace`로 확인, `data/events.db`에 영속).

---

## 1. 요구 사항 및 실행 환경

- Python 3.12 + `.wenv` 가상환경 (`D:\code\.wenv`, activegraph 1.10.0 포함)
- 별도 설치 없이 아래 두 가지 방식 중 하나로 실행합니다.

**방식 A — venv 활성화 후 실행 (PowerShell)**
```powershell
cd D:\code\brick-graph-agent
.\.wenv\Scripts\Activate.ps1
python -m src.agents.brick_agent --help
```

**방식 B — venv 파이썬을 직접 지정 (활성화 불필요, 권장)**
```powershell
cd D:\code\brick-graph-agent
.\.wenv\Scripts\python.exe -m src.agents.brick_agent --help
```

아래 매뉴얼의 예제는 모두 방식 B의 `python`을 `python`으로 표기합니다.

---

## 2. 빠른 시작 (5분 데모, API 키 불필요)

```powershell
# 1) SQL용 데모 DB 준비 (data/store_front.db — .okf/02_store_front 기반 시드)
python -m src.agents.brick_agent db seed

# 2) SQL Q&A (데이터베이스 조회)
python -m src.agents.brick_agent ask sql "How many stable tools are in the store front?"

# 3) OKF Q&A (지식베이스 질문)
python -m src.agents.brick_agent ask okf "How does prefill_phase relate to decode_phase?"

# 4) 역할 자동 분류 (router) — "How many..."는 sql로, "What is..."은 okf로
python -m src.agents.brick_agent ask "List the stable decisions"

# 5) 관측 가능성: LLM 호출 추적 포함
python -m src.agents.brick_agent ask okf "What is the kv_cache?" --trace
```

실행 결과 예시(SQL):
```
reader      : demo-sql-reader
schema      : D:\code\brick-graph-agent\data\store_front.db (5 tables)
question    : How many stable tools are in the store front?
SQL         : SELECT COUNT(*) FROM tools WHERE status = 'stable'
columns     : ['COUNT(*)']
  row       : (5,)
```

실행 결과 예시(OKF):
```
reader      : demo-okf-reader
kb          : D:\code\brick-graph-agent\.okf\01_nano_vllm
question    : How does prefill_phase relate to decode_phase?
answer      : prefill_phase is a prerequisite of decode_phase
transforms  : ['inject_concept_tree', 'inject_rules', 'trim_context']
concepts    : ['atomic.decode_phase', 'atomic.prefill_phase', 'composite.inference_model', ...]
```

---

## 3. CLI 명령 참조

```
brick-agent
├── ask [{sql|okf}] <QUESTION>     # 역할 지정 또는 자동 분류(router)
├── ask-sql (alias: sql) <Q>       # SQL Q&A 전용
├── ask-okf (alias: okf) <Q>       # OKF Q&A 전용
├── browse [--kb NAME|PATH] [--out PATH] [--no-open]  # KB 웹 대시보드 생성+열기
├── ingest  --kb-id <ID> --kb-path <PATH>   # OKF 번들 적재
├── lint    --kb-id <ID> [--kb-path <PATH>] # OKF 무결성 검사
├── db seed [--db <PATH>]          # data/store_front.db 시드(재생성)
└── mentor status                  # 파이프라인/가드레일/저장 run 현황
```

### 공통 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--real` | off | 실제 LLM(AnthropicReader) 사용. `ANTHROPIC_API_KEY` 필요. 미지정 시 키 불필요한 deterministic demo reader |
| `--store <PATH>` | `data/events.db` | 이벤트 스토어(SQLite) 경로 — 모든 run의 이벤트 로그 영속 |
| `--trace` | off | 실행 후 `llm.requested` / `llm.responded` 관측 트레일 출력 |
| `--db <PATH>` | `data/store_front.db` | SQL Q&A가 조회할 SQLite 파일 (ask-sql/ask 공통) |
| `--kb <NAME\|PATH>` | `01_nano_vllm` | OKF 번들 이름(→ `.okf/<name>`) 또는 경로 (ask-okf/ask 공통) |
| `--top-k <N>` | 10 | OKF 개념 검색 상위 N개 |

### 종료 코드

| 코드 | 의미 |
|------|------|
| `0` | 성공 |
| `1` | 드래프트/실행/인자 값 오류 (메시지 출력 후 종료) |
| `2` | CLI 사용법 오류 (argparse) |

---

## 4. SQL Q&A 서비스 (자연어 → DB 쿼리)

### 4.1 데이터베이스 준비

`ask sql`은 `--db`로 지정한 SQLite 파일을 **스키마 인트로스펙션**하여 사용합니다. 기본 파일 `data/store_front.db`는 `.okf/02_store_front` 번들의 YAML frontmatter에서 자동 생성되며, 없으면 `ask sql` 첫 실행 시 자동으로 시드됩니다. 명시적으로 다시 만들려면:

```powershell
python -m src.agents.brick_agent db seed
# seeded data/store_front.db: {'bundles': 3, 'tools': 5, 'skills': 3, 'decisions': 7, 'assets': 22}
```

생성되는 테이블:

| 테이블 | 주요 컬럼 | 설명 |
|--------|-----------|------|
| `bundles` | name(PK), status, description | `.okf` 루트 번들 (00_agent_model, 01_nano_vllm, 02_store_front) |
| `tools` | id(PK), name, kind, status, path, bundle(FK) | OKF 도구(okf_init, validator, visualizer, ...) |
| `skills` | id(PK), name, status, path, bundle(FK) | okf / validate / visualize 스킬 |
| `decisions` | id(PK), title, status, path, summary, bundle(FK) | 설계 결정 (ADR) |
| `assets` | (bundle, id)(PK), type, status, path | 번들 내 모든 노드 통합 뷰 |

### 4.2 데모 질문 (키 없이 바로 사용 가능)

| 질문 | 생성되는 SQL |
|------|--------------|
| `How many stable tools are in the store front?` | `SELECT COUNT(*) FROM tools WHERE status = 'stable'` |
| `How many tools are in the store front?` | `SELECT COUNT(*) FROM tools` |
| `List the stable decisions` | `SELECT title FROM decisions WHERE status = 'stable'` |
| `How many skills are in the store front?` | `SELECT COUNT(*) FROM skills` |
| `Which bundles are in the knowledge base?` | `SELECT name FROM bundles ORDER BY name` |

### 4.3 자신의 SQLite DB로 질문하기

자신의 DB 파일(`.db`)을 만들고 테이블·PK·FK를 정의하면, demo reader 대신 `--real`로 LLM을 사용해 자유로운 질문을 할 수 있습니다.

```powershell
# 예: 사용자 DB 생성
sqlite3 data/mydb.db "CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, price REAL); INSERT INTO products VALUES (1,'Laptop',1200.0),(2,'Mouse',25.0);"

# LLM으로 질문 (ANTHROPIC_API_KEY 필요)
python -m src.agents.brick_agent ask sql "What is the average price of products?" --db data/mydb.db --real --trace
```

`ask sql`의 동작 순서: **DB 스키마 인트로스펙션 → 스키마 그래프 인코딩(`schema.encoded`) → 컬럼 임베딩 스코어링(`columns.scored`) → 프롬프트 조립+변환(`prompt.assembled`) → LLM이 SQL 드래프트(`query.drafted`, llm.* 이벤트 기록) → SQLite 실행 → 결과 행 출력**.

> 참고: 가드레일 모듈(`src/runtime/guardrails.py`)에 SELECT-only 정책이 정의되어 있습니다(`mentor status`에서 확인). CLI는 드래프트된 SQL을 그대로 실행하므로, 신뢰할 수 없는 DB에는 `--real` 실행 시 주의하세요.

---

## 5. OKF Q&A 서비스 (지식베이스 질문)

### 5.1 지식베이스

- 기본 KB: `.okf/01_nano_vllm` (38개 개념 노드, 54개 전체 노드)
- `--kb`로 이름(→ `.okf/<name>`) 또는 경로 지정 가능. 예: `--kb 02_store_front`, `--kb .okf/00_agent_model`

### 5.2 데모 질문 (키 없이 바로 사용 가능)

| 질문 | 답변 |
|------|------|
| `How does prefill_phase relate to decode_phase?` | `prefill_phase is a prerequisite of decode_phase` |
| `What is the kv_cache?` | KV 캐시 개념 설명 (atomic.kv_cache 근거) |

### 5.3 동작 순서

`ask okf`는 다음 이벤트 체인을 구동합니다.

```
okf.ask.requested
  → okf.context.assembled   # 질문 임베딩 → top-K 개념 검색(--top-k)
  → llm.requested           # 프롬프트(prompt_hash) 기록
  → llm.responded           # 모델·답변·지연시간 기록
  → okf.answer.generated    # grounded 답변 + 적용된 transforms
```

컨텍스트는 **개념 트리 확장(부모/자식)** → **적용 규칙 선택** → **스키마 선택** → **프롬프트 변환 파이프라인**(`inject_concept_tree` → `inject_rules` → `trim_context`)을 거쳐 조립됩니다. 출력의 `concepts`는 사용된 개념 id 목록입니다.

### 5.4 관련 부가 명령

```powershell
# 번들 적재 (검증 + 그래프 객체 materialize, 이벤트 로그 저장)
python -m src.agents.brick_agent ingest --kb-id nano --kb-path .okf/01_nano_vllm

# 무결성 린트 (8개 디텍터: 순환/역방향 불일치/dangling/노이즈/증빙 부재/orphan 등)
python -m src.agents.brick_agent lint --kb-id nano --store data/events.db
# lint: valid=True n_errors=0  (Phase 8 수리 완료 — lint clean)
```

---

## 6. 역할 자동 분류 (router)

`ask "<질문>"`(역할 미지정)은 `src/agents/router.py::classify`로 역할을 자동 판별합니다.

```powershell
python -m src.agents.brick_agent ask "How many stable tools are in the store front?"
# role : sql (router) → SQL Q&A 실행

python -m src.agents.brick_agent ask "How does prefill_phase relate to decode_phase?"
# role : okf_ask → OKF Q&A 실행
```

분류 키워드 예: `select/how many/table/column` → sql, `okf/concept/rule/what is/how does` → okf, `ingest` → okf_ingest, `lint` → okf_lint, `mentor` → mentor.

---

## 6.5 OKF KB 웹 브라우징 (`browse`)

지식베이스를 **웹 대시보드**로 열어 그래프·문서를 시각적으로 탐색하고 학습할 수 있습니다
(vis-network 그래프 + 사이드바 노드 목록 + 마크다운 본문 뷰, 필터: ATOMIC/COMPOSITE/MODULE/META).

```powershell
# docs/01_nano_vllm/index.html 생성 + 기본 브라우저로 열기
python -m src.agents.brick_agent browse --kb 01_nano_vllm

# 다른 번들 / 특정 경로
python -m src.agents.brick_agent browse --kb 02_store_front
python -m src.agents.brick_agent browse --kb .okf/00_agent_model --out docs/agent-model.html

# 브라우저 없이 생성만 (CI 등)
python -m src.agents.brick_agent browse --kb 01_nano_vllm --no-open
```

- 출력 기본 위치: `docs/<kb 이름>/index.html` (`01_nano_vllm` → `docs/01_nano_vllm/index.html`)
- **앞으로/뒤로 네비게이션**: 사이드바 상단 `◀ 뒤로` / `앞으로 ▶` 버튼 또는 `Alt+←` / `Alt+→`로 방문한 노드를 이동 (브라우저 히스토리처럼 동작, 현재 위치 `n / N` 표시)
- **Learning Path 학습 링크**: `meta/learning_path.md`의 개념명은 실제 개념 파일(`../concepts/03_atomic/*.md` 등)로 연결되어 있어, 대시보드에서 클릭만으로 해당 개념 문서로 이동해 학습할 수 있습니다
- **인덱스 모듈 맵 링크**: `index.md`의 8개 Module 이름이 각 모듈 문서로, 핵심 개념이 실제 개념 파일로 연결
- **트리 하향 링크**: 모든 관계 블록(PREREQUISITES / PREREQUISITE_OF / COMPOSED_OF 등)의 개념 id가
  markdown 링크로 변환되어 module→composite→atomic 하향 트리를 클릭으로 탐색 가능 (그래프 엣지에도 반영)
- 그래프 노드를 클릭하면 해당 개념의 마크다운 본문이 표시되고, 본문 내 `.okf/` 링크는 노드 간 이동으로 동작합니다.
- `make browse KB=01_nano_vllm` 도 동일합니다.

---

## 7. 실제 LLM 사용 (`--real`)

demo reader는 결정적이어서 키 없이도 동작하지만, **미리 정의된 질문**만 답합니다. 자유로운 질문은 `--real`로 실제 LLM을 사용합니다.

```powershell
# Windows PowerShell
$env:ANTHROPIC_API_KEY = "sk-..."
python -m src.agents.brick_agent ask okf "Why is PagedAttention memory-efficient?" --real
python -m src.agents.brick_agent ask sql "How many deprecated tools exist?" --real
```

- 키가 없을 때 `--real` 사용 시: `AnthropicReader requires ANTHROPIC_API_KEY in the environment.` 에러로 안내됩니다.
- `--real` 미지정 + 데모 질문이 아닌 경우: `draft error: (empty SQL)` / 빈 답변이 나옵니다 — 이 경우 데모 질문을 쓰거나 `--real`을 켜세요.

---

## 8. 관측 가능성 (LLM observability)

### 8.1 `--trace` — LLM 호출 트레일

```powershell
python -m src.agents.brick_agent ask sql "How many stable tools are in the store front?" --trace
# ...
#   llm.req  : id=evt_048 model=demo-sql-reader hash=276e6283246b
#   llm.resp : caused_by=evt_048 latency=0.0000s error=- answer='SELECT COUNT(*) FROM tools ...'
```

- `llm.requested`: request_id, model, **prompt_hash**(내용 키), prompt
- `llm.responded`: caused_by(요청 이벤트 id), model, **answer**(기록됨 → 결정적 재생 가능), latency_seconds, error

### 8.2 이벤트 스토어 (`--store`)

모든 run(질문/적재/린트)의 전체 이벤트 로그가 `data/events.db`에 append-only로 영속됩니다.

```powershell
python -m src.agents.brick_agent mentor status
# sql pipeline : []  |  okf pipeline : [inject_concept_tree, inject_rules, trim_context]
# guardrails   : sql.allow_only_select=True okf.required_lint_before_ask=True mentor.min_improvement_threshold=0.02
# stored runs  : 5 (data/events.db)
# agent model  : history v1.4.0 (11 operators)
```

### 8.3 프로그램적 재생/캐시 (라이브러리)

```python
from src.runtime import event_store, replay

runs = event_store.list_runs("data/events.db")          # 저장된 run 목록
events = event_store.run_events("data/events.db", runs[0].run_id)
cache = replay.build_replay_cache(events)               # prompt_hash → 답변
rr = replay.ReplayReader(cache)                          # 결정적 재생 reader
graph, n = replay.replay_into_graph("data/events.db", runs[0].run_id)  # 로그→그래프 재구성
```

---

## 9. 전체 예제 세션

```powershell
cd D:\code\brick-graph-agent

# 1) 준비
python -m src.agents.brick_agent db seed

# 2) SQL Q&A 3종
python -m src.agents.brick_agent ask sql "How many tools are in the store front?"
python -m src.agents.brick_agent ask "List the stable decisions"
python -m src.agents.brick_agent sql "Which bundles are in the knowledge base?"

# 3) OKF Q&A 2종 (관측 트레일 포함)
python -m src.agents.brick_agent ask okf "How does prefill_phase relate to decode_phase?" --trace
python -m src.agents.brick_agent ask "What is the kv_cache?"

# 4) KB 관리
python -m src.agents.brick_agent ingest --kb-id nano --kb-path .okf/01_nano_vllm
python -m src.agents.brick_agent lint --kb-id nano

# 5) 상태 확인
python -m src.agents.brick_agent mentor status
```

---

## 10. 문제 해결

| 증상 | 원인/해결 |
|------|-----------|
| `draft error: (empty SQL)` 또는 빈 답변 | demo reader가 모르는 질문. 데모 질문(§4.2/§5.2)을 사용하거나 `--real`로 LLM 켜기 |
| `AnthropicReader requires ANTHROPIC_API_KEY` | `--real` 사용 시 `ANTHROPIC_API_KEY` 환경변수 설정 필요 |
| `no Reader registered for request_id` | 내부 오류(reader 등록 누락) — 버그 리포트 대상 |
| `exec error: OperationalError: no such table: ...` | `--db` 경로가 잘못됨. `db seed` 후 `data/store_front.db` 확인 |
| `lint: valid=True n_errors=0` | Phase 8에서 KB 무결성 수리 완료 (수리 전: 순환 7·역방향 53·dangling 14·증빙 부재 28 등 106건). `make repair`로 재수리 가능 |
| `FileNotFoundError: OKF knowledge base directory not found` | `--kb` 이름/경로 확인 (`.okf/<name>` 존재 여부) |
| 콘솔 출력 깨짐 | CLI는 UTF-8로 재구성합니다. 터미널 폰트/코드페이지 확인 |
| `UNIQUE constraint failed` (이벤트 스토어) | 같은 스토어 파일을 여러 run이 재사용 시 각 run은 unique run_id를 받으므로 정상적으로는 발생하지 않음. 스토어 파일 삭제 후 재시도 |

---

## 11. 검증 (테스트)

```powershell
python -m pytest tests/ -q        # 55 tests — 전체 스위트
python -m pytest tests/test_brick_cli.py -q        # CLI 전용
python -m pytest tests/test_phase5_observability.py -q  # 관측/재생 전용
```

---

## 참고: 관련 아티팩트

| 경로 | 설명 |
|------|------|
| `src/agents/brick_agent.py` | CLI 진입점 |
| `src/agents/router.py` | 역할 자동 분류 |
| `src/agents/okf/` | OKF 런타임 behaviors + agent (ingest/lint/ask) |
| `src/core/targets/sql/` | SQL 타겟 (스키마 인코딩 → 컬럼 스코어링 → 프롬프트 → 드래프트) |
| `src/core/targets/okf/` | OKF 타겟 (Phase 2: outcome/taxonomy/action_space/prompt_transforms/eval/target) |
| `src/runtime/` | 공유 서비스 (reader_registry, embedder, guardrails, history_logger, loader, event_store, observability, replay) |
| `data/store_front.db` | SQL Q&A 데모 DB (`.okf/02_store_front` 시드) |
| `data/events.db` | 이벤트 소싱 로그 (모든 run) |
| `.okf/00_agent_model/` | 통합 에이전트 스펙 (events.yaml, behaviors.spec.yaml, concepts/observability.md) |
| `brick-agent-plan.md` | 개발 계획 (Phase 1–8, Phase 1–6 완료) |
