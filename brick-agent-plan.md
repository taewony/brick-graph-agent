## Current State Assessment

We have built a robust **OKF Compiler** that:

- Validates YAML frontmatter and links across `.okf/` bundles.
- Builds an in-memory **IR graph** from knowledge nodes.
- Applies **history operators** (SPLIT, MERGE, etc.) to evolve the graph.
- Emits **behaviors.yaml** – a static, Cypher‑based executable control spec for ActiveGraph.

To build the full **Brick‑Agent** (text‑to‑SQL + OKF‑ingest/lint/ask + incremental‑learning mentor), we need to generalise the architecture.

---

## Revised Brick‑Agent Architecture

We will keep the existing OKF Compiler as the **knowledge ingestion and management layer**, but extend it to handle **multiple knowledge domains** (sales DB schema, OKF concepts/rules). On top of that, we place an **ActiveGraph Runtime** that executes event‑driven behaviours for each role. A **Mentor** observes outcomes and feeds improvements back into the knowledge graph via the compiler.

---

## 1. 수정의 핵심 원칙

agent_model로부터 “코드 생성”에서 “설정 로딩 + 기존 코드 조립”으로 전환한다.
이렇게 하면 유지보수 부담이 줄고, 이미 검증된 Self-Improving loop를 그대로 활용할 수 있다.
다만, agent_model은 전체 brick-agent의 핵심 사항을 명세하고 있어야 한다.

| 기존 계획 | 수정된 계획 |
|-----------|--------------|
| `agent_model` YAML → runtime Python code 생성 | `regimes` core를 runtime으로 재사용 |
| YAML이 실행 로직을 포함 | YAML은 **구조·이벤트·행동 계약**과 **감사 이력**만 기술 |
| Compiler가 전체 behavior 코드 생성 | Compiler는 `behaviors.yaml` **계약 명세**와 **handler 참조**만 생성 |
| 역할별 코드를 새로 작성 | SQL·Mentor는 `regimes`에서 재사용, OKF만 패턴 따라 신규 작성 |

- The YAML files (config/agent_model/*.yaml) will serve as declarative contracts for events, objects, and guardrails, but not as a source of Python code generation.

- The actual behaviours are defined in Python (reusing src/core/agent, src/core/loop, src/core/targets/sql) and registered directly with ActiveGraph.

- The OKF compiler continues to manage knowledge integrity and produce graph data; the runtime loader consumes that data into ActiveGraph.

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Brick-Agent Entry Point                │
│            (CLI / API – request routing & dispatch)        │
├─────────────────────────────────────────────────────────────┤
│              ActiveGraph Runtime (event bus)               │
│   – SQL behaviours   – OKF behaviours   – Mentor behaviours│
│   (실제 핸들러는 regimes.core에서 로드)                   │
├─────────────────────────────────────────────────────────────┤
│   Shared Services: Reader Registry, Embedder, Transform    │
│   Pipeline, Evaluator, History Logger, Guardrails          │
│   (regimes.core.agent / eval / loop 재사용)                │
├─────────────────────────────────────────────────────────────┤
│  OKF Compiler + IR (기존 유지) → behaviors.yaml (계약)    │
│  – .okf 지식베이스 검증/변환                              │
│  – 컴파일된 그래프 스펙 + handler 참조 생성                │
│  – runtime 이력은 .okf/*/history.yaml에 기록               │
└─────────────────────────────────────────────────────────────┘
```

- **SQL role** → `regimes.targets.sql` 전체 재사용  
- **Mentor role** → `regimes.loop` 전체 재사용  
- **OKF role** → `regimes.targets.longmemeval`의 구조를 참고하여 신규 작성  
- **공통 서비스** → `regimes.agent` 전체 재사용

---

## 3. 수정된 Project Structure 제안

기존 `regimes` 코드를 `src/core/` 아래로 통합하고,  
기존 OKF 컴파일러(`src/okf/`)와 새로 만드는 runtime/role 코드를 분리합니다.

```text
brick-agent/
├── src/
│   ├── core/                          # regimes 저장소에서 재사용/통합
│   │   ├── agent/                     # 공통 에이전트 코어
│   │   │   ├── agent.py
│   │   │   ├── behaviors.py
│   │   │   ├── build.py
│   │   │   ├── embedders.py
│   │   │   ├── events.py
│   │   │   ├── reader_transforms.py
│   │   │   ├── signals.py
│   │   │   ├── stoplist.py
│   │   │   ├── tokenize.py
│   │   │   ├── transforms.py
│   │   │   └── __init__.py
│   │   ├── eval/                      # 평가 추상화
│   │   │   ├── real.py
│   │   │   ├── types.py
│   │   │   └── __init__.py
│   │   ├── loop/                      # Self-Improving Mentor 루프
│   │   │   ├── attribute.py
│   │   │   ├── behaviors.py
│   │   │   ├── events.py
│   │   │   ├── gates.py
│   │   │   ├── hypothesize.py
│   │   │   ├── mock_eval.py
│   │   │   ├── regimes.py
│   │   │   ├── runner.py
│   │   │   └── __init__.py
│   │   └── targets/
│   │       ├── __init__.py
│   │       ├── sql/                   # Text-to-SQL 역할 (그대로 재사용)
│   │       │   ├── action_space.py
│   │       │   ├── eval.py
│   │       │   ├── exec.py
│   │       │   ├── hypothesize.py
│   │       │   ├── outcome.py
│   │       │   ├── prompt_transforms.py
│   │       │   ├── sql_parse.py
│   │       │   ├── target.py
│   │       │   ├── taxonomy.py
│   │       │   ├── agent/
│   │       │   │   ├── agent.py
│   │       │   │   ├── behaviors.py
│   │       │   │   ├── events.py
│   │       │   │   └── __init__.py
│   │       │   └── __init__.py
│   │       └── okf/                   # OKF 역할 (신규, SQL/longmemeval 패턴 참조)
│   │           ├── action_space.py
│   │           ├── eval.py
│   │           ├── outcome.py
│   │           ├── prompt_transforms.py
│   │           ├── target.py
│   │           ├── taxonomy.py
│   │           └── __init__.py
│   │
│   ├── okf/                           # 기존 OKF 컴파일러 (유지)
│   │   ├── validator.py
│   │   ├── ir.py
│   │   ├── history.py
│   │   ├── compiler.py
│   │   └── __init__.py
│   │
│   ├── runtime/                       # runtime glue (계약 로딩 + 공통 서비스)
│   │   ├── loader.py                  # behaviors.yaml → ActiveGraph 런타임에 메타데이터 로드
│   │   ├── reader_registry.py         # 질문/요청별 Reader 인스턴스 관리
│   │   ├── embedder.py                # core.agent.embedders 재노출 또는 wrapper
│   │   ├── history_logger.py          # runtime 연산을 .okf/*/history.yaml에 기록
│   │   ├── guardrails.py              # config/agent_model/guardrails.yaml 적용
│   │   └── __init__.py
│   │
│   ├── agents/                        # 역할별 behavior 래퍼/진입점
│   │   ├── router.py                  # REQUEST_RECEIVED → REQUEST_CLASSIFIED
│   │   ├── super_agent.py             # 최상위 CLI/API
│   │   ├── sql/
│   │   │   ├── behaviors.py           # core.targets.sql.agent.behaviors 재사용/래핑
│   │   │   └── __init__.py
│   │   ├── okf/
│   │   │   ├── behaviors.py           # OKF 전용 behavior 정의
│   │   │   └── __init__.py
│   │   └── mentor/
│   │       ├── behaviors.py           # core.loop.behaviors 재사용/래핑
│   │       └── __init__.py
│   │
│   └── tools/                         # 기존 도구 유지
│       ├── okf_link_check.py
│       └── okf_visualizer.py
│
├── config/
│   ├── agent_model/                   # 선언적 계약 (YAML)
│   │   ├── manifest.yaml
│   │   ├── events.yaml
│   │   ├── objects.yaml
│   │   ├── edges.yaml
│   │   ├── behaviors.yaml             # OKF 컴파일러가 생성 (handler 참조 포함)
│   │   ├── guardrails.yaml
│   │   ├── caching.yaml
│   │   ├── workflows/
│   │   │   ├── sql_workflow.yaml
│   │   │   ├── okf_workflow.yaml
│   │   │   └── mentor_workflow.yaml
│   │   └── history.yaml               # 모델 계약 변경 이력
│   └── targets/                       # 역할별 타겟 설정
│       ├── sql_target.yaml
│       └── okf_target.yaml
│
├── .okf/                              # 기존 OKF 지식베이스 (또는 knowledge/)
│   ├── 00_agent_model/
│   ├── 01_nano_vllm/
│   ├── 02_store_front/
│   ├── index.md
│   └── log.md
│
├── data/      # SQLite DB folder
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
│
└── README.md
```

---

## 4. `behaviors.yaml`과 Python Handler 연결

`behaviors.yaml`에는 이제 **실행 코드가 아닌 handler 참조**가 들어갑니다.

```yaml
behaviors:
  - name: sql_agent.encode_schema
    role: sql
    subscribes_to: sql.question.asked
    emits: sql.schema.encoded
    handler: agents.sql.behaviors::encode_schema   # Python 모듈 참조

  - name: okf_agent.ingest
    role: okf
    subscribes_to: okf.ingest.requested
    emits: okf.ingested
    handler: agents.okf.behaviors::ingest

  - name: mentor.observe
    role: mentor
    subscribes_to: [sql.evaluated, okf.answer.generated]
    emits: mentor.analyze
    handler: agents.mentor.behaviors::observe
```

- `runtime/loader.py`는 이 `handler` 참조를 읽어 ActiveGraph 런타임에 **실제 Python 함수를 바인딩**합니다.
- SQL과 Mentor의 handler는 대부분 `core.targets.sql.agent.behaviors`와 `core.loop.behaviors`에서 가져옵니다.
- OKF handler만 신규로 구현하면 됩니다.

---

Below is the updated **work-breakdown plan** for developing the **Brick‑Agent** (super‑agent) with a coding agent. It assumes the following existing components are reusable:

- `src/core/agent/` – shared runtime, embeddings, transforms
- `src/core/eval/` – evaluation abstraction
- `src/core/loop/` – mentor loop (self‑improvement)
- `src/core/targets/sql/` – complete text‑to‑SQL role
- `src/core/targets/longmemeval/` – pattern reference for OKF role
- `src/okf/` – existing OKF compiler, validator, IR, history
- `src/tools/` – link checker and visualizer

The remaining work is to **add the OKF role, integrate a request router, build runtime glue, and connect everything** into a unified super brick‑agent.

---

## Work Breakdown Plan

### Phase 1: Validate and Understand Existing Core (No Code Changes)

**Goal:** Ensure the imported `regimes` core works in the new repository.

- **Task 1.1:** Run existing tests (if any) for `src/core/agent/`, `src/core/loop/`, `src/core/targets/sql/`.
- **Task 1.2:** Create a minimal test script that:
  - Builds a `SqlTarget` using `src/core/targets/sql/target.py`.
  - Runs a simple SQL question with a `FakeReader` (or dummy Reader).
  - Confirms the event chain `question.asked → … → query.drafted` executes correctly.
- **Task 1.3:** Verify that the OKF compiler still works on the current `.okf/` bundles and produces `behaviors.yaml`.

**Acceptance:** Core SQL pipeline and OKF compiler are functional in the new structure.

---

### Phase 2: Design the OKF Target (New Code)

**Goal:** Create a complete `src/core/targets/okf/` package that mirrors the SQL target structure but for OKF knowledge.

- **Task 2.1:** Define `outcome.py` – a dataclass `OkfOutcome` containing:
  - `question_id`, `question`, `answer`, `context_parts` (concepts, rules, schema used), `applied_transforms`, `lint_errors`, `correct` (optional).
- **Task 2.2:** Define `taxonomy.py` – deterministic detectors for OKF structural integrity, e.g.:
  - `concept_orphan_detector`
  - `rule_schema_reference_detector`
  - `ambiguous_trigger_detector`
  - `cyclic_concept_detector`
  - These produce signals used by lint and evaluation.
- **Task 2.3:** Define `action_space.py` – encapsulates the OKF ask pipeline:
  - `OkfActionSpace` with a `build_prompt_parts(question, relevant_context)` method.
  - Uses `prompt_transforms` pipeline to modify prompt parts.
- **Task 2.4:** Define `prompt_transforms.py` – transforms specific to OKF ask, e.g.:
  - `inject_concept_tree` – add parent/child concepts to context
  - `inject_rules` – add applicable rules
  - `trim_context` – limit context length
- **Task 2.5:** Define `eval.py` – evaluation backend for OKF answers:
  - Compares generated answer against ground truth or checks rule adherence.
  - Returns correctness and signals.
- **Task 2.6:** Define `target.py` – `build_target` constructor that wires `OkfActionSpace`, `OkfTaxonomy`, and `OkfEvalBackend`.

**Acceptance:** A new OKF target exists and can be instantiated with a dummy knowledge base.

---

### Phase 3: Build OKF Runtime Behaviors (New Code)

**Goal:** Implement event-driven behaviors for OKF ingest, lint, and ask using the `@behavior` decorator.

- **Task 3.1:** Create `src/agents/okf/behaviors.py`.
- **Task 3.2:** Implement `okf_agent.ingest` behavior:
  - Triggered by `OKF_INGEST_REQUESTED`.
  - Reads the compiled OKF graph data (from `src/okf/ir.py` or a snapshot file) and populates ActiveGraph objects:
    - `concept` nodes, `rule` nodes, `schema_table`, `schema_column` nodes.
    - Relations: `concept_child`, `maps_to_column`, `uses_rule`, etc.
  - Emits `OKF_INGESTED` with summary counts.
- **Task 3.3:** Implement `okf_agent.lint` behavior:
  - Triggered by `OKF_LINT_REQUESTED`.
  - Runs validators from `src/core/targets/okf/taxonomy.py`.
  - Emits `OKF_LINTED` with issues list and validity flag.
- **Task 3.4:** Implement `okf_agent.assemble_context` behavior:
  - Triggered by `OKF_ASK_REQUESTED`.
  - Uses embedder to retrieve top‑K relevant concepts/rules.
  - Expands concept tree (parents, children).
  - Retrieves mapped schema objects.
  - Emits `OKF_CONTEXT_ASSEMBLED` with structured `context_parts`.
- **Task 3.5:** Implement `okf_agent.generate_answer` behavior:
  - Triggered by `OKF_CONTEXT_ASSEMBLED`.
  - Calls `OkfActionSpace.build_prompt_parts` then `prompt_transforms.apply_pipeline`.
  - Renders final prompt and calls `Reader.answer()`.
  - Emits `OKF_ANSWER_GENERATED` with answer and applied transforms.

**Acceptance:** A test can trigger `OKF_ASK_REQUESTED` and receive a grounded response from a fake reader.

---

### Phase 4: Build Runtime Services and Glue

**Goal:** Create support modules that connect OKF compiler output with ActiveGraph and manage shared resources.

- **Task 4.1:** Create `src/runtime/` package.
- **Task 4.2:** Implement `reader_registry.py`:
  - Process‑level dictionary keyed by `question_id`/`request_id`.
  - Functions: `set_reader`, `get_reader`, `clear_reader`.
- **Task 4.3:** Implement `embedder.py`:
  - Thin wrapper around `core.agent.embedders`.
  - Expose `embed(texts)` returning L2‑normalised vectors.
- **Task 4.4:** Implement `history_logger.py`:
  - Appends runtime operations (e.g., `MENTOR_APPLY`, `OPTIMIZE_PROMPT`) to the appropriate `.okf/*/history.yaml`.
- **Task 4.5:** Implement `guardrails.py`:
  - Loads rules from `config/agent_model/guardrails.yaml`.
  - Provides functions to check SQL safety, OKF lint requirements, mentor thresholds.
- **Task 4.6:** Implement `loader.py`:
  - Ingests the compiled OKF graph (from `src/okf/ir.py` or a pre‑compiled JSON).
  - Populates ActiveGraph with nodes/relations.
  - Provides a function to build the initial graph for a session.

**Acceptance:** All runtime services have unit tests.

---

### Phase 5: Request Router and Super‑Agent Entry Point

**Goal:** Provide a single interface that classifies requests and dispatches to the correct role.

- **Task 5.1:** Create `src/agents/router.py`:
  - Subscribes to `REQUEST_RECEIVED`.
  - Uses embeddings or a lightweight LLM call to classify the request into:
    - `sql` – ask a database query.
    - `okf_ingest` – ingest new knowledge.
    - `okf_lint` – lint knowledge base.
    - `okf_ask` – ask about OKF knowledge.
    - `mentor_status` – show learning status.
  - Emits `REQUEST_CLASSIFIED` with the role and original payload.
- **Task 5.2:** Create `src/agents/super_agent.py`:
  - Main CLI entry point.
  - Initialises ActiveGraph runtime with all behaviors (SQL, OKF, Mentor, Router).
  - Registers a Reader instance for each request.
  - Emits `REQUEST_RECEIVED` to start the chain.
  - Provides subcommands: `ask`, `ingest`, `lint`, `mentor`.
- **Task 5.3:** Ensure that the router can also trigger cross‑role flows (e.g., an OKF ask that requires SQL data).

**Acceptance:** Running `python -m src.agents.super_agent ask sql "..."` returns a SQL query; `... okf ask "..."` returns an OKF answer; `... mentor status` shows learning status.

---

### Phase 6: Mentor Integration (Adapt Existing Loop)

**Goal:** Ensure the existing self‑improvement loop can handle both SQL and OKF outcomes.

- **Task 6.1:** Review `src/core/loop/` to understand where regime classification and hypothesis generation happen.
- **Task 6.2:** Modify `loop/regimes.py` or `loop/hypothesize.py` (if necessary) to include OKF failure regimes, e.g.:
  - `wrong_concept_selected`
  - `missing_rule`
  - `context_too_broad`
- **Task 6.3:** Ensure the mentor observes both `SQL_EVALUATED` and `OKF_ANSWER_GENERATED` events.
- **Task 6.4:** Test that the mentor can propose a prompt transform after a series of SQL failures and that the transform is applied to subsequent SQL prompts.

**Acceptance:** A simulated failure run results in a mentor‑proposed improvement that is validated and applied, and the next run uses the improved prompt.

---

### Phase 7: Integration, Testing, and Documentation

**Goal:** End‑to‑end validation and packaging.

- **Task 7.1:** Write integration tests that cover:
  - SQL query generation.
  - OKF ingest → lint → ask.
  - Mentor learning cycle on a small dataset.
- **Task 7.2:** Update existing OKF visualizer/link checker to work with the extended graph (if needed).
- **Task 7.3:** Create `README.md` with architecture, setup instructions, and usage examples.
- **Task 7.4:** Add a `Makefile` or `pyproject.toml` scripts for common commands.

**Acceptance:** All tests pass; a new user can follow the README to run the super‑agent.