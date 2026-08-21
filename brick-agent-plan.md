## Current State Assessment

We have built a robust **OKF Compiler** that:

- Validates YAML frontmatter and links across `.okf/` bundles.
- Builds an in-memory **IR graph** from knowledge nodes.
- Applies **history operators** (SPLIT, MERGE, etc.) to evolve the graph.
- Emits **behaviors.yaml** – a static, Cypher‑based executable control spec for ActiveGraph.

To build the full **brick.agent** (사내 문서 → OKF wiki KB → Q&A + 자연어 기반 DB 쿼리/응답 + incremental‑learning mentor), we need to generalise the architecture **and secure LLM observability via event sourcing** so every agent decision is visible and replayable.

---

## Revised brick.agent Architecture

We keep the existing OKF Compiler as the **knowledge ingestion and management layer**, extended to handle **multiple knowledge domains** (DB schema, OKF concepts/rules). On top of that we place an **ActiveGraph Runtime** that executes event‑driven behaviours for each role. A **Mentor** observes outcomes and feeds improvements back via the knowledge graph and the prompt-transform pipeline. **Every LLM round-trip is recorded as `llm.requested` / `llm.responded` events** so the append‑only event log doubles as the observability store.

---

## 1. 수정의 핵심 원칙

agent_model로부터 “코드 생성”에서 “설정 로딩 + 기존 코드 조립”으로 전환한다.
이렇게 하면 유지보수 부담이 줄고, 이미 검증된 Self-Improving loop를 그대로 활용할 수 있다.
다만, agent_model은 전체 brick.agent의 핵심 사항을 명세하고 있어야 한다.

| 기존 계획 | 수정된 계획 |
|-----------|--------------|
| `agent_model` YAML → runtime Python code 생성 | `regimes` core를 runtime으로 재사용 |
| YAML이 실행 로직을 포함 | YAML은 **구조·이벤트·행동 계약**과 **감사 이력**만 기술 |
| Compiler가 전체 behavior 코드 생성 | Compiler는 `behaviors.yaml` **계약 명세**와 **handler 참조**만 생성 |
| 역할별 코드를 새로 작성 | SQL·Mentor는 `regimes`에서 재사용, OKF만 패턴 따라 신규 작성 |

- **Canonical 계약**: 통합 이벤트 어휘와 행동 스펙은 `.okf/00_agent_model/events.yaml` + `behaviors.spec.yaml`이 단일 원천(single source of truth)이며, `config/agent_model/*.yaml`은 guardrails/workflows 등 파생 설정만 담는다. (`events.yaml`은 레거시 이벤트 이름을 `aliases`로 매핑한다.)
- The YAML files serve as **declarative contracts** for events, objects, and guardrails, **not** as a source of Python code generation.
- The actual behaviours are defined in Python (reusing `src/core/agent`, `src/core/loop`, `src/core/targets/sql`, `src/core/targets/okf`) and registered directly with ActiveGraph.
- **Observability 원칙**: 모든 LLM 호출은 `llm.requested`/`llm.responded`로 로그에 남고(model, prompt_hash, cost_usd, cache_hit, latency_seconds), 실패는 payload 필드로 기록된다(예외가 아님). 상세 계약은 `.okf/00_agent_model/concepts/observability.md`.
- The OKF compiler continues to manage knowledge integrity and produce graph data; the runtime loader consumes that data into ActiveGraph.

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    brick.agent Entry Point                 │
│            (CLI / API – request routing & dispatch)        │
├─────────────────────────────────────────────────────────────┤
│              ActiveGraph Runtime (event bus)               │
│   – SQL behaviours   – OKF behaviours   – Mentor behaviours│
│   – Router behaviour (실제 핸들러는 src/core에서 로드)     │
├─────────────────────────────────────────────────────────────┤
│   Shared Services: Reader Registry, Embedder, Transform    │
│   Pipeline, Evaluator, History Logger, Guardrails          │
│   (src/core/agent / eval / loop 재사용)                    │
├─────────────────────────────────────────────────────────────┤
│   Observability Layer (event sourcing)                     │
│   – llm.requested/llm.responded (model·cost·latency·cache) │
│   – EventStore (SQLite) + Runtime.load/fork 재생           │
│   – trace.causal_chain · structured logging · OTel/Prom    │
├─────────────────────────────────────────────────────────────┤
│  OKF Compiler + IR (기존 유지) → behaviors.yaml (계약)     │
│  – .okf 지식베이스 검증/변환                               │
│  – 컴파일된 그래프 스펙 + handler 참조 생성                │
│  – runtime 이력은 .okf/*/history.yaml에 기록               │
└─────────────────────────────────────────────────────────────┘
```

- **SQL role** → `src/core/targets/sql` 전체 재사용 (+ `draft_query`에 `llm.*` 이벤트 emit 추가)
- **Mentor role** → `src/core/loop` 전체 재사용
- **OKF role** → `src/core/targets/okf` (Phase 2 완료) 위에 behaviors 신규 작성
- **공통 서비스** → `src/core/agent` 전체 재사용
- **Canonical 계약** → `.okf/00_agent_model/events.yaml`, `behaviors.spec.yaml`, `concepts/observability.md`

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
│   │       └── okf/                   # OKF 역할 (Phase 2 완료)
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
│   ├── runtime/                       # runtime glue (계약 로딩 + 공통 서비스 + 관측)
│   │   ├── loader.py                  # behaviors.spec.yaml → ActiveGraph 런타임에 handler 바인딩
│   │   ├── reader_registry.py         # 질문/요청별 Reader 인스턴스 관리
│   │   ├── embedder.py                # core.agent.embedders 재노출 또는 wrapper
│   │   ├── history_logger.py          # runtime 연산을 .okf/*/history.yaml에 기록 (canonical op 이름)
│   │   ├── guardrails.py              # config/agent_model/guardrails.yaml 적용
│   │   ├── event_store.py             # SQLiteEventStore + Runtime.load/fork 재생 (Phase 5)
│   │   ├── observability.py           # llm.* emit 헬퍼 + 구조화 로깅/OTel (Phase 5)
│   │   ├── replay.py                  # llm 응답 캐시 기반 결정적 재실험 (Phase 5)
│   │   └── __init__.py
│   │
│   ├── agents/                        # 역할별 behavior 래퍼/진입점
│   │   ├── router.py                  # request.received → request.classified
│   │   ├── brick_agent.py            # 최상위 CLI/API (--store/--trace 옵션)
│   │   ├── sql/
│   │   │   ├── behaviors.py           # core.targets.sql.agent.behaviors 재사용/래핑
│   │   │   └── __init__.py
│   │   ├── okf/
│   │   │   ├── behaviors.py           # OKF 전용 behavior 정의 (Phase 3)
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
│   ├── agent_model/                   # 파생 설정 (canonical은 .okf/00_agent_model/)
│   │   ├── manifest.yaml
│   │   ├── guardrails.yaml            # required_lint_before_ask 포함
│   │   ├── caching.yaml
│   │   ├── workflows/
│   │   │   ├── sql_workflow.yaml
│   │   │   ├── okf_workflow.yaml
│   │   │   └── mentor_workflow.yaml
│   │   └── history.yaml               # 모델 계약 변경 이력 (canonical과 동기화)
│   └── targets/                       # 역할별 타겟 설정
│       ├── sql_target.yaml
│       └── okf_target.yaml
│
├── .okf/                              # 기존 OKF 지식베이스 (또는 knowledge/)
│   ├── 00_agent_model/                # canonical 계약: events.yaml, behaviors.spec.yaml,
│   │                                  #   concepts/observability.md, history.yaml
│   ├── 01_nano_vllm/
│   ├── 02_store_front/
│   ├── index.md
│   └── log.md
│
├── data/      # SQLite DB folder (EventStore run.db 포함)
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
│
└── README.md
```

---

## 4. `behaviors.spec.yaml`과 Python Handler 연결

**Canonical 행동 스펙은 `.okf/00_agent_model/behaviors.spec.yaml`** 이며, 여기에는 **실행 코드가 아닌 handler 참조**가 들어갑니다.
(컴파일러가 생성하는 `.okf/00_agent_model/behaviors.yaml`은 지식 그래프 노드용 Cypher 패턴으로, 런타임 행동 스펙과 구분됩니다.)

```yaml
behaviors:
  - name: sql_agent.encode_schema
    role: sql
    subscribes_to: [question.asked]
    emits: schema.encoded
    handler: src.core.targets.sql.agent.behaviors::behavior_encode_schema

  - name: okf_agent.ingest
    role: okf
    subscribes_to: [okf.ingest.requested]
    emits: [okf.parsed, okf.validated, okf.loaded]
    handler: src.agents.okf.behaviors::ingest

  - name: okf_agent.generate_answer
    role: okf
    subscribes_to: [okf.context.assembled]
    emits: okf.answer.generated
    handler: src.agents.okf.behaviors::generate_answer
    observability:
      emits_llm_events: required   # reader.answer() → llm.requested/llm.responded
```

- `runtime/loader.py`는 이 `handler` 참조를 읽어 ActiveGraph 런타임에 **실제 Python 함수를 바인딩**합니다.
- SQL과 Mentor의 handler는 대부분 `src.core.targets.sql.agent.behaviors`와 `src.core.loop.behaviors`에서 가져옵니다.
- OKF handler만 신규로 구현하면 됩니다 (Phase 3).

---

Below is the updated **work-breakdown plan** for completing the **brick.agent** with a coding agent. **Phases 1–2 are ✅ COMPLETED** (SQL pipeline + OKF target + 20 tests passing). It assumes the following existing components are reusable:

- `src/core/agent/` – shared runtime, embeddings, transforms
- `src/core/eval/` – evaluation abstraction
- `src/core/loop/` – mentor loop (self‑improvement)
- `src/core/targets/sql/` – complete text‑to‑SQL role
- `src/core/targets/okf/` – **completed OKF role** (Phase 2)
- `src/okf/` – existing OKF compiler, validator, IR, history
- `.okf/00_agent_model/` – canonical contracts: `events.yaml`, `behaviors.spec.yaml`, `concepts/observability.md`
- `src/tools/` – link checker and visualizer

The remaining work is to **build OKF runtime behaviors, runtime glue, the observability layer, the request router, mentor integration, and end‑to‑end packaging** of the brick.agent.

---

## Work Breakdown Plan

### Phase 1: Validate and Understand Existing Core — ✅ COMPLETED

**Goal:** Ensure the imported `regimes` core works in the new repository.

- **Task 1.1:** Run existing tests for `src/core/agent/`, `src/core/loop/`, `src/core/targets/sql/`. — ✅ done (`tests/test_core_imports_and_sql_smoke.py` passing)
- **Task 1.2:** Create a minimal test script that builds a `SqlTarget` and confirms the event chain `question.asked → … → query.drafted`. — ✅ done
- **Task 1.3:** Verify that the OKF compiler still works on the current `.okf/` bundles and produces `behaviors.yaml`. — ✅ done

**Acceptance:** Core SQL pipeline and OKF compiler are functional in the new structure. — ✅ MET

---

### Phase 2: Design the OKF Target (New Code) — ✅ COMPLETED

**Goal:** Create a complete `src/core/targets/okf/` package that mirrors the SQL target structure but for OKF knowledge.

- **Task 2.1:** `outcome.py` – `OkfOutcome` (`question_id`, `question`, `answer`, `context_parts`, `applied_transforms`, `lint_errors`, `correct` optional). — ✅ done
- **Task 2.2:** `taxonomy.py` – deterministic detectors: `concept_orphan_detector`, `rule_schema_reference_detector`, `ambiguous_trigger_detector`, `cyclic_concept_detector` + `OkfTaxonomy`. — ✅ done
- **Task 2.3:** `action_space.py` – `OkfActionSpace.build_prompt_parts(question, relevant_context)` + prompt-transforms pipeline. — ✅ done
- **Task 2.4:** `prompt_transforms.py` – `inject_concept_tree`, `inject_rules`, `trim_context`. — ✅ done
- **Task 2.5:** `eval.py` – evaluation backend (gold-answer match or rule adherence) returning correctness + signals. — ✅ done
- **Task 2.6:** `target.py` – `build_target` wiring `OkfActionSpace`, `OkfTaxonomy`, `OkfEvalBackend`. — ✅ done

**Acceptance:** A new OKF target exists and can be instantiated with a '01_nano_vllm' knowledge base. — ✅ MET (`build_target(kb="01_nano_vllm")`; 20/20 tests passing under `.wenv`)

---

### Phase 3: Build OKF Runtime Behaviors (New Code)

**Goal:** Implement event-driven behaviors for OKF ingest, lint, and ask using the `@behavior` decorator, with event names from the canonical `events.yaml`.

- **Task 3.1:** Create `src/agents/okf/behaviors.py` (event constants imported from canonical `events.yaml`).
- **Task 3.2:** Implement `okf_agent.ingest` behavior:
  - Triggered by `okf.ingest.requested`.
  - Reuses `src/core/targets/okf/taxonomy.load_knowledge_graph()` (Phase 2 — body relationship parser 포함) to build the graph snapshot, then populates ActiveGraph objects:
    - `concept` nodes, `rule` nodes, `schema_table`, `schema_column` nodes.
    - Relations: `concept_child`, `maps_to_column`, `uses_rule`, `rule_references_schema`, etc.
  - Emits `okf.parsed` → `okf.validated` → `okf.loaded` with summary counts.
- **Task 3.3:** Implement `okf_agent.lint` behavior:
  - Triggered by `okf.lint.requested`.
  - Runs `lint_knowledge_graph()` from `src/core/targets/okf/taxonomy.py` (4 detectors).
  - Emits `okf.analyzed` → `okf.linted` with issues list and validity flag.
- **Task 3.4:** Implement `okf_agent.assemble_context` behavior:
  - Triggered by `okf.ask.requested`.
  - Uses embedder to retrieve top‑K relevant concepts/rules (embedder 호출은 `tool.requested/responded` emit 허용).
  - Expands concept tree (parents, children).
  - Retrieves mapped schema objects.
  - Emits `okf.context.assembled` with structured `context_parts` (concepts/rules/schema/relations).
- **Task 3.5:** Implement `okf_agent.generate_answer` behavior:
  - Triggered by `okf.context.assembled`.
  - Calls `OkfActionSpace.build_prompt_parts(question, context_parts)` — **파이프라인은 내부에서 실행됨** (`OkfPrompt.parts` + `applied_transforms` 반환) → `render_okf_prompt()`으로 최종 프롬프트 생성.
  - **LLM seam (observability)**: `reader.answer()` 호출을 `llm.requested` / `llm.responded` 이벤트로 감싼다 (model, prompt_hash, cost_usd, cache_hit, latency_seconds — 계약: `concepts/observability.md` §3.2).
  - Emits `okf.answer.generated` with answer, context_parts, applied transforms.

**Acceptance:** A test can trigger `okf.ask.requested` and receive a grounded response from a fake reader; the run log contains `llm.requested`/`llm.responded` for the answer call.

---

### Phase 4: Build Runtime Services and Glue — ✅ COMPLETED

**Goal:** Create support modules that connect OKF compiler output with ActiveGraph and manage shared resources.

- **Task 4.1:** Create `src/runtime/` package. — ✅ done
- **Task 4.2:** Implement `reader_registry.py`:
  - Process‑level dictionary keyed by `question_id`/`request_id`.
  - Functions: `set_reader`, `get_reader`, `clear_reader`, `clear_all_readers`, `call_reader` (timing + error capture).
  - Reader 호출 래퍼는 Phase 5의 `llm.*` emit 헬퍼와 결합.
  - SQL·OKF behaviors가 공유 레지스트리를 사용하도록 전환 완료.
- **Task 4.3:** Implement `embedder.py`:
  - Thin wrapper around `core.agent.embedders`.
  - Expose `embed(texts)` / `embed_one` / `cosine_similarity` / `rank_by_similarity` (L2‑normalised).
- **Task 4.4:** Implement `history_logger.py`:
  - Appends runtime operations (`ADD_BEHAVIOR`, `REWIRE_BEHAVIOR`, `OPTIMIZE_PROMPT`, `ADD_EVENT_TYPE`, `ADD_DOC`, `ADD_GUARDRAIL`, …) to the appropriate `.okf/*/history.yaml` (canonical operator names; non-canonical ops rejected).
- **Task 4.5:** Implement `guardrails.py`:
  - Loads rules from `config/agent_model/guardrails.yaml` (생성 완료) — defaults merge over built-ins.
  - Includes **`okf.required_lint_before_ask: true`**; provides `check_sql_safety`, `check_okf_lint_before_ask`, `check_mentor_promotion`.
- **Task 4.6:** Implement `loader.py`:
  - `load_kb_graph` (Phase 2 `load_knowledge_graph` 재사용), `populate_graph` (ActiveGraph objects/relations), `build_session` (세션 초기 그래프).

**Acceptance:** All runtime services have unit tests. — ✅ MET (`tests/test_runtime_services.py`, 15 tests)

---

### Phase 5: Observability & Event-Sourcing Persistence — ✅ COMPLETED

**Goal:** Make the event log the LLM observability store: durable, replayable, and cost/latency-visible.

- **Task 5.1:** `src/runtime/event_store.py` — SQLiteEventStore(`persist_to`) + run metadata (`list_runs`, `run_events`), store close hygiene. — ✅ done
- **Task 5.2:** `src/runtime/observability.py` — `llm.requested`/`llm.responded` emit 헬퍼 (`ask_with_observability`), `configure_logging`, `causal_chain_text`. — ✅ done
- **Task 5.3:** `src/runtime/replay.py` — `build_replay_cache`(prompt_hash→answer), `ReplayReader`(결정적 재생), `replay_into_graph`, `Runtime.load` 래퍼. — ✅ done
- **Task 5.4:** SQL `draft_query` — `reader.answer()`를 `llm.*` observability seam으로 래핑. — ✅ done
- SQL/OKF agent entrypoints에 `store_path`(persist_to) 배선 + run별 unique run_id. — ✅ done

**Acceptance:** 모든 LLM 호출이 로그에 남고(model·prompt_hash·answer·latency), 저장된 run을 재생할 수 있으며, replay cache가 기록된 답변을 결정적으로 서빙한다. — ✅ MET (`tests/test_phase5_observability.py`)

---

### Phase 6: Request Router and brick.agent Entry Point — ✅ COMPLETED

**Goal:** Provide a single interface that classifies requests and dispatches to the correct role.

- **Task 6.1:** `src/agents/router.py` — `classify()` (결정적 키워드 + 임베딩 폴백). — ✅ done
- **Task 6.2:** `src/agents/brick_agent.py` CLI — subcommands `ask sql` / `ask okf` / `ask`(router) / `ingest` / `lint` / `db seed` / `mentor status`, `--store`/`--trace`/`--real`, Reader demo/real 자동 선택. — ✅ done
- **Task 6.3:** Cross-role flow — OKF ask에서 SQL splice는 추후(Phase 8); 개별 역할 체인은 CLI로 독립 실행 가능. — ✅ (개별 Q&A)

**Acceptance:** `python -m src.agents.brick_agent ask sql "..."` returns a SQL query; `... okf ask "..."` returns an OKF answer; `... mentor status` shows learning status; `--trace` renders the LLM observability trail. — ✅ MET (`tests/test_brick_cli.py` + live CLI)

---

### Phase 7: Mentor Integration (Adapt Existing Loop) — ✅ COMPLETED

**Goal:** Ensure the existing self‑improvement loop can handle both SQL and OKF outcomes — and observe LLM cost/latency.

- **Task 7.1:** Review `src/core/loop/` to understand where regime classification and hypothesis generation happen. — ✅ done (loop은 target-agnostic; OKF/SQL target이 모든 seam 충족)
- **Task 7.2:** Map mentor diagnosis onto the **implemented `OkfTaxonomy`** regimes (`concept-cycle`, `rule-schema-mismatch`, `concept-orphan`, `ambiguous-trigger`, `unclassified`) — 병렬 택소노미 없이 기존 사용. 신규 실패 클러스터는 `OkfTaxonomy.register_regime`으로 추가 (예: `wrong-concept-selected`). — ✅ done + 테스트
- **Task 7.3:** Mentor observes `sql.evaluated` / `okf.answer.generated` (canonical) **plus `llm.responded`** — `observability.summarize_llm_usage(store)`로 LLM 호출 수·평균 지연·에러 집계, `mentor status`에 출력. — ✅ done
- **Task 7.4:** SQL 실패 시리즈 → mentor가 prompt transform 제안 → 검증 → 적용 → 이후 프롬프트가 개선본 사용. — ✅ done (테스트로 입증)

**Acceptance:** A simulated failure run results in a mentor‑proposed improvement that is validated (deterministic replay via Phase 5 — `ReplayReader`가 기록된 답변을 재호출 없이 서빙) and applied, and the next run uses the improved prompt. — ✅ MET (`tests/test_mentor_loop.py`)

---

### Phase 8: Integration, Testing, and Documentation — ✅ COMPLETED

**Goal:** End‑to‑end validation, KB integrity, and packaging.

- **Task 8.1:** Integration tests — SQL query generation, OKF ingest → lint → ask (lint-before-ask guardrail), observability (llm.* events + stored-run replay identical). — ✅ done (`tests/test_phase8_integration.py`)
- **Task 8.2:** Fix the concept cycles / dangling refs / one-sided relationships / noise / missing evidence in `.okf/01_nano_vllm` and re-run lint until clean. — ✅ done (`src/tools/okf_repair.py`, 멱등): 106 → **0 issues** (`valid=True n_errors=0`, `okf_validate.py` 0 errors), `log.md`에 기록
- **Task 8.3:** Update visualizer/link checker for the extended graph. — ✅ (body 관계 블록은 `dangling_reference_detector`가 담당; `okf_link_check`는 frontmatter 기준으로 동작 확인)
- **Task 8.4:** `README.md` — 실제 구현 기준 전면 개편 (Phase 1–8 상태, CLI, 관측, KB 감사 현황). — ✅ done
- **Task 8.5:** `Makefile` — test / seed / ask-sql / ask-okf / ingest / lint / mentor-status / repair / validate. — ✅ done

**Acceptance:** All tests pass; `okf lint` is clean; a new user can follow the README to run the brick.agent and inspect any answer's causal chain and LLM cost. — ✅ MET (**69/69 tests**, lint 0건, README/user-manual/Makefile 완비)
