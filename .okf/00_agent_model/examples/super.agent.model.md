Below is a **step-by-step systematic approach** to develop a **Super Agent** that combines three roles:

1. **Text-to-SQL** – natural language question → SQL query  
2. **OKF Ingest / Lint / Ask** – knowledge-base management and question answering  
3. **Incremental-Learning Mentor** – observes outcomes and continuously improves the other two

All roles are built on **ActiveGraph**, using its event-driven behaviors, typed graph objects, and relation model.

---

## Step 1 — Define the Unified Graph Schema

**Goal:** Create a single graph that holds all knowledge needed by the three roles.

### 1.1 Node Types

| Type               | Purpose                                                         | Key Fields |
|--------------------|-----------------------------------------------------------------|------------|
| `schema_table`     | Database table in the sales DB                                  | `name`, `description` |
| `schema_column`    | Column of a table                                               | `table`, `name`, `type`, `is_pk`, `description` |
| `concept`          | OKF concept (tree node)                                         | `id`, `name`, `description`, `parent`, `aliases` |
| `rule`             | Business rule or response template                              | `id`, `name`, `trigger`, `response_template`, `conditions`, `priority` |
| `question`         | A user request (SQL or OKF ask)                                 | `question_id`, `text`, `type` (`sql` / `okf_ask` / `meta`) |
| `sql_output`       | A generated SQL query                                           | `question_id`, `predicted_sql`, `gold_sql`, `correct` |
| `okf_answer`       | A generated OKF response                                        | `request_id`, `answer`, `context_parts` |
| `evaluation`       | Result of evaluation or user feedback                           | `outcome_id`, `correct`, `feedback`, `signals` |
| `feedback`         | Explicit or implicit feedback from users / evaluators           | `feedback_id`, `content`, `timestamp` |
| `prompt_transform` | A reusable prompt transformation (used by mentor)               | `name`, `code_ref`, `status` (`proposed` / `active` / `retired`) |
| `memory`           | Episodic or semantic memory for incremental learning            | `key`, `value`, `confidence` |

### 1.2 Relation Types

| Relation Type        | Source → Target          | Meaning |
|----------------------|--------------------------|---------|
| `has_column`         | `schema_table` → `schema_column` | Table contains column |
| `concept_child`      | `concept` → `concept`    | Tree hierarchy |
| `maps_to_column`     | `concept` → `schema_column` | Concept maps to DB column |
| `maps_to_table`      | `concept` → `schema_table`  | Concept maps to DB table |
| `uses_rule`          | `concept` → `rule`       | Concept uses a rule |
| `rule_references_schema` | `rule` → `schema_column` / `schema_table` | Rule references schema |
| `question_about`     | `question` → `concept` / `schema_table` | What the question is about |
| `produced`           | `question` → `sql_output` / `okf_answer` | Question leads to output |
| `evaluated_by`       | `sql_output` / `okf_answer` → `evaluation` | Output has evaluation |
| `generates_feedback` | `evaluation` → `feedback` | Evaluation creates feedback |
| `improves`           | `prompt_transform` → `sql_output` / `okf_answer` | Transform improves output |
| `learned_from`       | `memory` → `evaluation` / `feedback` | Memory derived from feedback |

**Why one unified schema?**  
A single graph allows cross-role reasoning: the SQL agent can use OKF concepts to enrich its schema understanding; the mentor can inspect all outcomes and feedback to propose improvements.

---

## Step 2 — Define the Event Vocabulary

Create a global event vocabulary covering all roles.

### 2.1 Core Lifecycle Events

| Event Constant           | Value                         | Description |
|--------------------------|-------------------------------|-------------|
| `REQUEST_RECEIVED`       | `"request.received"`          | User sends any request |
| `REQUEST_CLASSIFIED`     | `"request.classified"`        | Router decides role |
| `FINAL_RESPONSE_READY`   | `"response.ready"`            | Final output ready |

### 2.2 Text-to-SQL Events

| Event Constant       | Value                | Description |
|----------------------|----------------------|-------------|
| `QUESTION_ASKED`     | `"sql.question.asked"` | SQL question received |
| `SCHEMA_ENCODED`     | `"sql.schema.encoded"` | Schema objects in graph |
| `COLUMNS_SCORED`     | `"sql.columns.scored"` | Relevant columns selected |
| `PROMPT_ASSEMBLED`   | `"sql.prompt.assembled"` | Final LLM prompt built |
| `QUERY_DRAFTED`      | `"sql.query.drafted"`  | SQL generated |
| `SQL_EVALUATED`      | `"sql.evaluated"`      | SQL evaluated against gold |

### 2.3 OKF Events

| Event Constant         | Value                    | Description |
|------------------------|--------------------------|-------------|
| `OKF_INGEST_REQUESTED` | `"okf.ingest.requested"` | Ingest OKF data |
| `OKF_INGESTED`         | `"okf.ingested"`         | KB loaded |
| `OKF_LINT_REQUESTED`   | `"okf.lint.requested"`   | Lint KB |
| `OKF_LINTED`           | `"okf.linted"`           | Lint results |
| `OKF_ASK_REQUESTED`    | `"okf.ask.requested"`    | Ask KB question |
| `OKF_CONTEXT_ASSEMBLED`| `"okf.context.assembled"`| Relevant concepts/rules retrieved |
| `OKF_ANSWER_GENERATED` | `"okf.answer.generated"` | Answer produced |

### 2.4 Mentor Events

| Event Constant          | Value                     | Description |
|-------------------------|---------------------------|-------------|
| `MENTOR_OBSERVE`        | `"mentor.observe"`        | Mentor observes outcome |
| `MENTOR_ANALYZE`        | `"mentor.analyze"`        | Mentor analyses patterns |
| `MENTOR_PROPOSE`        | `"mentor.propose"`        | Mentor proposes an improvement |
| `MENTOR_VALIDATE`       | `"mentor.validate"`       | Improvement validated |
| `MENTOR_APPLY`          | `"mentor.apply"`          | Improvement promoted to active |

---

## Step 3 — Build the Shared Core Services

These services are used by all three roles.

### 3.1 Reader / LLM Service
- A process‑level registry keyed by `question_id` or `request_id` (as in the SQL agent).
- Supports multiple LLM backends (Fake, Anthropic, local).
- Interface:
  ```python
  class Reader:
      def answer(context, question, request_id) -> str
  ```

### 3.2 Embedder Service
- Shared embedding model (e.g., HashEmbedder with L2-normalized vectors).
- Used for column scoring, concept retrieval, and request classification.

### 3.3 Prompt Transform Pipeline
- Generic pipeline that can modify any `prompt_parts` dict.
- Used by SQL prompt assembly, OKF ask prompt, and mentor‑proposed transformations.
- Already implemented as `prompt_transforms`.

### 3.4 Evaluation / Taxonomy
- Deterministic detectors for SQL structure (`has_join`, `has_where`, etc.).
- OKF lint validators as a separate set of deterministic checks.
- Outcome summary projection for auditing.

### 3.5 Rule Engine
- For OKF rules, executes trigger conditions and returns response templates.
- Mentor can also use rule engine to propose new rules based on feedback.

---

## Step 4 — Implement Role Modules as Behavior Groups

Each role is a collection of ActiveGraph behaviors. They are registered at import time and share the same graph.

### 4.1 Text-to-SQL Module

Reuse the four SQL behaviors from the earlier model:

| Behavior Name                | Trigger Event       | Emit Event            |
|------------------------------|---------------------|-----------------------|
| `sql_agent.encode_schema`    | `SQL.QUESTION_ASKED`| `SQL.SCHEMA_ENCODED`  |
| `sql_agent.retrieve_relevant_columns` | `SQL.SCHEMA_ENCODED` | `SQL.COLUMNS_SCORED` |
| `sql_agent.prompt_pipeline`  | `SQL.COLUMNS_SCORED`| `SQL.PROMPT_ASSEMBLED`|
| `sql_agent.draft_query`      | `SQL.PROMPT_ASSEMBLED` | `SQL.QUERY_DRAFTED` |

**Enhancement:** Before `encode_schema`, optionally call the OKF module to enrich schema with concept metadata (e.g., column descriptions from concepts).

### 4.2 OKF Module

Implement three behaviors for ingest, lint, ask.

| Behavior Name                | Trigger Event            | Emit Event              |
|------------------------------|--------------------------|-------------------------|
| `okf_agent.ingest`           | `OKF_INGEST_REQUESTED`   | `OKF_INGESTED`          |
| `okf_agent.lint`             | `OKF_LINT_REQUESTED`     | `OKF_LINTED`            |
| `okf_agent.assemble_context` | `OKF_ASK_REQUESTED`      | `OKF_CONTEXT_ASSEMBLED` |
| `okf_agent.generate_answer`  | `OKF_CONTEXT_ASSEMBLED`  | `OKF_ANSWER_GENERATED`  |

These behaviors operate on the same graph, using `concept`, `rule`, `schema_table`, and `schema_column` nodes.

### 4.3 Incremental-Learning Mentor Module

The mentor is a meta‑agent that listens to outcomes and proposes improvements.

| Behavior Name            | Trigger Event        | Emit Event           |
|--------------------------|----------------------|----------------------|
| `mentor.observe`         | `SQL_EVALUATED`, `OKF_ANSWER_GENERATED`, `MENTOR_OBSERVE` | `MENTOR_ANALYZE` |
| `mentor.analyze`         | `MENTOR_ANALYZE`     | `MENTOR_PROPOSE`     |
| `mentor.validate`        | `MENTOR_PROPOSE`     | `MENTOR_VALIDATE`    |
| `mentor.apply`           | `MENTOR_VALIDATE`    | `MENTOR_APPLY`       |

**Mentor workflows:**

- **Observe:** Listen to `SQL_EVALUATED` (contains `correct`, `predicted_sql`, `gold_sql`, structural signals) and `OKF_ANSWER_GENERATED` (contains `answer`, `context_parts`).
- **Analyze:** Group failures by pattern (e.g., missing join, wrong table, wrong concept). Identify recurring schema elements or concepts that cause errors.
- **Propose:** Create a new `prompt_transform` node or a new `rule` node, or update an existing concept mapping.
- **Validate:** Run a hold‑out set of past evaluations to check if the proposed change improves accuracy.
- **Apply:** If validation passes, promote the transform/rule to active; otherwise retire it.

---

## Step 5 — Build the Request Router

A single entry point receives all user requests and routes them to the correct role.

### 5.1 Router Behavior

| Behavior Name        | Trigger Event       | Emit Event             |
|----------------------|---------------------|------------------------|
| `super_agent.router` | `REQUEST_RECEIVED`  | `REQUEST_CLASSIFIED`   |

**Routing logic:**

1. **Embed** the user request text.
2. **Classify** using a simple LLM call or embedding similarity against predefined prototypes:
   - `"SQL query"` → text‑to‑sql module
   - `"ingest knowledge"`, `"lint knowledge"`, `"ask about sales concepts"` → OKF module
   - `"improve the agent"`, `"show learning insights"` → mentor module
3. Emit `REQUEST_CLASSIFIED` with payload `{"role": "sql" | "okf" | "mentor", ...}`.
4. The corresponding module’s entry behavior listens to `REQUEST_CLASSIFIED` and starts its own event chain.

### 5.2 Dispatcher

After classification, the router can also **enrich** the payload with relevant graph context. For example, if role = SQL, it can attach the `question_id` and `reader` to the event payload. This avoids the need for each module to know about global registries.

---

## Step 6 — Define Cross‑Role Interaction Flows

The super agent’s power comes from **cross‑role synergy**. Define explicit event sequences that involve multiple modules.

### 6.1 SQL + OKF Flow

When a SQL question is asked, the OKF knowledge can improve column scoring and prompt assembly.

```
REQUEST_RECEIVED
  → REQUEST_CLASSIFIED (role=sql)
  → SQL.QUESTION_ASKED
  → OKF_CONTEXT_ASSEMBLED (retrieve concepts mapped to the schema)
  → SQL.SCHEMA_ENCODED (enriched with concept descriptions)
  → SQL.COLUMNS_SCORED (scoring uses concept embeddings)
  → SQL.PROMPT_ASSEMBLED
  → SQL.QUERY_DRAFTED
  → SQL_EVALUATED
  → MENTOR_OBSERVE
```

### 6.2 OKF Ask + SQL Flow

When an OKF ask request needs a database value, the SQL agent can generate the query and insert the result into the answer.

```
OKF_ASK_REQUESTED
  → OKF_CONTEXT_ASSEMBLED
  → SQL.QUESTION_ASKED (generate SQL to retrieve the data)
  → ... SQL chain ...
  → SQL.QUERY_DRAFTED
  → (execute SQL)
  → OKF_ANSWER_GENERATED (includes query result in response)
```

### 6.3 Mentor + Both

The mentor observes both SQL and OKF outcomes and proposes improvements to prompts, rules, or concept mappings.

```
SQL_EVALUATED / OKF_ANSWER_GENERATED
  → MENTOR_OBSERVE
  → MENTOR_ANALYZE
  → MENTOR_PROPOSE
  → MENTOR_VALIDATE
  → MENTOR_APPLY
```

The mentor may create a new `prompt_transform` that is automatically picked up by the next SQL prompt assembly.

---

## Step 7 — Implement Incremental Learning Loop

The mentor’s loop is the heart of continuous improvement.

### 7.1 Collect Feedback
- Evaluations from SQL agent (correct/incorrect, structural signals).
- User feedback on OKF answers (explicit thumbs up/down or implicit via follow‑up).
- Store all feedback as `feedback` nodes linked to `evaluation`.

### 7.2 Aggregate Patterns
- Use clustering or simple frequency analysis to find common failure patterns:
  - e.g., 80% of wrong SQL lack a join condition.
  - e.g., many OKF answers miss a specific rule.

### 7.3 Generate Candidates
- For SQL: propose a new prompt transform (e.g., `add_join_hint` that adds “Consider joining tables on foreign keys” to the prompt).
- For OKF: propose a new rule or update a concept’s aliases.

### 7.4 Validate Candidates
- Run candidate against a hold‑out set of past questions.
- Compare accuracy before and after.
- If accuracy improves by a threshold (e.g., >2%), promote candidate; otherwise discard.

### 7.5 Apply and Monitor
- Add the new transform to the prompt pipeline (via `promote`).
- The next SQL questions will automatically use it.
- Continue monitoring; if performance degrades, revert.

---

## Step 8 — Auditing and Explainability

Every response must be traceable to the graph objects and rules used.

### 8.1 SQL Agent Auditing
- `outcome_summary` already captures `applied_transforms`, structural signals.
- Add a link from `sql_output` to the specific `concept` nodes and `rule` nodes that influenced column selection or prompt.

### 8.2 OKF Agent Auditing
- `okf_answer` includes `context_parts` with exact concepts/rules used.
- Store `context_parts` in the graph for later inspection.

### 8.3 Mentor Auditing
- Every proposed change is a `prompt_transform` or `rule` node with a `status` field.
- Log all validate/apply/revert actions as edges.

---

## Step 9 — Testing & Validation

### 9.1 Unit Tests
- Each behavior individually: mock graph, emit input event, assert output event payload.
- Test prompt transform pipeline with mock transforms.

### 9.2 Integration Tests
- Full event chains for each role.
- Cross‑role flows (SQL + OKF, mentor + SQL).
- Use a FakeReader that returns known outputs for deterministic testing.

### 9.3 Regression Tests
- Store a set of past questions/requests with expected results.
- Run the entire agent and check that mentor changes do not degrade accuracy.

### 9.4 Lint Tests
- After mentor applies a change, run OKF lint to ensure graph consistency.

---

## Step 10 — Deployment & Packaging

Package the super agent as a single runtime with configurable behavior groups.

```python
from activegraph import Runtime

behaviors = [
    sql_agent.behaviors,
    okf_agent.behaviors,
    mentor.behaviors,
    router.behaviors,
]

runtime = Runtime(behaviors=behaviors)
runtime.run(seed_event={"type": REQUEST_RECEIVED, "payload": user_request})
```

- Use environment variables to enable/disable modules.
- Provide CLI commands:
  - `super-agent ask --role sql "Show total sales by region"`
  - `super-agent okf ingest --source kb.yaml`
  - `super-agent okf lint`
  - `super-agent mentor status`

---

## Summary

The **Super Agent** is built by:

1. **Unifying** all knowledge into one ActiveGraph schema.  
2. **Defining** a shared event vocabulary.  
3. **Extracting** common services (reader, embedder, prompt pipeline).  
4. **Implementing** three role modules as behavior groups.  
5. **Routing** requests with a classifier.  
6. **Connecting** roles through explicit cross‑role event flows.  
7. **Closing the loop** with an incremental‑learning mentor.  
8. **Ensuring** auditability and testing.

This approach leverages the existing event‑driven ActiveGraph architecture and extends it naturally to a multi‑role, self‑improving agent. It is modular, testable, and scalable — exactly what a coding agent tool would need to generate the implementation.

---

## Super-Agent Modeling Approach and sample SPECs

### folder structure

```
super_agent_model/
├── manifest.yaml              # 에이전트 모델 메타데이터 (버전, 역할, 설명)
├── events.yaml                # 전체 이벤트 타입 정의 (문자열 상수)
├── objects.yaml               # 그래프 노드 타입 정의 (스키마)
├── edges.yaml                 # 그래프 엣지 타입 정의
├── behaviors.yaml             # 행동 정의 (이벤트 구독 + 핸들러 참조)
├── guardrails.yaml            # 가드레일/검증 규칙 (정적)
├── caching.yaml               # 캐싱 전략 (선택)
├── workflows/                 # 복합 워크플로우 정의 (이벤트 체인)
│   ├── sql_workflow.yaml
│   ├── okf_workflow.yaml
│   └── mentor_workflow.yaml
├── concepts/                  # OKF 개념 트리 (선택적, 지식 구조)
├── rules/                     # OKF 규칙 (응답 규칙)
├── history.yaml               # 에이전트 모델 및 지식 구조 진화 기록
└── roles/                     # 역할별 설정
    ├── sql_role.yaml
    ├── okf_role.yaml
    └── mentor_role.yaml
```

### 4.2 events.yaml 예시

```yaml
events:
  request_received: "request.received"
  request_classified: "request.classified"
  sql_question_asked: "sql.question.asked"
  sql_schema_encoded: "sql.schema.encoded"
  sql_columns_scored: "sql.columns.scored"
  sql_prompt_assembled: "sql.prompt.assembled"
  sql_query_drafted: "sql.query.drafted"
  sql_evaluated: "sql.evaluated"
  okf_ingest_requested: "okf.ingest.requested"
  okf_ingested: "okf.ingested"
  okf_lint_requested: "okf.lint.requested"
  okf_linted: "okf.linted"
  okf_ask_requested: "okf.ask.requested"
  okf_context_assembled: "okf.context.assembled"
  okf_answer_generated: "okf.answer.generated"
  mentor_observe: "mentor.observe"
  mentor_analyze: "mentor.analyze"
  mentor_propose: "mentor.propose"
  mentor_validate: "mentor.validate"
  mentor_apply: "mentor.apply"
```

### 4.3 objects.yaml 예시 (노드 타입)

```yaml
objects:
  schema_table:
    fields: [name, description]
  schema_column:
    fields: [table, name, type, is_pk, description]
  concept:
    fields: [id, name, description, parent, aliases]
  rule:
    fields: [id, name, trigger, response_template, conditions, priority]
  question:
    fields: [question_id, text, type]   # type: sql | okf_ask | meta
  sql_output:
    fields: [question_id, predicted_sql, gold_sql, correct]
  okf_answer:
    fields: [request_id, answer, context_parts]
  evaluation:
    fields: [outcome_id, correct, feedback, signals]
  feedback:
    fields: [feedback_id, content, timestamp]
  prompt_transform:
    fields: [name, code_ref, status]    # proposed | active | retired
  memory:
    fields: [key, value, confidence]
```

### 4.4 edges.yaml 예시

```yaml
edges:
  has_column: {source: schema_table, target: schema_column}
  concept_child: {source: concept, target: concept}
  maps_to_column: {source: concept, target: schema_column}
  maps_to_table: {source: concept, target: schema_table}
  uses_rule: {source: concept, target: rule}
  rule_references_schema: {source: rule, target: schema_column | schema_table}
  question_about: {source: question, target: concept | schema_table}
  produced: {source: question, target: sql_output | okf_answer}
  evaluated_by: {source: sql_output | okf_answer, target: evaluation}
  generates_feedback: {source: evaluation, target: feedback}
  improves: {source: prompt_transform, target: sql_output | okf_answer}
  learned_from: {source: memory, target: evaluation | feedback}
```

### 4.5 behaviors.yaml 예시 (이벤트 구독 방식)

```yaml
behaviors:
  - name: sql_agent.encode_schema
    role: sql
    subscribes_to: sql.question.asked
    emits: sql.schema.encoded
    handler: sql_agent.encode_schema_handler   # 실제 코드 함수/클래스 참조
    description: "스키마 테이블/컬럼/FK를 그래프 객체로 변환"

  - name: sql_agent.retrieve_relevant_columns
    role: sql
    subscribes_to: sql.schema.encoded
    emits: sql.columns.scored
    handler: sql_agent.retrieve_relevant_columns_handler

  - name: okf_agent.ingest
    role: okf
    subscribes_to: okf.ingest.requested
    emits: okf.ingested
    handler: okf_agent.ingest_handler

  - name: mentor.observe
    role: mentor
    subscribes_to: [sql.evaluated, okf.answer.generated]
    emits: mentor.analyze
    handler: mentor.observe_handler
```

### 4.6 guardrails.yaml 예시

```yaml
guardrails:
  sql:
    - no_drop_table
    - max_columns_per_query: 20
  okf:
    - required_lint_before_ask: true
    - max_rules_per_answer: 10
  mentor:
    - min_validation_samples: 50
    - min_improvement_threshold: 0.02
```

### 4.7 history.yaml 예시

```yaml
history:
  - timestamp: "2026-08-14T09:00:00Z"
    operation: "ADD_BEHAVIOR"
    target: "mentor.propose_transform"
    reason: "멘토가 조인 누락 오류 패턴을 개선하기 위해 transform 제안"
  - timestamp: "2026-08-14T10:30:00Z"
    operation: "SPLIT_CONCEPT"
    target: "concept.sales_report"
    new_concepts: ["concept.regional_sales", "concept.product_sales"]
    reason: "판매 보고서 개념을 지역별/제품별로 분리"
```

---

## 5. 제안 모델의 장점

1. **선언적 + 이벤트 기반 결합**  
   YAML로 구조를 정의하고, 실제 동작은 이벤트 구독 방식으로 실행 → 정적 모델과 동적 워크플로우를 모두 만족

2. **역할별 관심사 분리**  
   `roles/`, `workflows/`를 통해 SQL, OKF, Mentor를 독립적으로 관리하면서도 하나의 그래프를 공유

3. **감사와 추적 가능**  
   `history.yaml`과 이벤트 페이로드에 `applied_transforms`, `context_parts` 등을 포함시켜 모든 결정을 재현 가능

4. **확장성**  
   새로운 역할(예: Reporting Agent)을 추가해도 기존 이벤트/객체 정의를 재사용 가능

5. **Coding Agent 친화적**  
   YAML 스펙이 명확하므로 다른 코딩 에이전트가 이를 해석하여 코드를 생성하기 쉬움