We can reverse-engineer the given code into a formal **Text-to-SQL Agent Model**. This model can be used as a specification for another coding agent tool to generate a similar agent.

---

## 1. Overview

The agent is an **event-driven, graph-based pipeline** that:

1. Receives a natural language question and database schema metadata.
2. Encodes schema objects (tables, columns, foreign keys) into a graph.
3. Selects the most relevant columns using embedding similarity.
4. Assembles a final LLM prompt through a configurable transformation pipeline.
5. Calls a `Reader` (LLM) to generate SQL.
6. Emits events at each stage; the final result is captured and later evaluated into a structured `SqlOutcome`.

The agent is built on **ActiveGraph**, using behaviors registered with `@behavior`. The behaviors are chained via events.

---

## 2. Event Vocabulary

Events are string constants defined in `events.py`.

| Constant           | Value               | Emitted by                     | Triggered by               |
|--------------------|---------------------|--------------------------------|----------------------------|
| `QUESTION_ASKED`   | `"question.asked"`  | (seed / entry point)          | –                          |
| `SCHEMA_ENCODED`   | `"schema.encoded"`  | `behavior_encode_schema`      | `question.asked`           |
| `COLUMNS_SCORED`   | `"columns.scored"`  | `behavior_score_columns`      | `schema.encoded`           |
| `PROMPT_ASSEMBLED` | `"prompt.assembled"`| `behavior_prompt_pipeline`    | `columns.scored`           |
| `QUERY_DRAFTED`    | `"query.drafted"`   | `behavior_draft_query`        | `prompt.assembled`         |

The execution chain is:
```
question.asked → schema.encoded → columns.scored → prompt.assembled → query.drafted
```

---

## 3. Graph Objects and Relations

The ActiveGraph stores the following node types:

### Node: `table`
- **type**: `"table"`
- **data**:
  - `name`: table name (string)

### Node: `column`
- **type**: `"column"`
- **data**:
  - `table`: parent table name (string)
  - `name`: column name without table prefix (string)
  - `qualified`: full qualified name, e.g., `"table.column"` (string)
  - `is_pk`: boolean, true if column is primary key

### Relation: `foreign_key`
- **type**: `"foreign_key"`
- **source**: column object ID (string)
- **target**: referenced column object ID (string)
- **data**:
  - `from`: qualified source column, e.g., `"sales.region"` (string)
  - `to`: qualified target column, e.g., `"regions.region_id"` (string)

---

## 4. Behaviors (Detailed Specification)

### 4.1 `sql_agent.encode_schema`

**Trigger:** `question.asked`  
**Emit:** `schema.encoded`

**Input payload (`question.asked`):**
```python
{
    "question_id": str,
    "question": str,           # natural language question
    "schema_id": str,          # optional
    "tables": list[str],       # table names
    "columns_by_table": dict[str, list[str]],  # table → column names
    "foreign_keys": list[tuple[str, str, str, str]],  # (table, col, ref_table, ref_col)
    "primary_keys": dict[str, str],  # table → primary key column
}
```

**Processing:**
- For each table name, add a `table` node.
- For each table, iterate its columns from `columns_by_table` and add a `column` node with `is_pk` determined from `primary_keys`.
- For each foreign key tuple `(t, c, rt, rc)`, find the corresponding column node IDs and add a `foreign_key` relation from `(t,c)` to `(rt,rc)`.
- Count `n_tables`, `n_columns`, `n_foreign_keys`.

**Emitted payload (`schema.encoded`):**
```python
{
    "question_id": str,
    "question": str,
    "schema_id": str,
    "n_tables": int,
    "n_columns": int,
    "n_foreign_keys": int,
}
```

---

### 4.2 `sql_agent.retrieve_relevant_columns`

**Trigger:** `schema.encoded`  
**Emit:** `columns.scored`

**Input payload:** from `schema.encoded`

**Processing:**
- Retrieve a shared embedder via `get_embedder()` (e.g., HashEmbedder with L2-normalized vectors).
- Iterate all `column` nodes from the graph (`ctx.view.objects(type="column")`) and collect `(object_id, qualified, table)`.
- Construct a list of texts: `[question] + [qualified for each column]`.
- Compute embeddings for all texts.
- Compute cosine similarity between question vector and each column vector (dot product because vectors are L2-normalized).
- Sort columns by descending similarity.
- Select top `top_k = 12` columns (if fewer than 12 columns, all are selected).
- Store scores as a dict mapping `qualified` → score.

**Emitted payload (`columns.scored`):**
```python
{
    "question_id": str,
    "question": str,
    "scorer_model": str,          # embedder model name
    "scores": dict[str, float],   # qualified column name → score
    "ranked": list[str],          # qualified column names sorted by score desc
    "selected_column_ids": list[str],  # top-k qualified names
}
```

---

### 4.3 `sql_agent.prompt_pipeline`

**Trigger:** `columns.scored`  
**Emit:** `prompt.assembled`

**Input payload:** from `columns.scored`

**Processing:**
1. **Select columns:**  
   Use `selected_column_ids` (qualified names) to filter all `column` nodes in the graph.
2. **Group selected columns by table:**  
   For each selected column, append its unqualified name to a list per table.
3. **Extract foreign-key hints:**  
   Iterate all `foreign_key` relations. If both `from` and `to` are in the selected set, add a hint string `"from -> to"`.
4. **Build schema text:**  
   For each table (sorted), create a line:  
   `"  {table}({col1}, {col2}, ...)"`.  
   Prepend `"Tables:\n"`. If no columns selected, use `"(no schema)"`.
5. **Build `schema_meta`:**  
   ```python
   {
       "tables": sorted(list of selected table names),
       "columns_by_table": {table: list of unqualified column names},
       "foreign_keys": fk_hints,  # list of "from -> to" strings
   }
   ```
6. **Initialize `prompt_parts`:**
   ```python
   {
       "schema": schema_text,   # string
       "instructions": "Write a single SQLite SELECT statement that answers the question. Return only the SQL — no prose.",
       "hints": fk_hints,       # list[str]
       "question": question,     # string
   }
   ```
7. **Run the prompt-transform pipeline:**  
   Call `prompt_transforms.apply_pipeline(prompt_parts=prompt_parts, question=question, schema_meta=schema_meta)`.
   This returns:
   ```python
   ({"prompt_parts": final_parts, "names": applied_transform_names}, errors)
   ```
   - `final_parts` has the same keys as `prompt_parts`, but values may have been modified by transforms.
   - `errors` is a list of `{"name": str, "error": str}` for transforms that raised.
8. **Assemble final prompt string:**
   ```text
   {final_parts['schema']}

   {optional Hints block if hints exist:
   Hints:
     hint1
     hint2
   }
   Instructions: {final_parts['instructions']}

   Question: {final_parts['question']}

   SQL:
   ```
   If `final_parts["hints"]` is empty or falsy, the Hints block is omitted.

**Emitted payload (`prompt.assembled`):**
```python
{
    "question_id": str,
    "question": str,
    "prompt": final_prompt,          # full string for LLM
    "prompt_parts": final_parts,     # dict with keys: schema, instructions, hints, question
    "schema_meta": schema_meta,      # dict with tables, columns_by_table, foreign_keys
    "applied_transforms": list[str], # names of transforms that succeeded
    "transform_errors": list[dict],  # errors from failed transforms
}
```

---

### 4.4 `sql_agent.draft_query`

**Trigger:** `prompt.assembled`  
**Emit:** `query.drafted`

**Input payload:** from `prompt.assembled`

**Processing:**
- Retrieve the `Reader` instance for this `question_id` from a process-level registry (`_READERS` dict).
  - The reader is registered by the entry point via `_set_reader(question_id, reader)` before emitting `question.asked`.
  - If no reader is found, set `drafter_error = "reader_missing: no Reader registered for question_id"`.
- If a reader exists, call:
  ```python
  sql = reader.answer(context=prompt, question=question, question_id=question_id)
  ```
  - `prompt` is the final assembled prompt string.
  - `question` is the original natural language question.
- Handle exceptions: set `drafter_error = f"{type(e).__name__}: {e}"` and `sql = ""`.
- If `sql` is a string, strip whitespace; otherwise set to empty string.
- Set `reader_name` from `getattr(reader, "name", "")` or empty.

**Emitted payload (`query.drafted`):**
```python
{
    "question_id": str,
    "predicted_sql": str,
    "drafter_error": str,
    "reader_name": str,
}
```

---

## 5. Prompt Transform Pipeline

The pipeline is a global ordered list of named callables.

### 5.1 Transform Signature

```python
def transform(prompt_parts: dict, question: str, schema_meta: dict) -> dict:
    ...
```

- **Input**: current `prompt_parts` (a dictionary with keys `schema`, `instructions`, `hints`, `question`), the original question, and schema metadata.
- **Output**: a new dictionary with **the same set of keys** (or a subset).  
  - If a key is omitted, the previous value is retained.
  - **New keys are not allowed** – raising an error.
  - Returned value must be a `dict`.

### 5.2 Pipeline Management

- `get_pipeline()` – returns a snapshot of the current pipeline entries.
- `promote(name, fn)` – append a transform (used by promotion gates).
- `revert(name)` – remove a transform by name.
- `clear()` – remove all transforms (test isolation).

### 5.3 Execution (`apply_pipeline`)

```python
def apply_pipeline(*, prompt_parts, question, schema_meta) -> (result, errors):
```

- Iterates through the pipeline entries in order.
- For each transform:
  - Calls `entry.fn(cur, question, schema_meta)`.
  - Validates return type and keys.
  - If successful, merges: `cur = {k: new_parts.get(k, cur[k]) for k in cur}`.
  - Appends transform name to `names`.
  - If it raises any exception, skips and appends `{"name": ..., "error": ...}` to `errors`.
- Returns:
  - `result = {"prompt_parts": cur, "names": names}`
  - `errors`

This ensures that a failing transform never breaks the whole assembly, and the audit trail is preserved.

---

## 6. Target and Evaluation Components

From the first code block, the full `SqlTarget` consists of:

### 6.1 `SqlTarget`
A composition object holding:
- `eval_backend`: instance of `SqlEvalBackend`
- `action_space`: instance of `SqlActionSpace`
- `taxonomy`: instance of `SqlTaxonomy`
- `name`: string, default `"sql"`

### 6.2 `SqlEvalBackend`
- Built with a `Reader` instance.
- Responsible for evaluating the predicted SQL against gold SQL.
- Produces an `SqlOutcome`.

### 6.3 `SqlActionSpace`
- Built with an `author` (default `StubSqlAuthor`) and a `SqlTaxonomy`.
- Contains the prompt-transform pipeline and SQL-shaped gates (not shown in detail).
- Shares the same `SqlTaxonomy` instance with the main target.

### 6.4 `SqlTaxonomy`
- Deterministic detectors for SQL structure, e.g., `has_join`, `has_where`, `has_group_by`.
- Used by evaluation to generate structural signals.

### 6.5 `SqlOutcome`
Fields (from `outcome_summary` mapping):
- `question_id`
- `question_type`
- `correct` (bool)
- `schema_id`
- `predicted_sql`
- `gold_sql`
- `exec_error` (optional)
- `predicted_tables` (list)
- `predicted_qualified_columns` (list of lists, e.g., `[[table, col], ...]`)
- `schema_tables` (list)
- `predicted_has_join`, `predicted_has_where`, `predicted_has_group_by` (bool)
- `gold_has_join`, `gold_has_where`, `gold_has_group_by` (bool)
- `applied_transforms` (list of transform names from the prompt pipeline)

The `outcome_summary` function converts an `SqlOutcome` into a dictionary for persistence, including the structural signals for auditability.

---

## 7. Reader Context Indirection

Because Python callables cannot be serialized in event payloads, the system uses a process-level dictionary (`_READERS`) keyed by `question_id`:

- `_set_reader(question_id, reader)` – called by the entry point before emitting `question.asked`.
- `_get_reader(question_id)` – used in `behavior_draft_query`.
- `_clear_reader(question_id)` – called after the query is drafted or the question is complete to avoid memory leaks.

**Important:** The entry point (`retrieve()`) must register the reader for each question before the event chain starts.

---

## 8. End-to-End Data Flow

```
Entry Point
   │
   ├─ Register reader in _READERS keyed by question_id
   ├─ Emit question.asked payload (question, schema info)
   │
   ▼
[behavior_encode_schema]
   │  Creates graph nodes/relations
   ▼
schema.encoded
   │
   ▼
[behavior_score_columns]
   │  Embeds question + columns, selects top-k
   ▼
columns.scored
   │
   ▼
[behavior_prompt_pipeline]
   │  Builds prompt_parts, runs transforms, assembles final prompt
   ▼
prompt.assembled
   │
   ▼
[behavior_draft_query]
   │  Calls Reader.answer(context=prompt)
   ▼
query.drafted
   │
   ▼
Evaluation (outside behavior chain)
   │  Uses SqlEvalBackend, SqlTaxonomy, SqlActionSpace
   ▼
SqlOutcome → outcome_summary → persisted report
```

---

## 9. Interface Contracts

### Reader
```python
class Reader:
    name: str   # optional, used for reader_name in query.drafted
    def answer(self, context: str, question: str, question_id: str) -> str:
        """Returns SQL as string."""
```

### Embedder
```python
def get_embedder() -> Embedder:
    ...

class Embedder:
    model: str
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Returns L2-normalized vectors for cosine similarity."""
```

### Prompt Transform
```python
def transform(prompt_parts: dict, question: str, schema_meta: dict) -> dict:
    """Returns modified prompt_parts with no new keys."""
```

### Behavior Registration
```python
@behavior(name="sql_agent.<behavior_name>", on=[EVENT])
def behavior_fn(event, graph, ctx):
    ...
```

---

## 10. Key Implementation Constraints

1. **No new keys in prompt transforms** – transforms must only modify existing keys.  
2. **Transforms are skipped on error** – their contribution is omitted; previous values persist.  
3. **The prompt pipeline records `applied_transforms`** – this is used in the final outcome for auditability.  
4. **Same `SqlTaxonomy` instance is shared** between `action_space` and `target` – ensures `eval_diff` and `diagnose` use the same registry.  
5. **Reader is stored in a process-level dict** – not serialized in events.  
6. **Embedding vectors must be L2-normalized** for the dot-product cosine similarity to be valid.  
7. **The default top-k is 12**, but if the schema has fewer columns, all are selected.  
8. **FK hints are only included if both endpoints are in the selected column set** – avoids irrelevant join suggestions.  
9. **The final prompt ends with `"SQL:"`** – instructs the LLM to output only SQL.  
10. **Errors in behaviors are captured in payload fields** (e.g., `drafter_error`, `transform_errors`) rather than raising exceptions, following the framework’s failure model.

---

This reverse-engineered model is complete enough to serve as a blueprint for another coding agent tool to generate an equivalent text-to-SQL agent. The next step would be to translate this into concrete class and function definitions in the target programming language and framework.

---

아래는 앞서 역설계한 SQL 에이전트 모델을 확장하여,  
**Sales-OKF 지식 베이스를 다루는 `Sale-Agent-Model`**을 정의한 것입니다.

이 모델은 OKF의 **ingest / lint / ask** 액션을 모두 처리할 수 있도록  
이벤트 기반 행동(behavior) 체인, 그래프 객체 모델, 프롬프트 변환 파이프라인, 평가/감사용 요약 구조를 포함합니다.

---

## 1. Sales-OKF 지식 베이스 구조

OKF(Open Knowledge Framework)는 다음 요소로 구성됩니다.

| 요소 | 설명 |
|------|------|
| **Concept** | 판매 도메인 개념. 예: `product`, `customer`, `region`, `order`, `refund_policy` |
| **Rule** | 고객 요청에 대응하는 규칙. 예: “환불 요청 시 `refund_policy`를 참조하라” |
| **Schema** | 영업 DB 스키마. 테이블·컬럼·관계 정보 |
| **Relation** | 개념-개념, 개념-스키마, 개념-규칙, 규칙-스키마 사이의 의미적 연결 |

**Concept 트리 구조 예시**

```
sales
├── product
│   ├── physical_product
│   └── digital_product
├── customer
│   ├── individual
│   └── corporate
├── order
│   ├── purchase_order
│   └── refund_order
└── policy
    ├── discount_policy
    └── refund_policy
```

**Rule 예시**

```yaml
rule: refund_request
description: 환불 요청 처리 규칙
condition: customer_intent == "refund"
actions:
  - refer_to: refund_policy
  - require_schema: [orders, refunds]
  - response_style: formal
```

**Relation 예시**

```yaml
- type: concept_to_schema
  from: order
  to: sales.orders
- type: concept_to_rule
  from: refund_order
  to: refund_request
- type: schema_to_rule
  from: sales.refunds
  to: refund_request
```

---

## 2. Sale-Agent-Model 개요

기존 SQL 에이전트와 동일하게 **Target / ActionSpace / Taxonomy / EvalBackend** 조합으로 정의합니다.

```
SalesTarget
├── eval_backend : SalesOkfEvalBackend
├── action_space : SalesOkfActionSpace
├── taxonomy     : SalesOkfTaxonomy
└── name         : "sales_okf"
```

- `SalesOkfEvalBackend`  
  OKF KB 저장소(그래프)와 평가 로직을 가진다.  
  `ingest`, `lint`, `ask`의 결과를 검증하고 `SalesOkfOutcome`을 생성한다.

- `SalesOkfActionSpace`  
  OKF 액션별 입력 페이로드를 검사하고, 행동 체인으로 연결한다.  
  프롬프트 변환 파이프라인을 포함한다.

- `SalesOkfTaxonomy`  
  OKF 구조·규칙·답변 품질을 결정적으로 탐지하는 디텍터 모음.  
  예: `concept_cycle_detected`, `rule_conflict_detected`, `missing_schema_reference`, `answer_has_policy_reference`

- `SalesOkfOutcome`  
  액션 수행 결과를 저장한다. `outcome_summary`를 통해 감사 가능한 요약으로 변환된다.

---

## 3. 이벤트 어휘

OKF 액션은 **세 개의 이벤트 체인**으로 구성됩니다.

### 3.1 Ingest 체인

| 이벤트 | 발생 주체 | 설명 |
|--------|-----------|------|
| `okf.ingest.requested` | 엔트리포인트 | OKF 파일/문서 수신 |
| `okf.parsed` | `behavior_parse_okf` | YAML/JSON 파싱 완료 |
| `okf.validated` | `behavior_validate_okf` | 스키마·중복·참조 무결성 검증 완료 |
| `okf.loaded` | `behavior_load_graph` | 그래프에 객체/관계 적재 완료 |

### 3.2 Lint 체인

| 이벤트 | 발생 주체 | 설명 |
|--------|-----------|------|
| `okf.lint.requested` | 엔트리포인트 | 린트 검사 요청 |
| `okf.analyzed` | `behavior_analyze_okf` | 구조·규칙·관계 분석 |
| `okf.linted` | `behavior_detect_okf_issues` | 이슈 목록 생성 |

### 3.3 Ask 체인

| 이벤트 | 발생 주체 | 설명 |
|--------|-----------|------|
| `customer.question.asked` | 엔트리포인트 | 고객 질문 수신 |
| `concepts.retrieved` | `behavior_retrieve_concepts` | 관련 개념 검색 |
| `rules.selected` | `behavior_select_rules` | 적용 가능한 규칙 선택 |
| `schema.selected` | `behavior_select_schema` | 필요 스키마 선택 |
| `prompt.assembled` | `behavior_prompt_pipeline` | 최종 프롬프트 조립 |
| `answer.drafted` | `behavior_draft_answer` | Reader가 답변 생성 |
| `sql.query_drafted` | `behavior_draft_sql` (선택) | SQL이 필요한 경우 |
| `answer.delivered` | `behavior_finalize_answer` | 최종 응답 포장 |

---

## 4. 그래프 객체 모델

OKF KB는 ActiveGraph 위에 다음과 같은 노드/관계로 표현됩니다.

| 노드 타입 | data 필드 |
|-----------|-----------|
| `concept` | `name`, `parent`, `path`, `description` |
| `rule` | `name`, `condition`, `actions`, `priority` |
| `schema_table` | `name`, `columns` |
| `schema_column` | `table`, `name`, `type`, `is_pk` |

| 관계 타입 | source → target | data 필드 |
|-----------|-----------------|-----------|
| `is_a` | concept → concept | `from`, `to` |
| `part_of` | concept → concept | `from`, `to` |
| `concept_to_rule` | concept → rule | `from`, `to` |
| `rule_to_rule` | rule → rule | `relation_type` (conflict, extends, requires) |
| `concept_to_schema` | concept → schema_table | `from`, `to` |
| `rule_to_schema` | rule → schema_table | `from`, `to` |
| `schema_fk` | schema_column → schema_column | `from`, `to` |

---

## 5. 행동(Behavior) 명세

### 5.1 `behavior_parse_okf`

**입력**: `okf.ingest.requested`  
**출력**: `okf.parsed`

- OKF 원본 데이터를 파싱하여 중간 표현(예: `OkfDocument`) 생성
- 트리 구조, 규칙, 스키마, 관계를 임시 객체로 변환
- 파싱 실패 시 `parse_errors`를 이벤트에 포함

### 5.2 `behavior_validate_okf`

**입력**: `okf.parsed`  
**출력**: `okf.validated`

- 개념 트리에서 부모 참조가 존재하는지 확인
- 규칙이 참조하는 개념·스키마가 실제 존재하는지 검사
- 중복 개념/규칙 탐지
- 순환 참조(concept cycle) 탐지
- 검증 결과를 `validation_errors`, `validated` 플래그로 반환

### 5.3 `behavior_load_graph`

**입력**: `okf.validated`  
**출력**: `okf.loaded`

- 검증된 개념/규칙/스키마/관계를 ActiveGraph 노드·에지로 변환
- 개념 트리의 부모-자식을 `is_a` 또는 `part_of` 관계로 저장
- 규칙-스키마 관계를 `rule_to_schema`로 저장
- 적재된 노드/에지 수를 이벤트에 기록

### 5.4 `behavior_analyze_okf`

**입력**: `okf.lint.requested`  
**출력**: `okf.analyzed`

- 전체 그래프를 순회하며 아래 항목 분석
  - 개념 트리 깊이, 분기 수
  - 규칙 간 충돌 관계 (`rule_to_rule` 관계 타입 확인)
  - 개념-스키마 매핑 누락
  - 규칙-스키마 매핑 누락
  - 사용되지 않는 개념/규칙 감지

### 5.5 `behavior_detect_okf_issues`

**입력**: `okf.analyzed`  
**출력**: `okf.linted`

- 분석 결과를 기반으로 이슈 목록 생성
  - `missing_reference`
  - `cycle_detected`
  - `rule_conflict`
  - `orphan_concept`
  - `unused_rule`
- 각 이슈는 `issue_type`, `severity`, `target`, `message` 포함
- 최종 `lint_report`를 이벤트에 실어 반환

### 5.6 `behavior_retrieve_concepts`

**입력**: `customer.question.asked`  
**출력**: `concepts.retrieved`

- 고객 질문을 임베딩하여 개념 노드의 텍스트 표현과 유사도 계산
- 트리 구조를 활용하여 상위/하위 개념 경로도 검색에 포함
- 상위 `top_k=10` 개념을 `selected_concept_ids`로 선택

### 5.7 `behavior_select_rules`

**입력**: `concepts.retrieved`  
**출력**: `rules.selected`

- 선택된 개념과 `concept_to_rule` 관계로 연결된 규칙을 수집
- 규칙 우선순위(`priority`)와 질문 의도 매칭도를 기준으로 정렬
- 상위 `top_rules`를 선택하여 이벤트에 기록

### 5.8 `behavior_select_schema`

**입력**: `rules.selected`  
**출력**: `schema.selected`

- 선택된 규칙이 참조하는 스키마 테이블·컬럼을 수집
- `concept_to_schema`, `rule_to_schema` 관계를 따라 탐색
- 최종적으로 필요한 스키마만 `selected_schema_ids`로 포함

### 5.9 `behavior_prompt_pipeline`

**입력**: `schema.selected`  
**출력**: `prompt.assembled`

- 선택된 개념·규칙·스키마를 이용해 `prompt_parts` 구성
  ```python
  {
      "system_instruction": "You are a sales assistant. Use only the provided OKF knowledge.",
      "concepts": [...],      # relevant concept names
      "rules": [...],         # relevant rule texts
      "schema": "...",        # relevant schema text
      "customer_question": question,
      "style_hint": "formal"  # from rule actions
  }
  ```
- 등록된 **prompt transform pipeline**을 실행하여 프롬프트 조각을 변환
- 적용된 transform 이름을 `applied_transforms`에 기록
- 최종 프롬프트 문자열을 조립

### 5.10 `behavior_draft_answer`

**입력**: `prompt.assembled`  
**출력**: `answer.drafted`

- `_READERS`에서 `question_id`에 해당하는 Reader를 꺼내 호출
- `reader.answer(context=final_prompt, question=question, question_id=question_id)`
- 실패 시 `drafter_error`에 오류 기록
- 생성된 답변 텍스트를 `predicted_answer`로 저장

### 5.11 `behavior_finalize_answer`

**입력**: `answer.drafted`  
**출력**: `answer.delivered`

- 답변에 적용된 규칙, 개념, 스키마를 메타데이터로 첨부
- 최종 응답 객체를 생성하고 `SalesOkfOutcome`으로 변환할 수 있는 payload 생성

---

## 6. OKF 액션별 처리 흐름 요약

### 6.1 Ingest

```
okf.ingest.requested
  → parse_okf → okf.parsed
  → validate_okf → okf.validated
  → load_graph → okf.loaded
```

**결과물**: ActiveGraph에 적재된 OKF KB

### 6.2 Lint

```
okf.lint.requested
  → analyze_okf → okf.analyzed
  → detect_okf_issues → okf.linted
```

**결과물**: `lint_report`와 이슈 목록

### 6.3 Ask

```
customer.question.asked
  → retrieve_concepts → concepts.retrieved
  → select_rules → rules.selected
  → select_schema → schema.selected
  → prompt_pipeline → prompt.assembled
  → draft_answer → answer.drafted
  → finalize_answer → answer.delivered
```

**결과물**: 고객 응답 텍스트 + 적용된 OKF 요소 목록

---

## 7. Target / ActionSpace / Taxonomy 명세

### 7.1 `SalesOkfTarget`

```python
@dataclass
class SalesOkfTarget:
    eval_backend: SalesOkfEvalBackend
    action_space: SalesOkfActionSpace = field(default_factory=SalesOkfActionSpace)
    taxonomy: SalesOkfTaxonomy = field(default_factory=SalesOkfTaxonomy)
    name: str = "sales_okf"

    def outcome_summary(self, outcome: SalesOkfOutcome) -> dict:
        return outcome_summary(outcome)
```

### 7.2 `SalesOkfActionSpace`

- `ingest_gate`: 입력 OKF 문서가 유효한 구조인지 검사
- `lint_gate`: 린트 요청에 필요한 스코프/대상이 있는지 확인
- `ask_gate`: 고객 질문과 필요한 컨텍스트가 존재하는지 확인
- `prompt_pipeline`: 프롬프트 변환 파이프라인 포함

### 7.3 `SalesOkfTaxonomy`

- `is_concept_cycle`: 개념 트리 순환 여부
- `is_rule_conflict`: 규칙 충돌 여부
- `has_missing_ref`: 참조 누락 여부
- `has_answer_policy`: 답변에 정책 참조가 있는지
- `has_sql_query`: SQL 생성 여부
- `response_style`: 응답 스타일(격식/비격식 등)

### 7.4 `SalesOkfOutcome`

```python
{
    "action": "ingest|lint|ask",
    "success": True,
    "n_concepts": 0,
    "n_rules": 0,
    "n_schema_tables": 0,
    "lint_report": [...],
    "selected_concepts": [...],
    "selected_rules": [...],
    "applied_transforms": [...],
    "predicted_answer": "...",
    "drafter_error": ""
}
```

`outcome_summary`는 이 중 감사에 필요한 신호를 골라 사전으로 반환합니다.

---

## 8. 왜 이런 모델을 제안하는가?

### 8.1 SQL 에이전트와 동일한 검증된 패턴 재사용

앞서 역설계한 SQL 에이전트는  
**이벤트 체인 + 행동 등록 + 그래프 객체 + 프롬프트 파이프라인 + 결과 요약**이라는 구조를 갖고 있습니다.  
이 구조는 OKF처럼 **여러 액션이 하나의 KB를 공유하고, 각 액션이 여러 단계를 거치는 경우**에 특히 적합합니다.

- **ingest**는 그래프를 구축하고
- **lint**는 그래프를 분석하며
- **ask**는 그래프에서 관련 정보를 검색하여 응답을 생성합니다.

모두 같은 그래프를 사용하므로 행동 체인으로 자연스럽게 연결할 수 있습니다.

### 8.2 액션별 책임 분리

`ingest`는 파싱·검증·적재,  
`lint`는 정적 분석·이슈 탐지,  
`ask`는 검색·프롬프트 조립·응답 생성으로 역할이 명확히 나뉩니다.

각각을 별도의 behavior로 등록하면  
- 개별 단계를 독립적으로 테스트할 수 있고
- 실패 지점을 이벤트 payload에 기록할 수 있으며
- 다른 액션에 영향을 주지 않고 수정할 수 있습니다.

### 8.3 프롬프트 변환 파이프라인 재사용

`ask` 액션에서 고객 응답 스타일, 포함할 규칙, 스키마 표시 방식 등을  
**prompt transforms**로 미세 조정할 수 있습니다.

예를 들어  
- `add_refund_policy_hint` transform
- `remove_unused_concepts` transform
- `enforce_formal_style` transform

등을 등록하면, 변환 과정이 모두 `applied_transforms`에 기록되어  
최종 응답이 왜 그렇게 생성되었는지 추적할 수 있습니다.

### 8.4 감사 가능한 결과 요약

`SalesOkfOutcome`에 적용된 규칙·개념·변환기 목록을 함께 저장하면  
나중에 “이 답변은 어떤 OKF 지식을 사용했는가?”를 정확히 확인할 수 있습니다.

이는 금융·영업처럼 규정 준수가 중요한 도메인에서 필수적입니다.

### 8.5 확장성

이벤트와 behavior를 추가하기만 하면  
- `ingest` 후 자동 `lint` 실행
- `ask` 중 SQL 생성이 필요할 때 서브 체인 호출
- 새로운 OKF 파일 버전 관리
등을 쉽게 확장할 수 있습니다.

---

## 9. 다른 코딩 에이전트 도구가 생성할 수 있는 계약 명세

위 모델을 바탕으로 다음과 같은 계약(interface)을 제시하면  
다른 코딩 에이전트 도구가 동일한 Sale-Agent를 생성할 수 있습니다.

```yaml
agent_name: sales_okf_agent
framework: activegraph

events:
  - question.asked
  - okf.ingest.requested
  - okf.parsed
  - okf.validated
  - okf.loaded
  - okf.lint.requested
  - okf.analyzed
  - okf.linted
  - concepts.retrieved
  - rules.selected
  - schema.selected
  - prompt.assembled
  - answer.drafted
  - answer.delivered

behaviors:
  - name: sales_agent.encode_okf
    on: [okf.ingest.requested]
    emits: okf.parsed
  - name: sales_agent.validate_okf
    on: [okf.parsed]
    emits: okf.validated
  - name: sales_agent.load_okf_graph
    on: [okf.validated]
    emits: okf.loaded
  - name: sales_agent.analyze_okf
    on: [okf.lint.requested]
    emits: okf.analyzed
  - name: sales_agent.detect_okf_issues
    on: [okf.analyzed]
    emits: okf.linted
  - name: sales_agent.retrieve_concepts
    on: [customer.question.asked]
    emits: concepts.retrieved
  - name: sales_agent.select_rules
    on: [concepts.retrieved]
    emits: rules.selected
  - name: sales_agent.select_schema
    on: [rules.selected]
    emits: schema.selected
  - name: sales_agent.prompt_pipeline
    on: [schema.selected]
    emits: prompt.assembled
  - name: sales_agent.draft_answer
    on: [prompt.assembled]
    emits: answer.drafted
  - name: sales_agent.finalize_answer
    on: [answer.drafted]
    emits: answer.delivered

target:
  name: sales_okf
  eval_backend: SalesOkfEvalBackend
  action_space: SalesOkfActionSpace
  taxonomy: SalesOkfTaxonomy

prompt_parts:
  - system_instruction
  - concepts
  - rules
  - schema
  - customer_question
  - style_hint
```

---

이렇게 정의된 **Sale-Agent-Model**은  
OKF 지식 베이스를 안전하게 수집(ingest)하고, 품질을 점검(lint)하며,  
고객 요청에 대해 감사 가능한 답변(ask)을 생성하는 에이전트를  
다른 도구에서도 동일하게 구현할 수 있도록 충분히 구체적인 청사진을 제공합니다.