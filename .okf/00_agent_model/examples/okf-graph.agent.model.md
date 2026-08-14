Here is a proposed **Sale-Agent-Model** that extends the earlier event-driven, graph-based SQL agent architecture to handle a Google OKF (Open Knowledge Framework) knowledge base containing:

- a **sales database schema**
- a **tree of sales concepts**  
- **response rules** for customer requests  
- **relations** between concepts, schema elements, and rules  

The model supports the three OKF actions: **ingest**, **lint**, and **ask**.

---

## 1. Overview

The Sale-Agent-Model is an **active-graph, event-driven agent** similar to the SQL agent described previously. Its core is a graph of **objects** (nodes) and **relations** (edges) representing all knowledge in the OKF. On top of this graph, three behaviours implement the actions:

| Action  | Behaviour                      | Purpose                                                                 |
|---------|--------------------------------|-------------------------------------------------------------------------|
| ingest  | `okf_agent.ingest`             | Load OKF data into the graph                                           |
| lint    | `okf_agent.lint`               | Validate the graph for consistency, completeness, and rule conflicts   |
| ask     | `okf_agent.ask`                | Answer a customer request using concepts, rules, and schema            |

The agent uses the same event vocabulary pattern as the SQL agent, with an explicit event chain per action.

---

## 2. Knowledge Base Structure (Graph Model)

The OKF knowledge base is represented as a typed property graph.

### Node types

| Type           | Data fields                                                                 |
|----------------|-----------------------------------------------------------------------------|
| `schema_table` | `name`, `description`                                                        |
| `schema_column`| `table`, `name`, `type`, `description`, `is_pk`, `is_fk`                     |
| `concept`      | `id`, `name`, `description`, `parent_concept`, `level`, `aliases`            |
| `rule`         | `id`, `name`, `trigger`, `response_template`, `conditions`, `priority`       |

### Relation types

| Relation type   | Source → Target       | Meaning                                                                 |
|-----------------|------------------------|-------------------------------------------------------------------------|
| `has_column`    | `schema_table` → `schema_column` | Table contains column                                     |
| `concept_child` | `concept` → `concept` | Parent-child relationship in the concept tree                           |
| `maps_to_column`| `concept` → `schema_column` | Concept is represented by a DB column                            |
| `maps_to_table` | `concept` → `schema_table`  | Concept is represented by a DB table                              |
| `uses_rule`     | `concept` → `rule`      | Concept invokes or uses a rule                                        |
| `rule_references_schema` | `rule` → `schema_column`/`schema_table` | Rule mentions schema objects                          |
| `related_to`    | `concept` ↔ `concept`    | Generic semantic relation between concepts                             |

This graph is the **single source of truth** for all downstream actions.

---

## 3. Event Vocabulary

The agent defines the following events:

| Constant                 | Value                    | Emitted by / used by |
|--------------------------|--------------------------|----------------------|
| `OKF_INGEST_REQUESTED`   | `"okf.ingest.requested"` | Entry point / external trigger |
| `OKF_INGESTED`           | `"okf.ingested"`         | Ingest behaviour      |
| `OKF_LINT_REQUESTED`     | `"okf.lint.requested"`   | Entry point / external trigger |
| `OKF_LINTED`             | `"okf.linted"`           | Lint behaviour        |
| `OKF_ASK_REQUESTED`      | `"okf.ask.requested"`    | Entry point / customer query |
| `OKF_CONTEXT_ASSEMBLED`  | `"okf.context.assembled"`| Ask behaviour – context retrieval |
| `OKF_ANSWER_GENERATED`   | `"okf.answer.generated"` | Ask behaviour – final response |

The event chains are:

```
ingest: okf.ingest.requested → okf.ingested
lint:   okf.lint.requested  → okf.linted
ask:    okf.ask.requested   → okf.context.assembled → okf.answer.generated
```

---

## 4. Ingest Action

### Behaviour: `okf_agent.ingest`

**Trigger:** `okf.ingest.requested`  
**Emit:** `okf.ingested`

**Input payload:**
```python
{
  "kb_id": str,
  "okf_source": str | dict,   # file path, URL, or in-memory OKF representation
  "format": str,              # e.g., "yaml", "json", "protobuf"
}
```

**Processing steps:**

1. **Parse the OKF source** into a flat list of tables, columns, concepts, rules, and relations.
2. **Create graph objects**:
   - For each table → `schema_table` node.
   - For each column → `schema_column` node, connected to its table via `has_column`.
   - For each concept → `concept` node. If a parent is present, add `concept_child` relation.
   - For each rule → `rule` node.
3. **Create mapping relations**:
   - From concept to column/table where the OKF defines a mapping (`maps_to_column`, `maps_to_table`).
   - From concept to rule where the concept uses a rule (`uses_rule`).
   - From rule to schema object when a rule mentions it (`rule_references_schema`).
4. **Emit `okf.ingested`** with summary counts:
   ```python
   {
     "kb_id": str,
     "n_tables": int,
     "n_columns": int,
     "n_concepts": int,
     "n_rules": int,
     "n_relations": int,
   }
   ```

**Why this design:**  
Ingest is idempotent and graph‑centred. By converting OKF into a typed graph, all later actions can operate on a uniform representation instead of parsing files repeatedly.

---

## 5. Lint Action

### Behaviour: `okf_agent.lint`

**Trigger:** `okf.lint.requested`  
**Emit:** `okf.linted`

**Input payload:**
```python
{
  "kb_id": str,
  "lint_config": dict   # optional rule overrides
}
```

**Processing steps:**

The lint behaviour runs a set of **deterministic validators** over the graph. Each validator checks a specific property and produces a `LintIssue` if it fails.

| Validator                         | Description                                                                 |
|-----------------------------------|-----------------------------------------------------------------------------|
| `orphan_concept_checker`          | Every concept must be reachable from the root of the concept tree.          |
| `missing_schema_mapping_checker`  | Every leaf concept should map to at least one column or table.              |
| `rule_schema_reference_checker`   | Rules that mention `{column}` or `{table}` must reference existing objects. |
| `ambiguous_rule_trigger_checker`  | No two rules may have identical trigger conditions but different priorities.|
| `cyclic_concept_checker`          | The concept tree must be acyclic.                                           |
| `pk_fk_consistency_checker`       | Foreign key columns must reference an existing primary key.                 |

Each validator returns a list of issues:
```python
{
  "validator": str,
  "severity": "error" | "warning",
  "object_id": str,
  "message": str,
}
```

**Emitted payload:**
```python
{
  "kb_id": str,
  "valid": bool,
  "n_errors": int,
  "n_warnings": int,
  "issues": list[dict],
}
```

**Why this design:**  
Lint is deliberately **deterministic and graph‑based**, not LLM‑based. This guarantees that structural problems in the OKF are caught early, before they propagate into ask responses. The same validator set can be reused in CI pipelines.

---

## 6. Ask Action

### Behaviour: `okf_agent.ask`

The ask action is split into two behaviours so that context retrieval and answer generation can be independently tested and audited.

#### 6.1 `okf_agent.assemble_context`

**Trigger:** `okf.ask.requested`  
**Emit:** `okf.context.assembled`

**Input payload:**
```python
{
  "request_id": str,
  "customer_request": str,   # natural language question or request
  "kb_id": str,
}
```

**Processing steps:**

1. **Embed the customer request** using the same embedder as the SQL agent (e.g., HashEmbedder with L2-normalised vectors).
2. **Retrieve relevant concepts** by cosine similarity over concept names, descriptions, and aliases. Select top‑K concepts (e.g., K=10).
3. **Expand concept tree context**: for each selected concept, include its parent and children up to a certain depth to preserve hierarchy.
4. **Retrieve rules** that are connected to the selected concepts via `uses_rule`, or whose trigger matches a simple keyword/embedding similarity.
5. **Retrieve schema objects** (`schema_table` and `schema_column`) mapped to the selected concepts.
6. **Assemble a structured `context_parts` dictionary**:
   ```python
   {
     "concepts": [ {id, name, description, parent} ],
     "rules": [ {id, name, trigger, response_template, conditions} ],
     "schema": {
        "tables": [ {name, description} ],
        "columns": [ {table, name, type, description} ]
     },
     "relations": [ {source, target, type} ]
   }
   ```

7. **Emit `okf.context.assembled`** with the context parts.

#### 6.2 `okf_agent.generate_answer`

**Trigger:** `okf.context.assembled`  
**Emit:** `okf.answer.generated`

**Input payload:** from `okf.context.assembled`

**Processing steps:**

1. **Render a final prompt** using the context parts. The prompt instructs the LLM to respond as a sales assistant, using the provided concepts, rules, and schema.
2. **Call a Reader** (LLM) via the same indirection mechanism used in the SQL agent (`_READERS` keyed by `request_id`).
3. **Capture the generated response**.

**Emitted payload:**
```python
{
  "request_id": str,
  "answer": str,             # natural language response (or possibly SQL if requested)
  "context_parts": dict,
  "applied_context_strategy": str,
  "error": str,              # empty if no error
}
```

**Why this design:**  
Splitting context assembly from answer generation allows:
- swapping the LLM without changing retrieval logic,
- auditing exactly what knowledge was used to produce an answer,
- adding transformations (like the SQL agent’s prompt‑transform pipeline) later if needed.

---

## 7. Why This Sale-Agent-Model Is Proposed

### 7.1 Alignment with OKF’s three actions
The model maps **ingest → graph construction**, **lint → deterministic validation**, **ask → context‑aware retrieval + generation**. This matches the OKF philosophy of separating knowledge ingestion, quality checking, and usage.

### 7.2 Single graph as shared knowledge base
All actions operate on the same graph. This avoids duplication and ensures that any change made during ingest is immediately visible to lint and ask. It also simplifies debugging – you can inspect the graph directly.

### 7.3 Reuse of proven event‑driven architecture
The earlier SQL agent already demonstrated a clean event chain (`question.asked → … → query.drafted`). Extending that same pattern to OKF actions reduces development risk and makes the system easier to understand.

### 7.4 Deterministic linting improves reliability
Rules, schema mappings, and concept hierarchies can contain errors. By making lint a **non‑LLM, graph‑checking step**, we catch those errors reliably. LLM‑based validation would be nondeterministic and expensive.

### 7.5 Auditable and explainable answers
Every ask response is traceable to the exact concepts, rules, and schema objects that were included in the context. This is critical for sales use cases where answers must be justified (e.g., pricing, eligibility, contract terms).

### 7.6 Extensibility
- New OKF node types or relations can be added without changing the core behaviours.
- The prompt‑transform pipeline from the SQL agent can be inserted between context assembly and answer generation if needed.
- Lint validators can be registered dynamically, just like prompt transforms.

### 7.7 Clear separation of concerns
- **Ingest** handles parsing and graph building.
- **Lint** handles quality assurance.
- **Ask** handles retrieval and response generation.
Each behaviour is small, testable, and independently deployable.

---

## 8. Conclusion

The **Sale-Agent-Model** extends the original SQL agent pattern into a full knowledge‑driven sales assistant. It uses a typed graph as the knowledge backbone and defines three explicit behaviours corresponding to OKF’s `ingest`, `lint`, and `ask`. This model is proposed because it is **auditable**, **reliable**, **extensible**, and **directly maps to the OKF action semantics**, while reusing the proven event‑driven architecture from the earlier agent.

If you want, I can also provide a concrete class/interface definition (e.g., Python dataclasses and behaviour signatures) for this model that another coding agent tool could use to generate the implementation.