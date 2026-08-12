---
name: activegraph-okf-modeler
description: Coding Agent MUST materialize all reusable capabilities into OKF Agent & Domain Model. Target Agent MUST be compilable and executable with fixed local LLM. Task NOT complete unless 7 DONE conditions pass.
---

# MANDATORY RULES

## 1. KNOWLEDGE vs BEHAVIOR vs DOMAIN

| Layer | Definition | Storage Location |
| :--- | :--- | :--- |
| **Knowledge** | What the system knows | `00_meta/`, `01_modules/`, `02_composite/`, `03_atomic/` |
| **Behavior** | What the Target Agent does | `04_agent_model/` (events, objects, edges, behaviors, guardrails, caching) |
| **Domain** | What the Agent reasons about (business concepts, rules, workflows) | `10_domain/` (concepts, relations, rules, workflows) |
| **Control** | When/under what conditions behavior executes | Represented via Event, Edge, Pattern, Trigger |

**RULE**: If a reusable capability is discovered, it MUST be modeled. Do NOT hide it in arbitrary code, one-off scripts, or Cloud session memory.

---

## 2. Agent & Domain Model

### 2.1 Agent Model (`.okf/00_agent_model/`)

```text
00_agent_model/
├── manifest.yaml          # bundle, model_version, local_llm
├── events.yaml            # Event types + payload + semantics
├── objects.yaml           # Graph Node types + properties
├── edges.yaml             # Typed edges + source/target/condition
├── behaviors.yaml         # id, pattern, triggers, prompt_template
├── guardrails.yaml        # rules + actions
├── caching.yaml           # strategy, TTL, key_fields, invalidation
└── history.yaml           # evolution record (operator + reason + validation)
```

### 2.2 Domain Model 

```text
domain_name/
├── concepts/              # 업무 개념 (customer, order, product, inquiry)
│   ├── _index.md
│   ├── customer.md
│   ├── order.md
│   ├── product.md
│   └── inquiry.md
├── relations/             # 개념 간 관계 (placed, contains, categorized_as)
│   ├── placed.md
│   └── contains.md
├── rules/                 # 의사결정 규칙 (priority, escalation)
│   ├── priority_rule.md
│   └── escalation_rule.md
└── workflows/             # 업무 프로세스 흐름 (BPMN-like)
    └── inquiry_handling.md
```

**RULE 1**: Agent Model files MUST be updated for every behavior-affecting change. Generated code is derived from these files.

**RULE 2**: Agent Model (`04_agent_model/`) MAY reference and utilize Domain Model (`10_domain/`).

- Example: `behavior.inquiry_handler` reads `domain/rules/priority_rule.md` to determine priority.

---

## 3. 13 EVOLUTION OPERATORS (MUST USE ONE OR MORE)

| Operator | Purpose |
| :--- | :--- |
| `ADD_BEHAVIOR` | New reusable behavior |
| `REMOVE_BEHAVIOR` | Remove obsolete behavior |
| `MODIFY_BEHAVIOR` | Change pattern/prompt/triggers |
| `SPLIT_BEHAVIOR` | Split overloaded behavior |
| `MERGE_BEHAVIORS` | Merge overlapping behaviors |
| `ADD_EVENT_TYPE` | New event trigger |
| `REMOVE_EVENT_TYPE` | Remove unused event |
| `ADD_OBJECT_TYPE` | New Graph Node type |
| `MODIFY_EDGE` | Change edge semantics/condition |
| `REWIRE_BEHAVIOR` | Change subscribed events/patterns |
| `ADD_GUARDRAIL` | New runtime constraint |
| `OPTIMIZE_PROMPT` | Refine LLM prompt |
| `ADD_CACHING` | Cache repeated/expensive operations |

**RULE 1**: Every Agent Model modification MUST be recorded with the operator(s) used and the reason in `history.yaml`.

**RULE 2**:

- When adding a Behavior that depends on a domain concept (e.g., Customer), that concept MUST exist in `domain/concepts/`.
- When a domain Rule or Workflow changes, the corresponding `00_agent_model/behaviors.yaml` entry MUST be optimized (`OPTIMIZE_PROMPT` or `MODIFY_BEHAVIOR`).

---

## 4. GOLDEN RULE (NEVER VIOLATE)

**MUST NOT**:

- Write one-off Python scripts for reusable Agent logic
- Keep important reasoning only in Cloud session memory
- Hide capability in undocumented helper functions
- Modify `src/agents/` code directly (except `generated/` folder)

**MUST**:

- Materialize capability into `00_agent_model/` YAML
- Use only the 13 approved operators
- Validate OKF and Agent & Domain Model schema after changes
- Ensure generated code compiles and executes with fixed local LLM

---

## 5. TARGET AGENT EXECUTION READINESS (MANDATORY CHECK)

The Target Agent MUST be able to execute **all** modeled behaviors using the fixed local LLM.

**Check**:

- `Agent Builder` can read all YAML files → no syntax errors
- Generated Python code imports without error
- Fixed local LLM can process the `prompt_template`
- No dependency on Cloud-only tools or hidden context

If any check fails, the task is **NOT** complete.

---

## 6. VALIDATION & REPLAY (MUST PASS)

For every Agent & Domain Model change, the following MUST pass:

1. `okf_validator.py .okf/00_nano_vllm --strict` → 0 errors
2. `python scripts/generate_local_agent.py .okf/00_nano_vllm/04_agent_model/` → success
3. `pytest tests/test_generated_agent.py` (if exists) → all pass
4. `python src/tools/okf_link_check.py . -m <affected_module>` → no broken links

---

## 7. DONE DECLARATION (ALL 7 MUST BE TRUE)

A task is **NOT DONE** merely because the requested KB change works.

The task is **DONE ONLY** when:

```
[ ] 1. OKF Change Validated (okf_validator --strict passes)
[ ] 2. Reusable Capability Identified (or explicitly declared NONE)
[ ] 3. Agent & Domain Model Updated (04_agent_model/ files changed)
[ ] 4. Evolution Operator Recorded (history.yaml updated with operator and reason)
[ ] 5. Target Agent Executable (generated code compiles and runs with local LLM)
[ ] 6. Replay/Evaluation Passed (link_check, tests, or simulations pass)
[ ] 7. Evolution Record Persisted (model_version incremented, history written)
```

---

## 8. REQUIRED COMPLETION REPORT (MUST OUTPUT BEFORE DONE)

Before declaring the task complete, the Cloud Agent MUST output:

```
[OK] 1. OKF Change Validated: <result>
[OK] 2. Reusable Capability Identified: <capability or NONE>
[OK] 3. Agent & Domain Model Updated: <files>
[OK] 4. Evolution Operators: <list>
[OK] 5. Target Agent Build: <success/fail>
[OK] 6. Replay/Evaluation: <pass/fail>
[OK] 7. Evolution Record: <version>

🎯 TASK COMPLETE
```

If any item is `[FAIL]`, **do NOT** declare the task complete.

---

## 9. CLOUD → LOCAL CONTRACT (IMMUTABLE)

The Target Agent MUST NOT depend on:

- Cloud Agent session memory
- Hidden Cloud reasoning
- Temporary files
- Cloud-only tools or prompts
- Undocumented scripts

**Rule**: If the Target Agent needs it, it MUST exist in one of:

- OKF Knowledge
- Agent & Domain Model
- Generated ActiveGraph runtime
- Target-Agent configuration

**Additional Rules**:

- If the Target Agent must execute a business rule (e.g., escalation policy), that rule MUST exist explicitly in `10_domain/rules/`.
- If the Target Agent must reason about a domain concept (e.g., Customer, Order), that concept MUST be defined in `10_domain/concepts/`.
- Temporary logic created by the Cloud Agent (e.g., "customer priority calculation") MUST NOT be transferred to the Target Agent unless modeled in `10_domain/rules/priority_rule.md`.

---

## 10. FUNDAMENTAL PRINCIPLE (READ ONCE, NEVER FORGET)

> **Cloud Agent session is temporary.**  
> **OKF bundle and Agent & Domain Model are the persistent source of truth.**  
> **Cloud designs and evolves. The bundle preserves. ActiveGraph compiles. The local LLM executes.**

**Therefore: Every reusable capability discovered during Cloud work MUST be materialized into the bundle-local ActiveGraph Agent & Domain Model or explicitly declared non-transferable.**