---

name: activegraph-okf-agent-modeler
description: >
Maintain an OKF bundle and evolve its embedded ActiveGraph Agent Model
so that every reusable capability discovered during Cloud Coding Agent work
can be reproduced and executed by a fixed-local-LLM Target Agent.
-----------------------------------------------------------------

# ActiveGraph OKF Agent Modeler

## 1. Purpose

This skill defines how a Cloud Coding Agent must work on an OKF knowledge bundle so that:

1. OKF knowledge can be added, modified, validated, and queried.
2. Reusable capabilities discovered during the work are explicitly modeled.
3. The corresponding ActiveGraph Agent Model is continuously updated.
4. The resulting Agent Model can be compiled into a Target Agent.
5. The Target Agent runs with the fixed local LLM.
6. Cloud-Agent-only reasoning or implementation is never implicitly required at runtime.

**Core principle:**

> Do not merely modify the OKF KB.
> Materialize every reusable capability discovered during the work into the ActiveGraph Agent Model.

The Cloud Coding Agent's implicit reasoning, temporary scripts, or session context are **not** part of the Target Agent unless explicitly materialized into the OKF bundle or its Agent Model.

---

# 2. Bundle-Local Agent Model

Every OKF bundle that has an associated Target Agent MUST contain its own Agent Model.

Example:

```text
.okf/
└── 00_nano_vllm/
    ├── 00_meta/
    ├── 01_modules/
    ├── 02_composite_concepts/
    ├── 03_atomic_concepts/
    └── 04_agent_model/
        ├── manifest.yaml
        ├── events.yaml
        ├── objects.yaml
        ├── edges.yaml
        ├── behaviors.yaml
        ├── guardrails.yaml
        ├── caching.yaml
        └── history.yaml
```

The Agent Model belongs to **this specific OKF bundle**.

It MUST NOT be treated as a global agent configuration unless explicitly designed as such.

The Agent Model may reference concepts, resources, and structures inside its parent bundle.

Example:

```yaml
behavior:
  id: query_schema
  knowledge_dependencies:
    - bundle: 00_nano_vllm
      concept: module
    - bundle: 00_nano_vllm
      concept: atomic_concept
```

This makes the relationship explicit:

```text
OKF Bundle
    │
    ├── Knowledge
    │
    └── Agent Model
          │
          ├── Events
          ├── Objects
          ├── Edges
          ├── Behaviors
          ├── Guardrails
          └── Caching
```

---

# 3. Knowledge vs Agent Model

Do not confuse knowledge with capability.

## Knowledge

Describes:

> What the system knows.

Stored primarily in the OKF knowledge structure.

Examples:

```text
Concept
Module
Document
Fact
Reference
Relationship
```

## Behavior

Describes:

> What the Target Agent does.

Examples:

```text
inspect_schema
retrieve_evidence
query_knowledge
validate_result
detect_contradiction
assemble_answer
```

## Control Model

Describes:

> When and under what conditions the behavior executes.

Represented through:

```text
Event
Typed Edge
Graph Pattern
Trigger
Guardrail
Policy
```

Therefore:

```text
              Agent
                │
       ┌────────┼────────┐
       ▼        ▼        ▼
   Knowledge  Procedure  Control
      │          │          │
     OKF      Behavior   Event/Edge
```

---

# 4. Capability Detection

Not every Cloud Coding Agent action requires an Agent Model change.

Before completing a task, determine:

> Did this task discover or create a reusable capability required by the Target Agent?

Examples of reusable capabilities:

* a repeatable OKF navigation procedure
* a query-planning procedure
* a schema-discovery procedure
* a validation procedure
* a contradiction-detection procedure
* a failure-recovery procedure
* a recurring reasoning pattern
* a new event-driven reaction
* a new guardrail
* a reusable caching strategy

If **NO**, complete the normal KB task.

If **YES**, the capability MUST be materialized into the Agent Model.

---

# 5. Golden Rule

**Never hide reusable Agent behavior inside arbitrary application code.**

Do NOT solve a reusable Agent capability only by:

```text
one-off Python script
temporary prompt
Cloud-Agent session context
hard-coded procedural logic
undocumented helper function
```

Instead:

```text
Observation
    ↓
Agent Model Change
    ↓
Evolution Operator
    ↓
Declarative Agent Model
    ↓
Generated ActiveGraph implementation
    ↓
Target-Agent validation
```

The declarative Agent Model is the source of truth.

Generated runtime code is derived from it.

---

# 6. Agent Model Files

## `manifest.yaml`

Contains:

```yaml
bundle: 00_nano_vllm
model_version: 1.0.0
agent_runtime: activegraph
local_llm: <fixed-model>
```

The version MUST be incremented when the Agent Model changes.

---

## `events.yaml`

Defines event types, payload semantics, and trigger conditions.

Example:

```yaml
events:
  - type: KNOWLEDGE_MODIFIED
    payload:
      path: string
      type: string
      frontmatter: object
      timestamp: string
    trigger: "OKF resource added or modified"
```

---

## `objects.yaml`

Defines Graph Node types and their properties.

Example:

```yaml
types:
  - name: Evidence
    properties:
      - source
      - concept_id
      - confidence
```

---

## `edges.yaml`

Defines typed graph relationships.

Example:

```yaml
edges:
  - name: SUPPORTS
    source_type: Evidence
    target_type: Claim
    condition: "evidence.confidence >= 0.8"
```

---

## `behaviors.yaml`

Defines Agent behaviors.

Each behavior SHOULD contain:

```yaml
- id: behavior.query_schema

  pattern: |
    MATCH ...

  triggers:
    - QUERY_REQUESTED

  inputs:
    - query

  outputs:
    - schema_context

  prompt_template: |
    ...

  knowledge_dependencies:
    - ...
```

---

## `guardrails.yaml`

Contains runtime constraints and validation rules.

---

## `caching.yaml`

Contains reusable caching strategies.

---

## `history.yaml`

Records every Agent Model evolution.

Example:

```yaml
- version: 1.2.0
  date: "2026-08-11"
  operators:
    - ADD_BEHAVIOR
    - REWIRE_BEHAVIOR
  reason: >
    Schema inspection was required before executing
    cross-concept queries.
  validation:
    status: passed
```

---

# 7. Approved Agent Model Evolution Operators

Cloud Coding Agent MUST use the following operators when modifying the Agent Model.

## 1. ADD_BEHAVIOR

Add a new reusable Agent behavior.

Required:

```text
id
pattern
prompt_template
triggers
```

Example:

```yaml
- id: behavior.query_schema
  pattern: |
    MATCH (c:Concept)
    WHERE c.requires_schema = true
    RETURN c
  triggers:
    - QUERY_REQUESTED
  prompt_template: |
    Inspect the relevant OKF schema before planning the query.
```

---

## 2. REMOVE_BEHAVIOR

Remove an obsolete behavior.

Before removal, verify that:

* no required behavior depends on it
* no active event triggers it
* no required workflow becomes disconnected

---

## 3. MODIFY_BEHAVIOR

Modify behavior attributes without changing its identity.

Possible changes:

```text
pattern
prompt_template
triggers
inputs
outputs
knowledge_dependencies
```

---

## 4. SPLIT_BEHAVIOR

Use when one behavior has multiple unrelated responsibilities.

Example:

```text
BEFORE

query_knowledge
    ├── schema discovery
    ├── query planning
    └── result validation
```

becomes:

```text
query_schema
query_planner
validate_query_result
```

The new behaviors MUST have narrower responsibilities and independently testable boundaries.

---

## 5. MERGE_BEHAVIORS

Merge behaviors when they have substantially overlapping responsibilities and independent execution provides no meaningful benefit.

The resulting behavior MUST preserve all required triggers, inputs, outputs, and guardrails.

---

## 6. ADD_EVENT_TYPE

Add an event when a meaningful state transition or externally observable occurrence must trigger one or more behaviors.

An event definition MUST specify:

```text
event name
issuer
payload
trigger condition
semantic meaning
```

Example:

```yaml
- type: KNOWLEDGE_MODIFIED
  issuer: Ingestor
  payload:
    path: string
    type: string
    timestamp: string
  trigger: "An OKF resource changes"
```

---

## 7. REMOVE_EVENT_TYPE

Remove an event only after verifying that:

* no active behavior requires it
* no guardrail depends on it
* no required workflow becomes disconnected

---

## 8. ADD_OBJECT_TYPE

Add a new Graph Node type when a concept requires independent state, identity, lifecycle, or relationship semantics inside the Agent runtime.

Define:

```text
name
properties
lifecycle
semantic purpose
```

Do not create an Object type merely because a piece of information exists.

---

## 9. MODIFY_EDGE

Modify the meaning or conditions of an existing typed edge.

Possible changes:

```text
source_type
target_type
condition
semantics
```

Changing an edge may alter multiple behaviors and MUST trigger regression validation.

---

## 10. REWIRE_BEHAVIOR

Change which events or graph patterns activate a behavior.

Examples:

```text
change subscribed event
change MATCH pattern
add trigger
remove trigger
change graph dependency
```

After rewiring, explicitly check for:

```text
cycles
dead behaviors
unreachable behaviors
unintended triggers
duplicate execution
```

---

## 11. ADD_GUARDRAIL

Add a rule that constrains behavior execution.

Example:

```yaml
- behavior_id: behavior.query_executor
  rule: "query_timeout < 30"
  action: "reject_and_log"
```

Guardrails SHOULD fail safely and SHOULD produce observable events/logs when violated.

---

## 12. OPTIMIZE_PROMPT

Optimize the prompt of an LLM-backed behavior.

Prompt changes MUST remain attached to the behavior identity.

Do not maintain important prompt knowledge only in Cloud Agent conversation history.

Record, where relevant:

```text
reason for change
failure evidence
expected improvement
validation result
```

---

## 13. ADD_CACHING

Add caching for repeated or expensive computation.

Define:

```text
behavior_id
strategy
TTL
key fields
invalidation events
```

Example:

```yaml
- behavior_id: behavior.query_schema
  strategy: ttl
  ttl_seconds: 3600
  key_fields:
    - bundle_version
    - query
  invalidation_events:
    - KNOWLEDGE_MODIFIED
```

Caching MUST NOT change the semantic result of the behavior.

---

# 8. Event and Behavior Design Principles

When deciding whether to add or modify an Event/Object/Behavior, prefer designs that maximize:

## Observability

Can the system determine what happened?

## Causality

Can the system determine why the behavior executed or failed?

## Controllability

Can the behavior be independently changed or replaced?

## Evaluability

Can its contribution be tested?

## Reusability

Can the same capability be applied to another task?

A good Agent Model is therefore not the one with the most objects or events.

It is the one that makes important capabilities:

```text
observable
causal
controllable
evaluatable
reusable
```

---

# 9. ActiveGraph Runtime Mapping

The declarative Agent Model is compiled into the ActiveGraph runtime.

Conceptually:

```text
events.yaml
objects.yaml
edges.yaml
behaviors.yaml
guardrails.yaml
caching.yaml
        │
        ▼
   Agent Builder
        │
        ▼
ActiveGraph Agent
        │
        ▼
Fixed Local LLM
```

Generated runtime code MUST be isolated from manually maintained code.

Preferred structure:

```text
src/
└── agents/
    ├── base/
    └── generated/
```

The generated Agent MUST be reproducible from the bundle-local Agent Model.

---

# 10. Cloud → Local Contract

The Cloud Coding Agent MUST assume that:

> The Target Agent has access only to explicitly persisted artifacts and the fixed local LLM.

Therefore the Target Agent MUST NOT depend on:

```text
Cloud Agent session memory
Cloud Agent hidden reasoning
temporary files
undocumented scripts
Cloud-only tools
Cloud-only prompts
```

If a capability is required by the Target Agent, it MUST exist in one or more of:

```text
OKF Knowledge
Agent Model
Generated ActiveGraph runtime
Target-Agent configuration
```

---

# 11. Validation and Replay

Whenever an Agent Model change is made:

1. Validate the OKF bundle.
2. Validate the Agent Model schema.
3. Generate the Target Agent.
4. Check for syntax/build errors.
5. Execute relevant Target-Agent tests.
6. Replay representative scenarios where available.
7. Compare behavior before and after the change.
8. Record the result in `history.yaml`.

Existing event logs SHOULD remain immutable and append-only so that important executions remain auditable and replayable.

---

# 12. Held-Out Evaluation Gate

Do not promote an Agent Model improvement solely because the Cloud Agent believes it is better.

For behavior-affecting changes:

```text
Candidate Agent
      │
      ▼
Validation Set
      │
      ▼
Held-Out Evaluation
      │
      ├── FAIL → reject/revise
      │
      └── PASS → promote
```

The evaluation SHOULD measure relevant dimensions such as:

```text
task success
answer correctness
evidence quality
robustness
latency
LLM calls
cost
failure rate
```

A change that improves one metric while seriously degrading another SHOULD NOT automatically replace the previous model.

---

# 13. Human Curator Boundary

The Cloud Coding Agent may propose and implement Agent Model changes, but high-impact structural changes MAY require human approval.

Examples:

```text
major ontology changes
destructive event removal
large topology changes
changes affecting multiple behaviors
changes that degrade held-out performance
```

The Agent Model history MUST preserve:

```text
what changed
why it changed
evidence
evaluation result
promotion/rejection status
```

---

# 14. Definition of Done

A task is **NOT DONE** merely because the requested OKF modification works.

The task is DONE only when all applicable conditions below are satisfied.

### 1. OKF Change Validated

The requested OKF change is correctly reflected in the target bundle and passes the available strict validation.

### 2. Reusable Capability Identified

The Cloud Agent explicitly determines whether the work discovered or modified a reusable capability.

If yes, that capability MUST be modeled.

### 3. Agent Model Updated

The corresponding files under:

```text
<bundle>/04_agent_model/
```

are updated.

### 4. Evolution Operator Recorded

Every Agent Model modification is represented by one or more approved operators:

```text
ADD_BEHAVIOR
REMOVE_BEHAVIOR
MODIFY_BEHAVIOR
SPLIT_BEHAVIOR
MERGE_BEHAVIORS
ADD_EVENT_TYPE
REMOVE_EVENT_TYPE
ADD_OBJECT_TYPE
MODIFY_EDGE
REWIRE_BEHAVIOR
ADD_GUARDRAIL
OPTIMIZE_PROMPT
ADD_CACHING
```

### 5. Target-Agent Readiness

The current Agent Model can be compiled into a runnable ActiveGraph Agent that uses the fixed local LLM.

### 6. Replay / Evaluation Passed

Relevant tests, replay scenarios, link validation, or simulations pass.

### 7. Evolution Record Persisted

The Agent Model version is incremented and the change is recorded in:

```text
history.yaml
```

---

# 15. Required Completion Report

Before declaring the task complete, the Cloud Coding Agent MUST report:

```text
[OK] OKF Change Validated
[OK] Reusable Capability Identified: <capability or NONE>
[OK] Agent Model Updated: <files>
[OK] Evolution Operators:
    - <operator>
    - <operator>
[OK] Target Agent Build/Generation: <result>
[OK] Replay/Evaluation: <result>
[OK] Evolution Record: <version>

TASK COMPLETE
```

If any required item fails, do NOT declare the task complete.

---

# 16. Fundamental Design Rule

The purpose of this skill is not merely to make the Cloud Coding Agent better at editing an OKF KB.

Its purpose is to maintain a continuous path:

```text
Cloud Coding Work
       │
       ▼
OKF Knowledge
       │
       +
       │
Reusable Capability
       │
       ▼
Bundle-Local Agent Model
       │
       ▼
ActiveGraph Compilation
       │
       ▼
Fixed Local LLM
       │
       ▼
Target Agent
```

Therefore:

> **Every reusable capability discovered during Cloud Coding Agent work must either be materialized into the bundle-local ActiveGraph Agent Model or explicitly declared non-transferable.**

The Cloud Agent session is temporary.

The OKF bundle and its Agent Model are the persistent source of truth.

**Cloud designs and evolves.
The bundle preserves.
ActiveGraph compiles.
The local LLM executes.**
