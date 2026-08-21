"""OKF runtime behaviors (brick-agent-plan Phase 3).

Four behaviors registered with `@behavior`, following the canonical
`.okf/00_agent_model/events.yaml` vocabulary:

    ingest:  okf.ingest.requested
               → okf.parsed → okf.validated → okf.loaded
    lint:    okf.lint.requested
               → okf.analyzed → okf.linted
    ask:     okf.ask.requested
               → okf.context.assembled
               → llm.requested → llm.responded   (observability seam)
               → okf.answer.generated

The knowledge graph is loaded with the Phase 2 loader
(`src.core.targets.okf.taxonomy.load_knowledge_graph`) and cached per
`kb_id` (process-level side table — Python objects can't ride event
payloads, same indirection the SQL agent uses for Readers). The ask
chain satisfies the observability contract: `reader.answer()` is wrapped
in `llm.requested` / `llm.responded` events (model, prompt_hash,
latency_seconds).
"""

from __future__ import annotations

import threading
from typing import Any

from activegraph import behavior

from src.agents.okf import events as E
from src.core.agent.embedders import get_embedder
from src.core.targets.okf.action_space import OkfActionSpace
from src.core.targets.okf.prompt_transforms import render_okf_prompt
from src.core.targets.okf.taxonomy import (
    OkfKnowledgeGraph,
    lint_knowledge_graph,
)
from src.runtime import loader, observability
from src.runtime.reader_registry import (
    clear_reader as _clear_reader,
    get_reader as _get_reader,
    set_reader as _set_reader,
)

# ===========================================================================
# Per-kb indirection (KB snapshots can't ride event payloads)
# ===========================================================================

_KBS: dict[str, OkfKnowledgeGraph] = {}
_KB_LOCK = threading.Lock()


def _set_kb(kb_id: str, graph: OkfKnowledgeGraph) -> None:
    with _KB_LOCK:
        _KBS[kb_id] = graph


def _get_kb(kb_id: str) -> OkfKnowledgeGraph | None:
    with _KB_LOCK:
        return _KBS.get(kb_id)


def clear_kb(kb_id: str | None = None) -> None:
    """Drop cached KB snapshots. Test isolation only."""
    with _KB_LOCK:
        if kb_id is None:
            _KBS.clear()
        else:
            _KBS.pop(kb_id, None)


def _kb_for(payload: dict[str, Any]) -> OkfKnowledgeGraph | None:
    """KB snapshot for the payload's kb_id; loads + caches on demand
    from kb_path / okf_source when the ingest chain hasn't run."""
    kb_id = payload.get("kb_id", "")
    graph = _get_kb(kb_id)
    if graph is not None:
        return graph
    src = payload.get("kb_path") or payload.get("okf_source")
    if src:
        try:
            graph = loader.load_kb_graph(src)
        except Exception:  # noqa: BLE001 — caller records the failure
            return None
        _set_kb(kb_id, graph)
    return graph


def _emit_missing_kb(graph, etype: str, payload: dict[str, Any]) -> None:
    graph.emit(etype, {**payload, "error": "kb_missing: no graph for kb_id"})


# ===========================================================================
# 1) Ingest chain
# ===========================================================================


@behavior(name="okf_agent.ingest", on=[E.OKF_INGEST_REQUESTED])
def behavior_ingest(event, graph, ctx) -> None:  # noqa: ARG001
    """Parse → validate → load the OKF bundle into the graph.

    Reuses `load_knowledge_graph` (Phase 2; parses body relationship
    blocks) for the snapshot, runs the deterministic detectors as the
    validation step, then materializes concept/rule/schema objects and
    relations on the ActiveGraph."""
    payload = event.payload
    kb_id = payload.get("kb_id", "")
    src = payload.get("kb_path") or payload.get("okf_source")

    if not src:
        _emit_missing_kb(graph, E.OKF_PARSED, {"kb_id": kb_id, "n_documents": 0})
        _emit_missing_kb(graph, E.OKF_VALIDATED, {"kb_id": kb_id, "valid": False})
        _emit_missing_kb(graph, E.OKF_LOADED, {"kb_id": kb_id, "valid": False})
        return

    try:
        snapshot = loader.load_kb_graph(src)
    except Exception as e:  # noqa: BLE001 — failure recorded in payload
        err = f"{type(e).__name__}: {e}"
        graph.emit(E.OKF_PARSED, {"kb_id": kb_id, "n_documents": 0, "parse_errors": [{"error": err}]})
        graph.emit(E.OKF_VALIDATED, {"kb_id": kb_id, "valid": False, "validation_errors": [{"error": err}]})
        graph.emit(E.OKF_LOADED, {"kb_id": kb_id, "valid": False, "error": err, "n_concepts": 0, "n_rules": 0, "n_schema_tables": 0, "n_relations": 0, "n_objects": 0})
        return

    _set_kb(kb_id, snapshot)

    graph.emit(E.OKF_PARSED, {
        "kb_id": kb_id,
        "n_documents": len(snapshot.nodes),
        "parse_errors": [],
    })

    issues = lint_knowledge_graph(snapshot)
    graph.emit(E.OKF_VALIDATED, {
        "kb_id": kb_id,
        "valid": len(issues) == 0,
        "validation_errors": list(issues),
    })

    n_objects, n_relations = loader.populate_graph(graph, snapshot)
    graph.emit(E.OKF_LOADED, {
        "kb_id": kb_id,
        "n_concepts": len(snapshot.concept_ids),
        "n_rules": len(snapshot.rule_ids),
        "n_schema_tables": len(snapshot.schema_ids),
        "n_relations": n_relations,
        "n_objects": n_objects,
        "valid": len(issues) == 0,
        "error": "",
    })


# ===========================================================================
# 2) Lint chain
# ===========================================================================


@behavior(name="okf_agent.lint", on=[E.OKF_LINT_REQUESTED])
def behavior_lint(event, graph, ctx) -> None:  # noqa: ARG001
    """Run the four deterministic OKF detectors and emit the issue list."""
    payload = event.payload
    kb_id = payload.get("kb_id", "")
    snapshot = _kb_for(payload)
    if snapshot is None:
        _emit_missing_kb(graph, E.OKF_ANALYZED, {"kb_id": kb_id})
        _emit_missing_kb(graph, E.OKF_LINTED, {"kb_id": kb_id, "valid": False})
        return

    stats = {
        "n_concepts": len(snapshot.concept_ids),
        "n_rules": len(snapshot.rule_ids),
        "n_schema": len(snapshot.schema_ids),
        "n_nodes": len(snapshot.nodes),
    }
    graph.emit(E.OKF_ANALYZED, {"kb_id": kb_id, "stats": stats})

    issues = lint_knowledge_graph(snapshot)
    graph.emit(E.OKF_LINTED, {
        "kb_id": kb_id,
        "valid": len(issues) == 0,
        "n_errors": len(issues),
        "n_warnings": 0,
        "issues": list(issues),
    })


# ===========================================================================
# 3) Ask chain
# ===========================================================================


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


@behavior(name="okf_agent.assemble_context", on=[E.OKF_ASK_REQUESTED])
def behavior_assemble_context(event, graph, ctx) -> None:  # noqa: ARG001
    """Embed the question, retrieve top-K concepts, expand the concept
    tree (parents/children), select trigger-matched rules and the schema
    objects they reference, then emit structured context_parts."""
    payload = event.payload
    request_id = payload.get("request_id", "")
    question = payload.get("question", "")
    kb_id = payload.get("kb_id", "")
    top_k = int(payload.get("top_k", 10))

    snapshot = _kb_for(payload)
    if snapshot is None:
        _emit_missing_kb(graph, E.OKF_CONTEXT_ASSEMBLED, {
            "request_id": request_id,
            "question": question,
            "context_parts": {"concepts": [], "rules": [], "schema": [], "question": question},
        })
        return

    concepts = snapshot.concept_ids
    embedder = get_embedder()
    texts = [question] + [
        f"{c} {snapshot.nodes[c].title} {' '.join(sorted(snapshot.nodes[c].triggers))}"
        for c in concepts
    ]
    vecs = embedder.embed(texts)
    q_vec = vecs[0]
    scores = {c: _cosine(q_vec, v) for c, v in zip(concepts, vecs[1:])}
    ranked = sorted(scores, key=lambda c: -scores[c])
    selected = ranked[:top_k]

    # Expand the concept tree: prerequisites (parents) + prerequisite_of
    # (children) as raw ids; the prompt pipeline tags them (parent)/(child).
    expanded: list[str] = []
    for c in selected:
        n = snapshot.nodes[c]
        if c not in expanded:
            expanded.append(c)
        for p in sorted(n.prerequisites):
            if p not in expanded:
                expanded.append(p)
        for ch in sorted(n.prerequisite_of):
            if ch not in expanded:
                expanded.append(ch)

    # Rules: normalized trigger phrase appears in the question.
    q_lower = question.lower()
    rules = [
        rid for rid in snapshot.rule_ids
        if any(t and t in q_lower for t in snapshot.nodes[rid].triggers)
    ]

    # Schema: referenced by the expanded concepts / selected rules.
    used_ids = set(expanded) | set(rules)
    schema_ids: set[str] = set()
    for nid in used_ids:
        for ref in snapshot.nodes[nid].schema_refs:
            full = snapshot.resolve(ref)
            if full is not None and full in snapshot.schema_ids:
                schema_ids.add(full)

    context_parts = {
        "concepts": expanded,
        "rules": rules,
        "schema": sorted(schema_ids),
        "question": question,
    }

    graph.emit(E.OKF_CONTEXT_ASSEMBLED, {
        "request_id": request_id,
        "question": question,
        "kb_id": kb_id,
        "context_parts": context_parts,
        "selected_concept_ids": selected,
        "expanded_concept_ids": [c for c in expanded if c not in selected],
        "scorer_model": embedder.model,
        "scores": {c: round(scores[c], 6) for c in ranked[:top_k]},
        "applied_context_strategy": "embed_topk+tree_expand+trigger_rules",
        "error": "",
    })


@behavior(name="okf_agent.generate_answer", on=[E.OKF_CONTEXT_ASSEMBLED])
def behavior_generate_answer(event, graph, ctx) -> None:  # noqa: ARG001
    """Run the OKF ask pipeline (build_prompt_parts → render), call the
    Reader inside the llm.requested / llm.responded observability seam,
    and emit the grounded answer."""
    payload = event.payload
    request_id = payload.get("request_id", "")
    question = payload.get("question", "")
    context_parts = dict(payload.get("context_parts", {}) or {})
    kb_id = payload.get("kb_id", "")
    error = str(payload.get("error", "") or "")
    answer = ""
    applied: tuple[str, ...] = ()

    if not error:
        snapshot = _get_kb(kb_id)
        aspace = OkfActionSpace(knowledge_graph=snapshot)
        try:
            prompt = aspace.build_prompt_parts(question, context_parts)
            rendered = render_okf_prompt(prompt.parts)
            applied = prompt.applied_transforms

            reader = _get_reader(request_id)
            if reader is None:
                error = "reader_missing: no Reader registered for request_id"
            else:
                answer, error, _latency, _req, _resp = observability.ask_with_observability(
                    graph,
                    reader,
                    request_id=request_id,
                    question=question,
                    context=rendered,
                )
        except Exception as e:  # noqa: BLE001 — ask pipeline failure
            error = f"{type(e).__name__}: {e}"

    if not isinstance(answer, str):
        answer = str(answer or "")

    graph.emit(E.OKF_ANSWER_GENERATED, {
        "request_id": request_id,
        "answer": answer,
        "context_parts": context_parts,
        "applied_transforms": list(applied),
        "error": error,
    })
