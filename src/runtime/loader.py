"""Runtime graph loader — OKF bundle → OkfKnowledgeGraph → ActiveGraph.

Phase 2's `load_knowledge_graph` (body relationship parser 포함) is the
single parsing entry; this module adds the ActiveGraph materialization
(concept/rule/schema objects + relations) and a session builder for the
router / behaviors.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from activegraph import FrozenClock, Graph, IDGen

from src.core.targets.okf.taxonomy import OkfKnowledgeGraph, load_knowledge_graph


def load_kb_graph(kb_path: str | Path) -> OkfKnowledgeGraph:
    """Load one OKF bundle into an OkfKnowledgeGraph snapshot."""
    return load_knowledge_graph(kb_path)


def populate_graph(graph: Graph, snapshot: OkfKnowledgeGraph) -> tuple[int, int]:
    """Materialize snapshot nodes/edges as ActiveGraph objects/relations.

    Relations derived deterministically from the snapshot:
      - `concept_child`           : prerequisite → concept (parent),
                                    composite → parts,
                                    concept → prerequisite_of dependents
      - `maps_to`                 : concept → schema node (schema_refs)
      - `rule_references_schema`  : rule → schema node (schema_refs)
      - `contradicts`             : concept → concept
    Returns (n_objects, n_relations).
    """
    node_obj: dict[str, str] = {}
    for nid, n in sorted(snapshot.nodes.items()):
        if n.is_schema:
            obj_type = "schema_column" if "column" in n.type.lower() else "schema_table"
        elif n.is_rule:
            obj_type = "rule"
        elif n.is_concept:
            obj_type = "concept"
        else:
            obj_type = "kb_node"
        o = graph.add_object(type=obj_type, data={
            "id": n.id,
            "title": n.title,
            "status": n.status,
            "path": n.path,
            "type": n.type,
        })
        node_obj[nid] = o.id

    n_relations = 0
    for nid, n in sorted(snapshot.nodes.items()):
        src = node_obj.get(nid)
        if src is None:
            continue
        if n.is_concept:
            for p in sorted(n.prerequisites):
                dst = node_obj.get(p)
                if dst:
                    graph.add_relation(source=dst, target=src, type="concept_child",
                                       data={"from": p, "to": nid})
                    n_relations += 1
            for c in sorted(n.composed_of):
                dst = node_obj.get(c)
                if dst:
                    graph.add_relation(source=src, target=dst, type="concept_child",
                                       data={"from": nid, "to": c})
                    n_relations += 1
            for ch in sorted(n.prerequisite_of):
                dst = node_obj.get(ch)
                if dst:
                    graph.add_relation(source=src, target=dst, type="concept_child",
                                       data={"from": nid, "to": ch})
                    n_relations += 1
        for ref in sorted(n.schema_refs):
            full = snapshot.resolve(ref)
            dst = node_obj.get(full) if full is not None else None
            if dst is None:
                continue
            rel_type = "maps_to" if n.is_concept else ("rule_references_schema" if n.is_rule else "references")
            graph.add_relation(source=src, target=dst, type=rel_type,
                               data={"from": nid, "to": ref})
            n_relations += 1
        for x in sorted(n.contradicts):
            dst = node_obj.get(x)
            if dst:
                graph.add_relation(source=src, target=dst, type="contradicts",
                                   data={"from": nid, "to": x})
                n_relations += 1
    return len(node_obj), n_relations


@dataclass
class SessionGraph:
    """A ready-to-run ActiveGraph session for one knowledge base."""

    graph: Graph
    snapshot: OkfKnowledgeGraph
    n_objects: int
    n_relations: int


def build_session(
    kb_path: str | Path,
    *,
    run_id: str = "brick-runtime-session",
    frozen_t: str = "2026-01-01T00:00:00Z",
) -> SessionGraph:
    """Load a KB and populate a fresh ActiveGraph graph with its
    objects/relations — the initial graph for a session (router / CLI)."""
    graph = Graph(ids=IDGen(), clock=FrozenClock(frozen_t), run_id=run_id)
    snapshot = load_kb_graph(kb_path)
    n_objects, n_relations = populate_graph(graph, snapshot)
    return SessionGraph(
        graph=graph,
        snapshot=snapshot,
        n_objects=n_objects,
        n_relations=n_relations,
    )
