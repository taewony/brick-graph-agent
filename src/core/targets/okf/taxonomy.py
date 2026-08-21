"""OKF knowledge graph model + structural-integrity detectors + taxonomy.

Mirrors the SQL target's structure (`src.core.targets.sql.taxonomy`) at
the taxonomy level — a `RegimeTaxonomy` implementation whose
`classify(outcome)` maps an `OkfOutcome` to a named regime — but the OKF
detector story is two-layered:

  1. KB-level detectors (the task-named functions `concept_orphan_detector`,
     `rule_schema_reference_detector`, `ambiguous_trigger_detector`,
     `cyclic_concept_detector`). Each is PURE over an `OkfKnowledgeGraph`
     and returns a list of lint signals
     `[{"code": ..., "node": ..., "detail": ...}, ...]`. These are the
     signals the OKF eval backend records as `OkfOutcome.lint_errors`.

  2. Per-outcome regime detectors (`detect_*` below). PURE over an
     `OkfOutcome` — they read the lint codes the eval backend attached,
     exactly like SQL detectors read the structural fields the SQL eval
     attached. `OkfTaxonomy.classify` walks these in priority order.

Why the two layers? The mentor loop classifies per-question outcomes and
can never re-read the KB (no I/O in the loop), so the KB-level signals
must be materialized onto each outcome by the eval backend before
diagnose runs. The KB-level detectors stay the single source of truth
for what an integrity issue IS; the eval backend just fans their output
onto the outcomes that touched the affected nodes.

The graph loader (`load_knowledge_graph`) parses OKF bundles the way the
existing `src/okf/validator.py` does, with one important difference:
relationship blocks in real bundles (e.g. `.okf/01_nano_vllm`) live in
the BODY (`- **PREREQUISITES**: ...`) rather than the frontmatter, and
appear in several shapes (inline backticked lists, bare comma lists,
multi-line indented backtick items, `(explanation)` suffixes, dotted
ids).
The validator only reads frontmatter relationships, so this loader has
its own small body parser that tolerates all those shapes.
"""

from __future__ import annotations

import re
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from src.core.loop.regimes import Regime, HistogramRow
from src.core.targets.okf.outcome import OkfLintError, OkfOutcome


# ---------------------------------------------------------------------------
# Graph model
# ---------------------------------------------------------------------------


def _normalize_trigger(text: str) -> str:
    """Deterministic trigger-phrase key: lowercase, keep [a-z0-9] and
    Hangul, drop everything else. Two concepts sharing a normalized
    phrase are ambiguous retrievers."""
    return re.sub(r"[^a-z0-9\uac00-\ud7af]+", "", (text or "").lower())


def _is_concept_type(t: str) -> bool:
    return "concept" in (t or "").lower()


def _is_rule_type(t: str) -> bool:
    return "rule" in (t or "").lower()


def _is_schema_type(t: str) -> bool:
    return any(k in (t or "").lower() for k in ("table", "column", "schema"))


@dataclass
class OkfNode:
    """One node of an OKF bundle.

    `prerequisites` / `composed_of` / `prerequisite_of` / `contradicts`
    hold RESOLVED node ids (see `OkfKnowledgeGraph.resolve`). The RAW
    references are preserved in `raw_relationships` so the audit
    detectors can flag dangling refs, and non-id prose captured from
    malformed relationship blocks lives in `relationship_noise`. `evidence`
    carries the frontmatter `sources` resource URLs (or empty when the
    node makes claims without any evidence document)."""

    id: str
    stem: str
    type: str
    title: str
    status: str
    path: str
    prerequisites: frozenset[str] = frozenset()
    composed_of: frozenset[str] = frozenset()
    prerequisite_of: frozenset[str] = frozenset()
    contradicts: frozenset[str] = frozenset()
    triggers: frozenset[str] = frozenset()
    schema_refs: frozenset[str] = frozenset()
    # Raw (unresolved) relationship references + non-id noise + evidence.
    raw_relationships: dict[str, frozenset[str]] = field(default_factory=dict)
    relationship_noise: dict[str, tuple[str, ...]] = field(default_factory=dict)
    evidence: frozenset[str] = frozenset()

    @property
    def is_concept(self) -> bool:
        return _is_concept_type(self.type)

    @property
    def is_rule(self) -> bool:
        return _is_rule_type(self.type)

    @property
    def is_schema(self) -> bool:
        return _is_schema_type(self.type)


@dataclass
class OkfKnowledgeGraph:
    """In-memory snapshot of one OKF bundle. `nodes` is keyed by full
    node id (frontmatter `id`, else file stem); `by_stem` maps bare
    stems to full ids so body references like `` `kv_cache` `` resolve
    to `atomic.kv_cache`."""

    name: str
    nodes: dict[str, OkfNode] = field(default_factory=dict)
    by_stem: dict[str, str] = field(default_factory=dict)

    def resolve(self, name: str) -> str | None:
        """Full node id for a reference, or None if it doesn't resolve."""
        name = (name or "").strip().strip("`")
        if not name:
            return None
        if name in self.nodes:
            return name
        if name in self.by_stem:
            return self.by_stem[name]
        # Last-segment fallback: "atomic.kv_cache" -> "kv_cache" stem.
        tail = name.split(".")[-1]
        if tail and tail in self.by_stem:
            return self.by_stem[tail]
        return None

    @property
    def concept_ids(self) -> tuple[str, ...]:
        return tuple(sorted(nid for nid, n in self.nodes.items() if n.is_concept))

    @property
    def rule_ids(self) -> tuple[str, ...]:
        return tuple(sorted(nid for nid, n in self.nodes.items() if n.is_rule))

    @property
    def schema_ids(self) -> tuple[str, ...]:
        return tuple(sorted(nid for nid, n in self.nodes.items() if n.is_schema))

    def snapshot(self) -> dict[str, Any]:
        """JSON-safe view of the graph for prompt-transform `kb_meta`."""
        nodes = {
            nid: {
                "id": n.id,
                "type": n.type,
                "title": n.title,
                "prerequisites": sorted(n.prerequisites),
                "composed_of": sorted(n.composed_of),
                "prerequisite_of": sorted(n.prerequisite_of),
                "triggers": sorted(n.triggers),
                "schema_refs": sorted(n.schema_refs),
            }
            for nid, n in self.nodes.items()
        }
        return {"name": self.name, "nodes": nodes, "by_stem": dict(self.by_stem)}


# ---------------------------------------------------------------------------
# Bundle loading
# ---------------------------------------------------------------------------

# Relationship header: "- **PREREQUISITES**:" (optionally with inline ids).
# Names may contain spaces (e.g. "SYNERGY WITH") so indented items under
# such blocks attach to the right relation instead of the previous one.
_REL_HEADER = re.compile(r"^\s*-\s*\*\*([A-Z][A-Z0-9_ ]*)\*\*:?(.*)$")
# Relationship item line (indented): "  - `inference_only`"
_REL_ITEM = re.compile(r"^\s+-\s+(.+)$")
# A plausible node-id token: no whitespace, starts alnum.
_ID_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$")
# A markdown link used as a relationship item: `[id](target)`.
# Non-anchored: scanned anywhere in the chunk (links may follow other text).
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")

# Frontmatter relationship keys (validator-compatible) merged into the
# body-derived relationship map.
_FM_REL_KEYS = ("prerequisites", "composed_of", "prerequisite_of", "contradicts")
_FM_ALIAS_KEYS = ("aliases", "triggers")
_FM_SCHEMA_REF_KEYS = ("schema_refs", "maps_to", "maps_to_schema", "references_schema")

# Body markers for rule -> schema references.
_BODY_SCHEMA_REF = re.compile(r"\b(?:schema|maps_to_column|references_schema)\s*[:=]\s*([\w.]+)")


def _strip_parentheticals(text: str) -> str:
    """Remove balanced (...) groups — including commas inside them, which
    naive comma-splitting would corrupt (e.g. "`dynamic_batcher`
    (Composite Concept, Module 03)")."""
    out: list[str] = []
    depth = 0
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif depth == 0:
            out.append(ch)
    return "".join(out)


def _extract_ids(remainder: str) -> tuple[list[str], list[str]]:
    """Extract plausible node-id tokens from a relationship header
    remainder or an item line.

    Accepts backticked ids, bare ids, comma-separated lists,
    parenthetical-suffixed forms, and MARKDOWN LINKS (`[id](target)`)
    — the id is the link text. Links are scanned FIRST (their targets
    contain parentheses that parenthetical-stripping would destroy);
    parentheticals are then stripped from the gaps between links before
    comma-splitting, so a trailing explanation like "(Module 05, 블록
    할당 풀)" cannot leak a comma-split noise token. Returns (ids,
    noise): `noise` holds tokens that don't look like node ids (prose
    paragraphs, frontmatter snippets, mixed text)."""
    ids: list[str] = []
    noise: list[str] = []

    def _bare(tok: str) -> None:
        tok = tok.strip().strip("`").strip()
        if not tok:
            return
        if _ID_TOKEN_RE.match(tok):
            ids.append(tok)
        else:
            noise.append(tok)

    def _gap(text: str) -> None:
        for tok in _strip_parentheticals(text).split(","):
            _bare(tok)

    pos = 0
    for lm in _MD_LINK_RE.finditer(remainder or ""):
        _gap(remainder[pos:lm.start()])
        link_text = lm.group(0)
        inner = link_text[1:link_text.find("]")].strip().strip("`").strip()
        if inner and _ID_TOKEN_RE.match(inner):
            ids.append(inner)
        elif inner:
            noise.append(inner)
        pos = lm.end()
    _gap((remainder or "")[pos:])
    return ids, noise


def _parse_relationships(body: str) -> tuple[dict[str, set[str]], dict[str, list[str]]]:
    """Parse `- **RELATION**: ...` blocks from a concept body.

    Returns (rels, noise): `rels` = {RELATION_NAME: {raw id, ...}},
    `noise` = {RELATION_NAME: [non-id item, ...]} for the four supported
    relations."""
    rels: dict[str, set[str]] = {}
    noise: dict[str, list[str]] = {}
    current: str | None = None
    for raw in body.splitlines():
        m = _REL_HEADER.match(raw)
        if m:
            current = m.group(1)
            rest = m.group(2)
            if rest.strip():
                ids, bad = _extract_ids(rest)
                rels.setdefault(current, set()).update(ids)
                if bad:
                    noise.setdefault(current, []).extend(bad)
            continue
        m = _REL_ITEM.match(raw)
        if m and current is not None:
            ids, bad = _extract_ids(m.group(1))
            rels.setdefault(current, set()).update(ids)
            if bad:
                noise.setdefault(current, []).extend(bad)
        elif raw.strip() and current is not None:
            # A non-item, non-blank line (heading, prose paragraph, table)
            # terminates the current relationship block — otherwise an
            # indented bullet list anywhere later in the file would be
            # swallowed into the last relationship header.
            current = None
    return rels, noise


def _parse_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """Split OKF markdown into (frontmatter dict, body). Returns
    ({}, content) when there's no well-formed YAML block."""
    if not content.startswith("---"):
        return {}, content
    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}, content
    try:
        import yaml

        fm = yaml.safe_load(parts[1]) or {}
    except Exception:  # noqa: BLE001 — malformed YAML degrades to empty fm
        return {}, parts[2].strip()
    return fm, parts[2].strip()


def _fm_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


def _source_resources(fm: dict[str, Any]) -> list[str]:
    """Resource URLs from frontmatter `sources` — the evidence links.

    `sources` is normally a list of {id, resource, title} dicts (OKF
    v0.2); bare strings are accepted too."""
    out: list[str] = []
    raw = fm.get("sources")
    if raw is None:
        return out
    items = raw if isinstance(raw, list) else [raw]
    for s in items:
        if isinstance(s, dict):
            res = s.get("resource")
            if res:
                out.append(str(res))
        elif isinstance(s, str) and s.strip():
            out.append(s.strip())
    return out


def load_knowledge_graph(kb_root: str | Path) -> OkfKnowledgeGraph:
    """Load one OKF bundle directory into an `OkfKnowledgeGraph`.

    Two passes: (1) parse every `*.md` under `kb_root` into nodes keyed
    by full id, (2) resolve body/frontmatter relationship references to
    full ids via the id/stem map. Unresolvable references are dropped
    from the resolved sets (the node keeps its own id, so it can still
    be flagged as an orphan)."""
    root = Path(kb_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"OKF knowledge base directory not found: {root}")
    graph = OkfKnowledgeGraph(name=str(root))

    # Pass 1: parse files.
    rels_by_id: dict[str, dict[str, set[str]]] = {}
    for p in sorted(root.rglob("*.md")):
        content = p.read_text(encoding="utf-8")
        fm, body = _parse_frontmatter(content)
        rels, noise = _parse_relationships(body)
        rel_path = p.relative_to(root)
        node_id = str(fm.get("id") or p.stem)
        stem = p.stem
        if node_id in graph.nodes:
            continue  # id collision — first file wins (validator flags it)
        # Merge frontmatter relationship fields into the body-derived map.
        # Frontmatter values are cleaned the same way as body items
        # (parentheticals stripped, backticks removed, comma lists split).
        for k in _FM_REL_KEYS:
            if k in fm:
                ids, _ = _extract_ids(", ".join(_fm_str_list(fm.get(k))))
                rels.setdefault(k.upper(), set()).update(ids)
        rels_by_id[node_id] = rels
        triggers = {_normalize_trigger(str(fm.get("title", stem)))}
        triggers.update(_normalize_trigger(a) for a in _fm_str_list(fm.get("aliases")))
        triggers.update(_normalize_trigger(a) for a in _fm_str_list(fm.get("triggers")))
        schema_refs: set[str] = set()
        for k in _FM_SCHEMA_REF_KEYS:
            ids, _ = _extract_ids(", ".join(_fm_str_list(fm.get(k))))
            schema_refs.update(ids)
        schema_refs.update(_BODY_SCHEMA_REF.findall(body))
        graph.nodes[node_id] = OkfNode(
            id=node_id,
            stem=stem,
            type=str(fm.get("type", "")),
            title=str(fm.get("title", stem)),
            status=str(fm.get("status", "")),
            path=str(rel_path).replace("\\", "/"),
            triggers=frozenset(t for t in triggers if t),
            schema_refs=frozenset(schema_refs),
            raw_relationships={k: frozenset(v) for k, v in rels.items()},
            relationship_noise={k: tuple(v) for k, v in noise.items()},
            evidence=frozenset(_source_resources(fm)),
        )
        if stem not in graph.by_stem:
            graph.by_stem[stem] = node_id

    # Pass 2: resolve relationship references.
    def _resolve_set(raw_ids: set[str]) -> frozenset[str]:
        resolved = set()
        for rid in raw_ids:
            full = graph.resolve(rid)
            if full is not None:
                resolved.add(full)
        return frozenset(resolved)

    for nid, n in graph.nodes.items():
        rels = rels_by_id.get(nid, {})
        n.prerequisites = _resolve_set(rels.get("PREREQUISITES", set()))
        n.composed_of = _resolve_set(rels.get("COMPOSED_OF", set()))
        n.prerequisite_of = _resolve_set(rels.get("PREREQUISITE_OF", set()))
        n.contradicts = _resolve_set(rels.get("CONTRADICTS", set()))
    return graph


# ---------------------------------------------------------------------------
# KB-level detectors — the task-named structural-integrity signals
# ---------------------------------------------------------------------------


def concept_orphan_detector(graph: OkfKnowledgeGraph) -> list[OkfLintError]:
    """Concepts with no parent/child relationships of their own AND not
    referenced by any other node's relationship lists. Such concepts are
    unreachable from the KB's navigation and can never be grounded by a
    tree-expansion transform."""
    referenced: set[str] = set()
    for n in graph.nodes.values():
        referenced |= n.prerequisites | n.composed_of | n.prerequisite_of
    issues: list[OkfLintError] = []
    for nid, n in graph.nodes.items():
        if not n.is_concept:
            continue
        if not (n.prerequisites or n.composed_of or n.prerequisite_of) and nid not in referenced:
            issues.append({
                "code": "concept_orphan",
                "node": nid,
                "detail": (
                    f"concept {nid!r} has no parent/child relationships "
                    "and is not referenced by any other node"
                ),
            })
    return issues


def rule_schema_reference_detector(graph: OkfKnowledgeGraph) -> list[OkfLintError]:
    """Rules whose schema references (frontmatter `schema_refs`/
    `maps_to`/`references_schema`, or body `schema:` mentions) do not
    resolve to a schema-table/schema-column node. A rule that points at
    a non-existent schema cell can never be grounded in the schema the
    ask pipeline injects."""
    schema_ids = set(graph.schema_ids)
    issues: list[OkfLintError] = []
    for nid, n in graph.nodes.items():
        if not n.is_rule:
            continue
        for ref in sorted(n.schema_refs):
            full = graph.resolve(ref)
            if full is None or full not in schema_ids:
                issues.append({
                    "code": "rule_schema_reference",
                    "node": nid,
                    "detail": (
                        f"rule {nid!r} references schema {ref!r} which "
                        "does not resolve to a schema node"
                    ),
                })
    return issues


def ambiguous_trigger_detector(graph: OkfKnowledgeGraph) -> list[OkfLintError]:
    """Trigger phrases (normalized titles/aliases) owned by more than
    one concept. Retrieval keyed on such a phrase cannot tell the
    concepts apart, so the ask pipeline has no way to pick the right
    one without disambiguation."""
    by_trigger: dict[str, list[str]] = defaultdict(list)
    for nid, n in graph.nodes.items():
        if not n.is_concept:
            continue
        for t in n.triggers:
            by_trigger[t].append(nid)
    issues: list[OkfLintError] = []
    for t in sorted(by_trigger):
        owners = sorted(set(by_trigger[t]))
        if len(owners) > 1:
            issues.append({
                "code": "ambiguous_trigger",
                "node": owners[0],
                "detail": (
                    f"trigger phrase {t!r} matches multiple concepts: "
                    + ", ".join(owners)
                ),
            })
    return issues


def _concept_edges(graph: OkfKnowledgeGraph) -> dict[str, set[str]]:
    """Directed edges over concept ids:
      - X.prerequisites ∋ Y  ->  Y -> X   (Y must come before X)
      - X.composed_of ∋ Y    ->  Y -> X   (parts before composite)
      - X.prerequisite_of ∋ Z ->  X -> Z   (explicit inverse)
    """
    edges: dict[str, set[str]] = defaultdict(set)
    for n in graph.nodes.values():
        for prereq in n.prerequisites:
            edges[prereq].add(n.id)
        for part in n.composed_of:
            edges[part].add(n.id)
        for dep in n.prerequisite_of:
            edges[n.id].add(dep)
    return edges


def cyclic_concept_detector(graph: OkfKnowledgeGraph) -> list[OkfLintError]:
    """Cycles in the concept prerequisite/composition DAG. A cycle means
    no topological study order exists — concept-tree expansion would
    loop forever."""
    edges = _concept_edges(graph)
    issues: list[OkfLintError] = []
    visited: set[str] = set()
    rec_stack: list[str] = []
    on_stack: set[str] = set()

    def dfs(nid: str) -> None:
        visited.add(nid)
        rec_stack.append(nid)
        on_stack.add(nid)
        for nxt in sorted(edges.get(nid, ())):
            if nxt not in visited:
                dfs(nxt)
            elif nxt in on_stack:
                start = rec_stack.index(nxt)
                cycle = rec_stack[start:] + [nxt]
                issues.append({
                    "code": "cyclic_concept",
                    "node": nxt,
                    "detail": "concept dependency cycle: " + " -> ".join(cycle),
                })
        on_stack.remove(nid)
        rec_stack.pop()

    for nid in sorted(graph.concept_ids):
        if nid not in visited:
            dfs(nid)
    return issues


def inverse_relationship_detector(graph: OkfKnowledgeGraph) -> list[OkfLintError]:
    """One-sided relationship declarations — the logical contradiction.

    X declares `PREREQUISITE_OF: Y` but Y's `prerequisites` does not list
    X (and the mirror: X lists prerequisite Y but Y never declares
    `PREREQUISITE_OF: X`). A learning-order graph written this way is
    internally inconsistent: the learner cannot trust either direction."""
    issues: list[OkfLintError] = []
    for nid, n in graph.nodes.items():
        if not n.is_concept:
            continue
        for y in sorted(n.prerequisite_of):
            yn = graph.nodes.get(y)
            if yn is None or not yn.is_concept:
                continue
            if nid not in yn.prerequisites:
                issues.append({
                    "code": "inverse_relationship",
                    "node": nid,
                    "detail": (
                        f"{nid!r} declares PREREQUISITE_OF {y!r} but {y!r} "
                        "does not list it in prerequisites"
                    ),
                })
        for y in sorted(n.prerequisites):
            yn = graph.nodes.get(y)
            if yn is None or not yn.is_concept:
                continue
            if nid not in yn.prerequisite_of:
                issues.append({
                    "code": "inverse_relationship",
                    "node": nid,
                    "detail": (
                        f"{nid!r} lists prerequisite {y!r} but {y!r} never "
                        f"declares PREREQUISITE_OF {nid!r}"
                    ),
                })
    return issues


def dangling_reference_detector(graph: OkfKnowledgeGraph) -> list[OkfLintError]:
    """Relationship targets that resolve to no node.

    The OKFValidator only checks frontmatter relationships + wiki/markdown
    links, so body-block references (`- **PREREQUISITES**: ...`) that
    point at non-existent ids (e.g. `composite.distributed_executor`)
    are invisible to it. This detector walks the raw parse."""
    issues: list[OkfLintError] = []
    for nid, n in graph.nodes.items():
        for rel, raw_ids in sorted(n.raw_relationships.items()):
            for rid in sorted(raw_ids):
                if graph.resolve(rid) is None:
                    issues.append({
                        "code": "dangling_reference",
                        "node": nid,
                        "detail": (
                            f"relationship {rel} references {rid!r} which "
                            "resolves to no node"
                        ),
                    })
    return issues


def relationship_noise_detector(graph: OkfKnowledgeGraph) -> list[OkfLintError]:
    """Relationship blocks containing prose / frontmatter snippets instead
    of node ids — a malformed markdown list that a naive parser would
    swallow into the graph (e.g. an `- **COMPOSED_OF**:` block followed by
    a description paragraph)."""
    issues: list[OkfLintError] = []
    for nid, n in graph.nodes.items():
        for rel, items in sorted(n.relationship_noise.items()):
            for item in items:
                issues.append({
                    "code": "relationship_noise",
                    "node": nid,
                    "detail": f"relationship {rel} contains non-id item {item[:80]!r}",
                })
    return issues


def missing_evidence_detector(graph: OkfKnowledgeGraph) -> list[OkfLintError]:
    """Concept/rule nodes with no frontmatter `sources` — claims made
    without any evidence document. Under OKF v0.2 trust, `verified`
    without `sources` is an unsupported attestation: the statement is
    asserted but no document backs it."""
    issues: list[OkfLintError] = []
    for nid, n in graph.nodes.items():
        if not (n.is_concept or n.is_rule):
            continue
        if not n.evidence:
            issues.append({
                "code": "missing_evidence",
                "node": nid,
                "detail": f"{nid!r} has no `sources` evidence in frontmatter",
            })
    return issues


def lint_knowledge_graph(graph: OkfKnowledgeGraph) -> list[OkfLintError]:
    """Run all KB-level detectors; concatenated signal list."""
    return (
        cyclic_concept_detector(graph)
        + inverse_relationship_detector(graph)
        + dangling_reference_detector(graph)
        + relationship_noise_detector(graph)
        + concept_orphan_detector(graph)
        + rule_schema_reference_detector(graph)
        + ambiguous_trigger_detector(graph)
        + missing_evidence_detector(graph)
    )


# ---------------------------------------------------------------------------
# Per-outcome regime detectors — pure over OkfOutcome
# ---------------------------------------------------------------------------


def _lint_codes(o: OkfOutcome) -> set[str]:
    codes: set[str] = set()
    for err in o.lint_errors:
        if isinstance(err, dict):
            code = err.get("code")
            if isinstance(code, str):
                codes.add(code)
        elif isinstance(err, str):
            codes.add(err)
    return codes


def detect_concept_cycle(o: OkfOutcome) -> bool:
    if o.correct:
        return False
    return "cyclic_concept" in _lint_codes(o)


def detect_rule_schema_mismatch(o: OkfOutcome) -> bool:
    if o.correct:
        return False
    return "rule_schema_reference" in _lint_codes(o)


def detect_concept_orphan(o: OkfOutcome) -> bool:
    if o.correct:
        return False
    return "concept_orphan" in _lint_codes(o)


def detect_ambiguous_trigger(o: OkfOutcome) -> bool:
    if o.correct:
        return False
    return "ambiguous_trigger" in _lint_codes(o)


def detect_unclassified(o: OkfOutcome) -> bool:  # noqa: ARG001
    return True


# ---------------------------------------------------------------------------
# Built-in taxonomy
# ---------------------------------------------------------------------------

# Seam-reachability mirrors the SQL target's split: a regime is
# optimizable / seam-reachable iff a prompt-transform in the OKF ask
# pipeline can address it. The wall is the KB itself — a concept cycle
# can't be fixed by prompt edits.
_BUILTIN: list[Regime] = [
    Regime(
        name="concept-cycle",
        detector=detect_concept_cycle,
        optimizable=False,
        seam_reachable=False,
        description=(
            "The knowledge base has a cyclic prerequisite/composition "
            "dependency. Prompt-transforms cannot fix KB authoring; the "
            "wall is an OKF bundle edit."
        ),
    ),
    Regime(
        name="rule-schema-mismatch",
        detector=detect_rule_schema_mismatch,
        optimizable=True,
        seam_reachable=True,
        description=(
            "An applicable rule references schema cells that don't "
            "resolve in the knowledge base. A prompt-transform that "
            "injects the correct schema (or drops the bad rule) can fix "
            "it."
        ),
    ),
    Regime(
        name="concept-orphan",
        detector=detect_concept_orphan,
        optimizable=True,
        seam_reachable=True,
        description=(
            "A concept with no parent/child relationships was used — "
            "grounding is impossible. A prompt-transform that injects "
            "the surrounding concept tree can fix it."
        ),
    ),
    Regime(
        name="ambiguous-trigger",
        detector=detect_ambiguous_trigger,
        optimizable=True,
        seam_reachable=True,
        description=(
            "A trigger phrase matches multiple concepts, so retrieval "
            "picked the wrong one. A prompt-transform that disambiguates "
            "or trims the competing concepts can fix it."
        ),
    ),
    Regime(
        name="unclassified",
        detector=detect_unclassified,
        optimizable=False,
        seam_reachable=False,
        description="Catch-all for outcomes no other detector matches.",
    ),
]


PRIORITY: tuple[str, ...] = (
    "concept-cycle",
    "rule-schema-mismatch",
    "concept-orphan",
    "ambiguous-trigger",
    "unclassified",
)


# ---------------------------------------------------------------------------
# Taxonomy adapter (implements regimes.target.RegimeTaxonomy)
# ---------------------------------------------------------------------------


@dataclass
class OkfTaxonomy:
    """Per-instance OKF taxonomy state. Mirrors `SqlTaxonomy`: state
    lives on the instance so multiple OkfTargets can coexist without
    sharing a global registry."""

    name: str = "okf"
    _registry: dict[str, Regime] = field(default_factory=dict)
    _priority: list[str] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self):
        if not self._registry:
            self._registry = {r.name: r for r in _BUILTIN}
            self._priority = list(PRIORITY)

    def REGIMES(self) -> dict[str, Regime]:  # noqa: N802
        with self._lock:
            return dict(self._registry)

    def classify(self, outcome: Any) -> Regime:
        with self._lock:
            for name in self._priority:
                r = self._registry[name]
                if r.detector(outcome):
                    return r
            return self._registry["unclassified"]

    def histogram(self, outcomes: Sequence[Any], *, failures_only: bool = True) -> list[HistogramRow]:
        # `correct is False` — an unjudged outcome (correct=None) is
        # neither a failure nor a win, so it must not count in either.
        target = [o for o in outcomes if (not failures_only) or (o.correct is False)]
        by_regime: dict[str, list[Any]] = {n: [] for n in self._priority}
        for o in target:
            r = self.classify(o)
            by_regime.setdefault(r.name, []).append(o)
        with self._lock:
            rows = []
            for name in self._priority:
                r = self._registry[name]
                members = by_regime.get(name, [])
                rows.append(HistogramRow(
                    regime=name,
                    count=len(members),
                    optimizable=r.optimizable,
                    seam_reachable=r.seam_reachable,
                    qids=tuple(o.question_id for o in members),
                ))
        return rows

    def is_seam_reachable(self, regime_name: str) -> bool:
        with self._lock:
            r = self._registry.get(regime_name)
        return bool(r and r.seam_reachable)

    def format_histogram(self, rows: Sequence[HistogramRow], *, n_failures: int, n_total: int) -> str:
        lines = [
            f"OKF regime histogram (failures={n_failures} / total={n_total}):",
            f"  {'regime':<24s}  {'count':>5s}  {'opt':>4s}  {'seam':>5s}",
        ]
        for r in rows:
            flag_opt = "yes" if r.optimizable else "no"
            flag_seam = "yes" if r.seam_reachable else "no"
            lines.append(
                f"  {r.regime:<24s}  {r.count:>5d}  {flag_opt:>4s}  {flag_seam:>5s}"
            )
        return "\n".join(lines)

    def name_wall(self, counts: Mapping[str, int]) -> str:
        reg = self.REGIMES()
        fragments: list[str] = []
        for name, c in sorted(counts.items()):
            if c <= 0:
                continue
            r = reg.get(name)
            if r is None or (r.optimizable and r.seam_reachable):
                continue
            if name == "concept-cycle":
                fix = "edit the OKF knowledge base (break the cycle), not the prompt"
            else:
                fix = "outside the prompt-transform action space"
            fragments.append(f"{name}={c} → {fix}")
        return "; ".join(fragments) if fragments else "no remaining failures"

    def register_regime(
        self,
        name: str,
        detector: Callable[[Any], bool],
        *,
        optimizable: bool,
        seam_reachable: bool,
        description: str = "",
        priority_after: str = "ambiguous-trigger",
    ) -> None:
        """LLM-proposed regime hook. Same shape as LME's
        `register_regime` but lives on the OkfTaxonomy instance."""
        with self._lock:
            if name in self._registry:
                raise ValueError(f"regime already registered: {name!r}")
            self._registry[name] = Regime(
                name=name, detector=detector,
                optimizable=optimizable, seam_reachable=seam_reachable,
                description=description,
            )
            try:
                idx = self._priority.index(priority_after)
            except ValueError:
                idx = len(self._priority) - 1
            self._priority.insert(idx + 1, name)
