"""The prompt-transform pipeline for the OKF target.

Mirrors `src.core.targets.sql.prompt_transforms` exactly in shape: an
ordered list of `(name, callable)` entries the OKF ask pipeline walks on
every `OkfActionSpace.build_prompt_parts` call. Each entry's callable
has signature

    transform(prompt_parts: dict, question: str, kb_meta: dict) -> dict

and must return a new dict over the same keys as the input. The seam
guards this signature at the OKF ActionSpace's static-analysis gate
before promotion.

The pipeline ships with the three built-in OKF ask transforms, in order:

    inject_concept_tree -> inject_rules -> trim_context

so a freshly built OKF target already grounds answers in the concept
tree and applicable rules under a bounded context. `promote`/`revert`
let the mentor loop append/remove additional transforms on top; `clear`
drops everything (test isolation) and `reset` restores the built-ins.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable

PromptTransform = Callable[[dict, str, dict], dict]

# Context budget for trim_context: cap on the combined length of the
# concepts + rules + schema sections, in characters.
MAX_CONTEXT_CHARS = 4000


@dataclass(frozen=True)
class PipelineEntry:
    name: str
    fn: PromptTransform


_LOCK = threading.Lock()
_PIPELINE: list[PipelineEntry] = []


# ---------------------------------------------------------------------------
# Built-in transforms
# ---------------------------------------------------------------------------


def _kb_node(kb_meta: dict[str, Any], node_id: str) -> dict[str, Any] | None:
    """Look up one node's snapshot inside kb_meta (may be absent)."""
    nodes = kb_meta.get("nodes") if isinstance(kb_meta, dict) else None
    if not isinstance(nodes, dict):
        return None
    node = nodes.get(node_id)
    return node if isinstance(node, dict) else None


def _dedupe(items: list[Any]) -> list[Any]:
    seen: set[Any] = set()
    out: list[Any] = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out


def inject_concept_tree(prompt_parts: dict, question: str, kb_meta: dict) -> dict:
    """Expand every concept id in `prompt_parts["concepts"]` with its
    parents (prerequisites) and children (prerequisite_of) from kb_meta,
    as tagged entries `<id> (parent)` / `<id> (child)`. This is what
    makes a retrieved concept reachable/groundable — an orphan concept
    stays alone and the taxonomy can flag the gap."""
    out = dict(prompt_parts)
    concepts = list(out.get("concepts", []))
    expanded: list[str] = []
    for cid in concepts:
        if not isinstance(cid, str):
            expanded.append(cid)
            continue
        node = _kb_node(kb_meta, cid)
        expanded.append(cid)
        if node is None:
            continue
        for parent in sorted(node.get("prerequisites", ())):
            expanded.append(f"{parent} (parent)")
        for child in sorted(node.get("prerequisite_of", ())):
            expanded.append(f"{child} (child)")
    out["concepts"] = _dedupe(expanded)
    return out


def inject_rules(prompt_parts: dict, question: str, kb_meta: dict) -> dict:
    """Append applicable rule ids from kb_meta to `prompt_parts["rules"]`.

    A rule is applicable when a normalized trigger phrase appears in the
    question (case-insensitive substring) or the rule's concept set
    overlaps the context's concepts. Deterministic; no I/O."""
    out = dict(prompt_parts)
    rules = list(out.get("rules", []))
    existing = set(rules)
    q_lower = (question or "").lower()
    nodes = kb_meta.get("nodes") if isinstance(kb_meta, dict) else None
    if isinstance(nodes, dict):
        for rid, node in sorted(nodes.items()):
            if not isinstance(node, dict):
                continue
            if not ("rule" in str(node.get("type", "")).lower()):
                continue
            if rid in existing:
                continue
            triggers = node.get("triggers", ())
            trigger_hit = any(str(t) and str(t) in q_lower for t in triggers)
            if trigger_hit:
                rules.append(rid)
                existing.add(rid)
    out["rules"] = _dedupe(rules)
    return out


def trim_context(prompt_parts: dict, question: str, kb_meta: dict) -> dict:
    """Cap the combined length of the concepts + rules + schema sections
    at `MAX_CONTEXT_CHARS`. Drops the tail of each section (order
    preserved) so the question/instructions/hints always survive."""
    budget = int(kb_meta.get("max_context_chars", MAX_CONTEXT_CHARS)) if isinstance(kb_meta, dict) else MAX_CONTEXT_CHARS
    out = dict(prompt_parts)
    remaining = max(1, int(budget))
    for key in ("concepts", "rules", "schema"):
        if key not in out:
            continue  # never introduce keys the input didn't have
        items = list(out.get(key, []))
        kept: list[Any] = []
        for it in items:
            s = it if isinstance(it, str) else str(it)
            if remaining <= 0:
                break
            kept.append(it)
            remaining -= len(s) + 1  # +1 for the newline in the renderer
        out[key] = kept
    return out


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def _reset_pipeline() -> None:
    global _PIPELINE
    _PIPELINE = [
        PipelineEntry(name="inject_concept_tree", fn=inject_concept_tree),
        PipelineEntry(name="inject_rules", fn=inject_rules),
        PipelineEntry(name="trim_context", fn=trim_context),
    ]


_reset_pipeline()


def get_pipeline() -> list[PipelineEntry]:
    """Snapshot of the current pipeline."""
    with _LOCK:
        return list(_PIPELINE)


def promote(name: str, fn: PromptTransform) -> None:
    """Append a transform. The loop's promotion gate calls this only
    after all four lifecycle gates pass."""
    with _LOCK:
        _PIPELINE.append(PipelineEntry(name=name, fn=fn))


def revert(name: str) -> None:
    """Remove a transform by name (built-in or promoted)."""
    with _LOCK:
        _PIPELINE[:] = [e for e in _PIPELINE if e.name != name]


def clear() -> None:
    """Drop every transform, built-ins included. Test isolation only."""
    with _LOCK:
        _PIPELINE.clear()


def reset() -> None:
    """Restore the three built-in transforms. Test isolation only."""
    with _LOCK:
        _reset_pipeline()


def apply_pipeline(
    *,
    prompt_parts: dict[str, Any],
    question: str,
    kb_meta: dict[str, Any],
) -> tuple[dict, list[dict]]:
    """Walk the pipeline. Returns (result, errors).

    `result` is {"prompt_parts": dict, "names": list[str]}.
    `errors` is a list of {"name": str, "error": str} for transforms
    that raised at runtime. A raising transform's contribution is
    skipped; the previous prompt_parts carry forward."""
    cur: dict[str, Any] = dict(prompt_parts)
    names: list[str] = []
    errors: list[dict] = []
    for entry in get_pipeline():
        try:
            new_parts = entry.fn(cur, question, kb_meta)
            if not isinstance(new_parts, dict):
                raise TypeError(
                    f"transform {entry.name!r} returned "
                    f"{type(new_parts).__name__}, expected dict"
                )
            extra = set(new_parts) - set(cur)
            if extra:
                raise ValueError(
                    f"transform {entry.name!r} introduced unknown "
                    f"prompt_parts keys: {sorted(extra)[:3]}..."
                )
            # Re-key to the same shape; missing keys keep their previous
            # value (transforms may filter, not invent).
            cur = {k: new_parts.get(k, cur[k]) for k in cur}
            names.append(entry.name)
        except Exception as e:  # noqa: BLE001 — agent runtime path
            errors.append({"name": entry.name, "error": f"{type(e).__name__}: {e}"})
    return ({"prompt_parts": cur, "names": names}, errors)


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------


def render_okf_prompt(parts: dict[str, Any]) -> str:
    """Deterministic text render of OKF prompt parts for the Reader."""
    sections: list[str] = []

    def _bullet(key: str, heading: str) -> None:
        items = parts.get(key) or []
        if items:
            lines = [f"  - {it}" for it in items]
            sections.append(f"{heading}:\n" + "\n".join(lines))

    _bullet("concepts", "Concepts")
    _bullet("rules", "Rules")
    _bullet("schema", "Schema")

    instructions = parts.get("instructions", "")
    if instructions:
        sections.append(f"Instructions: {instructions}")

    hints = parts.get("hints") or []
    if hints:
        sections.append("Hints:\n  " + "\n  ".join(str(h) for h in hints))

    sections.append(f"Question: {parts.get('question', '')}")
    sections.append("Answer:")
    return "\n\n".join(sections)
