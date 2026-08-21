"""OKF KB repair tool — Phase 8 cleanup of `.okf/01_nano_vllm`.

Applies three deterministic repair passes over a bundle so
`lint_knowledge_graph` reaches zero issues:

  1. BODY_EDITS  — targeted relationship-block fixes (cycle-breaking
     direction corrections, dangling-id re-pointing, noise removal).
  2. EVIDENCE    — add frontmatter `sources` (vLLM PagedAttention paper)
     to every concept/rule node that lacks evidence.
  3. INVERSE     — mirror one-sided relationship declarations into the
     target node's frontmatter (prerequisites / prerequisite_of), so the
     graph is dual-declared and consistent.

Run:  python -m src.tools.okf_repair [bundle] [--dry-run]
After repair: `lint_knowledge_graph` should return [] (lint clean).
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from src.core.targets.okf.taxonomy import load_knowledge_graph

# vLLM PagedAttention paper — the real evidence source for the nano-vLLM KB.
VLLM_PAPER = {
    "id": "vllm-paper",
    "resource": "https://arxiv.org/abs/2209.06155",
    "title": "vLLM - Easy, Fast and Cheap LLM Serving with PagedAttention",
}

_MD_LINK_RE = re.compile(r"\[[^\]]*\]\([^)]*\)")
_ID_IN_TEXT_RE = re.compile(r"`([^`]+)`|([A-Za-z0-9][A-Za-z0-9_.\-]*)")
_REL_HEADER_LINE_RE = re.compile(r"^(\s*-\s*\*\*[A-Z][A-Z0-9_ ]*\*\*:)(.*)$")
_ITEM_LINE_RE = re.compile(r"^(\s+-\s+)(.*)$")

# (relative path, old substring, new substring) — applied with
# old.replace(new) once; a miss raises so silent drift is impossible.
BODY_EDITS: list[tuple[str, str, str]] = [
    # ---- dangling re-points ----
    ("concepts/01_module/module_05_memory_management.md",
     "- **PREREQUISITE_OF**: `distributed_serving` (Module 7)",
     "- **PREREQUISITE_OF**: `composite.distributed_serving` (Module 7)"),
    ("concepts/01_module/module_06_prefix_caching.md",
     "- **PREREQUISITE_OF**: `distributed_serving` (Module 7)",
     "- **PREREQUISITE_OF**: `composite.distributed_serving` (Module 7)"),
    ("concepts/01_module/module_07_distributed_serving.md",
     "`prefix_cache_manager`",
     "`composite.prefix_cache_manager`"),
    ("concepts/03_atomic/master_worker.md",
     "- **PREREQUISITE_OF**: `composite.distributed_executor`",
     "- **PREREQUISITE_OF**: `composite.distributed_serving`"),
    ("concepts/03_atomic/shared_memory_ipc.md",
     "- **PREREQUISITE_OF**: `composite.distributed_executor`",
     "- **PREREQUISITE_OF**: `composite.distributed_serving`"),
    ("concepts/03_atomic/tensor_parallelism.md",
     "- **PREREQUISITE_OF**: `composite.distributed_executor` (Module 07)",
     "- **PREREQUISITE_OF**: `composite.distributed_serving` (Module 07)"),
    # ---- cycle-breaking direction corrections (COMPOSED_OF -> PREREQUISITE_OF) ----
    ("concepts/03_atomic/continuous_batching.md",
     "- **COMPOSED_OF**: `iteration_level_scheduling` (배치 단계 스케줄링)",
     "- **PREREQUISITE_OF**: `iteration_level_scheduling` (배치 단계 스케줄링)"),
    ("concepts/03_atomic/block_allocator.md",
     "- **COMPOSED_OF**: `swap_manager` (스와핑 지원)",
     "- **PREREQUISITE_OF**: `swap_manager` (스와핑 지원)"),
    ("concepts/03_atomic/swap_manager.md",
     "- **COMPOSED_OF**: `block_manager` (블록 추적)",
     "- **PREREQUISITE_OF**: `composite.block_manager` (블록 추적)"),
    ("concepts/03_atomic/memory_pool.md",
     "- **COMPOSED_OF**: max_batch_size,  `block_allocator` (블록 할당기), `swap_manager` (스와핑 관리자)",
     "- **PREREQUISITE_OF**: `max_batch_size`, `block_allocator` (블록 할당기), `swap_manager` (스와핑 관리자)"),
    ("concepts/03_atomic/fault_tolerance.md",
     "- **COMPOSED_OF**: `distributed_serving` (전체 서비스 레이어에서 장애 내성을 활용)",
     "- **PREREQUISITE_OF**: `composite.distributed_serving` (전체 서비스 레이어에서 장애 내성을 활용)"),
    ("concepts/03_atomic/load_balancer.md",
     "- **COMPOSED_OF**: `fault_tolerance` (장애 시 재시도·전환)",
     "- **PREREQUISITE_OF**: `fault_tolerance` (장애 시 재시도·전환)"),
    # ---- noise removal (prose moved inside parentheticals) ----
    ("concepts/03_atomic/paged_kv_cache.md",
     "  - `atomic.memory_pool` (Module 05) – 블록을 할당할 물리적 메모리 풀",
     "  - `atomic.memory_pool` (Module 05, 블록을 할당할 물리적 메모리 풀)"),
]


@dataclass
class RepairReport:
    body_edits: list[str] = field(default_factory=list)
    evidence_added: list[str] = field(default_factory=list)
    inverse_added: list[tuple[str, str, list[str]]] = field(default_factory=list)


def _split_frontmatter(content: str) -> tuple[str, str, str]:
    """(head, frontmatter_yaml, rest) for a `---`-delimited file."""
    parts = content.split("---", 2)
    if len(parts) < 3:
        return "", "", content
    return parts[0], parts[1], parts[2]


def _apply_body_edits(bundle: Path, report: RepairReport) -> None:
    for rel, old, new in BODY_EDITS:
        p = bundle / rel
        text = p.read_text(encoding="utf-8")
        if old in text:
            p.write_text(text.replace(old, new, 1), encoding="utf-8")
            report.body_edits.append(rel)
        elif new not in text:
            raise RuntimeError(f"body edit target not found in {rel}: {old!r}")


def _apply_evidence(bundle: Path, graph, report: RepairReport) -> None:
    for nid, n in sorted(graph.nodes.items()):
        if not (n.is_concept or n.is_rule):
            continue
        if n.evidence:
            continue
        p = bundle / n.path
        if not p.is_file():
            continue
        content = p.read_text(encoding="utf-8")
        head, fm_yaml, rest = _split_frontmatter(content)
        if not fm_yaml.strip():
            continue
        fm = yaml.safe_load(fm_yaml) or {}
        if isinstance(fm, dict) and fm.get("sources"):
            continue
        sources_block = (
            "sources:\n"
            "  - id: vllm-paper\n"
            "    resource: https://arxiv.org/abs/2209.06155\n"
            "    title: vLLM - Easy, Fast and Cheap LLM Serving with PagedAttention\n"
        )
        new_fm = fm_yaml.rstrip() + "\n" + sources_block
        p.write_text(f"{head}---{new_fm}---{rest}", encoding="utf-8")
        report.evidence_added.append(nid)


def _apply_inverse(bundle: Path, graph, report: RepairReport) -> None:
    """Mirror one-sided edges into the target node's frontmatter.

    For each concept X with X.PREREQUISITE_OF ∋ Y where Y does not list X
    in prerequisites → add X to Y's frontmatter `prerequisites`.
    For each X.PREREQUISITES ∋ Y where Y never declares PREREQUISITE_OF X
    → add X to Y's frontmatter `prerequisite_of`."""
    # Collect per-target additions: {node_id: {"prerequisites": [...], "prerequisite_of": [...]}}
    additions: dict[str, dict[str, set[str]]] = {}
    for nid, n in graph.nodes.items():
        if not n.is_concept:
            continue
        for y in sorted(n.prerequisite_of):
            yn = graph.nodes.get(y)
            if yn is None or not yn.is_concept:
                continue
            if nid not in yn.prerequisites:
                additions.setdefault(y, {}).setdefault("prerequisites", set()).add(nid)
        for y in sorted(n.prerequisites):
            yn = graph.nodes.get(y)
            if yn is None or not yn.is_concept:
                continue
            if nid not in yn.prerequisite_of:
                additions.setdefault(y, {}).setdefault("prerequisite_of", set()).add(nid)

    for node_id, by_key in sorted(additions.items()):
        n = graph.nodes.get(node_id)
        if n is None:
            continue
        p = bundle / n.path
        if not p.is_file():
            continue
        content = p.read_text(encoding="utf-8")
        head, fm_yaml, rest = _split_frontmatter(content)
        if not fm_yaml.strip():
            raise RuntimeError(f"no frontmatter to mirror into: {n.path}")
        fm = yaml.safe_load(fm_yaml) or {}
        fm = dict(fm)
        added_keys: list[str] = []
        for key, ids in sorted(by_key.items()):
            existing = set(fm.get(key) or ())
            fresh = sorted(ids - existing)
            if not fresh:
                continue
            fm[key] = sorted(existing | set(fresh))
            added_keys.append(f"{key}:{','.join(fresh)}")
        if not added_keys:
            continue
        new_fm = yaml.safe_dump(fm, sort_keys=False, allow_unicode=True).rstrip()
        p.write_text(f"{head}---\n{new_fm}\n---{rest}", encoding="utf-8")
        report.inverse_added.append((node_id, n.path, added_keys))


def repair(bundle: str | Path = ".okf/01_nano_vllm") -> RepairReport:
    root = Path(bundle).resolve()
    report = RepairReport()
    _apply_body_edits(root, report)
    graph = load_knowledge_graph(root)
    _apply_evidence(root, graph, report)
    graph = load_knowledge_graph(root)  # reload after evidence
    _apply_inverse(root, graph, report)
    return report


# ---------------------------------------------------------------------------
# Linkify pass — module → composite → atomic downward tree links
# ---------------------------------------------------------------------------


def _linkify_gap(gap: str, file_dir: str, id_to_path: dict[str, str]) -> str:
    """Wrap resolvable id tokens in a text gap as markdown links."""
    out: list[str] = []
    pos = 0
    for m in _ID_IN_TEXT_RE.finditer(gap):
        raw = m.group(0)
        idc = m.group(1) or m.group(2)
        out.append(gap[pos:m.start()])
        if idc in id_to_path:
            rel = os.path.relpath(id_to_path[idc], start=file_dir).replace("\\", "/")
            out.append(f"[`{idc}`]({rel})")
        else:
            out.append(raw)
        pos = m.end()
    out.append(gap[pos:])
    return "".join(out)


def _rel_link(chunk: str, file_dir: str, id_to_path: dict[str, str]) -> str:
    """Linkify id tokens in a chunk; keep existing markdown links, but
    repair broken file:/// links whose link text resolves to an id."""
    out: list[str] = []
    pos = 0
    for lm in _MD_LINK_RE.finditer(chunk):
        out.append(_linkify_gap(chunk[pos:lm.start()], file_dir, id_to_path))
        link = lm.group(0)
        m = re.match(r"\[([^\]]+)\]\([^)]+\)", link)
        if m:
            text = m.group(1).strip().strip("`").strip()
            if text in id_to_path:
                rel = os.path.relpath(id_to_path[text], start=file_dir).replace("\\", "/")
                out.append(f"[`{text}`]({rel})")
                pos = lm.end()
                continue
        out.append(link)
        pos = lm.end()
    out.append(_linkify_gap(chunk[pos:], file_dir, id_to_path))
    return "".join(out)


def linkify_relationships(bundle: str | Path = ".okf/01_nano_vllm") -> RepairReport:
    """Convert id tokens (backticked / bare / broken file:/// links) in
    relationship blocks and prose into markdown links to the target
    node's file, so module→composite→atomic tree links work in markdown
    viewers and the dashboard. Idempotent: existing links are kept."""
    root = Path(bundle).resolve()
    graph = load_knowledge_graph(root)
    id_to_path = {nid: n.path for nid, n in graph.nodes.items()}
    for stem, nid in graph.by_stem.items():
        id_to_path.setdefault(stem, id_to_path.get(nid))
    report = RepairReport()

    for p in sorted(root.rglob("*.md")):
        content = p.read_text(encoding="utf-8")
        head, fm_yaml, rest = _split_frontmatter(content)
        if not rest.strip():
            continue
        file_dir = str(p.parent.relative_to(root)).replace("\\", "/")
        in_fence = False
        current_rel: str | None = None
        out: list[str] = []
        changed = False
        for raw in rest.splitlines():
            s = raw.strip()
            if s.startswith("```") or s.startswith("~~~"):
                in_fence = not in_fence
                out.append(raw)
                continue
            if in_fence:
                out.append(raw)
                continue
            m = _REL_HEADER_LINE_RE.match(raw)
            if m:
                current_rel = m.group(1)
                new = current_rel + _rel_link(m.group(2), file_dir, id_to_path)
                changed |= new != raw
                out.append(new)
                continue
            m = _ITEM_LINE_RE.match(raw)
            if m and current_rel is not None:
                new = m.group(1) + _rel_link(m.group(2), file_dir, id_to_path)
                changed |= new != raw
                out.append(new)
                continue
            # Prose: linkify backticked/bare id mentions + broken links too.
            new = _rel_link(raw, file_dir, id_to_path)
            changed |= new != raw
            out.append(new)
        if changed:
            p.write_text(f"{head}---{fm_yaml}---" + "\n".join(out), encoding="utf-8")
            report.body_edits.append(str(p.relative_to(root)))
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("bundle", nargs="?", default=".okf/01_nano_vllm")
    ap.add_argument("--dry-run", action="store_true", help="report changes without writing")
    ap.add_argument("--linkify", action="store_true",
                    help="linkify relationship/prose id tokens into markdown links")
    args = ap.parse_args(argv)
    if args.dry_run:
        print("dry-run is not supported (script writes in place); remove --dry-run to run")
        return 2
    report = linkify_relationships(args.bundle) if args.linkify else repair(args.bundle)
    print(f"body edits      : {len(report.body_edits)}")
    for f in report.body_edits[:20]:
        print("  ", f)
    if not args.linkify:
        print(f"evidence added  : {len(report.evidence_added)}")
        print(f"inverse mirrored: {len(report.inverse_added)} nodes")
    graph = load_knowledge_graph(args.bundle)
    from src.core.targets.okf.taxonomy import lint_knowledge_graph

    issues = lint_knowledge_graph(graph)
    print(f"lint after      : {len(issues)} issues")
    for i in issues[:10]:
        print("  ", i.get("code"), i.get("node"), str(i.get("detail", ""))[:60])
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
