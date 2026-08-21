"""Agent-Model evolution logger.

Appends canonical evolution operators to `.okf/*/history.yaml`
(per `activegraph_okf_agent_modeler.md`): ADD_BEHAVIOR, REWIRE_BEHAVIOR,
OPTIMIZE_PROMPT, ADD_EVENT_TYPE, ADD_DOC, ... Each record carries
{id, timestamp, op, target, reason, ...} and the file's version is
bumped (patch). Used at runtime by the mentor / router / tooling.

NOTE: rewrite via yaml.safe_dump drops comment lines in the source file
(standard pyyaml behaviour); semantic content is preserved.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

# Canonical operator names (activegraph_okf_agent_modeler.md §7 + the
# ADD_DOC / RENAME operators used by the agent-model work).
CANONICAL_OPS: frozenset[str] = frozenset({
    "ADD_BEHAVIOR",
    "REMOVE_BEHAVIOR",
    "MODIFY_BEHAVIOR",
    "SPLIT_BEHAVIOR",
    "MERGE_BEHAVIORS",
    "ADD_EVENT_TYPE",
    "REMOVE_EVENT_TYPE",
    "ADD_OBJECT_TYPE",
    "MODIFY_EDGE",
    "REWIRE_BEHAVIOR",
    "ADD_GUARDRAIL",
    "OPTIMIZE_PROMPT",
    "ADD_CACHING",
    "ADD_DOC",
    "RENAME",
})

_OP_ID_RE = re.compile(r"^op_(\d+)$")


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_history(path: str | Path) -> dict[str, Any]:
    """Load a history.yaml; missing file → fresh skeleton."""
    p = Path(path)
    if not p.is_file():
        return {"version": "1.0.0", "updated_at": _now_iso(), "history": []}
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    data.setdefault("version", "1.0.0")
    data.setdefault("updated_at", "")
    data.setdefault("history", [])
    return data


def _next_op_id(history: list[dict[str, Any]]) -> str:
    max_n = 0
    for rec in history:
        m = _OP_ID_RE.match(str(rec.get("id", "")))
        if m:
            max_n = max(max_n, int(m.group(1)))
    return f"op_{max_n + 1:03d}"


def bump_version(version: str) -> str:
    """X.Y.Z → X.Y.(Z+1)."""
    parts = (version or "1.0.0").split(".")
    while len(parts) < 3:
        parts.append("0")
    parts[2] = str(int(parts[2]) + 1)
    return ".".join(parts)


def append_operator(
    path: str | Path,
    *,
    op: str,
    target: str,
    reason: str,
    timestamp: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Append one canonical evolution operator record.

    Returns the appended record. Raises ValueError for non-canonical op
    names (typo protection — the history is an audit contract)."""
    if op not in CANONICAL_OPS:
        raise ValueError(
            f"non-canonical history operator {op!r}; "
            f"use one of {sorted(CANONICAL_OPS)}"
        )
    p = Path(path)
    data = read_history(p)
    record: dict[str, Any] = {
        "id": _next_op_id(data["history"]),
        "timestamp": timestamp or _now_iso(),
        "op": op,
        "target": target,
    }
    record.update(extra)
    record["reason"] = reason
    data["history"].append(record)
    data["version"] = bump_version(data["version"])
    data["updated_at"] = timestamp or _now_iso()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True) or "",
        encoding="utf-8",
    )
    return record
