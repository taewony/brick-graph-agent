"""Durable event store + run metadata (brick-agent-plan Phase 5.1).

Wraps activegraph's `SQLiteEventStore` (via the `persist_to=` sugar) and
exposes run-level helpers: build a Runtime with a durable store, read a
run's events back, and list stored runs. The event log is the source of
truth — the graph is a projection rebuilt by replay (see `replay.py`).
"""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from activegraph import Event, FrozenClock, Graph, IDGen, Runtime

DEFAULT_FROZEN_T = "2026-01-01T00:00:00Z"


@dataclass(frozen=True)
class StoredRun:
    """One run recorded in a store (from the store's `runs` table)."""

    run_id: str
    path: str
    n_events: int
    label: str = ""
    parent_run_id: str | None = None


def new_runtime(
    behaviors: Iterable[Any],
    *,
    persist_to: str | Path | None = None,
    run_id: str | None = None,
    frozen_t: str = DEFAULT_FROZEN_T,
    replay_llm_cache: bool = False,
) -> tuple[Graph, Runtime]:
    """Build a Graph + Runtime, optionally attached to a durable
    SQLiteEventStore via `persist_to`."""
    graph = Graph(
        ids=IDGen(),
        clock=FrozenClock(frozen_t),
        run_id=run_id or f"run-{uuid.uuid4().hex[:12]}",
    )
    kw: dict[str, Any] = {
        "behaviors": list(behaviors),
        "replay_llm_cache": replay_llm_cache,
    }
    if persist_to is not None:
        p = Path(persist_to)
        p.parent.mkdir(parents=True, exist_ok=True)
        kw["persist_to"] = str(p)
    rt = Runtime(graph, **kw)
    return graph, rt


def run_events(store_path: str | Path, run_id: str) -> list[Event]:
    """Read one run's event log back from the store."""
    from activegraph.store.sqlite import SQLiteEventStore

    store = SQLiteEventStore(str(store_path), run_id)
    return list(store.iter_events())


def list_runs(store_path: str | Path) -> list[StoredRun]:
    """List recorded runs (best-effort; empty when the store is absent
    or has no runs table yet)."""
    p = Path(store_path)
    if not p.is_file():
        return []
    conn = sqlite3.connect(str(p))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT * FROM runs ORDER BY created_at").fetchall()
        has_events = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='events'"
        ).fetchone() is not None
    except sqlite3.OperationalError:
        return []
    out: list[StoredRun] = []
    for r in rows:
        rec = dict(r)
        n = 0
        if has_events:
            n = conn.execute(
                "SELECT COUNT(*) AS c FROM events WHERE run_id=?", (rec.get("run_id"),)
            ).fetchone()["c"]
        out.append(StoredRun(
            run_id=rec.get("run_id", ""),
            path=str(p),
            n_events=n,
            label=rec.get("label") or "",
            parent_run_id=rec.get("parent_run_id"),
        ))
    return out
