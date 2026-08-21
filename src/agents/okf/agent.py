"""OKF agent entrypoints: ingest / lint / ask on the ActiveGraph runtime.

Mirrors the SQL agent's entry pattern (`src.core.targets.sql.agent.agent`):
importing `behaviors` registers the `okf_agent.*` behaviors as a side
effect; each entrypoint pins the runtime to that snapshot, seeds the
canonical event, drains the runtime, and returns a trace carrying the
full event log + the terminal payload + the LLM observability trail.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from activegraph import Event, FrozenClock, Graph, IDGen, Runtime, get_registry

# Importing this module registers the four okf_agent behaviors as a side
# effect. We capture the snapshot once at import time.
from src.agents.okf import behaviors as _behaviors_module  # noqa: F401
from src.agents.okf import events as E
from src.agents.okf.behaviors import clear_kb
from src.runtime.reader_registry import (
    clear_reader as _clear_reader,
    set_reader as _set_reader,
)


DEFAULT_FROZEN_T = "2026-01-01T00:00:00Z"
DETERMINISTIC_RUN_ID = "brick-okf-agent-determ"

_OKF_BEHAVIORS_SNAPSHOT = [
    b for b in get_registry() if b.name.startswith("okf_agent.")
]


@dataclass
class OkfTrace:
    """One OKF agent run: the full event log + the terminal payload +
    the LLM observability trail (llm.requested / llm.responded)."""

    kind: str
    events: list[Event]
    run_id: str
    payload: dict[str, Any] | None = None        # terminal event payload
    llm_events: list[dict[str, Any]] = field(default_factory=list)


def _last_payload(events: list[Event], etype: str) -> dict[str, Any] | None:
    for ev in reversed(events):
        if ev.type == etype:
            return dict(ev.payload)
    return None


def _seed(graph: Graph, etype: str, payload: dict[str, Any]) -> Event:
    ev = Event(
        id=graph.ids.event(),
        type=etype,
        payload=dict(payload),
        actor="caller",
        caused_by=None,
        timestamp=graph.clock.now(),
    )
    graph.emit(ev)
    return ev


def _new_runtime(frozen_t: str, store_path: str | Path | None = None) -> tuple[Graph, Runtime]:
    # A stored run needs a UNIQUE run_id per store file (the store's
    # events table keys on (id, run_id)); only the in-memory path keeps
    # the deterministic id.
    run_id = DETERMINISTIC_RUN_ID if store_path is None else f"brick-okf-{uuid.uuid4().hex[:12]}"
    graph = Graph(
        ids=IDGen(),
        clock=FrozenClock(frozen_t),
        run_id=run_id,
    )
    kw: dict[str, Any] = {"behaviors": _OKF_BEHAVIORS_SNAPSHOT}
    if store_path is not None:
        p = Path(store_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        kw["persist_to"] = str(p)
    rt = Runtime(graph, **kw)
    return graph, rt


def _close_store(rt: Runtime) -> None:
    """Release the SQLiteEventStore connection so the file can be
    removed/rotated on Windows."""
    store = getattr(rt.graph, "store", None)
    if store is not None:
        try:
            store.close()
        except Exception:  # noqa: BLE001 — best-effort hygiene
            pass


def _llm_trail(events: list[Event]) -> list[dict[str, Any]]:
    trail = []
    for ev in events:
        if ev.type in (E.LLM_REQUESTED, E.LLM_RESPONDED):
            rec = dict(ev.payload)
            rec["id"] = ev.id
            rec["event_type"] = ev.type
            trail.append(rec)
    return trail


# ---------------------------------------------------------------------------
# Entrypoints
# ---------------------------------------------------------------------------


def ingest(
    *,
    kb_id: str,
    kb_path: str | Path,
    request_id: str | None = None,
    frozen_t: str = DEFAULT_FROZEN_T,
    store_path: str | Path | None = None,
) -> OkfTrace:
    """Parse → validate → load one OKF bundle into the graph."""
    graph, rt = _new_runtime(frozen_t, store_path=store_path)
    _seed(graph, E.OKF_INGEST_REQUESTED, {
        "kb_id": kb_id,
        "kb_path": str(kb_path),
        "request_id": request_id or kb_id,
    })
    rt.run_until_idle()
    _close_store(rt)
    return OkfTrace(
        kind="ingest",
        events=list(graph.events),
        run_id=rt.run_id,
        payload=_last_payload(graph.events, E.OKF_LOADED),
    )


def lint(
    *,
    kb_id: str,
    kb_path: str | Path | None = None,
    request_id: str | None = None,
    frozen_t: str = DEFAULT_FROZEN_T,
    store_path: str | Path | None = None,
) -> OkfTrace:
    """Run the deterministic OKF detectors over a knowledge base."""
    graph, rt = _new_runtime(frozen_t, store_path=store_path)
    _seed(graph, E.OKF_LINT_REQUESTED, {
        "kb_id": kb_id,
        "kb_path": str(kb_path) if kb_path is not None else None,
        "request_id": request_id or kb_id,
    })
    rt.run_until_idle()
    _close_store(rt)
    return OkfTrace(
        kind="lint",
        events=list(graph.events),
        run_id=rt.run_id,
        payload=_last_payload(graph.events, E.OKF_LINTED),
    )


def ask(
    *,
    request_id: str,
    question: str,
    kb_id: str,
    kb_path: str | Path | None = None,
    reader: Any = None,
    top_k: int = 10,
    frozen_t: str = DEFAULT_FROZEN_T,
    store_path: str | Path | None = None,
) -> OkfTrace:
    """Ask a grounded question: assemble context → LLM (observability
    seam) → generated answer."""
    graph, rt = _new_runtime(frozen_t, store_path=store_path)
    if reader is not None:
        _set_reader(request_id, reader)
    try:
        _seed(graph, E.OKF_ASK_REQUESTED, {
            "request_id": request_id,
            "question": question,
            "kb_id": kb_id,
            "kb_path": str(kb_path) if kb_path is not None else None,
            "top_k": top_k,
        })
        rt.run_until_idle()
    finally:
        _clear_reader(request_id)
    _close_store(rt)
    return OkfTrace(
        kind="ask",
        events=list(graph.events),
        run_id=rt.run_id,
        payload=_last_payload(graph.events, E.OKF_ANSWER_GENERATED),
        llm_events=_llm_trail(graph.events),
    )


__all__ = ["OkfTrace", "ask", "clear_kb", "ingest", "lint"]
