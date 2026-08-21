"""Deterministic replay + LLM response cache (brick-agent-plan Phase 5.3).

The llm.* events recorded by the observability seam make re-runs
deterministic: `build_replay_cache` harvests prompt_hash → answer from a
run's log, and `ReplayReader` serves those answers instead of calling the
LLM again (the held-out validation gate in the agent-modeler discipline).
`replay_into_graph` / `load_run_runtime` rebuild a graph/runtime from a
stored run.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable

from activegraph import Event, FrozenClock, Graph, IDGen, Runtime
from activegraph.store.base import replay_into

from src.runtime.event_store import run_events

LLM_REQUESTED = "llm.requested"
LLM_RESPONDED = "llm.responded"


def build_replay_cache(events: Iterable[Event]) -> dict[str, str]:
    """prompt_hash → recorded answer, harvested from llm.requested /
    llm.responded pairs (responded.caused_by → requested.id; error-shaped
    responses are not reusable output)."""
    events_list = list(events)
    by_id = {e.id: e for e in events_list}
    cache: dict[str, str] = {}
    for e in events_list:
        if e.type != LLM_RESPONDED or e.payload.get("error"):
            continue
        req = by_id.get(e.payload.get("caused_by"))
        if req is None or req.type != LLM_REQUESTED:
            continue
        h = req.payload.get("prompt_hash")
        answer = e.payload.get("answer")
        if h and isinstance(answer, str):
            cache[h] = answer
    return cache


class ReplayReader:
    """Reader that serves recorded answers by prompt hash.

    On a cache hit the recorded answer is returned — no LLM call, so a
    held-out validation run replays deterministically. On a miss it
    delegates to `fallback` (live reader) or returns a marker."""

    name = "replay-reader"

    def __init__(self, cache: dict[str, str] | None = None, fallback: Any = None):
        self.cache = dict(cache or {})
        self.fallback = fallback

    def answer(self, *, context: str, question: str, question_id: str) -> str:
        h = hashlib.sha256((context or "").encode("utf-8")).hexdigest()
        if h in self.cache:
            return self.cache[h]
        if self.fallback is not None:
            return self.fallback.answer(
                context=context, question=question, question_id=question_id,
            )
        return "[replay-miss: no recorded answer for this prompt]"


def replay_into_graph(
    store_path: str,
    run_id: str,
    graph: Graph | None = None,
) -> tuple[Graph, int]:
    """Rebuild a graph from a stored run's event log (no listeners fired).

    Returns (graph, n_events_replayed)."""
    events = run_events(store_path, run_id)
    if graph is None:
        graph = Graph(
            ids=IDGen(),
            clock=FrozenClock("2026-01-01T00:00:00Z"),
            run_id=run_id,
        )
    n = replay_into(graph, events)
    return graph, n


def load_run_runtime(
    store_path: str | Path,
    run_id: str,
    *,
    behaviors: Iterable[Any] | None = None,
    replay_strict: bool = False,
) -> Runtime:
    """Reopen a stored run as a live Runtime (activegraph native replay)."""
    return Runtime.load(
        str(store_path), run_id,
        behaviors=behaviors,
        replay_strict=replay_strict,
    )
