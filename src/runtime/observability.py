"""LLM observability helpers (brick-agent-plan Phase 5.2).

Every Reader call in brick.agent goes through the observability seam:
`llm.requested` (model, prompt_hash, prompt) → `llm.responded` (caused_by,
model, answer, latency_seconds, error, cost_usd?, cache_hit?). The
recorded `answer` makes the log self-sufficient for deterministic replay
(see `replay.py`). Also re-exports the runtime's structured logging and
causal-chain renderer.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from activegraph import Event

from src.runtime.reader_registry import call_reader

LLM_REQUESTED = "llm.requested"
LLM_RESPONDED = "llm.responded"


def prompt_hash(prompt: str) -> str:
    """Content key for LLM prompts (deterministic replay + caching)."""
    return hashlib.sha256((prompt or "").encode("utf-8")).hexdigest()


def emit_llm_request(graph, *, request_id: str, model: str, prompt: str) -> Event:
    return graph.emit(LLM_REQUESTED, {
        "request_id": request_id,
        "model": model,
        "prompt_hash": prompt_hash(prompt),
        "prompt": prompt,
    })


def emit_llm_response(
    graph,
    *,
    request_id: str,
    caused_by: str,
    model: str,
    answer: str,
    latency_seconds: float,
    error: str = "",
    cost_usd: float | None = None,
    cache_hit: bool | None = None,
) -> Event:
    payload: dict[str, Any] = {
        "request_id": request_id,
        "caused_by": caused_by,
        "model": model,
        "answer": answer,
        "latency_seconds": latency_seconds,
        "error": error,
    }
    if cost_usd is not None:
        payload["cost_usd"] = cost_usd
    if cache_hit is not None:
        payload["cache_hit"] = cache_hit
    return graph.emit(LLM_RESPONDED, payload)


def ask_with_observability(
    graph,
    reader: Any,
    *,
    request_id: str,
    question: str,
    context: str,
) -> tuple[str, str, float, Event, Event]:
    """The full LLM observability seam.

    Emits `llm.requested` → invokes `reader.answer()` (timing + error
    capture) → emits `llm.responded` with the recorded answer.

    Returns (answer, error, latency_seconds, llm_requested_event,
    llm_responded_event)."""
    model = getattr(reader, "name", "")
    req = emit_llm_request(graph, request_id=request_id, model=model, prompt=context)
    answer, error, latency = call_reader(
        reader, context=context, question=question, question_id=request_id,
    )
    resp = emit_llm_response(
        graph,
        request_id=request_id,
        caused_by=req.id,
        model=model,
        answer=answer,
        latency_seconds=latency,
        error=error,
    )
    return answer, error, latency, req, resp


def causal_chain_text(graph, object_id: str) -> str:
    """Render the causal chain from an object back to its origin
    (including any LLM round-trips with model + cost)."""
    try:
        from activegraph.trace.causal import causal_chain

        return causal_chain(graph, object_id)
    except Exception as e:  # noqa: BLE001 — renderer is best-effort
        return f"(causal chain unavailable: {type(e).__name__}: {e})"


def configure_logging(**kwargs: Any) -> Any:
    """Opinionated JSON-line structured logging (activegraph schema:
    run_id, event_id, behavior, model, cost_usd, latency_seconds, ...)."""
    from activegraph.observability.logging import configure_logging as _cfg

    return _cfg(**kwargs)


def summarize_llm_usage(store_path: str | Path | None = None) -> dict[str, Any]:
    """Aggregate the LLM observability trail across stored runs.

    Reads every `llm.responded` event from the event store and returns
    call counts, average latency, and error count — the consumption side
    of the mentor's cost/latency observation (Phase 7.3)."""
    from src.runtime.event_store import list_runs, run_events

    if store_path is None:
        store_path = ""
    n_calls = 0
    n_errors = 0
    latencies: list[float] = []
    for run in list_runs(store_path):
        for ev in run_events(store_path, run.run_id):
            if ev.type != LLM_RESPONDED:
                continue
            n_calls += 1
            if ev.payload.get("error"):
                n_errors += 1
            lat = ev.payload.get("latency_seconds")
            if isinstance(lat, (int, float)):
                latencies.append(float(lat))
    return {
        "n_runs": len(list_runs(store_path)),
        "n_llm_calls": n_calls,
        "avg_latency_seconds": round(sum(latencies) / len(latencies), 4) if latencies else 0.0,
        "n_llm_errors": n_errors,
    }
