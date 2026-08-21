"""Shared per-request Reader registry + invocation wrapper.

The single process-level side table for Readers keyed by
`question_id` / `request_id` — Python callables cannot ride event
payloads, so entry points register the Reader here and behaviors read
it back (the indirection the SQL agent pioneered, now shared by every
role). Phase 5's `llm.requested` / `llm.responded` observability helper
wraps `call_reader`.
"""

from __future__ import annotations

import threading
import time
from typing import Any

_LOCK = threading.Lock()
_REGISTRY: dict[str, Any] = {}


def set_reader(request_id: str, reader: Any) -> None:
    """Register a Reader for a request before the event chain starts."""
    with _LOCK:
        _REGISTRY[request_id] = reader


def get_reader(request_id: str) -> Any:
    """The Reader registered for a request, or None."""
    with _LOCK:
        return _REGISTRY.get(request_id)


def clear_reader(request_id: str) -> None:
    """Drop one request's Reader (memory hygiene after the chain)."""
    with _LOCK:
        _REGISTRY.pop(request_id, None)


def clear_all_readers() -> None:
    """Drop every Reader. Test isolation only."""
    with _LOCK:
        _REGISTRY.clear()


def call_reader(
    reader: Any,
    *,
    context: str,
    question: str,
    question_id: str,
) -> tuple[str, str, float]:
    """Invoke `reader.answer()` with timing + error capture.

    Returns (answer, error, latency_seconds). `error` is "" on success;
    exceptions are captured (framework failure model — never raised).
    """
    t0 = time.perf_counter()
    try:
        answer = reader.answer(
            context=context, question=question, question_id=question_id,
        )
        error = ""
    except Exception as e:  # noqa: BLE001 — runtime path
        answer = ""
        error = f"{type(e).__name__}: {e}"
    latency = time.perf_counter() - t0
    if not isinstance(answer, str):
        answer = str(answer or "")
    return answer, error, latency
