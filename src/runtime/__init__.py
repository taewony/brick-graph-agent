"""brick.agent runtime glue (brick-agent-plan Phase 4 + 5).

Shared services every role (sql / okf / mentor / router) uses:

    reader_registry  — per-request Reader side table + invocation wrapper
    embedder         — thin wrapper over src.core.agent.embedders
    history_logger   — canonical operator appends to .okf/*/history.yaml
    guardrails       — config/agent_model/guardrails.yaml enforcement
    loader           — OKF bundle → OkfKnowledgeGraph → ActiveGraph objects
    event_store      — durable SQLiteEventStore + run metadata (Phase 5)
    observability    — llm.requested/llm.responded seam + logging/causal (Phase 5)
    replay           — deterministic replay + LLM response cache (Phase 5)
"""

from __future__ import annotations

from src.runtime import (
    embedder,
    event_store,
    guardrails,
    history_logger,
    loader,
    observability,
    reader_registry,
    replay,
)

__all__ = [
    "embedder",
    "event_store",
    "guardrails",
    "history_logger",
    "loader",
    "observability",
    "reader_registry",
    "replay",
]
