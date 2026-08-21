"""Canonical OKF agent events.

Event names MUST match `.okf/00_agent_model/events.yaml` (the single
source of truth). The `llm.*` / `tool.*` events are the observability
seam: every Reader call in the ask chain is recorded as
`llm.requested` / `llm.responded`.
"""

from __future__ import annotations

# ---- okf ingest chain ----
OKF_INGEST_REQUESTED = "okf.ingest.requested"
OKF_PARSED = "okf.parsed"
OKF_VALIDATED = "okf.validated"
OKF_LOADED = "okf.loaded"

# ---- okf lint chain ----
OKF_LINT_REQUESTED = "okf.lint.requested"
OKF_ANALYZED = "okf.analyzed"
OKF_LINTED = "okf.linted"

# ---- okf ask chain ----
OKF_ASK_REQUESTED = "okf.ask.requested"
OKF_CONTEXT_ASSEMBLED = "okf.context.assembled"
OKF_ANSWER_GENERATED = "okf.answer.generated"

# ---- observability seam (shared with sql / mentor) ----
LLM_REQUESTED = "llm.requested"
LLM_RESPONDED = "llm.responded"
TOOL_REQUESTED = "tool.requested"
TOOL_RESPONDED = "tool.responded"

__all__ = [
    "OKF_INGEST_REQUESTED",
    "OKF_PARSED",
    "OKF_VALIDATED",
    "OKF_LOADED",
    "OKF_LINT_REQUESTED",
    "OKF_ANALYZED",
    "OKF_LINTED",
    "OKF_ASK_REQUESTED",
    "OKF_CONTEXT_ASSEMBLED",
    "OKF_ANSWER_GENERATED",
    "LLM_REQUESTED",
    "LLM_RESPONDED",
    "TOOL_REQUESTED",
    "TOOL_RESPONDED",
]
