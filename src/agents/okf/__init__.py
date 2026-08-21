"""OKF agent — event-driven runtime behaviors (brick-agent-plan Phase 3).

Public surface:
    from src.agents.okf import ask, ingest, lint, OkfTrace
    from src.agents.okf import events as E
"""

from __future__ import annotations

from src.agents.okf import events as E  # noqa: F401  (re-export)
from src.agents.okf.agent import OkfTrace, ask, clear_kb, ingest, lint

__all__ = [
    "E",
    "OkfTrace",
    "ask",
    "clear_kb",
    "ingest",
    "lint",
]
