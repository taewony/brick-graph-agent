"""Request router — classifies natural-language requests into roles
(brick-agent-plan Phase 6.1).

`classify` is deterministic (keyword-first, embedding fallback optional)
and returns one of the canonical roles: sql / okf_ingest / okf_lint /
okf_ask / mentor. The CLI's generic `ask` subcommand uses it to dispatch
to the right role chain.
"""

from __future__ import annotations

from src.runtime.embedder import cosine_similarity, embed_one

ROLES = ("sql", "okf_ingest", "okf_lint", "okf_ask", "mentor")

_SQL_MARKERS = (
    "select", "query the database", "database query", "sql", "how many",
    "count ", "table", "column", "db",
)
_INGEST_MARKERS = ("ingest", "load knowledge", "add knowledge", "import bundle")
_LINT_MARKERS = ("lint", "validate", "check integrity", "integrity check")
_MENTOR_MARKERS = ("mentor", "improve", "learning status", "how is the agent doing")

_PROTOTYPES = {
    "sql": "database query select table column count",
    "okf_ingest": "ingest load knowledge bundle okf",
    "okf_lint": "lint validate check integrity okf",
    "okf_ask": "okf knowledge concept rule answer explain",
    "mentor": "mentor improve learning status transform",
}


def classify(text: str, *, use_embedding: bool = False) -> str:
    """Map a user request to a role.

    Keyword heuristics first (deterministic); with `use_embedding=True`,
    a tie or no-marker falls back to embedding-similarity against role
    prototypes."""
    t = (text or "").lower()

    if any(m in t for m in _SQL_MARKERS):
        return "sql"
    if any(m in t for m in _INGEST_MARKERS):
        return "okf_ingest"
    if any(m in t for m in _LINT_MARKERS):
        return "okf_lint"
    if any(m in t for m in _MENTOR_MARKERS):
        return "mentor"
    if any(k in t for k in ("okf", "knowledge", "concept", "rule", "what is", "how does", "explain")):
        return "okf_ask"

    if use_embedding:
        q = embed_one(text)
        best, best_score = "okf_ask", -1.0
        for role, proto in _PROTOTYPES.items():
            s = cosine_similarity(q, embed_one(proto))
            if s > best_score:
                best, best_score = role, s
        return best
    return "okf_ask"
