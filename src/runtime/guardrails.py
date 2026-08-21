"""Guardrails service — config/agent_model/guardrails.yaml enforcement.

Rules merge over built-in defaults (a missing file never breaks the
runtime). Checks:

    check_sql_safety(sql)                — SELECT-only + forbidden keywords
    check_okf_lint_before_ask(lint)      — ask requires a clean lint pass
    check_mentor_promotion(diff)         — improvement must clear the floor
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_GUARDRAILS: dict[str, Any] = {
    "sql": {
        "allow_only_select": True,
        "forbidden_keywords": [
            "drop", "alter", "delete", "insert", "update",
            "truncate", "pragma", "attach", "detach",
        ],
        "max_columns_per_query": 20,
    },
    "okf": {
        "required_lint_before_ask": True,
        "max_rules_per_answer": 10,
    },
    "mentor": {
        "min_validation_samples": 50,
        "min_improvement_threshold": 0.02,
    },
}

# config/agent_model/guardrails.yaml (repo root / config)
DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config" / "agent_model" / "guardrails.yaml"


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_guardrails(path: str | Path | None = None) -> dict[str, Any]:
    """Load guardrails, merging the YAML file over the built-in defaults."""
    p = Path(path) if path is not None else DEFAULT_PATH
    if not p.is_file():
        return DEFAULT_GUARDRAILS
    try:
        override = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001 — malformed config degrades to defaults
        return DEFAULT_GUARDRAILS
    return _merge(DEFAULT_GUARDRAILS, override)


# ---------------------------------------------------------------------------
# Checks — each returns (ok: bool, reasons: list[str])
# ---------------------------------------------------------------------------


def check_sql_safety(sql: str, rules: dict[str, Any] | None = None) -> tuple[bool, list[str]]:
    rules = rules or load_guardrails()["sql"]
    reasons: list[str] = []
    lower = (sql or "").lower()
    if rules.get("allow_only_select", True) and not lower.lstrip().startswith("select"):
        reasons.append("sql must be a single SELECT statement")
    for kw in rules.get("forbidden_keywords", []):
        if re_search_word(kw, lower):
            reasons.append(f"forbidden keyword: {kw}")
    return (len(reasons) == 0, reasons)


def re_search_word(keyword: str, text: str) -> bool:
    import re

    return re.search(rf"\b{re.escape(keyword)}\b", text) is not None


def check_okf_lint_before_ask(
    lint_payload: dict[str, Any] | None,
    rules: dict[str, Any] | None = None,
) -> tuple[bool, list[str]]:
    rules = rules or load_guardrails()["okf"]
    reasons: list[str] = []
    if rules.get("required_lint_before_ask", True):
        if lint_payload is None:
            reasons.append("okf lint result missing (required_lint_before_ask)")
        elif not lint_payload.get("valid", False):
            reasons.append(
                f"okf lint not clean: n_errors={lint_payload.get('n_errors', '?')}"
            )
    return (len(reasons) == 0, reasons)


def check_mentor_promotion(
    diff: Any,
    rules: dict[str, Any] | None = None,
) -> tuple[bool, list[str]]:
    rules = rules or load_guardrails()["mentor"]
    reasons: list[str] = []
    floor = float(rules.get("min_improvement_threshold", 0.02))
    delta = getattr(diff, "overall_delta", None)
    if delta is None:
        reasons.append("diff has no overall_delta")
    elif delta < floor:
        reasons.append(f"overall delta {delta:+.4f} < floor {floor:+.4f}")
    return (len(reasons) == 0, reasons)
