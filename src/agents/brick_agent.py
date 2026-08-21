"""brick.agent CLI (brick-agent-plan Phase 6.2, wired with Phase 5
observability + persistence).

Usage:
    python -m src.agents.brick_agent ask sql "QUESTION" [--db PATH] [--real] [--store PATH] [--trace]
    python -m src.agents.brick_agent ask okf "QUESTION" [--kb BUNDLE] [--real] [--store PATH] [--trace]
    python -m src.agents.brick_agent ask "QUESTION"                    # router-classified
    python -m src.agents.brick_agent ingest --kb-id ID --kb-path PATH [--store PATH]
    python -m src.agents.brick_agent lint --kb-id ID [--kb-path PATH] [--store PATH]
    python -m src.agents.brick_agent db seed [--db PATH]
    python -m src.agents.brick_agent mentor status [--store PATH]

Readers: deterministic demo by default (no keys); `--real` uses
AnthropicReader (requires ANTHROPIC_API_KEY). Every run is persisted to
the event store (default data/events.db) including the llm.requested /
llm.responded observability trail.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
DEFAULT_EVENTS_DB = DATA / "events.db"
DEFAULT_SQL_DB = DATA / "store_front.db"
DEFAULT_KB = "01_nano_vllm"
OKF_ROOT = ROOT / ".okf"

# Deterministic demo tables (no LLM key needed).
_DEMO_SQL: dict[str, str] = {
    "How many stable tools are in the store front?": "SELECT COUNT(*) FROM tools WHERE status = 'stable'",
    "How many tools are in the store front?": "SELECT COUNT(*) FROM tools",
    "List the stable decisions": "SELECT title FROM decisions WHERE status = 'stable'",
    "How many skills are in the store front?": "SELECT COUNT(*) FROM skills",
    "Which bundles are in the knowledge base?": "SELECT name FROM bundles ORDER BY name",
}
_DEMO_OKF: dict[str, str] = {
    "How does prefill_phase relate to decode_phase?": "prefill_phase is a prerequisite of decode_phase",
    "What is the kv_cache?": (
        "kv_cache stores the key/value matrices of past tokens so attention "
        "does not recompute them (see concept atomic.kv_cache)."
    ),
}


class _DemoSqlReader:
    """Returns the gold SQL for known demo questions, else empty."""

    name = "demo-sql-reader"

    def __init__(self, table: dict[str, str]):
        self.table = dict(table)

    def answer(self, *, context: str, question: str, question_id: str) -> str:
        return self.table.get(question, "")


class _DemoOkfReader:
    """Returns the gold answer for known demo questions, else empty."""

    name = "demo-okf-reader"

    def __init__(self, table: dict[str, str]):
        self.table = dict(table)

    def answer(self, *, context: str, question: str, question_id: str) -> str:
        return self.table.get(question, "")


# ---------------------------------------------------------------------------
# SQL support
# ---------------------------------------------------------------------------


def introspect_schema(db_path: str | Path) -> dict:
    """Read tables / columns / PKs / FKs from a SQLite file."""
    conn = sqlite3.connect(str(db_path))
    try:
        tables = [
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        columns_by_table: dict[str, list[str]] = {}
        primary_keys: dict[str, str] = {}
        foreign_keys: list[tuple[str, str, str, str]] = []
        for t in tables:
            info = conn.execute(f"PRAGMA table_info({t})").fetchall()
            columns_by_table[t] = [r[1] for r in info]
            pks = [r[1] for r in info if r[5] > 0]
            if pks:
                primary_keys[t] = pks[0]
            for fk in conn.execute(f"PRAGMA foreign_key_list({t})").fetchall():
                # (id, seq, table, from, to, ...)
                foreign_keys.append((t, fk[3], fk[2], fk[4]))
        return {
            "tables": tables,
            "columns_by_table": columns_by_table,
            "primary_keys": primary_keys,
            "foreign_keys": foreign_keys,
        }
    finally:
        conn.close()


def _execute_sql(db_path: str | Path, sql: str) -> tuple[list[str], list[tuple]]:
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.execute(sql)
        if cur.description is None:
            return [], [("rowcount", cur.rowcount)]
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
        return cols, rows
    finally:
        conn.close()


def _sql_reader(real: bool, question: str):
    if real:
        from src.core.eval.real import build_real_reader

        return build_real_reader()
    return _DemoSqlReader(_DEMO_SQL)


def _okf_reader(real: bool, question: str):
    if real:
        from src.core.eval.real import build_real_reader

        return build_real_reader()
    return _DemoOkfReader(_DEMO_OKF)


def _resolve_kb(kb: str) -> Path:
    p = Path(kb)
    if p.is_dir():
        return p
    return OKF_ROOT / kb


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def _cmd_ask_sql(args) -> int:
    db = Path(args.db or DEFAULT_SQL_DB)
    if not db.is_file():
        from src.tools.seed_store_front import seed

        seed(db_path=db)

    schema = introspect_schema(db)
    reader = _sql_reader(args.real, args.question)
    qid = f"cli-sql-{uuid.uuid4().hex[:8]}"

    from src.core.targets.sql.agent.agent import retrieve

    trace = retrieve(
        {
            "question_id": qid,
            "question": args.question,
            "schema_id": db.name,
            **schema,
        },
        reader=reader,
        store_path=args.store or DEFAULT_EVENTS_DB,
    )
    sql = trace.drafted.predicted_sql.strip()
    error = trace.drafted.drafter_error

    print(f"reader      : {getattr(reader, 'name', '?')}")
    print(f"schema      : {db} ({len(schema['tables'])} tables)")
    print(f"question    : {args.question}")
    if error:
        print(f"draft error : {error}")
        return 1
    if not sql:
        print("draft error : (empty SQL)")
        return 1
    print(f"SQL         : {sql}")
    try:
        cols, rows = _execute_sql(db, sql)
    except sqlite3.Error as e:
        print(f"exec error  : {type(e).__name__}: {e}")
        return 1
    print(f"columns     : {cols}")
    for r in rows:
        print(f"  row       : {r}")

    if args.trace:
        _print_llm_trail_from_events(trace.events)
    return 0


def _cmd_ask_okf(args) -> int:
    kb_path = _resolve_kb(args.kb)
    reader = _okf_reader(args.real, args.question)
    request_id = f"cli-okf-{uuid.uuid4().hex[:8]}"

    from src.agents.okf import ask

    trace = ask(
        request_id=request_id,
        question=args.question,
        kb_id=f"cli-{kb_path.name}",
        kb_path=kb_path,
        reader=reader,
        top_k=args.top_k,
        store_path=args.store or DEFAULT_EVENTS_DB,
    )
    p = trace.payload or {}
    print(f"reader      : {getattr(reader, 'name', '?')}")
    print(f"kb          : {kb_path}")
    print(f"question    : {args.question}")
    if p.get("error"):
        print(f"error       : {p['error']}")
        return 1
    print(f"answer      : {p.get('answer', '')}")
    print(f"transforms  : {p.get('applied_transforms', [])}")
    concepts = (p.get("context_parts") or {}).get("concepts", [])
    print(f"concepts    : {concepts[:6]}{' ...' if len(concepts) > 6 else ''}")

    if args.trace:
        for rec in trace.llm_events:
            _print_llm_record(rec)
    return 0


def _cmd_ask(args) -> int:
    from src.agents.router import classify

    role = classify(args.question)
    print(f"role        : {role} (router)")
    if role == "sql":
        return _cmd_ask_sql(args)
    return _cmd_ask_okf(args)


def _cmd_ingest(args) -> int:
    from src.agents.okf import ingest

    trace = ingest(
        kb_id=args.kb_id,
        kb_path=args.kb_path,
        store_path=args.store or DEFAULT_EVENTS_DB,
    )
    print(f"ingest      : {trace.payload or {}}")
    return 0


def _cmd_lint(args) -> int:
    from src.agents.okf import lint

    trace = lint(
        kb_id=args.kb_id,
        kb_path=args.kb_path,
        store_path=args.store or DEFAULT_EVENTS_DB,
    )
    p = trace.payload or {}
    print(f"lint        : valid={p.get('valid')} n_errors={p.get('n_errors')}")
    for issue in p.get("issues", [])[:10]:
        print(f"  - [{issue.get('code')}] {issue.get('node')}: {issue.get('detail', '')[:80]}")
    return 0


def _cmd_db_seed(args) -> int:
    from src.tools.seed_store_front import seed

    db = seed(db_path=args.db or DEFAULT_SQL_DB)
    conn = sqlite3.connect(db)
    counts = {
        t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        for t in ("bundles", "tools", "skills", "decisions", "assets")
    }
    print(f"seeded {db}: {counts}")
    return 0


def _cmd_browse(args) -> int:
    """Generate the OKF KB web dashboard (docs/<bundle>/index.html) and
    open it in the default browser so the knowledge can be browsed and
    studied visually."""
    from src.tools.okf_visualizer import build_okf_visualizer

    kb_path = _resolve_kb(args.kb)
    out = Path(args.out) if args.out else ROOT / "docs" / kb_path.name / "index.html"
    build_okf_visualizer(kb_path, out)
    print(f"dashboard   : {out}")

    if not args.no_open:
        import webbrowser

        uri = out.resolve().as_uri()
        try:
            webbrowser.open(uri)
            print(f"opened      : {uri}")
        except Exception as e:  # noqa: BLE001 — headless envs can't open
            print(f"open failed : {e}")
            print(f"open manually: {out.resolve()}")
    return 0


def _cmd_mentor_status(args) -> int:
    from src.core.targets.sql import prompt_transforms as sql_pipe
    from src.core.targets.okf import prompt_transforms as okf_pipe
    from src.runtime import event_store, guardrails, history_logger, observability

    print(f"sql pipeline   : {[e.name for e in sql_pipe.get_pipeline()]}")
    print(f"okf pipeline   : {[e.name for e in okf_pipe.get_pipeline()]}")
    g = guardrails.load_guardrails()
    print(f"guardrails     : sql.allow_only_select={g['sql']['allow_only_select']} "
          f"okf.required_lint_before_ask={g['okf']['required_lint_before_ask']} "
          f"mentor.min_improvement_threshold={g['mentor']['min_improvement_threshold']}")
    store = str(args.store or DEFAULT_EVENTS_DB)
    runs = event_store.list_runs(store)
    print(f"stored runs    : {len(runs)} ({store})")
    usage = observability.summarize_llm_usage(store)
    print(f"llm usage      : calls={usage['n_llm_calls']} "
          f"avg_latency={usage['avg_latency_seconds']}s errors={usage['n_llm_errors']}")
    hist = history_logger.read_history(ROOT / ".okf" / "00_agent_model" / "history.yaml")
    print(f"agent model    : history v{hist.get('version')} ({len(hist.get('history', []))} operators)")
    return 0


# ---------------------------------------------------------------------------
# Observability printing
# ---------------------------------------------------------------------------


def _print_llm_trail_from_events(events) -> None:
    for ev in events:
        if ev.type in ("llm.requested", "llm.responded"):
            rec = dict(ev.payload)
            rec["id"] = ev.id
            rec["event_type"] = ev.type
            _print_llm_record(rec)


def _print_llm_record(rec: dict) -> None:
    etype = rec.get("event_type", "")
    if etype == "llm.requested":
        print(f"  llm.req      : id={rec.get('id')} model={rec.get('model')} hash={(rec.get('prompt_hash') or '')[:12]}")
    elif etype == "llm.responded":
        ans = str(rec.get("answer", ""))[:60]
        print(f"  llm.resp     : caused_by={rec.get('caused_by')} latency={rec.get('latency_seconds'):.4f}s "
              f"error={rec.get('error') or '-'} answer={ans!r}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="brick-agent", description="brick.agent CLI")
    sub = ap.add_subparsers(dest="command", required=True)

    p_ask = sub.add_parser("ask", help="Ask a question (router-classified unless a role is given)")
    p_ask.add_argument("role", nargs="?", choices=["sql", "okf"], help="optional explicit role")
    p_ask.add_argument("question")
    _add_common(p_ask)
    p_ask.add_argument("--db", default=str(DEFAULT_SQL_DB))
    p_ask.add_argument("--kb", default=DEFAULT_KB)
    p_ask.add_argument("--top-k", type=int, default=10)

    p_sql = sub.add_parser("ask-sql", aliases=["sql"], help="Natural-language DB query")
    p_sql.add_argument("question")
    _add_common(p_sql)
    p_sql.add_argument("--db", default=str(DEFAULT_SQL_DB))

    p_okf = sub.add_parser("ask-okf", aliases=["okf"], help="OKF knowledge Q&A")
    p_okf.add_argument("question")
    _add_common(p_okf)
    p_okf.add_argument("--kb", default=DEFAULT_KB)
    p_okf.add_argument("--top-k", type=int, default=10)

    p_ingest = sub.add_parser("ingest", help="Ingest an OKF bundle")
    p_ingest.add_argument("--kb-id", required=True)
    p_ingest.add_argument("--kb-path", required=True)
    _add_common(p_ingest)

    p_lint = sub.add_parser("lint", help="Lint an OKF bundle")
    p_lint.add_argument("--kb-id", required=True)
    p_lint.add_argument("--kb-path", default=None)
    _add_common(p_lint)

    p_db = sub.add_parser("db", help="Database helpers")
    p_db_sub = p_db.add_subparsers(dest="db_command", required=True)
    p_seed = p_db_sub.add_parser("seed", help="Seed data/store_front.db from .okf/02_store_front")
    p_seed.add_argument("--db", default=str(DEFAULT_SQL_DB))

    p_mentor = sub.add_parser("mentor", help="Mentor loop helpers")
    p_mentor_sub = p_mentor.add_subparsers(dest="mentor_command", required=True)
    p_status = p_mentor_sub.add_parser("status", help="Show learning/observability status")
    _add_common(p_status)

    p_browse = sub.add_parser("browse", help="Generate + open the OKF KB web dashboard")
    p_browse.add_argument("--kb", default=DEFAULT_KB, help="OKF bundle name or path")
    p_browse.add_argument("--out", default=None, help="output HTML path (default docs/<kb>/index.html)")
    p_browse.add_argument("--no-open", action="store_true", help="generate only, do not open the browser")

    return ap


def _add_common(p) -> None:
    p.add_argument("--real", action="store_true", help="use AnthropicReader (requires ANTHROPIC_API_KEY)")
    p.add_argument("--store", default=str(DEFAULT_EVENTS_DB), help="event store SQLite path")
    p.add_argument("--trace", action="store_true", help="print the llm.* observability trail")


def main(argv: list[str] | None = None) -> int:
    # Console-encoding safety (Windows cp949 chokes on em-dashes etc.).
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    cmd = args.command
    if cmd == "ask":
        if args.role == "sql":
            return _cmd_ask_sql(args)
        if args.role == "okf":
            return _cmd_ask_okf(args)
        return _cmd_ask(args)
    if cmd in ("ask-sql", "sql"):
        return _cmd_ask_sql(args)
    if cmd in ("ask-okf", "okf"):
        return _cmd_ask_okf(args)
    if cmd == "ingest":
        return _cmd_ingest(args)
    if cmd == "lint":
        return _cmd_lint(args)
    if cmd == "db":
        return _cmd_db_seed(args)
    if cmd == "browse":
        return _cmd_browse(args)
    if cmd == "mentor":
        return _cmd_mentor_status(args)
    print(f"unknown command: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
