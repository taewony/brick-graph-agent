"""brick.agent CLI tests — okf/sql 개별 Q&A + db seed + mentor status."""

import sqlite3
from pathlib import Path

from src.agents.brick_agent import main
from src.agents.router import classify

KB = Path(__file__).resolve().parents[1] / ".okf" / "01_nano_vllm"

SQL_Q = "How many stable tools are in the store front?"
OKF_Q = "How does prefill_phase relate to decode_phase?"
OKF_A = "prefill_phase is a prerequisite of decode_phase"


def test_router_classify():
    assert classify(SQL_Q) == "sql"
    assert classify(OKF_Q) == "okf_ask"
    assert classify("lint the knowledge base") == "okf_lint"
    assert classify("ingest a new bundle") == "okf_ingest"
    assert classify("mentor status") == "mentor"


def test_cli_db_seed(ws_tmp):
    db = ws_tmp / "store_front.db"
    rc = main(["db", "seed", "--db", str(db)])
    assert rc == 0
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT COUNT(*) FROM tools").fetchone()[0] == 5
    assert conn.execute("SELECT COUNT(*) FROM skills").fetchone()[0] == 3
    assert conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] >= 5
    assert conn.execute("SELECT COUNT(*) FROM bundles").fetchone()[0] >= 3


def test_cli_ask_sql_demo(capsys, ws_tmp):
    db = ws_tmp / "store_front.db"
    store = ws_tmp / "events.db"
    rc = main(["ask", "sql", SQL_Q, "--db", str(db), "--store", str(store)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "SELECT COUNT(*) FROM tools WHERE status = 'stable'" in out
    assert "row" in out  # execution result printed


def test_cli_ask_sql_trace(capsys, ws_tmp):
    db = ws_tmp / "store_front.db"
    store = ws_tmp / "events.db"
    rc = main(["ask", "sql", SQL_Q, "--db", str(db), "--store", str(store), "--trace"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "llm.req" in out and "llm.resp" in out


def test_cli_ask_okf_demo(capsys, ws_tmp):
    store = ws_tmp / "events.db"
    rc = main(["ask", "okf", OKF_Q, "--kb", str(KB), "--store", str(store)])
    out = capsys.readouterr().out
    assert rc == 0
    assert OKF_A in out
    assert "inject_concept_tree" in out


def test_cli_ask_okf_trace(capsys, ws_tmp):
    store = ws_tmp / "events.db"
    rc = main(["ask", "okf", OKF_Q, "--kb", str(KB), "--store", str(store), "--trace"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "llm.req" in out and "llm.resp" in out


def test_cli_ask_router_dispatch(capsys, ws_tmp):
    db = ws_tmp / "store_front.db"
    store = ws_tmp / "events.db"
    rc = main(["ask", SQL_Q, "--db", str(db), "--store", str(store)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "role        : sql (router)" in out


def test_cli_ingest_and_lint(capsys, ws_tmp):
    store = ws_tmp / "events.db"
    rc = main(["ingest", "--kb-id", "cli-nano", "--kb-path", str(KB), "--store", str(store)])
    assert rc == 0
    out1 = capsys.readouterr().out
    assert "n_concepts" in out1

    rc = main(["lint", "--kb-id", "cli-nano", "--store", str(store)])
    out2 = capsys.readouterr().out
    assert rc == 0
    assert "n_errors" in out2


def test_cli_mentor_status(capsys, ws_tmp):
    store = ws_tmp / "events.db"
    rc = main(["mentor", "status", "--store", str(store)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "sql pipeline" in out and "okf pipeline" in out
    assert "guardrails" in out and "stored runs" in out and "agent model" in out


def test_cli_browse_generates_dashboard(ws_tmp):
    out = ws_tmp / "dashboard.html"
    rc = main(["browse", "--kb", str(KB), "--out", str(out), "--no-open"])
    assert rc == 0
    assert out.is_file() and out.stat().st_size > 10_000
    text = out.read_text(encoding="utf-8")
    assert "vis-network" in text
    assert "atomic.prefill_phase" in text
    # Browser-style back/forward navigation is embedded.
    assert "btn-back" in text and "btn-forward" in text
    assert "function goBack" in text and "function goForward" in text
    assert "ArrowLeft" in text  # Alt+← shortcut
