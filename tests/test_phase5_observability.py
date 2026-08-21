"""Phase 5 acceptance tests: event store, observability seam, replay.

Acceptance (brick-agent-plan.md Phase 5): all LLM calls land in the log
(model·prompt_hash·answer·latency), a stored run replays, and the
replay cache serves recorded answers deterministically.
"""

import hashlib
from pathlib import Path

from src.agents.okf import ask as okf_ask
from src.core.targets.okf.eval import FakeOkfReader
from src.core.targets.sql.agent.agent import retrieve
from src.core.targets.sql.eval import FakeSqlReader
from src.runtime import event_store, observability, replay

KB = Path(__file__).resolve().parents[1] / ".okf" / "01_nano_vllm"

GOLD_Q = "How does prefill_phase relate to decode_phase?"
GOLD_A = "prefill_phase is a prerequisite of decode_phase"


def _gold_reader(qid):
    return FakeOkfReader({qid: (GOLD_A, GOLD_A, "")})


def test_event_store_persists_okf_ask_run(ws_tmp):
    store = ws_tmp / "events.db"
    trace = okf_ask(
        request_id="r1", question=GOLD_Q, kb_id="nano", kb_path=KB,
        reader=_gold_reader("r1"), store_path=store,
    )
    assert trace.payload["answer"] == GOLD_A

    runs = event_store.list_runs(store)
    assert len(runs) >= 1
    events = event_store.run_events(store, runs[0].run_id)
    types = [e.type for e in events]
    assert "llm.requested" in types
    assert "llm.responded" in types
    assert "okf.answer.generated" in types
    # The observability record carries model/hash/answer/latency.
    resp = next(e for e in events if e.type == "llm.responded")
    assert resp.payload["answer"] == GOLD_A
    assert "latency_seconds" in resp.payload


def test_observability_helpers():
    assert observability.prompt_hash("hello") == hashlib.sha256(b"hello").hexdigest()
    text = observability.causal_chain_text(None, "obj-1")
    assert "unavailable" in text  # graceful fallback


def test_replay_cache_and_reader(ws_tmp):
    store = ws_tmp / "events.db"
    okf_ask(
        request_id="r2", question=GOLD_Q, kb_id="nano", kb_path=KB,
        reader=_gold_reader("r2"), store_path=store,
    )
    runs = event_store.list_runs(store)
    events = event_store.run_events(store, runs[0].run_id)

    cache = replay.build_replay_cache(events)
    assert cache, "replay cache must harvest prompt_hash -> answer"

    rr = replay.ReplayReader(cache)
    req = next(e for e in events if e.type == "llm.requested")
    recorded_prompt = req.payload["prompt"]
    assert rr.answer(context=recorded_prompt, question=GOLD_Q, question_id="r2") == GOLD_A
    # A miss without fallback returns the deterministic marker.
    assert rr.answer(context="some other prompt", question="q", question_id="x") == "[replay-miss: no recorded answer for this prompt]"

    # Fallback delegates to the live reader on a miss.
    with_fallback = replay.ReplayReader(cache, fallback=_gold_reader("r2"))
    assert with_fallback.answer(context="other", question="q", question_id="r2") == GOLD_A


def test_replay_into_graph_and_load_runtime(ws_tmp):
    store = ws_tmp / "events.db"
    okf_ask(
        request_id="r3", question=GOLD_Q, kb_id="nano", kb_path=KB,
        reader=_gold_reader("r3"), store_path=store,
    )
    run_id = event_store.list_runs(store)[0].run_id

    graph, n = replay.replay_into_graph(store, run_id)
    assert n >= 1 and graph.run_id == run_id

    rt = replay.load_run_runtime(store, run_id)
    assert rt.graph.run_id == run_id


def test_sql_draft_query_emits_llm_seam(ws_tmp):
    store = ws_tmp / "sql-events.db"
    instance = {
        "question_id": "q1",
        "question_type": "count",
        "question": "How many users are there?",
        "schema_id": "smoke",
        "tables": ["users"],
        "columns_by_table": {"users": ["id", "name"]},
        "foreign_keys": [],
        "primary_keys": {"users": "id"},
    }
    trace = retrieve(
        instance,
        reader=FakeSqlReader({"q1": ("SELECT COUNT(*) FROM users", "", "")}),
        store_path=store,
    )
    types = [e.type for e in trace.events]
    assert "llm.requested" in types and "llm.responded" in types
    assert trace.drafted.predicted_sql == "SELECT COUNT(*) FROM users"
    runs = event_store.list_runs(store)
    assert len(runs) >= 1
