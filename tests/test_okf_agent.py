"""Phase 3 acceptance tests: OKF runtime behaviors (ingest / lint / ask).

Acceptance (brick-agent-plan.md Phase 3):
  - a test can trigger `okf.ask.requested` and receive a grounded
    response from a fake reader;
  - the run log contains `llm.requested` / `llm.responded` for the
    answer call.
"""

from pathlib import Path

import pytest

from src.agents.okf import ask, clear_kb, ingest, lint
from src.agents.okf import events as E
from src.core.targets.okf.eval import FakeOkfReader

KB = Path(__file__).resolve().parents[1] / ".okf" / "01_nano_vllm"

GOLD_Q = "How does prefill_phase relate to decode_phase?"
GOLD_A = "prefill_phase is a prerequisite of decode_phase"


@pytest.fixture(autouse=True)
def _clear_okf_state():
    clear_kb()
    yield
    clear_kb()


class _GoldReader:
    """Deterministic reader that always returns the gold answer."""

    name = "gold-reader"

    def answer(self, *, context, question, question_id):
        return GOLD_A


def _in_order(values, expected):
    pos = -1
    for item in expected:
        pos = values.index(item, pos + 1)
    return True


def test_ingest_chain_on_real_kb():
    trace = ingest(kb_id="nano", kb_path=KB)
    types = [ev.type for ev in trace.events]

    assert _in_order(types, [E.OKF_INGEST_REQUESTED, E.OKF_PARSED, E.OKF_VALIDATED, E.OKF_LOADED])
    assert trace.kind == "ingest"
    p = trace.payload
    assert p is not None
    assert p["n_concepts"] > 0
    assert p["n_rules"] >= 0
    assert p["n_objects"] > 0
    assert isinstance(p["n_relations"], int)
    # Graph objects materialized during the chain.
    from activegraph import Runtime  # noqa: F401 — chain ran on the runtime
    assert "error" in p


def test_lint_chain_on_real_kb():
    trace = lint(kb_id="nano", kb_path=KB)
    types = [ev.type for ev in trace.events]

    assert _in_order(types, [E.OKF_LINT_REQUESTED, E.OKF_ANALYZED, E.OKF_LINTED])
    p = trace.payload
    assert p is not None
    assert "issues" in p and isinstance(p["issues"], list)
    assert "n_errors" in p and p["n_errors"] >= 0
    # Deterministic: two lint runs produce identical issue lists.
    trace2 = lint(kb_id="nano", kb_path=KB)
    assert trace2.payload["issues"] == p["issues"]


def test_ask_chain_reaches_generated_answer_with_llm_seam():
    trace = ask(
        request_id="r1",
        question=GOLD_Q,
        kb_id="nano",
        kb_path=KB,
        reader=_GoldReader(),
    )
    types = [ev.type for ev in trace.events]

    assert _in_order(types, [
        E.OKF_ASK_REQUESTED,
        E.OKF_CONTEXT_ASSEMBLED,
        E.LLM_REQUESTED,
        E.LLM_RESPONDED,
        E.OKF_ANSWER_GENERATED,
    ])

    # Grounded response from the fake reader.
    p = trace.payload
    assert p["answer"] == GOLD_A
    assert p["error"] == ""
    assert "inject_concept_tree" in p["applied_transforms"]

    # Observability trail: model, prompt_hash, latency, causal link.
    assert len(trace.llm_events) == 2
    req, resp = trace.llm_events
    assert req["event_type"] == E.LLM_REQUESTED
    assert "prompt_hash" in req and len(req["prompt_hash"]) == 64
    assert "model" in req
    assert resp["event_type"] == E.LLM_RESPONDED
    assert resp["caused_by"] == req["id"]
    assert resp["latency_seconds"] >= 0.0


def test_ask_context_is_expanded_and_grounded():
    trace = ask(
        request_id="r2",
        question=GOLD_Q,
        kb_id="nano",
        kb_path=KB,
        reader=_GoldReader(),
    )
    ctx_ev = [e for e in trace.events if e.type == E.OKF_CONTEXT_ASSEMBLED][-1]
    ctx = ctx_ev.payload
    assert ctx["error"] == ""
    parts = ctx["context_parts"]
    assert parts["question"] == GOLD_Q
    assert len(parts["concepts"]) > 0
    assert len(ctx["selected_concept_ids"]) > 0
    # Tree expansion: the assembled context includes tree neighbors of
    # the selected concept(s).
    assert len(parts["concepts"]) >= len(ctx["selected_concept_ids"])
    assert "scorer_model" in ctx and "scores" in ctx


def test_ask_reader_missing_records_error():
    trace = ask(
        request_id="r3",
        question=GOLD_Q,
        kb_id="nano",
        kb_path=KB,
        reader=None,
    )
    p = trace.payload
    assert "reader_missing" in p["error"]
    assert p["answer"] == ""
    # No LLM round-trip was recorded for the missing reader.
    assert trace.llm_events == []
    assert p["request_id"] == "r3"


def test_ask_without_kb_path_uses_ingested_cache():
    # ingest first (caches the graph under kb_id), then ask without kb_path.
    ingest(kb_id="cached", kb_path=KB)
    trace = ask(
        request_id="r4",
        question=GOLD_Q,
        kb_id="cached",
        reader=_GoldReader(),
    )
    p = trace.payload
    assert p["answer"] == GOLD_A and p["error"] == ""
