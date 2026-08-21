"""Phase 4 acceptance tests: shared runtime services and glue.

Acceptance (brick-agent-plan.md Phase 4): all runtime services have
unit tests — reader_registry, embedder, history_logger, guardrails,
loader.
"""

from pathlib import Path

import pytest

from src.runtime import guardrails, history_logger, loader, reader_registry
from src.runtime.embedder import (
    cosine_similarity,
    embed,
    embed_one,
    rank_by_similarity,
)

KB = Path(__file__).resolve().parents[1] / ".okf" / "01_nano_vllm"

# pytest's default tmp_path lives outside the sandbox; use a fresh
# workspace-local dir instead.
WS_TMP = Path(__file__).resolve().parents[1] / ".runtime-tests"


@pytest.fixture
def ws_tmp_dir():
    d = WS_TMP / "history"
    d.mkdir(parents=True, exist_ok=True)
    yield d
    import shutil

    shutil.rmtree(WS_TMP, ignore_errors=True)


# ---------------------------------------------------------------------------
# reader_registry
# ---------------------------------------------------------------------------


def test_reader_registry_set_get_clear():
    class R:
        name = "r"

    reader_registry.set_reader("q1", R())
    assert reader_registry.get_reader("q1").name == "r"
    assert reader_registry.get_reader("missing") is None
    reader_registry.clear_reader("q1")
    assert reader_registry.get_reader("q1") is None


def test_reader_registry_clear_all():
    class R:
        name = "r"

    reader_registry.set_reader("a", R())
    reader_registry.set_reader("b", R())
    reader_registry.clear_all_readers()
    assert reader_registry.get_reader("a") is None
    assert reader_registry.get_reader("b") is None


def test_call_reader_success_and_error():
    class Good:
        name = "good"

        def answer(self, *, context, question, question_id):
            return f"answer:{question_id}"

    answer, error, latency = reader_registry.call_reader(
        Good(), context="ctx", question="q", question_id="id1",
    )
    assert answer == "answer:id1" and error == "" and latency >= 0.0

    class Bad:
        name = "bad"

        def answer(self, *, context, question, question_id):
            raise RuntimeError("boom")

    answer, error, latency = reader_registry.call_reader(
        Bad(), context="ctx", question="q", question_id="id2",
    )
    assert answer == "" and "RuntimeError" in error and latency >= 0.0


# ---------------------------------------------------------------------------
# embedder
# ---------------------------------------------------------------------------


def test_embed_returns_l2_normalized_vectors():
    vecs = embed(["hello world", "another text"])
    assert len(vecs) == 2 and len(vecs[0]) > 0
    for v in vecs:
        norm = sum(x * x for x in v) ** 0.5
        assert abs(norm - 1.0) < 1e-6


def test_embed_one_and_cosine():
    v = embed_one("kv cache memory")
    assert len(v) > 0
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-6


def test_rank_by_similarity_orders_related_first():
    ranked = rank_by_similarity(
        "kv cache memory pool",
        ["kv cache", "pizza toppings", "block allocator memory"],
    )
    assert len(ranked) == 3
    assert ranked[0][0] != "pizza toppings"  # token-overlap ranking


# ---------------------------------------------------------------------------
# history_logger
# ---------------------------------------------------------------------------


def test_history_logger_append_roundtrip(ws_tmp_dir):
    p = ws_tmp_dir / "history.yaml"
    rec1 = history_logger.append_operator(
        p, op="ADD_BEHAVIOR", target="okf_agent.foo", reason="test",
        timestamp="2026-08-14T00:00:00Z",
    )
    assert rec1["id"] == "op_001" and rec1["op"] == "ADD_BEHAVIOR"

    rec2 = history_logger.append_operator(
        p, op="ADD_EVENT_TYPE", target="llm.requested", reason="test",
        timestamp="2026-08-14T00:00:01Z",
    )
    assert rec2["id"] == "op_002"

    data = history_logger.read_history(p)
    assert data["version"] == "1.0.2"
    assert len(data["history"]) == 2
    assert data["history"][-1]["target"] == "llm.requested"


def test_history_logger_rejects_non_canonical_op(ws_tmp_dir):
    with pytest.raises(ValueError):
        history_logger.append_operator(
            ws_tmp_dir / "h.yaml", op="DELETE_EVERYTHING", target="x", reason="r",
        )


def test_history_logger_bump_version():
    assert history_logger.bump_version("1.2.0") == "1.2.1"
    assert history_logger.bump_version("1.0") == "1.0.1"


# ---------------------------------------------------------------------------
# guardrails
# ---------------------------------------------------------------------------


def test_guardrails_load_defaults_and_config():
    rules = guardrails.load_guardrails()
    assert rules["sql"]["allow_only_select"] is True
    assert rules["okf"]["required_lint_before_ask"] is True
    assert rules["mentor"]["min_improvement_threshold"] == 0.02
    # The repo config file exists and merges over defaults.
    assert guardrails.DEFAULT_PATH.is_file()
    merged = guardrails.load_guardrails(guardrails.DEFAULT_PATH)
    assert merged["sql"]["allow_only_select"] is True


def test_check_sql_safety():
    ok, reasons = guardrails.check_sql_safety("SELECT id FROM users WHERE tier='vip'")
    assert ok and reasons == []

    ok, reasons = guardrails.check_sql_safety("DROP TABLE users")
    assert not ok
    assert any("forbidden keyword: drop" in r for r in reasons)

    ok, reasons = guardrails.check_sql_safety("SELECT 1; DELETE FROM logs")
    assert not ok


def test_check_okf_lint_before_ask():
    ok, reasons = guardrails.check_okf_lint_before_ask({"valid": True, "n_errors": 0})
    assert ok and reasons == []

    ok, reasons = guardrails.check_okf_lint_before_ask({"valid": False, "n_errors": 3})
    assert not ok and any("lint not clean" in r for r in reasons)

    ok, reasons = guardrails.check_okf_lint_before_ask(None)
    assert not ok


def test_check_mentor_promotion():
    class Diff:
        def __init__(self, delta):
            self.overall_delta = delta

    ok, _ = guardrails.check_mentor_promotion(Diff(0.05))
    assert ok
    ok, reasons = guardrails.check_mentor_promotion(Diff(0.0))
    assert not ok and any("floor" in r for r in reasons)


# ---------------------------------------------------------------------------
# loader
# ---------------------------------------------------------------------------


def test_load_kb_graph_and_populate():
    snapshot = loader.load_kb_graph(KB)
    assert len(snapshot.concept_ids) > 0

    from activegraph import FrozenClock, Graph, IDGen

    graph = Graph(ids=IDGen(), clock=FrozenClock("2026-01-01T00:00:00Z"), run_id="t")
    n_objects, n_relations = loader.populate_graph(graph, snapshot)
    assert n_objects > 0
    assert n_relations >= 0
    assert len(graph.objects(type="concept")) > 0


def test_build_session():
    session = loader.build_session(KB)
    assert session.n_objects > 0
    assert session.snapshot is not None
    assert session.graph.run_id == "brick-runtime-session"
