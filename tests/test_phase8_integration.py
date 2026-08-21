"""Phase 8 integration tests (brick-agent-plan Phase 8.1).

Covers: SQL query generation, OKF ingest → lint → ask with the
lint-before-ask guardrail, and observability (every ask records
llm.requested/llm.responded; a stored run replays identically).
"""

from pathlib import Path

from src.agents.okf import ask as okf_ask
from src.agents.okf import ingest as okf_ingest
from src.agents.okf import lint as okf_lint
from src.core.targets.okf.eval import FakeOkfReader
from src.runtime import event_store, guardrails, replay

KB = Path(__file__).resolve().parents[1] / ".okf" / "01_nano_vllm"

QUESTION = "How does prefill_phase relate to decode_phase?"
GOLD = "prefill_phase is a prerequisite of decode_phase"


def test_okf_ingest_lint_ask_integration(ws_tmp):
    store = ws_tmp / "events.db"
    kb_id = "integ-nano"

    # ingest → lint (clean KB now) → ask
    okf_ingest(kb_id=kb_id, kb_path=KB, store_path=store)
    lint_result = okf_lint(kb_id=kb_id, store_path=store)
    p = lint_result.payload or {}
    assert p.get("valid") is True, p.get("issues")

    ask_result = okf_ask(
        request_id="i1", question=QUESTION, kb_id=kb_id,
        reader=FakeOkfReader({"i1": (GOLD, GOLD, "")}), store_path=store,
    )
    assert ask_result.payload["answer"] == GOLD
    assert ask_result.payload["error"] == ""


def test_lint_before_ask_guardrail():
    # Clean lint payload → ask allowed; dirty → blocked.
    ok, _ = guardrails.check_okf_lint_before_ask({"valid": True, "n_errors": 0})
    assert ok
    ok, reasons = guardrails.check_okf_lint_before_ask({"valid": False, "n_errors": 3})
    assert not ok and reasons


def test_sql_query_generation_integration(ws_tmp):
    from src.core.targets.sql.agent.agent import retrieve
    from src.core.targets.sql.eval import FakeSqlReader

    store = ws_tmp / "sql.db"
    instance = {
        "question_id": "s-int",
        "question_type": "count",
        "question": "How many users are there per name?",
        "schema_id": "m",
        "tables": ["users"],
        "columns_by_table": {"users": ["id", "name"]},
        "foreign_keys": [],
        "primary_keys": {"users": "id"},
    }
    trace = retrieve(
        instance,
        reader=FakeSqlReader({
            "s-int": ("SELECT name, COUNT(*) FROM users GROUP BY name", "", ""),
        }),
        store_path=store,
    )
    assert trace.drafted.predicted_sql == "SELECT name, COUNT(*) FROM users GROUP BY name"
    assert trace.drafted.drafter_error == ""


def test_observability_run_replays_identically(ws_tmp):
    store = ws_tmp / "events.db"
    okf_ask(
        request_id="r1", question=QUESTION, kb_id="nano", kb_path=KB,
        reader=FakeOkfReader({"r1": (GOLD, GOLD, "")}), store_path=store,
    )
    run = event_store.list_runs(store)[0]
    events = event_store.run_events(store, run.run_id)
    types = [e.type for e in events]
    assert "llm.requested" in types and "llm.responded" in types

    # Replay the stored run into a fresh graph — event stream identical.
    graph, n = replay.replay_into_graph(store, run.run_id)
    assert n == len(events)
    replayed_types = [e.type for e in graph.events]
    assert sorted(replayed_types) == sorted(types)

    # The replay cache serves the recorded answer without an LLM call.
    cache = replay.build_replay_cache(events)
    rr = replay.ReplayReader(cache, fallback=_Boom())
    req = next(e for e in events if e.type == "llm.requested")
    assert rr.answer(context=req.payload["prompt"], question=QUESTION, question_id="r1") == GOLD


def test_learning_path_links_resolve():
    """Learning Path의 개념명은 실제 개념 md 파일로 연결되어야 하며
    (대시보드/마크다운 뷰어에서 클릭으로 학습 가능), 깨진 옛 경로가
    남아 있어서는 안 된다."""
    import re

    lp = KB / "meta" / "learning_path.md"
    text = lp.read_text(encoding="utf-8")
    assert "00_nano_vllm" not in text  # old broken bundle path gone

    targets = re.findall(r"\]\(\.\./([^)#]+\.md)\)", text)
    assert targets, "learning_path.md must contain relative concept links"
    for t in targets:
        assert (KB / t).is_file(), f"learning path link target missing: {t}"
    # The key concept links are present.
    assert "../concepts/03_atomic/prefill_phase.md" in text
    assert "../concepts/02_composite/autoregressive_loop.md" in text


def test_index_module_map_links_resolve():
    """메인 인덱스의 8개 Module 이름은 해당 모듈 문서로 연결되어야 한다."""
    import re

    idx = KB / "index.md"
    text = idx.read_text(encoding="utf-8")
    targets = re.findall(r"\]\((concepts/01_module/module_[^)]+\.md)\)", text)
    assert len(targets) == 8, targets
    for t in targets:
        assert (KB / t).is_file(), f"module map link target missing: {t}"
    assert "[Module 00](" in text and "[Module 07](" in text
    assert "[`distributed_serving`](concepts/02_composite/distributed_serving_system.md)" in text


def test_concept_tree_downward_links_resolve():
    """module → composite → atomic 하향 트리 링크: 모든 관계 블록의
    링크 타깃 파일이 실제로 존재해야 한다."""
    import re

    n_checked = 0
    for p in KB.rglob("*.md"):
        text = p.read_text(encoding="utf-8")
        for m in re.finditer(r"\[`([^`]+)`\]\(([^)]+)\)", text):
            target = m.group(2)
            if target.startswith(("http", "#")):
                continue
            assert (p.parent / target).resolve().is_file(), f"{p.name}: broken tree link {target!r}"
            n_checked += 1
    assert n_checked > 20, "expected a rich set of concept tree links"


class _Boom:
    name = "boom"

    def answer(self, *, context, question, question_id):
        raise AssertionError("LLM was re-called during replay")
