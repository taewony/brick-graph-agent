"""Phase 7 acceptance tests: Mentor integration (adapt the loop).

Acceptance (brick-agent-plan.md Phase 7): a simulated failure run results
in a mentor-proposed improvement that is validated and applied, and the
next run uses the improved prompt. Plus Task 7.2 (register_regime), 7.3
(LLM observability consumption), and replay-driven validation (Phase 5).
"""

import pytest

from src.agents.okf import ask as okf_ask
from src.core.loop import run_loop
from src.core.targets.okf import build_target as build_okf_target
from src.core.targets.okf import prompt_transforms as okf_pipe
from src.core.targets.okf.action_space import ORPHAN_UNLOCK
from src.core.targets.okf.eval import FakeOkfReader
from src.core.targets.okf.outcome import OkfOutcome
from src.core.targets.okf.taxonomy import OkfTaxonomy
from src.core.targets.sql import build_target as build_sql_target
from src.core.targets.sql import prompt_transforms as sql_pipe
from src.core.targets.sql.eval import FakeSqlReader
from src.core.targets.sql.hypothesize import AGGREGATION_UNLOCK
from src.runtime import event_store, observability, replay


def _sql_instances():
    return [
        {
            "question_id": "s1",
            "question_type": "count",
            "question": "How many users are there per name?",
            "schema_id": "m",
            "tables": ["users"],
            "columns_by_table": {"users": ["id", "name"]},
            "foreign_keys": [],
            "primary_keys": {"users": "id"},
            "schema_ddl": "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT);",
            "seed_rows": [
                "INSERT INTO users (id, name) VALUES (1, 'Ada');",
                "INSERT INTO users (id, name) VALUES (2, 'Grace');",
            ],
            # Gold uses GROUP BY → wrong-aggregation when predicted omits it.
            "gold_sql": "SELECT name, COUNT(*) FROM users GROUP BY name",
            "gold_result_set": (("Ada", 1), ("Grace", 1)),
        }
    ]


def _sql_reader():
    # Baseline: wrong SQL (no GROUP BY) → wrong-aggregation regime.
    # After the stub transform injects AGGREGATION_UNLOCK, gold is returned.
    return FakeSqlReader({
        "s1": ("SELECT name, COUNT(*) FROM users GROUP BY name", "SELECT name FROM users", AGGREGATION_UNLOCK),
    })


@pytest.fixture(autouse=True)
def _reset_pipelines():
    okf_pipe.reset()
    sql_pipe.clear()
    yield
    okf_pipe.reset()
    sql_pipe.clear()


# ---------------------------------------------------------------------------
# Task 7.4: SQL mentor cycle — propose → validate → apply → next run improved
# ---------------------------------------------------------------------------


def test_sql_mentor_cycle_promotes_transform_and_next_run_uses_it():
    target = build_sql_target(reader=_sql_reader())
    instances = _sql_instances()

    baseline = target.eval_backend.run_on_split(instances)
    assert baseline.overall_accuracy() == 0.0

    report = run_loop(target=target, instances=instances)

    # Mentor proposed, validated (eval-diff), and promoted.
    assert len(report.promotions) == 1, report.transform_log
    promo = report.promotions[0]
    assert promo["target_regime"] == "wrong-aggregation"
    assert promo["overall_delta"] > 0.0
    assert report.stopped["reason"] == "no_optimizable_regime_remaining"
    assert report.transform_log[-1]["status"] == "promoted"

    # The next run uses the improved prompt (transform installed).
    after = target.eval_backend.run_on_split(instances)
    assert after.overall_accuracy() == 1.0
    assert promo["name"] in after.outcomes[0].applied_transforms


# ---------------------------------------------------------------------------
# Task 7.2 integration: OKF mentor cycle on the implemented OkfTaxonomy
# ---------------------------------------------------------------------------


def _orphan_kb(ws_tmp):
    """A tiny KB whose concepts are all orphans (no relationships)."""
    kb = ws_tmp / "kb"
    kb.mkdir(parents=True)
    (kb / "index.md").write_text("---\ntype: Index\n---\n# index\n", encoding="utf-8")
    for cid in ("atomic.a", "atomic.b"):
        stem = cid.split(".")[-1]
        (kb / f"{stem}.md").write_text(
            f"---\ntype: AtomicConcept\nid: {cid}\ntitle: {cid}\n---\n# {cid}\n",
            encoding="utf-8",
        )
    return kb


def _okf_instances():
    return [
        {
            "question_id": "m1",
            "question_type": "okf",
            "question": "Explain atomic.a",
            "gold_answer": "answer A",
            "relevant_context": {"concepts": ["atomic.a"], "rules": [], "schema": []},
        },
        {
            "question_id": "m2",
            "question_type": "okf",
            "question": "Explain atomic.b",
            "gold_answer": "answer B",
            "relevant_context": {"concepts": ["atomic.b"], "rules": [], "schema": []},
        },
    ]


def _okf_reader():
    # Baseline wrong; flips to gold when the concept-orphan hint is injected.
    return FakeOkfReader({
        "m1": ("answer A", "wrong A", ORPHAN_UNLOCK),
        "m2": ("answer B", "wrong B", ORPHAN_UNLOCK),
    })


def test_okf_mentor_cycle_promotes_transform(ws_tmp):
    kb = _orphan_kb(ws_tmp)
    target = build_okf_target(kb=kb, reader=_okf_reader())
    instances = _okf_instances()

    baseline = target.eval_backend.run_on_split(instances)
    assert baseline.overall_accuracy() == 0.0
    # Failures classify into the optimizable concept-orphan regime.
    assert OkfTaxonomy().classify(baseline.outcomes[0]).name == "concept-orphan"

    report = run_loop(target=target, instances=instances)

    assert len(report.promotions) == 1, report.transform_log
    promo = report.promotions[0]
    assert promo["target_regime"] == "concept-orphan"
    assert promo["overall_delta"] > 0.0
    assert report.stopped["reason"] == "no_optimizable_regime_remaining"

    after = target.eval_backend.run_on_split(instances)
    assert after.overall_accuracy() == 1.0
    assert promo["name"] in after.outcomes[0].applied_transforms


# ---------------------------------------------------------------------------
# Task 7.2: extend the OKF taxonomy via register_regime (no parallel taxonomy)
# ---------------------------------------------------------------------------


def test_okf_taxonomy_register_regime():
    tax = OkfTaxonomy()
    tax.register_regime(
        "wrong-concept-selected",
        lambda o: o.question == "trick question",
        optimizable=True,
        seam_reachable=True,
        description="Retrieval picked the wrong concept (candidate regime).",
    )
    o = OkfOutcome(question_id="q", question="trick question", correct=False, lint_errors=())
    assert tax.classify(o).name == "wrong-concept-selected"
    assert tax.is_seam_reachable("wrong-concept-selected") is True
    # The built-in regimes are untouched.
    assert tax.classify(OkfOutcome(question_id="q", correct=True, lint_errors=())).name == "unclassified"


# ---------------------------------------------------------------------------
# Task 7.3: mentor consumes the LLM observability trail
# ---------------------------------------------------------------------------


def test_llm_usage_summary_from_store(ws_tmp):
    from pathlib import Path

    kb = Path(__file__).resolve().parents[1] / ".okf" / "01_nano_vllm"
    store = ws_tmp / "events.db"
    okf_ask(
        request_id="u1",
        question="How does prefill_phase relate to decode_phase?",
        kb_id="nano",
        kb_path=kb,
        reader=FakeOkfReader({"u1": ("prefill_phase is a prerequisite of decode_phase", "", "")}),
        store_path=store,
    )
    usage = observability.summarize_llm_usage(store)
    assert usage["n_runs"] >= 1
    assert usage["n_llm_calls"] >= 1
    assert usage["avg_latency_seconds"] >= 0.0
    assert usage["n_llm_errors"] == 0


# ---------------------------------------------------------------------------
# Acceptance: validation is deterministic via Phase 5 replay (no LLM re-call)
# ---------------------------------------------------------------------------


def test_replay_driven_validation_no_llm_recall(ws_tmp):
    from pathlib import Path

    kb = Path(__file__).resolve().parents[1] / ".okf" / "01_nano_vllm"
    question = "How does prefill_phase relate to decode_phase?"
    gold = "prefill_phase is a prerequisite of decode_phase"
    store = ws_tmp / "events.db"

    # 1) Record a run (the validation set) with a gold reader.
    okf_ask(
        request_id="v1", question=question, kb_id="nano", kb_path=kb,
        reader=FakeOkfReader({"v1": (gold, "", "")}), store_path=store,
    )
    runs = event_store.list_runs(store)
    events = event_store.run_events(store, runs[0].run_id)
    cache = replay.build_replay_cache(events)
    assert cache

    # 2) Re-run the SAME question with a replay reader whose fallback is a
    #    reader that FAILS if called → a cache hit proves no LLM re-call.
    class _BoomReader:
        name = "boom"

        def answer(self, *, context, question, question_id):
            raise AssertionError("LLM was re-called during deterministic replay")

    replay_reader = replay.ReplayReader(cache, fallback=_BoomReader())
    trace = okf_ask(
        request_id="v2", question=question, kb_id="nano", kb_path=kb,
        reader=replay_reader,
    )
    assert trace.payload["answer"] == gold
    assert trace.payload["error"] == ""
