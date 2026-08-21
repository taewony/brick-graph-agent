"""Phase 2 acceptance tests for the OKF target package.

These run in the project env (they import the full OKF target, whose
action space pulls in `src.core.loop.gates` and thus the activegraph
runtime, exactly like the SQL target tests).

Coverage:
  - build_target instantiates against the real `.okf/01_nano_vllm` KB
  - the ask pipeline (inject_concept_tree -> inject_rules ->
    trim_context) grounds context in the concept tree
  - the four KB-level taxonomy detectors produce signals
  - OkfTaxonomy classifies per-outcome lint codes into regimes
  - static / sandbox gates accept OKF-shaped transforms
  - eval_diff wiring moves the mock-mode needle (promote -> revert)
  - judgment (gold text / rule adherence) and outcome_summary
"""

from pathlib import Path

import pytest

from src.core.eval.types import EvalResult
from src.core.targets.okf import (
    OkfActionSpace,
    OkfEvalBackend,
    OkfKnowledgeGraph,
    OkfOutcome,
    OkfTaxonomy,
    FakeOkfReader,
    build_target,
    concept_orphan_detector,
    cyclic_concept_detector,
    load_knowledge_graph,
    outcome_summary,
    rule_schema_reference_detector,
)
from src.core.targets.okf import prompt_transforms as pt
from src.core.targets.okf.eval import judge_okf_answer
from src.core.targets.okf.prompt_transforms import (
    MAX_CONTEXT_CHARS,
    apply_pipeline,
    inject_concept_tree,
    render_okf_prompt,
)
from src.core.targets.okf.taxonomy import (
    OkfNode,
    ambiguous_trigger_detector,
    lint_knowledge_graph,
)

KB = Path(__file__).resolve().parents[1] / ".okf" / "01_nano_vllm"

_VALID_TRANSFORM = (
    "def transform(prompt_parts, question, kb_meta):\n"
    "    out = dict(prompt_parts)\n"
    "    hints = list(out.get('hints', []))\n"
    "    hints.append('extra hint')\n"
    "    out['hints'] = hints\n"
    "    return out\n"
)


@pytest.fixture(autouse=True)
def _reset_pipeline():
    pt.reset()
    yield
    pt.reset()


# ---------------------------------------------------------------------------
# 2.6 acceptance: instantiate against the 01_nano_vllm knowledge base
# ---------------------------------------------------------------------------


def test_build_target_with_01_nano_vllm_knowledge_base():
    target = build_target(kb=KB)

    assert target.name == "okf"
    assert isinstance(target.taxonomy, OkfTaxonomy)
    assert isinstance(target.action_space, OkfActionSpace)
    assert isinstance(target.eval_backend, OkfEvalBackend)
    # The graph is loaded and shared by backend + action space.
    graph = target.eval_backend.graph
    assert len(graph.concept_ids) > 0
    assert target.action_space.knowledge_graph is graph
    # A bare bundle name resolves against .okf/ too (acceptance path).
    from src.core.targets.okf.target import resolve_kb_path

    bare = build_target(kb="01_nano_vllm")
    assert isinstance(bare.eval_backend.graph, OkfKnowledgeGraph)
    assert resolve_kb_path("01_nano_vllm").name == "01_nano_vllm"


def test_lint_knowledge_graph_is_deterministic_on_real_kb():
    issues_a = lint_knowledge_graph(load_knowledge_graph(KB))
    issues_b = lint_knowledge_graph(load_knowledge_graph(KB))
    assert issues_a == issues_b
    assert all(isinstance(i, dict) and "code" in i and "node" in i for i in issues_a)


# ---------------------------------------------------------------------------
# 2.3/2.4 ask pipeline: build_prompt_parts + prompt transforms
# ---------------------------------------------------------------------------


def test_build_prompt_parts_expands_concept_tree():
    aspace = OkfActionSpace(knowledge_graph=load_knowledge_graph(KB))
    prompt = aspace.build_prompt_parts(
        "How does prefill relate to decode?",
        {"concepts": ["atomic.prefill_phase"], "rules": [], "schema": []},
    )
    concepts = prompt.parts["concepts"]
    assert "atomic.inference_only (parent)" in concepts
    assert {"atomic.decode_phase (child)", "composite.inference_model (child)"} <= set(concepts)
    assert prompt.applied_transforms == ("inject_concept_tree", "inject_rules", "trim_context")
    assert prompt.transform_errors == ()
    # The rendered prompt is grounded in the tree.
    assert "atomic.inference_only (parent)" in render_okf_prompt(prompt.parts)


def test_inject_concept_tree_handles_empty_kb_meta():
    base = {"concepts": ["c.x"], "rules": [], "schema": [], "question": "q", "instructions": "i", "hints": []}
    out = inject_concept_tree(dict(base), "q", {})
    assert out["concepts"] == ["c.x"]  # passthrough, no new keys


def test_trim_context_bounds_context():
    long = {"concepts": [f"concept_{i}" * 200 for i in range(50)], "rules": [], "schema": []}
    untrimmed = sum(len(c) for c in long["concepts"])
    trimmed = pt.trim_context(dict(long), "q", {})
    total = sum(len(c) for c in trimmed["concepts"])
    assert total < untrimmed
    assert total <= MAX_CONTEXT_CHARS + max(len(c) for c in long["concepts"])


def test_pipeline_registry_promote_revert_clear_reset():
    assert [e.name for e in pt.get_pipeline()] == ["inject_concept_tree", "inject_rules", "trim_context"]
    pt.promote("extra", lambda parts, q, kb: dict(parts))
    assert [e.name for e in pt.get_pipeline()][-1] == "extra"
    pt.revert("extra")
    assert [e.name for e in pt.get_pipeline()][-1] == "trim_context"
    pt.clear()
    assert pt.get_pipeline() == []
    pt.reset()
    assert len(pt.get_pipeline()) == 3


def test_apply_pipeline_skips_raising_transform():
    pt.promote("boom", lambda parts, q, kb: (_ for _ in ()).throw(RuntimeError("nope")))
    base = {"concepts": [], "rules": [], "schema": [], "question": "q", "instructions": "i", "hints": []}
    result, errors = apply_pipeline(prompt_parts=dict(base), question="q", kb_meta={})
    assert result["names"] == ["inject_concept_tree", "inject_rules", "trim_context"]
    assert errors and errors[-1]["name"] == "boom"
    assert result["prompt_parts"]["question"] == "q"  # prior parts carry forward


# ---------------------------------------------------------------------------
# 2.2 taxonomy detectors + OkfTaxonomy
# ---------------------------------------------------------------------------


def _synthetic_graph() -> OkfKnowledgeGraph:
    g = OkfKnowledgeGraph(name="syn")
    g.nodes = {
        "c.alpha": OkfNode(id="c.alpha", stem="alpha", type="AtomicConcept", title="Alpha", status="draft", path="a.md", triggers=frozenset({"alpha"})),
        "c.beta": OkfNode(id="c.beta", stem="beta", type="AtomicConcept", title="Beta", status="draft", path="b.md", triggers=frozenset({"beta"}), prerequisites=frozenset({"c.alpha"})),
        "c.gamma": OkfNode(id="c.gamma", stem="gamma", type="CompositeConcept", title="Beta", status="draft", path="c.md", triggers=frozenset({"beta"}), composed_of=frozenset({"c.beta"})),
        "c.delta": OkfNode(id="c.delta", stem="delta", type="AtomicConcept", title="Delta", status="draft", path="d.md", triggers=frozenset({"delta"})),
        "r.one": OkfNode(id="r.one", stem="one", type="Rule", title="Rule One", status="draft", path="r.md", schema_refs=frozenset({"s.nope"})),
        "s.customers": OkfNode(id="s.customers", stem="customers", type="SchemaTable", title="customers", status="draft", path="s.md"),
    }
    g.by_stem = {"alpha": "c.alpha", "beta": "c.beta", "gamma": "c.gamma", "delta": "c.delta", "one": "r.one", "customers": "s.customers"}
    return g


def test_concept_orphan_detector():
    issues = concept_orphan_detector(_synthetic_graph())
    assert any(i["code"] == "concept_orphan" and i["node"] == "c.delta" for i in issues)
    # c.alpha is referenced as a prerequisite of c.beta -> not an orphan.
    assert not any(i["node"] == "c.alpha" for i in issues)


def test_rule_schema_reference_detector():
    issues = rule_schema_reference_detector(_synthetic_graph())
    assert any(i["code"] == "rule_schema_reference" and i["node"] == "r.one" for i in issues)


def test_ambiguous_trigger_detector():
    issues = ambiguous_trigger_detector(_synthetic_graph())
    assert any(i["code"] == "ambiguous_trigger" for i in issues)


def test_cyclic_concept_detector():
    g = OkfKnowledgeGraph(name="cyc")
    g.nodes = {
        "c.x": OkfNode(id="c.x", stem="x", type="AtomicConcept", title="X", status="draft", path="x.md", prerequisites=frozenset({"c.y"})),
        "c.y": OkfNode(id="c.y", stem="y", type="AtomicConcept", title="Y", status="draft", path="y.md", prerequisites=frozenset({"c.x"})),
    }
    g.by_stem = {"x": "c.x", "y": "c.y"}
    issues = cyclic_concept_detector(g)
    assert any(i["code"] == "cyclic_concept" for i in issues)


def test_inverse_relationship_detector():
    from src.core.targets.okf.taxonomy import inverse_relationship_detector

    g = OkfKnowledgeGraph(name="inv")
    g.nodes = {
        # A declares PREREQUISITE_OF B, but B never lists A as prerequisite.
        "c.a": OkfNode(id="c.a", stem="a", type="AtomicConcept", title="A", status="draft", path="a.md",
                      prerequisite_of=frozenset({"c.b"})),
        "c.b": OkfNode(id="c.b", stem="b", type="AtomicConcept", title="B", status="draft", path="b.md"),
    }
    g.by_stem = {"a": "c.a", "b": "c.b"}
    issues = inverse_relationship_detector(g)
    assert any(i["code"] == "inverse_relationship" and i["node"] == "c.a" for i in issues)

    # Consistent pair: B lists A as prerequisite → no issue.
    g.nodes["c.b"].prerequisites = frozenset({"c.a"})
    issues = inverse_relationship_detector(g)
    assert not any(i["node"] == "c.a" for i in issues)


def test_dangling_reference_detector():
    from src.core.targets.okf.taxonomy import dangling_reference_detector

    g = OkfKnowledgeGraph(name="dang")
    g.nodes = {
        "c.a": OkfNode(
            id="c.a", stem="a", type="AtomicConcept", title="A", status="draft", path="a.md",
            raw_relationships={"PREREQUISITES": frozenset({"ghost.concept"}), "COMPOSED_OF": frozenset({"c.b"})},
        ),
        "c.b": OkfNode(id="c.b", stem="b", type="AtomicConcept", title="B", status="draft", path="b.md"),
    }
    g.by_stem = {"a": "c.a", "b": "c.b"}
    issues = dangling_reference_detector(g)
    assert any(i["code"] == "dangling_reference" and i["node"] == "c.a" and "ghost.concept" in i["detail"] for i in issues)
    assert not any("c.b" in i["detail"] for i in issues)


def test_relationship_noise_detector():
    from src.core.targets.okf.taxonomy import relationship_noise_detector

    g = OkfKnowledgeGraph(name="noise")
    g.nodes = {
        "c.a": OkfNode(
            id="c.a", stem="a", type="AtomicConcept", title="A", status="draft", path="a.md",
            relationship_noise={"COMPOSED_OF": ("**All-Reduce**를 통해 모든 GPU의 부분 어텐션 결과를 합산합니다.",)},
        ),
    }
    issues = relationship_noise_detector(g)
    assert any(i["code"] == "relationship_noise" and i["node"] == "c.a" for i in issues)


def test_missing_evidence_detector():
    from src.core.targets.okf.taxonomy import missing_evidence_detector

    g = OkfKnowledgeGraph(name="evid")
    g.nodes = {
        "c.a": OkfNode(id="c.a", stem="a", type="AtomicConcept", title="A", status="draft", path="a.md"),
        "c.b": OkfNode(id="c.b", stem="b", type="AtomicConcept", title="B", status="draft", path="b.md",
                       evidence=frozenset({"https://example.com/source"})),
    }
    issues = missing_evidence_detector(g)
    assert any(i["code"] == "missing_evidence" and i["node"] == "c.a" for i in issues)
    assert not any(i["node"] == "c.b" for i in issues)


def test_real_kb_audit_is_clean():
    """Phase 8 acceptance: the 01_nano_vllm KB has been repaired — lint
    reports zero structural issues (cycles, dangling refs, one-sided
    relationships, noise, missing evidence all resolved)."""
    graph = load_knowledge_graph(KB)
    issues = lint_knowledge_graph(graph)
    assert issues == [], f"expected lint-clean KB, got {len(issues)} issues: {issues[:3]}"


def test_taxonomy_classify_and_histogram():
    tax = OkfTaxonomy()
    cases = [
        ({"code": "concept_orphan", "node": "c.delta"}, "concept-orphan"),
        ({"code": "rule_schema_reference", "node": "r.one"}, "rule-schema-mismatch"),
        ({"code": "ambiguous_trigger", "node": "c.gamma"}, "ambiguous-trigger"),
        ({"code": "cyclic_concept", "node": "c.x"}, "concept-cycle"),
    ]
    for err, expected in cases:
        o = OkfOutcome(question_id="q", correct=False, lint_errors=(err,))
        assert tax.classify(o).name == expected

    clean = OkfOutcome(question_id="q", correct=True, lint_errors=())
    assert tax.classify(clean).name == "unclassified"

    rows = tax.histogram([
        OkfOutcome(question_id="o1", correct=False, lint_errors=({"code": "concept_orphan"},)),
        OkfOutcome(question_id="o2", correct=False, lint_errors=({"code": "cyclic_concept"},)),
        OkfOutcome(question_id="o3", correct=True, lint_errors=()),
        OkfOutcome(question_id="o4", correct=None, lint_errors=()),  # unjudged: not a failure
    ])
    by = {r.regime: r.count for r in rows}
    assert by["concept-orphan"] == 1
    assert by["concept-cycle"] == 1
    assert sum(r.count for r in rows) == 2
    assert tax.is_seam_reachable("concept-orphan") is True
    assert tax.is_seam_reachable("concept-cycle") is False
    assert "knowledge base" in tax.name_wall({"concept-cycle": 1})


# ---------------------------------------------------------------------------
# 2.5 judgment + eval backend
# ---------------------------------------------------------------------------


def test_judge_okf_answer():
    correct, signals = judge_okf_answer("  Prefill Phase is the compute-bound phase. ", gold_answer="prefill phase is the compute-bound phase")
    assert correct is True and signals["match"] is True

    correct, _ = judge_okf_answer("short", gold_answer="prefill phase")
    assert correct is False

    correct, _ = judge_okf_answer(
        "use rule.vip_customer and schema.customers.tier",
        gold_rules=("rule.vip_customer", "schema.customers.tier"),
    )
    assert correct is True

    correct, signals = judge_okf_answer("mentions rule.other only", gold_rules=("rule.vip_customer",))
    assert correct is False and signals["missing_rules"] == ["rule.vip_customer"]

    correct, _ = judge_okf_answer("no gold at all", gold_answer="", gold_rules=())
    assert correct is None  # unjudged


def test_end_to_end_fake_reader_and_promotion_needle():
    instances = [
        {
            "question_id": "okf-q1",
            "question_type": "okf",
            "question": "How does prefill_phase relate to decode_phase?",
            "gold_answer": "prefill_phase is a prerequisite of decode_phase",
            "relevant_context": {"concepts": ["atomic.prefill_phase"], "rules": [], "schema": []},
        },
        {
            "question_id": "okf-q2",
            "question_type": "okf",
            "question": "Which concept composes decoder_layer?",
            "gold_answer": "sampling, kv_cache, decoder_layer",
            "relevant_context": {"concepts": ["composite.decoder_layer"], "rules": [], "schema": []},
        },
    ]
    reader = FakeOkfReader({
        "okf-q1": ("prefill_phase is a prerequisite of decode_phase", "wrong answer one", "Include the parent and child concepts"),
        "okf-q2": ("sampling, kv_cache, decoder_layer", "wrong answer two", ""),
    })
    target = build_target(kb=KB, reader=reader)

    baseline = target.eval_backend.run_on_split(instances)
    o1, o2 = baseline.outcomes
    assert o1.correct is False and o2.correct is False
    assert baseline.aggregate["overall_accuracy"] == 0.0
    assert o1.kb_name.endswith("01_nano_vllm")
    assert "inject_concept_tree" in o1.applied_transforms
    assert o1.hypothesis == o1.answer == "wrong answer one"
    assert any("(child)" in str(c) for c in o1.context_parts["concepts"])

    # Mock-mode needle: promote the stub-authored transform; the unlock
    # phrase flips q1 to correct.
    draft = target.action_space.draft(dominant_regime="concept-orphan", failures=[o1])
    fn = target.action_space.compile(draft.source)
    diff = target.action_space.eval_diff(
        fn=fn,
        fn_name=draft.name,
        target_regime=draft.target_regime,
        baseline=baseline,
        eval_backend=target.eval_backend,
        instances=instances,
    )
    assert diff.overall_delta > 0
    assert diff.target_regime == "concept-orphan"
    assert diff.taxonomy_name == "okf"
    # q1 transitions out of its baseline regime into correct; q2 stays put.
    assert any(t[0] == "okf-q1" and t[2] == "correct" for t in diff.transitions), diff.transitions
    assert not any(t[0] == "okf-q2" for t in diff.transitions), diff.transitions
    # eval_diff reverts after the run; the pipeline is clean again.
    assert draft.name not in [e.name for e in pt.get_pipeline()]


def test_outcome_summary_projection():
    o = OkfOutcome(
        question_id="q1", correct=False,
        question="question", answer="answer",
        context_parts={"concepts": ["c.a"], "rules": ["r.b"], "schema": []},
        applied_transforms=("inject_concept_tree",),
        lint_errors=({"code": "concept_orphan", "node": "c.a"},),
        kb_name="kb",
    )
    s = outcome_summary(o)
    assert s["question_id"] == "q1"
    assert s["correct"] is False
    assert s["n_concepts"] == 1 and s["n_rules"] == 1
    assert s["lint_errors"][0]["code"] == "concept_orphan"
    assert s["applied_transforms"] == ["inject_concept_tree"]


# ---------------------------------------------------------------------------
# 2.3 gates wiring
# ---------------------------------------------------------------------------


def test_static_and_sandbox_gates_okf_shape():
    aspace = OkfActionSpace()

    sr = aspace.static_gate(_VALID_TRANSFORM)
    assert sr.passed, sr.reasons

    bad_sig = "def transform(a, b):\n    return {}\n"
    assert not aspace.static_gate(bad_sig).passed

    fn = aspace.compile(_VALID_TRANSFORM)
    probe = {
        "prompt_parts": {"concepts": [], "rules": [], "schema": [], "question": "q", "instructions": "i", "hints": []},
        "question": "q",
        "kb_meta": {},
    }
    sgr = aspace.sandbox_gate(fn, probes=[probe])
    assert sgr.passed, sgr.reasons


def test_build_probes_reads_context_parts():
    aspace = OkfActionSpace()
    baseline = EvalResult(
        outcomes=[
            OkfOutcome(
                question_id="q1", correct=False, question="question?",
                context_parts={"concepts": ["c.a"], "rules": [], "schema": [], "question": "question?", "instructions": "i", "hints": []},
            )
        ],
        aggregate={},
        backend="okf",
    )
    probes = aspace.build_probes(baseline)
    assert probes and set(probes[0]) == {"prompt_parts", "question", "kb_meta"}
    assert probes[0]["question"] == "question?"
    assert "concepts" in probes[0]["prompt_parts"]
