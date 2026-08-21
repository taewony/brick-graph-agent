"""OKF target — a concrete `regimes.target.Target` for OKF knowledge.

An OKF ask agent: knowledge base -> structural-integrity lint (the four
taxonomy detectors) -> context assembly through the prompt-transforms
ask pipeline (inject_concept_tree -> inject_rules -> trim_context) ->
Reader answer -> judgment against gold text or rule adherence. The same
loop machinery from `src.core.loop` drives it without modification.

Public surface:
    from src.core.targets.okf import build_target, OkfTarget, OkfOutcome
"""

from __future__ import annotations

from src.core.targets.okf.action_space import (
    OKF_IMPORT_WHITELIST,
    OKF_SIGNATURE_PARAMS,
    OkfActionSpace,
    OkfPrompt,
    StubOkfAuthor,
)
from src.core.targets.okf.eval import FakeOkfReader, OkfEvalBackend
from src.core.targets.okf.outcome import OkfContextParts, OkfLintError, OkfOutcome
from src.core.targets.okf.prompt_transforms import (
    inject_concept_tree,
    inject_rules,
    render_okf_prompt,
    trim_context,
)
from src.core.targets.okf.target import OkfTarget, build_target, outcome_summary
from src.core.targets.okf.taxonomy import (
    OkfKnowledgeGraph,
    OkfNode,
    OkfTaxonomy,
    ambiguous_trigger_detector,
    concept_orphan_detector,
    cyclic_concept_detector,
    dangling_reference_detector,
    inverse_relationship_detector,
    lint_knowledge_graph,
    load_knowledge_graph,
    missing_evidence_detector,
    relationship_noise_detector,
    rule_schema_reference_detector,
)

__all__ = [
    "FakeOkfReader",
    "OKF_IMPORT_WHITELIST",
    "OKF_SIGNATURE_PARAMS",
    "OkfActionSpace",
    "OkfContextParts",
    "OkfEvalBackend",
    "OkfKnowledgeGraph",
    "OkfLintError",
    "OkfNode",
    "OkfOutcome",
    "OkfPrompt",
    "OkfTarget",
    "OkfTaxonomy",
    "StubOkfAuthor",
    "ambiguous_trigger_detector",
    "build_target",
    "concept_orphan_detector",
    "cyclic_concept_detector",
    "dangling_reference_detector",
    "inject_concept_tree",
    "inject_rules",
    "inverse_relationship_detector",
    "lint_knowledge_graph",
    "load_knowledge_graph",
    "missing_evidence_detector",
    "outcome_summary",
    "relationship_noise_detector",
    "render_okf_prompt",
    "rule_schema_reference_detector",
    "trim_context",
]
