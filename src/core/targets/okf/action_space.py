"""OkfActionSpace: the OKF ask pipeline + the OKF-shaped gates.

The OKF ask pipeline is the synchronous seam Phase 3's behaviors will
call: `build_prompt_parts(question, relevant_context)` assembles the
base prompt parts (concepts / rules / schema / question / instructions /
hints) and runs the `prompt_transforms` pipeline (by default
inject_concept_tree -> inject_rules -> trim_context) over them,
returning an `OkfPrompt` carrying the final parts and the applied
transform names.

The four lifecycle gates delegate to the shared `regimes.loop.gates`
(the same target-agnostic gate bodies the SQL target uses) with OKF
knobs:

  - static_gate:          (sig_params, import_whitelist) -> OKF set
  - sandbox_gate:         (call_fn, value_validator) -> prompt-transform shape
  - eval_diff:            (install, revert, taxonomy) -> OKF pipeline + OkfTaxonomy
  - promotion_decision:   (per_type_floors, overall_floor_delta)

Per-target configuration lives on this ActionSpace instance; the gate
bodies are shared. `StubOkfAuthor` (deterministic authoring) also lives
here to keep the package to the six Phase-2 modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from src.core.eval.types import EvalResult
from src.core.loop import gates as _gates
from src.core.target import (
    DraftedChange,
    EvalDiff,
    PromotionDecision,
    SandboxResult,
    StaticResult,
)
from src.core.targets.okf import prompt_transforms as _pipeline
from src.core.targets.okf.taxonomy import OkfKnowledgeGraph, OkfTaxonomy


OKF_SIGNATURE_PARAMS: tuple[str, ...] = ("prompt_parts", "question", "kb_meta")
OKF_IMPORT_WHITELIST: frozenset[str] = frozenset({"math", "string"})


def _okf_call_fn(fn: Callable, probe: Mapping[str, Any]) -> dict:
    """OKF prompt-transform call shape:
    `fn(prompt_parts, question, kb_meta)`. Matches the seam in
    `OkfActionSpace.build_prompt_parts`."""
    return fn(
        dict(probe.get("prompt_parts", {})),
        probe.get("question", ""),
        dict(probe.get("kb_meta", {})),
    )


def _okf_value_validator(out: Mapping[str, Any]) -> None:
    """OKF prompt-transforms can return any JSON-shaped values
    (strings, lists, dicts). No additional invariant beyond "is a
    dict and keys ⊆ input keys" — both enforced by sandbox_gate
    itself."""
    return None


@dataclass(frozen=True)
class OkfPrompt:
    """The result of one OKF ask-pipeline run: the final prompt parts
    plus the audit trail of applied transforms."""

    parts: dict[str, Any]
    applied_transforms: tuple[str, ...] = ()
    transform_errors: tuple[dict[str, Any], ...] = ()


# ---------------------------------------------------------------------------
# Stub authoring — deterministic transform drafts keyed by regime.
# The unlock phrases are what FakeOkfReader looks for to flip a question
# to correct in mock mode (same mechanism as StubSqlAuthor).
# ---------------------------------------------------------------------------

ORPHAN_UNLOCK = "Include the parent and child concepts of every retrieved concept."
RULE_SCHEMA_UNLOCK = "Apply only rules whose schema references resolve in the knowledge base."
AMBIGUOUS_UNLOCK = "Pick exactly one concept per trigger phrase before answering."


_STUB_LIBRARY: dict[str, tuple[str, str, str]] = {
    "concept-orphan": (
        "stub_concept_tree_hint",
        (
            "def transform(prompt_parts, question, kb_meta):\n"
            "    out = dict(prompt_parts)\n"
            "    hints = list(out.get('hints', []))\n"
            f"    hints.append({ORPHAN_UNLOCK!r})\n"
            "    out['hints'] = hints\n"
            "    return out\n"
        ),
        "Inject a concept-tree hint so grounding includes the parents "
        "and children of every retrieved concept.",
    ),
    "rule-schema-mismatch": (
        "stub_rule_schema_hint",
        (
            "def transform(prompt_parts, question, kb_meta):\n"
            "    out = dict(prompt_parts)\n"
            "    hints = list(out.get('hints', []))\n"
            f"    hints.append({RULE_SCHEMA_UNLOCK!r})\n"
            "    out['hints'] = hints\n"
            "    return out\n"
        ),
        "Inject a rule-schema hint so the LLM only applies rules whose "
        "schema references actually exist in the knowledge base.",
    ),
    "ambiguous-trigger": (
        "stub_disambiguate_hint",
        (
            "def transform(prompt_parts, question, kb_meta):\n"
            "    out = dict(prompt_parts)\n"
            "    hints = list(out.get('hints', []))\n"
            f"    hints.append({AMBIGUOUS_UNLOCK!r})\n"
            "    out['hints'] = hints\n"
            "    return out\n"
        ),
        "Inject a disambiguation hint so the LLM picks exactly one "
        "concept per ambiguous trigger phrase.",
    ),
}

_TARGET_PRIORITY: tuple[str, ...] = (
    "concept-orphan",
    "rule-schema-mismatch",
    "ambiguous-trigger",
)


@dataclass
class StubOkfAuthor:
    name: str = "stub"

    def draft(
        self,
        *,
        dominant_regime: str,
        failures: Iterable[Any],  # noqa: ARG002 — parity with LLM author
    ) -> DraftedChange:
        if dominant_regime in _STUB_LIBRARY:
            n, src, rat = _STUB_LIBRARY[dominant_regime]
            return DraftedChange(
                name=n, source=src, target_regime=dominant_regime,
                author=self.name, rationale=rat,
            )
        for r in _TARGET_PRIORITY:
            n, src, rat = _STUB_LIBRARY[r]
            return DraftedChange(
                name=n, source=src, target_regime=r,
                author=self.name, rationale=rat,
            )
        raise RuntimeError("StubOkfAuthor has no library entries")  # pragma: no cover


# ---------------------------------------------------------------------------
# Action space
# ---------------------------------------------------------------------------


@dataclass
class OkfActionSpace:
    """Implements `regimes.target.ActionSpace` for the OKF target.

    The OKF action space is "append a Python prompt-transform to the
    OKF ask pipeline" — the analog of the SQL prompt-edit seam and the
    LME score-transform pipeline."""

    author: Any = field(default_factory=StubOkfAuthor)
    taxonomy: OkfTaxonomy = field(default_factory=OkfTaxonomy)
    knowledge_graph: OkfKnowledgeGraph | None = None
    signature_params: tuple[str, ...] = OKF_SIGNATURE_PARAMS
    import_whitelist: frozenset[str] = OKF_IMPORT_WHITELIST
    expected_fn: str = "transform"
    per_type_floors: Mapping[str, float] = field(default_factory=dict)
    overall_floor_delta: float = 0.0
    confirm_threshold: float = 0.0
    n_probe_outcomes: int = 5
    sandbox_time_budget_s: float = 2.0

    # ---- ask pipeline ----------------------------------------------------

    def kb_snapshot(self) -> dict[str, Any]:
        """JSON-safe kb_meta for transforms / probes. Empty when no
        knowledge graph is attached (transforms must passthrough)."""
        if self.knowledge_graph is None:
            return {}
        return self.knowledge_graph.snapshot()

    def assemble_base_parts(self, question: str, relevant_context: Mapping[str, Any]) -> dict[str, Any]:
        """Pipeline-free base prompt parts from the question + the
        retrieved relevant context (concepts / rules / schema). Phase 3
        behaviors can call this directly; build_prompt_parts layers the
        pipeline on top."""
        relevant = dict(relevant_context or {})
        return {
            "concepts": list(relevant.get("concepts", [])),
            "rules": list(relevant.get("rules", [])),
            "schema": list(relevant.get("schema", [])),
            "question": question,
            "instructions": (
                "Answer the question using only the concepts, rules and "
                "schema provided. Cite the concept/rule ids you use."
            ),
            "hints": [],
        }

    def build_prompt_parts(self, question: str, relevant_context: Mapping[str, Any]) -> OkfPrompt:
        """The OKF ask pipeline: assemble base parts, run the
        prompt-transforms pipeline, return the final parts + audit."""
        base = self.assemble_base_parts(question, relevant_context)
        applied, errors = _pipeline.apply_pipeline(
            prompt_parts=base,
            question=question,
            kb_meta=self.kb_snapshot(),
        )
        return OkfPrompt(
            parts=applied["prompt_parts"],
            applied_transforms=tuple(applied["names"]),
            transform_errors=tuple(errors),
        )

    # ---- authoring --------------------------------------------------------

    def draft(self, *, dominant_regime: str, failures: Sequence[Any]) -> DraftedChange:
        return self.author.draft(
            dominant_regime=dominant_regime, failures=list(failures),
        )

    # ---- gates ------------------------------------------------------------

    def static_gate(self, source: str) -> StaticResult:
        return _gates.static_gate(
            source,
            expected_fn=self.expected_fn,
            signature_params=self.signature_params,
            import_whitelist=self.import_whitelist,
        )

    def compile(self, source: str) -> Callable:
        return _gates.compile_transform(source, expected_fn=self.expected_fn)

    def sandbox_gate(
        self, fn: Callable, *, probes: Sequence[Mapping[str, Any]]
    ) -> SandboxResult:
        """Delegates to the shared `gates.sandbox_gate` with the OKF
        call_fn (prompt-transform signature) and a no-op value
        validator (prompt_parts hold strings/lists, not floats)."""
        return _gates.sandbox_gate(
            fn,
            probes=list(probes),
            time_budget_s=self.sandbox_time_budget_s,
            call_fn=_okf_call_fn,
            value_validator=_okf_value_validator,
        )

    def build_probes(self, baseline: EvalResult) -> list[dict[str, Any]]:
        probes: list[dict[str, Any]] = []
        for o in baseline.outcomes[: self.n_probe_outcomes]:
            probes.append({
                "prompt_parts": dict(getattr(o, "context_parts", {}) or {}),
                "question": getattr(o, "question", ""),
                "kb_meta": self.kb_snapshot(),
            })
        return probes

    # ---- install / revert -------------------------------------------------

    def install(self, name: str, fn: Callable) -> None:
        _pipeline.promote(name, fn)

    def revert(self, name: str) -> None:
        _pipeline.revert(name)

    # ---- eval-diff --------------------------------------------------------

    def eval_diff(
        self,
        *,
        fn: Callable,
        fn_name: str,
        target_regime: str,
        baseline: EvalResult,
        eval_backend: Any,
        instances: Sequence[Any],
    ) -> EvalDiff:
        """Delegates to the shared `gates.eval_diff` with the OKF
        prompt-transform install/revert seam and the OKF taxonomy."""
        return _gates.eval_diff(
            fn=fn, fn_name=fn_name, target_regime=target_regime,
            baseline=baseline, eval_backend=eval_backend,
            instances=list(instances),
            install=self.install, revert=self.revert,
            taxonomy=self.taxonomy,
        )

    # ---- promotion decision ----------------------------------------------

    def promotion_decision(self, diff: EvalDiff) -> PromotionDecision:
        return _gates.promotion_decision(
            diff,
            per_type_floors=self.per_type_floors,
            overall_floor_delta=self.overall_floor_delta,
        )
