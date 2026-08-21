"""OkfEvalBackend + OKF readers/judging.

OkfEvalBackend.run_on_split(instances) -> EvalResult:
  - Loads the OKF knowledge graph once (from `kb_path`) and lints it
    with the four taxonomy detectors; the issues become
    `OkfOutcome.lint_errors` on the outcomes that touched the affected
    nodes (so diagnose is pure over the outcome — no KB re-reads in the
    loop).
  - For each instance: assemble prompt parts via the action space's
    ask pipeline (concept tree + rules + trim), render the prompt, ask
    the Reader, judge the answer against gold text or rule adherence,
    construct an OkfOutcome.

FakeOkfReader: deterministic. Carries a per-question (gold, default-
wrong, unlock_phrase) tuple. When `unlock_phrase` is present in the
assembled prompt it returns gold; otherwise the default wrong. The
unlock phrase is how prompt-transforms move the needle in mock mode: a
promoted transform that injects the right hint flips the matching
questions to correct (same mechanism as FakeSqlReader).
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from src.core.eval.types import EvalResult
from src.core.targets.okf import prompt_transforms as _pipeline
from src.core.targets.okf.action_space import OkfActionSpace
from src.core.targets.okf.outcome import OkfOutcome
from src.core.targets.okf.prompt_transforms import render_okf_prompt
from src.core.targets.okf.taxonomy import (
    OkfKnowledgeGraph,
    lint_knowledge_graph,
    load_knowledge_graph,
)


@dataclass
class FakeOkfReader:
    """Deterministic test reader.

    `table[qid]` = (gold_answer, default_wrong_answer, unlock_phrase).
    If the assembled prompt contains the unlock phrase (case-sensitive
    substring) we return gold_answer; otherwise default_wrong. Questions
    whose default_wrong already equals gold (or whose unlock phrase is
    empty) are simply correct on baseline."""

    table: dict[str, tuple[str, str, str]] = field(default_factory=dict)
    name: str = "fake-okf-reader-v1"

    def answer(self, *, context: str, question: str, question_id: str) -> str:  # noqa: ARG002
        gold, wrong, unlock = self.table.get(question_id, ("", "", ""))
        if unlock and unlock in context:
            return gold
        return wrong or gold


# ---------------------------------------------------------------------------
# Judging
# ---------------------------------------------------------------------------


def normalize_answer(text: str) -> str:
    """Case/whitespace fold for deterministic text comparison."""
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def judge_okf_answer(
    answer: str,
    *,
    gold_answer: str = "",
    gold_rules: Iterable[str] = (),
) -> tuple[bool | None, dict[str, Any]]:
    """Judge an OKF answer.

    Correctness sources (checked in order):
      - gold_answer present  -> normalized text match (containment or
                                equality in either direction);
      - gold_rules present   -> rule adherence: every rule id must be
                                mentioned in the answer text;
      - neither             -> `correct=None` (not judged yet).

    Returns (correct, signals) where signals carries {"match",
    "missing_rules", "adhered_rules"} for audit."""
    gold_rules = [str(r) for r in gold_rules]
    norm = normalize_answer(answer)
    signals: dict[str, Any] = {"match": None, "missing_rules": [], "adhered_rules": []}

    if gold_answer:
        gold_norm = normalize_answer(gold_answer)
        matched = bool(gold_norm) and (
            gold_norm in norm or norm in gold_norm or gold_norm == norm
        )
        signals["match"] = matched

    missing = [r for r in gold_rules if r not in norm]
    signals["missing_rules"] = missing
    signals["adhered_rules"] = [r for r in gold_rules if r in norm]
    rule_ok = (not gold_rules) or not missing

    if gold_answer and gold_rules:
        correct = bool(signals["match"]) and rule_ok
    elif gold_answer:
        correct = bool(signals["match"])
    elif gold_rules:
        correct = rule_ok
    else:
        correct = None
    return correct, signals


# ---------------------------------------------------------------------------
# Eval backend
# ---------------------------------------------------------------------------


@dataclass
class OkfEvalBackend:
    """The OKF eval backend. Composes a Reader + the OKF ask pipeline +
    the KB graph (with lint); runs them over a list of instances and
    returns OkfOutcome records ready for diagnose."""

    reader: Any
    kb_path: str | Path
    action_space: OkfActionSpace | None = None
    name: str = "okf-eval-backend-v1"

    def __post_init__(self) -> None:
        self.kb_path = Path(self.kb_path)
        # Share the graph instance the caller (build_target) already
        # loaded onto the action space — single load, identical object
        # for both the backend and the action space.
        if self.action_space is not None and self.action_space.knowledge_graph is not None:
            self._graph = self.action_space.knowledge_graph
            return
        self._graph = load_knowledge_graph(self.kb_path)
        if self.action_space is None:
            self.action_space = OkfActionSpace(knowledge_graph=self._graph)
        else:
            self.action_space.knowledge_graph = self._graph

    @property
    def graph(self) -> OkfKnowledgeGraph:
        return self._graph

    def _fallback_context(self) -> dict[str, Any]:
        """Deterministic mock-mode fallback when an instance carries no
        `relevant_context`: all concept ids, no rules/schema. Phase 3's
        retrieval behavior will populate real contexts."""
        return {
            "concepts": list(self._graph.concept_ids),
            "rules": [],
            "schema": [],
        }

    @staticmethod
    def _used_node_ids(parts: dict[str, Any]) -> set[str]:
        """Concept/rule ids referenced by the final prompt parts
        (stripping the " (parent)"/" (child)" tags inject_concept_tree
        adds)."""
        used: set[str] = set()
        for c in parts.get("concepts", ()):
            if isinstance(c, str):
                base = c.split(" (parent)")[0].split(" (child)")[0]
                used.add(base)
        for r in parts.get("rules", ()):
            if isinstance(r, str):
                used.add(r)
        return used

    def run_on_split(
        self,
        instances: Iterable[dict[str, Any]],
        *,
        run_dir: str | Path | None = None,
    ) -> EvalResult:
        instances = list(instances)
        issues = lint_knowledge_graph(self._graph)
        outcomes: list[OkfOutcome] = []

        for inst in instances:
            qid = inst["question_id"]
            qtype = inst.get("question_type", "okf")
            question = inst["question"]

            relevant_context = inst.get("relevant_context") or self._fallback_context()
            prompt = self.action_space.build_prompt_parts(question, relevant_context)
            rendered = render_okf_prompt(prompt.parts)

            reader_error = ""
            try:
                answer = self.reader.answer(
                    context=rendered, question=question, question_id=qid,
                )
            except Exception as e:  # noqa: BLE001 — runtime path
                answer = ""
                reader_error = f"{type(e).__name__}: {e}"
            if not isinstance(answer, str):
                answer = str(answer or "")

            correct, signals = judge_okf_answer(
                answer,
                gold_answer=inst.get("gold_answer", ""),
                gold_rules=inst.get("gold_rules", ()),
            )
            if reader_error:
                correct = False

            used = self._used_node_ids(prompt.parts)
            if used:
                outcome_issues = tuple(
                    i for i in issues
                    if isinstance(i, dict) and i.get("node") in used
                )
            else:
                outcome_issues = tuple(issues)

            outcomes.append(OkfOutcome(
                question_id=qid,
                question_type=qtype,
                is_abstention=False,
                answer_session_ids=(),
                correct=correct,
                judge_label="okf-1" if correct else "okf-0",
                judge_raw=signals,
                hypothesis=answer,
                run_id="",
                error=(reader_error or None),
                score_error="",
                applied_transforms=prompt.applied_transforms,
                question=question,
                answer=answer,
                context_parts=dict(prompt.parts),
                lint_errors=outcome_issues,
                kb_name=self._graph.name,
            ))

        per_type_correct: dict[str, int] = defaultdict(int)
        per_type_total: dict[str, int] = defaultdict(int)
        n_errors = 0
        for o in outcomes:
            per_type_total[o.question_type] += 1
            if o.correct:
                per_type_correct[o.question_type] += 1
            if o.error:
                n_errors += 1

        aggregate = {
            "version": "regimes-okf-eval-v1",
            "n": len(outcomes),
            "overall_accuracy": (
                sum(1 for o in outcomes if o.correct) / len(outcomes)
                if outcomes else 0.0
            ),
            "per_type_accuracy": {
                t: per_type_correct[t] / per_type_total[t]
                for t in sorted(per_type_total)
            },
            "n_errors": n_errors,
            "kb": self._graph.name,
            "n_lint_issues": len(issues),
        }

        if run_dir is not None:
            rd = Path(run_dir)
            rd.mkdir(parents=True, exist_ok=True)
            (rd / "aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n")

        return EvalResult(
            outcomes=outcomes,
            aggregate=aggregate,
            backend="okf",
            run_dir=(str(run_dir) if run_dir is not None else None),
            config={
                "reader": getattr(self.reader, "name", ""),
                "applied_transforms": [e.name for e in _pipeline.get_pipeline()],
                "kb": self._graph.name,
                "n_lint_issues": len(issues),
            },
        )
