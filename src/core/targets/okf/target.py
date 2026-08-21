"""OkfTarget — the concrete `Target` for the OKF knowledge agent.

Bundles the four pieces: an `OkfEvalBackend` (reader + KB graph + lint),
an `OkfActionSpace` (ask pipeline + prompt-transform gates), an
`OkfTaxonomy` (deterministic integrity detectors), and a per-outcome
`outcome_summary` projection.

`build_target` is the convenience constructor: turn `(kb, reader,
author)` into a full Target. The knowledge base is referenced by bundle
name ("01_nano_vllm") resolved against `.okf/` (or a custom `kb_root`),
or by an explicit path."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.core.target import ActionSpace, EvalBackend, RegimeTaxonomy
from src.core.targets.okf.action_space import OkfActionSpace, StubOkfAuthor
from src.core.targets.okf.eval import FakeOkfReader, OkfEvalBackend
from src.core.targets.okf.outcome import OkfOutcome
from src.core.targets.okf.taxonomy import OkfTaxonomy, load_knowledge_graph


def resolve_kb_path(kb: str | Path, kb_root: str | Path | None = None) -> Path:
    """Resolve a knowledge-base reference to a directory path.

    - Explicit paths (Path-like, contains a separator, or absolute) are
      used as-is.
    - Bare bundle names (e.g. "01_nano_vllm") resolve against
      `kb_root` (default `.okf`).
    Raises FileNotFoundError when the directory doesn't exist."""
    kb_root_path = Path(kb_root) if kb_root is not None else Path(".okf")
    candidate = Path(kb)
    if (
        candidate.is_absolute()
        or candidate.suffix
        or candidate.exists()
        or any(sep in str(kb) for sep in ("/", "\\"))
    ):
        path = candidate
    else:
        path = kb_root_path / str(kb)
    if not path.is_dir():
        raise FileNotFoundError(
            f"OKF knowledge base directory not found: {path} "
            f"(resolved from {kb!r}, kb_root={kb_root_path})"
        )
    return path


def outcome_summary(o: OkfOutcome) -> dict[str, Any]:
    """Self-justifying per-question summary for persistence — carries
    the lint signals and context-shape facts the OKF detectors used to
    assign the regime label so every label in the persisted report is
    auditable."""
    return {
        "question_id": o.question_id,
        "question_type": o.question_type,
        "correct": o.correct,
        "kb_name": o.kb_name,
        "question": o.question,
        "answer": o.answer,
        "applied_transforms": list(o.applied_transforms),
        "n_concepts": len(o.context_parts.get("concepts", ())),
        "n_rules": len(o.context_parts.get("rules", ())),
        "n_schema": len(o.context_parts.get("schema", ())),
        "lint_errors": [
            err if isinstance(err, str) else dict(err)
            for err in o.lint_errors
        ],
    }


@dataclass
class OkfTarget:
    """Concrete OKF `Target`. Composition only."""

    eval_backend: EvalBackend
    action_space: ActionSpace = field(default_factory=OkfActionSpace)
    taxonomy: RegimeTaxonomy = field(default_factory=OkfTaxonomy)
    name: str = "okf"

    def outcome_summary(self, outcome: OkfOutcome) -> dict[str, Any]:
        return outcome_summary(outcome)


def build_target(
    *,
    kb: str | Path,
    reader: Any = None,
    author: Any = None,
    kb_root: str | Path | None = None,
) -> OkfTarget:
    """Build an OkfTarget from a knowledge base + an optional Reader and
    author.

    The graph is loaded ONCE here and shared by the action space (its
    `kb_meta` for prompt-transforms) and the eval backend (context
    fallback + lint); the action space and the target SHARE a single
    OkfTaxonomy instance so eval_diff sees the same registry the loop's
    diagnose step does (the SQL pattern)."""
    kb_path = resolve_kb_path(kb, kb_root=kb_root)
    graph = load_knowledge_graph(kb_path)
    tax = OkfTaxonomy()
    action_space = OkfActionSpace(
        author=author if author is not None else StubOkfAuthor(),
        taxonomy=tax,
        knowledge_graph=graph,
    )
    return OkfTarget(
        eval_backend=OkfEvalBackend(
            reader=reader if reader is not None else FakeOkfReader(),
            kb_path=kb_path,
            action_space=action_space,
        ),
        action_space=action_space,
        taxonomy=tax,
    )
