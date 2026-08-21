"""OkfOutcome — per-question record for the OKF target.

The OKF role needs to report both answer-generation artifacts and
knowledge-integrity signals. Subclassing the shared Outcome keeps the
mentor/eval loop able to handle OKF outcomes without a separate result
contract, while the OKF-specific fields below carry the grounded context
and lint state used by later taxonomy/eval work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.core.eval.types import Outcome


OkfContextParts = dict[str, Any]
OkfLintError = dict[str, Any] | str


@dataclass(frozen=True)
class OkfOutcome(Outcome):
    """OKF-specific outcome fields.

    `correct=None` means the answer has not been judged yet. Once an OKF
    eval backend checks the answer against ground truth or rule
    adherence, it can set `correct` to a concrete bool.
    """

    question_id: str = ""
    question_type: str = "okf"
    is_abstention: bool = False
    answer_session_ids: tuple[str, ...] = ()
    correct: bool | None = None

    question: str = ""
    answer: str = ""
    # The final prompt parts the answer was grounded in — carries
    # "concepts", "rules", "schema" (plus "question", "instructions",
    # "hints"). This IS the prompt_parts dict the OKF action space
    # produced, so the taxonomy's per-outcome detectors and the loop's
    # build_probes can both read it without re-running the pipeline.
    context_parts: OkfContextParts = field(default_factory=dict)
    lint_errors: tuple[OkfLintError, ...] = ()
    # Knowledge-base audit tag: the resolved bundle path the outcome was
    # produced against (e.g. ".okf/01_nano_vllm").
    kb_name: str = ""

    def __post_init__(self) -> None:
        if self.answer and not self.hypothesis:
            object.__setattr__(self, "hypothesis", self.answer)
