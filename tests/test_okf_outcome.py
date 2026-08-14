from src.core.eval.types import Outcome
from src.core.targets.okf import OkfOutcome


def test_okf_outcome_captures_task_2_1_fields():
    outcome = OkfOutcome(
        question_id="okf-q1",
        question="Which rules map VIP customers to schema columns?",
        answer="VIP customers are mapped by rule.vip_customer.",
        context_parts={
            "concepts": ["concept.vip_customer"],
            "rules": ["rule.vip_customer"],
            "schema": ["customers.tier"],
        },
        applied_transforms=("inject_rules",),
        lint_errors=({"code": "missing_schema_reference", "node": "rule.vip_customer"},),
    )

    assert isinstance(outcome, Outcome)
    assert outcome.question_id == "okf-q1"
    assert outcome.question_type == "okf"
    assert outcome.correct is None
    assert outcome.question.startswith("Which rules")
    assert outcome.answer == "VIP customers are mapped by rule.vip_customer."
    assert outcome.hypothesis == outcome.answer
    assert outcome.context_parts["concepts"] == ["concept.vip_customer"]
    assert outcome.applied_transforms == ("inject_rules",)
    assert outcome.lint_errors[0]["code"] == "missing_schema_reference"
