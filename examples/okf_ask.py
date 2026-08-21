"""examples/okf_ask.py — ask the OKF target a question against the KB.

Runs the full Phase-2 OKF ask pipeline end-to-end:

    build_target(kb="01_nano_vllm")
        -> OkfActionSpace.build_prompt_parts(question, relevant_context)
           (inject_concept_tree -> inject_rules -> trim_context)
        -> render_okf_prompt(...)
        -> reader.answer(...)
        -> judge_okf_answer(...)  -> OkfOutcome

Usage (from the repo root, with the .wenv interpreter):

    .wenv\\Scripts\\python.exe examples/okf_ask.py                        # deterministic demo
    .wenv\\Scripts\\python.exe examples/okf_ask.py --prompt-only          # print only the assembled prompt
    .wenv\\Scripts\\python.exe examples/okf_ask.py --question "..." --concept atomic.kv_cache
    .wenv\\Scripts\\python.exe examples/okf_ask.py --real --question "..."   # real LLM (needs ANTHROPIC_API_KEY)

The default demo uses FakeOkfReader so the whole ask -> judge loop is
verifiable without any API key. `--real` swaps in AnthropicReader.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.targets.okf import FakeOkfReader, build_target  # noqa: E402
from src.core.targets.okf.prompt_transforms import render_okf_prompt  # noqa: E402

DEFAULT_QUESTION = "How does prefill_phase relate to decode_phase?"
DEFAULT_CONCEPT = "atomic.prefill_phase"
# Demo gold answers for the canned questions (deterministic judge).
_GOLD: dict[str, str] = {
    DEFAULT_QUESTION: "prefill_phase is a prerequisite of decode_phase",
}


def main() -> int:
    ap = argparse.ArgumentParser(description="Ask the OKF target a question.")
    ap.add_argument("--question", default=DEFAULT_QUESTION, help="question text")
    ap.add_argument("--concept", default=DEFAULT_CONCEPT,
                    help="concept id to ground the context on")
    ap.add_argument("--kb", default="01_nano_vllm",
                    help="knowledge base name (resolved under .okf/) or path")
    ap.add_argument("--prompt-only", action="store_true",
                    help="print the assembled prompt and exit (no reader/judge)")
    ap.add_argument("--real", action="store_true",
                    help="use AnthropicReader instead of the deterministic fake")
    args = ap.parse_args()

    gold = _GOLD.get(args.question, "")
    if args.real:
        from src.core.eval.real import build_real_reader

        reader = build_real_reader()
        note = "reader=AnthropicReader (real LLM)"
    else:
        # unlock="" -> always returns gold (or "" when no gold is known).
        reader = FakeOkfReader({"ask-1": (gold, "", "")})
        note = "reader=FakeOkfReader (deterministic demo)"

    target = build_target(kb=args.kb, reader=reader)
    graph = target.eval_backend.graph
    relevant = {"concepts": [args.concept], "rules": [], "schema": []}

    if args.prompt_only:
        prompt = target.action_space.build_prompt_parts(args.question, relevant)
        print(render_okf_prompt(prompt.parts))
        print("\napplied transforms:", prompt.applied_transforms)
        return 0

    result = target.eval_backend.run_on_split([{
        "question_id": "ask-1",
        "question_type": "okf",
        "question": args.question,
        "gold_answer": gold,
        "relevant_context": relevant,
    }])
    o = result.outcomes[0]

    print("=" * 72)
    print(f"KB        : {o.kb_name}  ({len(graph.concept_ids)} concepts, {len(graph.nodes)} nodes)")
    print(f"QUESTION  : {o.question}")
    print(f"READER    : {note}")
    print("-" * 72)
    print(render_okf_prompt(o.context_parts))
    print("-" * 72)
    print(f"ANSWER    : {o.answer}")
    print(f"CORRECT   : {o.correct}")
    print(f"JUDGE     : {o.judge_raw}")
    print(f"TRANSFORMS: {list(o.applied_transforms)}")
    if o.lint_errors:
        print(f"LINT      : {[e.get('code') for e in o.lint_errors]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
