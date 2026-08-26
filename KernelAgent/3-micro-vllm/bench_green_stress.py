"""Adversarial Green Context stress benchmark.

Level 1 experiment for Paper #1: keep one latency-sensitive decode request
active while repeatedly injecting large prefill requests. The main metric is
protected decode completion gap, which includes prefill-induced pauses between
visible decode tokens in the current sequential engine loop.

Run on the RTX 5070 target PC, not on the host preparation machine.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from random import randint, seed


LOWER_IS_BETTER = {
    "decode_step_p50_ms",
    "decode_step_p95_ms",
    "decode_step_p99_ms",
    "decode_step_max_ms",
    "decode_gap_p50_ms",
    "decode_gap_p95_ms",
    "decode_gap_p99_ms",
    "decode_gap_max_ms",
    "prefill_step_p95_ms",
    "prefill_step_max_ms",
    "total_time",
}
METRICS = tuple(sorted(LOWER_IS_BETTER | {"throughput"}))


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    data = sorted(values)
    idx = (len(data) - 1) * (q / 100.0)
    lo = int(idx)
    hi = min(lo + 1, len(data) - 1)
    weight = idx - lo
    return data[lo] * (1.0 - weight) + data[hi] * weight


def pct_delta(base: float, target: float) -> float:
    return ((target - base) / base) * 100.0 if base else 0.0


def random_prompt(length: int) -> list[int]:
    return [randint(0, 10000) for _ in range(length)]


def run_stress_case(args: argparse.Namespace, use_green: bool) -> dict:
    os.environ["NANO_VLLM_USE_GREEN_CONTEXTS"] = "1" if use_green else "0"
    os.environ["NANO_VLLM_USE_CUTILE"] = "1"
    os.environ["NANO_VLLM_GREEN_CONTEXT_API"] = args.green_api
    os.environ["NANO_VLLM_PREFILL_SMS"] = str(args.prefill_sms)
    os.environ["NANO_VLLM_DECODE_SMS"] = str(args.decode_sms)

    from nanovllm import LLM, SamplingParams

    seed(args.seed)
    model_path = os.path.expanduser(args.model)
    llm = LLM(model_path, enforce_eager=True, max_model_len=args.max_model_len)

    protected_prompt = random_prompt(args.decode_prompt_tokens)
    protected_params = SamplingParams(
        temperature=0.6,
        ignore_eos=True,
        max_tokens=args.decode_output_tokens,
    )
    interference_params = SamplingParams(
        temperature=0.6,
        ignore_eos=True,
        max_tokens=args.prefill_output_tokens,
    )

    llm.add_request(protected_prompt, protected_params)

    step_index = 0
    total_tokens_processed = 0
    injected_prefills = 0
    protected_decode_steps = 0
    next_inject_decode_step = args.first_inject_after_decode_steps
    protected_finished = False

    decode_step_ms: list[float] = []
    decode_gap_ms: list[float] = []
    prefill_step_ms: list[float] = []
    injection_records: list[dict] = []
    event_trace: list[dict] = []
    last_decode_end = None

    start_total = time.time()
    while not llm.is_finished():
        if (
            not protected_finished
            and injected_prefills < args.injections
            and protected_decode_steps >= next_inject_decode_step
        ):
            llm.add_request(random_prompt(args.prefill_prompt_tokens), interference_params)
            injected_prefills += 1
            injection_records.append(
                {
                    "injection_index": injected_prefills,
                    "step_index_before": step_index,
                    "protected_decode_steps_before": protected_decode_steps,
                    "wall_time_s": time.time() - start_total,
                }
            )
            next_inject_decode_step += args.inject_every_decode_steps

        step_index += 1
        t0 = time.time()
        outputs, num_tokens = llm.step()
        t1 = time.time()
        step_ms = (t1 - t0) * 1000.0

        if num_tokens > 0:
            kind = "prefill"
            total_tokens_processed += num_tokens
            prefill_step_ms.append(step_ms)
        else:
            kind = "decode"
            decoded = -num_tokens
            total_tokens_processed += decoded
            decode_step_ms.append(step_ms)
            if last_decode_end is not None:
                decode_gap_ms.append((t1 - last_decode_end) * 1000.0)
            last_decode_end = t1
            if not protected_finished:
                protected_decode_steps += 1

        for seq_id, _token_ids in outputs:
            if seq_id == 0:
                protected_finished = True

        if args.trace_steps:
            event_trace.append(
                {
                    "step": step_index,
                    "kind": kind,
                    "step_ms": step_ms,
                    "num_tokens": num_tokens,
                    "protected_decode_steps": protected_decode_steps,
                    "injected_prefills": injected_prefills,
                    "outputs": [seq_id for seq_id, _ in outputs],
                }
            )

    total_time = time.time() - start_total
    runner = getattr(llm, "model_runner", None)

    median_gap = percentile(decode_gap_ms, 50.0)
    over_2x_median = sum(1 for value in decode_gap_ms if median_gap > 0 and value > 2.0 * median_gap)

    result = {
        "workload": "green_context_stress_level1",
        "model": model_path,
        "seed": args.seed,
        "decode_prompt_tokens": args.decode_prompt_tokens,
        "decode_output_tokens": args.decode_output_tokens,
        "prefill_prompt_tokens": args.prefill_prompt_tokens,
        "prefill_output_tokens": args.prefill_output_tokens,
        "injections_requested": args.injections,
        "injections_completed": injected_prefills,
        "inject_every_decode_steps": args.inject_every_decode_steps,
        "first_inject_after_decode_steps": args.first_inject_after_decode_steps,
        "total_tokens": total_tokens_processed,
        "total_time": total_time,
        "throughput": total_tokens_processed / total_time if total_time > 0 else 0.0,
        "decode_steps": len(decode_step_ms),
        "decode_gaps": len(decode_gap_ms),
        "prefill_steps": len(prefill_step_ms),
        "decode_step_p50_ms": percentile(decode_step_ms, 50.0),
        "decode_step_p95_ms": percentile(decode_step_ms, 95.0),
        "decode_step_p99_ms": percentile(decode_step_ms, 99.0),
        "decode_step_max_ms": max(decode_step_ms) if decode_step_ms else 0.0,
        "decode_gap_p50_ms": percentile(decode_gap_ms, 50.0),
        "decode_gap_p95_ms": percentile(decode_gap_ms, 95.0),
        "decode_gap_p99_ms": percentile(decode_gap_ms, 99.0),
        "decode_gap_max_ms": max(decode_gap_ms) if decode_gap_ms else 0.0,
        "decode_gap_over_2x_median_count": over_2x_median,
        "prefill_step_p95_ms": percentile(prefill_step_ms, 95.0),
        "prefill_step_max_ms": max(prefill_step_ms) if prefill_step_ms else 0.0,
        "injections": injection_records,
        "green_requested_api": os.environ.get("NANO_VLLM_GREEN_CONTEXT_API", "auto"),
        "green_enabled": bool(getattr(runner, "use_green_contexts", False)),
        "green_api_type": getattr(runner, "green_api_type", None),
        "green_prefill_sms": getattr(runner, "green_prefill_sms", None),
        "green_decode_sms": getattr(runner, "green_decode_sms", None),
        "green_split_layout_width": getattr(runner, "green_split_layout_width", None),
        "green_prefill_resource_source": getattr(runner, "green_prefill_resource_source", None),
    }
    if args.trace_steps:
        result["trace"] = event_trace
    print("RESULT_JSON:" + json.dumps(result, ensure_ascii=False))
    return result


def parse_result(stdout: str) -> dict:
    for line in stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            return json.loads(line.removeprefix("RESULT_JSON:"))
    raise RuntimeError(f"RESULT_JSON not found in child output:\n{stdout}")


def run_child(script: Path, args: argparse.Namespace, use_green: bool) -> dict:
    cmd = [
        sys.executable,
        str(script),
        "--run-case",
        "--model",
        args.model,
        "--max-model-len",
        str(args.max_model_len),
        "--decode-prompt-tokens",
        str(args.decode_prompt_tokens),
        "--decode-output-tokens",
        str(args.decode_output_tokens),
        "--prefill-prompt-tokens",
        str(args.prefill_prompt_tokens),
        "--prefill-output-tokens",
        str(args.prefill_output_tokens),
        "--injections",
        str(args.injections),
        "--inject-every-decode-steps",
        str(args.inject_every_decode_steps),
        "--first-inject-after-decode-steps",
        str(args.first_inject_after_decode_steps),
        "--green-api",
        args.green_api,
        "--prefill-sms",
        str(args.prefill_sms),
        "--decode-sms",
        str(args.decode_sms),
        "--seed",
        str(args.seed),
    ]
    if use_green:
        cmd.append("--use-green")
    if args.trace_steps:
        cmd.append("--trace-steps")

    env = os.environ.copy()
    env["NANO_VLLM_USE_CUTILE"] = "1"
    proc = subprocess.run(
        cmd,
        cwd=str(script.parent),
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"child failed with exit code {proc.returncode}\n"
            f"cmd={' '.join(cmd)}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    result = parse_result(proc.stdout)
    result["child_stdout_tail"] = proc.stdout.splitlines()[-10:]
    return result


def summarize(rows: list[dict]) -> dict:
    summary = {}
    for metric in METRICS:
        base_values = [row["baseline"][metric] for row in rows]
        green_values = [row["green"][metric] for row in rows]
        deltas = [row["delta_pct"][metric] for row in rows]
        summary[metric] = {
            "baseline_mean": statistics.mean(base_values),
            "green_mean": statistics.mean(green_values),
            "delta_pct_mean": statistics.mean(deltas),
            "delta_pct_median": statistics.median(deltas),
            "delta_pct_std": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
            "improved_runs": sum(1 for delta in deltas if delta < 0) if metric in LOWER_IS_BETTER else sum(1 for delta in deltas if delta > 0),
            "runs": len(rows),
        }
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Green Context adversarial prefill/decode stress benchmark.")
    parser.add_argument("--run-case", action="store_true", help="Internal child mode: run one baseline/green case and print RESULT_JSON.")
    parser.add_argument("--use-green", action="store_true", help="Internal child mode: enable Green Contexts.")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--jsonl", default="green_context_stress_cuda_core_32_16.jsonl")
    parser.add_argument("--model", default="~/huggingface/Qwen3-0.6B/")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--decode-prompt-tokens", type=int, default=32)
    parser.add_argument("--decode-output-tokens", type=int, default=256)
    parser.add_argument("--prefill-prompt-tokens", type=int, default=3072)
    parser.add_argument("--prefill-output-tokens", type=int, default=1)
    parser.add_argument("--injections", type=int, default=12)
    parser.add_argument("--inject-every-decode-steps", type=int, default=8)
    parser.add_argument("--first-inject-after-decode-steps", type=int, default=4)
    parser.add_argument("--green-api", choices=["auto", "pytorch", "cuda_core"], default="cuda_core")
    parser.add_argument("--prefill-sms", type=int, default=32)
    parser.add_argument("--decode-sms", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trace-steps", action="store_true", help="Include per-step trace in RESULT_JSON. This makes JSONL larger.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.run_case:
        run_stress_case(args, args.use_green)
        return

    script = Path(__file__)
    output_path = Path(args.jsonl)
    if not output_path.is_absolute():
        output_path = script.parent / output_path

    rows = []
    with output_path.open("a", encoding="utf-8") as f:
        for index in range(1, args.repeats + 1):
            baseline = run_child(script, args, use_green=False)
            green = run_child(script, args, use_green=True)
            row = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "run_index": index,
                "baseline": baseline,
                "green": green,
                "delta_pct": {metric: pct_delta(baseline[metric], green[metric]) for metric in METRICS},
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            rows.append(row)
            print(
                f"run {index:02d}: "
                f"enabled={green.get('green_enabled')} api={green.get('green_api_type')} "
                f"gap_p99_delta={row['delta_pct']['decode_gap_p99_ms']:+.2f}% "
                f"gap_max_delta={row['delta_pct']['decode_gap_max_ms']:+.2f}% "
                f"throughput_delta={row['delta_pct']['throughput']:+.2f}%"
            )

    print("\nsummary")
    summary = summarize(rows)
    for metric in METRICS:
        item = summary[metric]
        print(
            f"{metric:>24}: baseline={item['baseline_mean']:.2f} "
            f"green={item['green_mean']:.2f} "
            f"delta_mean={item['delta_pct_mean']:+.2f}% "
            f"delta_median={item['delta_pct_median']:+.2f}% "
            f"std={item['delta_pct_std']:.2f}% "
            f"improved={item['improved_runs']}/{item['runs']}"
        )
    print(f"raw_jsonl={output_path}")


if __name__ == "__main__":
    main()
