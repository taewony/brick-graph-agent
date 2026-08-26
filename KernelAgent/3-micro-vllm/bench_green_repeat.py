"""Repeated Green Context benchmark driver.

This script is intended for the target PC with CUDA/PyTorch installed. It runs
bench_green.py in isolated subprocesses, records raw JSONL evidence, and prints
aggregate statistics suitable for paper decisions.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


METRICS = ("ttft", "p50_itl", "p99_itl", "throughput", "total_time")
LOWER_IS_BETTER = {"ttft", "p50_itl", "p99_itl", "total_time"}


def parse_result(stdout: str) -> dict:
    for line in stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            return json.loads(line.removeprefix("RESULT_JSON:"))
    raise RuntimeError(f"RESULT_JSON not found in child output:\n{stdout}")


def run_child(script: Path, mode: str, env: dict[str, str]) -> dict:
    proc = subprocess.run(
        [sys.executable, str(script), mode],
        cwd=str(script.parent),
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"child failed for {mode} with exit code {proc.returncode}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    result = parse_result(proc.stdout)
    result["child_stdout_tail"] = proc.stdout.splitlines()[-8:]
    return result


def pct_delta(base: float, target: float) -> float:
    return ((target - base) / base) * 100.0 if base else 0.0


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
            "delta_pct_std": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
            "improved_runs": sum(1 for delta in deltas if delta < 0) if metric in LOWER_IS_BETTER else sum(1 for delta in deltas if delta > 0),
            "runs": len(rows),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Repeat Green Context benchmark and write JSONL evidence.")
    parser.add_argument("--repeats", type=int, default=10, help="Number of paired baseline/green runs.")
    parser.add_argument("--green-api", choices=["auto", "pytorch", "cuda_core"], default="cuda_core")
    parser.add_argument("--prefill-sms", type=int, default=32)
    parser.add_argument("--decode-sms", type=int, default=16)
    parser.add_argument("--jsonl", default="green_context_results_cuda_core.jsonl")
    args = parser.parse_args()

    script = Path(__file__).with_name("bench_green.py")
    output_path = Path(args.jsonl)
    if not output_path.is_absolute():
        output_path = Path(__file__).parent / output_path

    env = os.environ.copy()
    env["NANO_VLLM_USE_CUTILE"] = "1"
    env["NANO_VLLM_GREEN_CONTEXT_API"] = args.green_api
    env["NANO_VLLM_PREFILL_SMS"] = str(args.prefill_sms)
    env["NANO_VLLM_DECODE_SMS"] = str(args.decode_sms)

    rows = []
    with output_path.open("a", encoding="utf-8") as f:
        for index in range(1, args.repeats + 1):
            baseline = run_child(script, "--run-baseline", env)
            green = run_child(script, "--run-green", env)
            row = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "run_index": index,
                "requested_api": args.green_api,
                "prefill_sms": args.prefill_sms,
                "decode_sms": args.decode_sms,
                "baseline": baseline,
                "green": green,
                "delta_pct": {metric: pct_delta(baseline[metric], green[metric]) for metric in METRICS},
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            rows.append(row)
            print(
                f"run {index:02d}: "
                f"api={green.get('green_api_type')} enabled={green.get('green_enabled')} "
                f"ttft_delta={row['delta_pct']['ttft']:+.2f}% "
                f"p99_delta={row['delta_pct']['p99_itl']:+.2f}% "
                f"throughput_delta={row['delta_pct']['throughput']:+.2f}%"
            )

    summary = summarize(rows)
    print("\nsummary")
    for metric, item in summary.items():
        print(
            f"{metric:>10}: baseline={item['baseline_mean']:.2f} "
            f"green={item['green_mean']:.2f} "
            f"delta={item['delta_pct_mean']:+.2f}% "
            f"std={item['delta_pct_std']:.2f}% "
            f"improved={item['improved_runs']}/{item['runs']}"
        )
    print(f"raw_jsonl={output_path}")


if __name__ == "__main__":
    main()
