#!/usr/bin/env python3
"""micro-vLLM 논문 실험 하니스 (baseline tracking).

논문 실행조건(experiment_conditions.md)을 지키며 처리량 벤치마크(bench.py)를 실행하고,
실행조건(환경/모델/flags) + 결과를 JSON으로 기록해 기준선 변동을 추적한다.

사용 예:
  python run_experiment.py --use-cutile --cutile-prefill-strategy hybrid
  python run_experiment.py --use-cutile --cutile-cudagraph --repeats 3 --out-jsonl baseline.jsonl
"""
import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def find_git_root() -> Path | None:
    d = SCRIPT_DIR
    while d != d.parent:
        if (d / ".git").exists():
            return d
        d = d.parent
    return None


def run_cmd(cmd: list[str], cwd: Path | None = None) -> str:
    try:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {e}"


def get_env_info() -> dict:
    info: dict = {}
    root = find_git_root()
    if root is not None:
        info["git_root"] = str(root)
        info["git_commit"] = run_cmd(["git", "rev-parse", "--short", "HEAD"], cwd=root)
        info["git_dirty"] = run_cmd(["git", "status", "--porcelain"], cwd=root) != ""
    else:
        info["git_root"] = None
        info["git_commit"] = "unknown"
        info["git_dirty"] = None
    info["nvidia_smi"] = run_cmd([
        "nvidia-smi",
        "--query-gpu=name,driver_version,clocks.sm,temperature.gpu,power.draw,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader",
    ])
    info["os"] = sys.platform
    info["python"] = sys.version.split()[0]
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            prop = torch.cuda.get_device_properties(0)
            info["gpu_name"] = prop.name
            info["gpu_total_mem_gb"] = round(prop.total_memory / (1024 ** 3), 1)
            info["gpu_sm_count"] = getattr(prop, "multi_processor_count", None)
    except Exception as e:  # noqa: BLE001
        info["torch"] = f"ERROR: {e}"
    try:
        import cuda
        info["cuda_python"] = getattr(cuda, "__version__", "unknown")
    except Exception:  # noqa: BLE001
        pass
    return info


def get_model_info(model_path: str) -> dict:
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(os.path.expanduser(model_path))
        num_heads = getattr(cfg, "num_attention_heads", 1)
        hidden_size = getattr(cfg, "hidden_size", 1)
        return {
            "arch": getattr(cfg, "architectures", ["Unknown"])[0],
            "hidden_size": hidden_size,
            "num_layers": getattr(cfg, "num_hidden_layers", None),
            "num_heads": num_heads,
            "num_kv_heads": getattr(cfg, "num_key_value_heads", None),
            "head_dim": getattr(cfg, "head_dim", None) or (hidden_size // num_heads),
            "vocab_size": getattr(cfg, "vocab_size", None),
            "torch_dtype": str(getattr(cfg, "torch_dtype", None)),
        }
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def build_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [sys.executable, str(SCRIPT_DIR / "bench.py")]
    if args.use_cutile:
        cmd += ["--use-cutile"]
    cmd += ["--cutile-prefill-strategy", args.cutile_prefill_strategy]
    if args.cutile_cudagraph:
        cmd += ["--cutile-cudagraph"]
    if args.enforce_eager:
        cmd += ["--enforce-eager"]
    cmd += [
        "--num-seqs", str(args.num_seqs),
        "--max-input-len", str(args.max_input_len),
        "--max-output-len", str(args.max_output_len),
        "--seed", str(args.seed),
        "--model-path", args.model_path,
        "--max-model-len", str(args.max_model_len),
        "--graph-mode", args.graph_mode,
    ]
    return cmd


def run_bench(cmd: list[str]) -> dict:
    proc = subprocess.run(cmd, cwd=str(SCRIPT_DIR), capture_output=True, text=True)
    result: dict = {}
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            try:
                result = json.loads(line[len("RESULT_JSON:"):])
            except json.JSONDecodeError:
                pass
    if proc.returncode != 0:
        result["error"] = True
        result["stderr_tail"] = proc.stderr.splitlines()[-20:] if proc.stderr else []
    result["exit_code"] = proc.returncode
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="micro-vLLM throughput benchmark with condition tracking.")
    parser.add_argument("--use-cutile", action="store_true")
    parser.add_argument("--cutile-prefill-strategy", choices=["hybrid", "direct", "padded"], default="hybrid")
    parser.add_argument("--cutile-cudagraph", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--num-seqs", type=int, default=256)
    parser.add_argument("--max-input-len", type=int, default=1024)
    parser.add_argument("--max-output-len", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-path", default="~/huggingface/Qwen3-0.6B/")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--graph-mode", choices=["persistent", "copy"], default="persistent", help="CUDA Graph buffer mode (Tier 3a A/B)")
    parser.add_argument("--compare-graph-modes", action="store_true", help="Run both persistent & copy modes and print a comparison")
    parser.add_argument("--repeats", type=int, default=1, help="Number of repeats per mode")
    parser.add_argument("--out-jsonl", default="baseline.jsonl", help="Append records to this JSONL (relative to script dir)")
    args = parser.parse_args()

    env = get_env_info()
    model = get_model_info(args.model_path)
    out_path = SCRIPT_DIR / args.out_jsonl

    modes = ["persistent", "copy"] if args.compare_graph_modes else [args.graph_mode]
    throughputs: dict[str, list[float]] = {}

    for mode in modes:
        args.graph_mode = mode
        cmd = build_cmd(args)
        for i in range(args.repeats):
            result = run_bench(cmd)
            record = {
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "env": env,
                "model": model,
                "cmd": " ".join(cmd),
                "graph_mode": mode,
                "repeat": i + 1,
                "result": result,
            }
            print(json.dumps(record, ensure_ascii=False))
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            tp = (result.get("throughput_tok_s") or 0.0)
            throughputs.setdefault(mode, []).append(tp)

    print(f"appended {len(modes) * args.repeats} record(s) to {out_path}")

    if args.compare_graph_modes and len(modes) == 2:
        def mean(vals):
            return sum(vals) / len(vals) if vals else 0.0
        tp_copy = mean(throughputs.get("copy", []))
        tp_persist = mean(throughputs.get("persistent", []))
        print("\n=== graph-mode comparison ===")
        print(f"copy (구버전)     : {tp_copy:.2f} tok/s")
        print(f"persistent (신버전): {tp_persist:.2f} tok/s")
        if tp_copy > 0:
            delta = (tp_persist - tp_copy) / tp_copy * 100.0
            print(f"delta             : {delta:+.2f}%")


if __name__ == "__main__":
    main()
