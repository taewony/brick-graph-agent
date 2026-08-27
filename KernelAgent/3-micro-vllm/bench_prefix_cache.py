import argparse
import json
import os
import statistics
import subprocess
import time
from pathlib import Path
from random import Random


def get_nvidia_smi():
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,clocks.sm,temperature.gpu,power.draw,utilization.gpu,memory.used,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip()
    except Exception as e:
        return f"ERROR: {e}"




VOCAB_LIMIT = 10000


def percentile(values, q):
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = (len(ordered) - 1) * (q / 100.0)
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def clear_persistent_prefix_cache(llm):
    manager = llm.scheduler.block_manager
    if llm.scheduler.waiting or llm.scheduler.running:
        raise RuntimeError("cannot clear prefix cache while requests are active")
    manager.hash_to_block_id.clear()
    for block in manager.blocks:
        if block.ref_count == 0:
            block.hash = -1
            block.token_ids = []


def make_prompt(static_prefix, dynamic_suffix_len, rng, mutate_prefix=False, request_index=0):
    prompt = list(static_prefix)
    if mutate_prefix:
        prompt[0] = (prompt[0] + request_index + 1) % VOCAB_LIMIT
    prompt.extend(rng.randint(0, VOCAB_LIMIT - 1) for _ in range(dynamic_suffix_len))
    return prompt


def step_with_metrics(llm):
    import torch
    seqs, is_prefill = llm.scheduler.schedule()
    cached_tokens = sum(seq.num_cached_tokens for seq in seqs) if is_prefill else 0
    prompt_tokens = sum(len(seq) for seq in seqs) if is_prefill else len(seqs)
    computed_prefill_tokens = sum(len(seq) - seq.num_cached_tokens for seq in seqs) if is_prefill else 0

    t0 = time.perf_counter()
    token_ids = llm.model_runner.call("run", seqs, is_prefill)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    llm.scheduler.postprocess(seqs, token_ids)
    outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
    return {
        "is_prefill": is_prefill,
        "elapsed_ms": elapsed_ms,
        "scheduled_sequences": len(seqs),
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "computed_prefill_tokens": computed_prefill_tokens,
        "decode_tokens": 0 if is_prefill else len(seqs),
        "outputs": outputs,
    }


def run_one_request(llm, prompt, sampling_params):
    llm.add_request(prompt, sampling_params)
    started = time.perf_counter()
    ttft_ms = None
    prompt_tokens = len(prompt)
    cached_tokens = 0
    computed_prefill_tokens = 0
    decode_latencies = []
    output_tokens = 0
    step_count = 0

    while not llm.is_finished():
        step_count += 1
        metrics = step_with_metrics(llm)
        if metrics["is_prefill"]:
            ttft_ms = metrics["elapsed_ms"]
            cached_tokens += metrics["cached_tokens"]
            computed_prefill_tokens += metrics["computed_prefill_tokens"]
        else:
            decode_latencies.append(metrics["elapsed_ms"])
        for _, token_ids in metrics["outputs"]:
            output_tokens += len(token_ids)

    e2e_ms = (time.perf_counter() - started) * 1000.0
    if output_tokens == 0:
        output_tokens = sampling_params.max_tokens
    return {
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "computed_prefill_tokens": computed_prefill_tokens,
        "cache_hit_ratio": cached_tokens / prompt_tokens if prompt_tokens else 0.0,
        "ttft_ms": ttft_ms or 0.0,
        "e2e_ms": e2e_ms,
        "decode_p50_itl_ms": percentile(decode_latencies, 50.0),
        "decode_p99_itl_ms": percentile(decode_latencies, 99.0),
        "decode_steps": len(decode_latencies),
        "output_tokens": output_tokens,
        "throughput_tok_s": output_tokens / (e2e_ms / 1000.0) if e2e_ms > 0 else 0.0,
        "step_count": step_count,
    }


def summarize(condition, rows):
    def mean(key):
        return statistics.mean(row[key] for row in rows) if rows else 0.0

    def stdev(key):
        return statistics.stdev(row[key] for row in rows) if len(rows) > 1 else 0.0

    return {
        "type": "summary",
        "condition": condition,
        "requests": len(rows),
        "mean_cache_hit_ratio": mean("cache_hit_ratio"),
        "mean_cached_tokens": mean("cached_tokens"),
        "mean_computed_prefill_tokens": mean("computed_prefill_tokens"),
        "mean_ttft_ms": mean("ttft_ms"),
        "std_ttft_ms": stdev("ttft_ms"),
        "mean_e2e_ms": mean("e2e_ms"),
        "std_e2e_ms": stdev("e2e_ms"),
        "mean_decode_p50_itl_ms": mean("decode_p50_itl_ms"),
        "mean_decode_p99_itl_ms": mean("decode_p99_itl_ms"),
        "mean_throughput_tok_s": mean("throughput_tok_s"),
        "std_throughput_tok_s": stdev("throughput_tok_s"),
    }


def run_condition(llm, condition, static_prefix, args, out_rows):
    from nanovllm import SamplingParams
    condition_offsets = {"no_cache": 101, "warm_cache": 202, "prefix_changed": 303}
    rng = Random(args.seed + condition_offsets[condition])
    sampling_params = SamplingParams(
        temperature=args.temperature,
        ignore_eos=True,
        max_tokens=args.max_tokens,
    )

    if condition == "warm_cache":
        clear_persistent_prefix_cache(llm)
        prime_prompt = make_prompt(static_prefix, args.dynamic_suffix_tokens, rng)
        prime_metrics = run_one_request(llm, prime_prompt, sampling_params)
        prime_record = {
            "type": "prime",
            "condition": condition,
            "request_index": -1,
            **prime_metrics,
        }
        out_rows.append(prime_record)
        print(json.dumps(prime_record, sort_keys=True))
    else:
        clear_persistent_prefix_cache(llm)

    measured = []
    for i in range(args.requests):
        if condition == "no_cache":
            clear_persistent_prefix_cache(llm)
        prompt = make_prompt(
            static_prefix,
            args.dynamic_suffix_tokens,
            rng,
            mutate_prefix=(condition == "prefix_changed"),
            request_index=i,
        )
        metrics = run_one_request(llm, prompt, sampling_params)
        record = {
            "type": "request",
            "condition": condition,
            "request_index": i,
            "static_prefix_tokens": args.static_prefix_tokens,
            "dynamic_suffix_tokens": args.dynamic_suffix_tokens,
            "max_tokens": args.max_tokens,
            **metrics,
        }
        measured.append(record)
        out_rows.append(record)
        print(json.dumps(record, sort_keys=True))

    summary = summarize(condition, measured)
    out_rows.append(summary)
    print("SUMMARY_JSON:" + json.dumps(summary, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser(description="Fixed-prefix KV-cache benchmark for micro-vllm paper experiments.")
    parser.add_argument("--model-path", default=os.path.expanduser("~/huggingface/Qwen3-0.6B/"))
    parser.add_argument("--use-cutile", action="store_true")
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--static-prefix-tokens", type=int, default=2048)
    parser.add_argument("--dynamic-suffix-tokens", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--conditions", default="no_cache,warm_cache,prefix_changed")
    parser.add_argument("--out-jsonl", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.use_cutile:
        os.environ["NANO_VLLM_USE_CUTILE"] = "1"
    else:
        os.environ.pop("NANO_VLLM_USE_CUTILE", None)

    if args.static_prefix_tokens + args.dynamic_suffix_tokens > args.max_model_len:
        raise ValueError("static prefix plus dynamic suffix must fit within max_model_len")

    rng = Random(args.seed)
    from nanovllm import LLM, SamplingParams
    static_prefix = [rng.randint(0, VOCAB_LIMIT - 1) for _ in range(args.static_prefix_tokens)]
    print(json.dumps({
        "type": "benchmark_start",
        "model_path": args.model_path,
        "backend": "cutile" if args.use_cutile else "default",
        "requests": args.requests,
        "static_prefix_tokens": args.static_prefix_tokens,
        "dynamic_suffix_tokens": args.dynamic_suffix_tokens,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "conditions": args.conditions.split(","),
    }, sort_keys=True))

    llm = LLM(args.model_path, enforce_eager=False, max_model_len=args.max_model_len)
    llm.generate(["Benchmark warmup"], SamplingParams(max_tokens=1, ignore_eos=True), use_tqdm=False)

    rows = []
    for condition in [part.strip() for part in args.conditions.split(",") if part.strip()]:
        if condition not in {"no_cache", "warm_cache", "prefix_changed"}:
            raise ValueError(f"unknown condition: {condition}")
        run_condition(llm, condition, static_prefix, args, rows)

    gpu_after = get_nvidia_smi()
    rows.append({"type": "gpu_after", "nvidia_smi_after": gpu_after})
    print(json.dumps({"type": "gpu_after", "nvidia_smi_after": gpu_after}, sort_keys=True))

    if args.out_jsonl:
        out_path = Path(args.out_jsonl)
        with out_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps({"type": "benchmark_written", "path": str(out_path), "rows": len(rows)}, sort_keys=True))



if __name__ == "__main__":
    main()





