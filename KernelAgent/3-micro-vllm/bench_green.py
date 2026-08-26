import os
import sys
import time
import json
import argparse
import subprocess
from random import randint, seed

# Simple self-contained percentile function to avoid numpy dependency
def get_percentile(data, q):
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = (len(sorted_data) - 1) * (q / 100.0)
    floor_idx = int(idx)
    ceil_idx = min(floor_idx + 1, len(sorted_data) - 1)
    weight = idx - floor_idx
    return sorted_data[floor_idx] * (1.0 - weight) + sorted_data[ceil_idx] * weight


def run_benchmark_workload(use_green: bool):
    os.environ["NANO_VLLM_USE_GREEN_CONTEXTS"] = "1" if use_green else "0"
    os.environ["NANO_VLLM_USE_CUTILE"] = "1"
    
    from nanovllm import LLM, SamplingParams
    
    seed(42)
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    
    # Initialize engine in eager mode for native Windows execution
    llm = LLM(path, enforce_eager=True, max_model_len=4096)
    
    # Dynamic prompts:
    # 1. Latency-sensitive client: short prompt, decodes 100 tokens
    prompt_decode = [randint(0, 10000) for _ in range(30)]
    sampling_params_decode = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=100)
    
    # 2. Heavy prefill client: massive prompt (2048 tokens), runs only prefill + 5 tokens
    prompt_prefill = [randint(0, 10000) for _ in range(2048)]
    sampling_params_prefill = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=5)
    
    # Submit first request
    llm.add_request(prompt_decode, sampling_params_decode)
    
    decode_latencies = []
    ttft = 0.0
    total_tokens_processed = 0
    step_idx = 0
    
    start_total_time = time.time()
    
    while not llm.is_finished():
        step_idx += 1
        
        # Inject heavy prefill client at Step 5
        if step_idx == 5:
            llm.add_request(prompt_prefill, sampling_params_prefill)
            
        t0 = time.time()
        outputs, num_tokens = llm.step()
        t1 = time.time()
        
        step_time_ms = (t1 - t0) * 1000.0
        
        # Accumulate metrics
        if num_tokens > 0:
            # Prefill step
            total_tokens_processed += num_tokens
            # If it's the heavy prompt injected at step 5
            if step_idx >= 5:
                ttft = step_time_ms
        else:
            # Decode step
            total_tokens_processed += (-num_tokens)
            # Record ITL (ignore step 1 which includes decode prompt prefill)
            if step_idx > 1:
                decode_latencies.append(step_time_ms)
                
    total_time = time.time() - start_total_time
    throughput = total_tokens_processed / total_time if total_time > 0 else 0.0
    
    p50_itl = get_percentile(decode_latencies, 50.0)
    p99_itl = get_percentile(decode_latencies, 99.0)
    
    # Print JSON output to stdout for parent process to parse
    runner = getattr(llm, "model_runner", None)
    result = {
        "ttft": ttft,
        "p50_itl": p50_itl,
        "p99_itl": p99_itl,
        "throughput": throughput,
        "total_tokens": total_tokens_processed,
        "total_time": total_time,
        "green_requested_api": os.environ.get("NANO_VLLM_GREEN_CONTEXT_API", "auto"),
        "green_enabled": bool(getattr(runner, "use_green_contexts", False)),
        "green_api_type": getattr(runner, "green_api_type", None),
        "green_prefill_sms": getattr(runner, "green_prefill_sms", None),
        "green_decode_sms": getattr(runner, "green_decode_sms", None),
        "green_split_layout_width": getattr(runner, "green_split_layout_width", None),
        "green_prefill_resource_source": getattr(runner, "green_prefill_resource_source", None),
    }
    print("RESULT_JSON:" + json.dumps(result))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-baseline", action="store_true")
    parser.add_argument("--run-green", action="store_true")
    args = parser.parse_args()
    
    if args.run_baseline:
        run_benchmark_workload(use_green=False)
        return
    elif args.run_green:
        run_benchmark_workload(use_green=True)
        return
        
    print("=============================================================")
    print("🚀 Running Dedicated SM Resource Isolation (Green Contexts) Benchmark")
    print("=============================================================")
    
    # 1. Run Baseline Subprocess
    print("\n⏱️ Running Baseline Configuration (Green Contexts OFF)...")
    env = os.environ.copy()
    proc_base = subprocess.run(
        [sys.executable, __file__, "--run-baseline"],
        capture_output=True,
        text=True,
        env=env
    )
    
    # 2. Run Green Contexts Subprocess
    print("\n🟢 Running Target Configuration (Green Contexts ON)...")
    proc_green = subprocess.run(
        [sys.executable, __file__, "--run-green"],
        capture_output=True,
        text=True,
        env=env
    )
    
    # Helper to parse result JSON from subprocess stdout
    def parse_proc_output(proc):
        if proc.returncode != 0:
            print(f"❌ Subprocess failed with exit code {proc.returncode}")
            print(f"Stdout:\n{proc.stdout}")
            print(f"Stderr:\n{proc.stderr}")
            sys.exit(1)
            
        for line in proc.stdout.splitlines():
            if line.startswith("RESULT_JSON:"):
                return json.loads(line.replace("RESULT_JSON:", "", 1))
        
        print(f"❌ Could not find result JSON in stdout. Raw output:\n{proc.stdout}\nStderr:\n{proc.stderr}")
        sys.exit(1)

    res_base = parse_proc_output(proc_base)
    res_green = parse_proc_output(proc_green)
    
    # 3. Print Comparison Report
    print("\n" + "=" * 70)
    print("📊 BENCHMARK COMPARISON REPORT: BASELINE VS GREEN CONTEXTS")
    print("=" * 70)
    print(f"Model: Qwen2.5-3B-Instruct (Eager cuTile Backend)")
    print(f"Workload: Concurrent 2048-token Prefill + 100-token Decode Client")
    print("-" * 70)
    
    ttft_diff = ((res_green['ttft'] - res_base['ttft']) / res_base['ttft']) * 100.0 if res_base['ttft'] > 0 else 0.0
    p99_diff = ((res_green['p99_itl'] - res_base['p99_itl']) / res_base['p99_itl']) * 100.0 if res_base['p99_itl'] > 0 else 0.0
    tput_diff = ((res_green['throughput'] - res_base['throughput']) / res_base['throughput']) * 100.0 if res_base['throughput'] > 0 else 0.0
    
    print(f"{'Metric':<30} | {'Baseline (Green OFF)':<20} | {'Target (Green ON)':<18} | {'Delta':<10}")
    print("-" * 70)
    print(f"{'TTFT (Prefill Latency)':<30} | {res_base['ttft']:>16.2f} ms | {res_green['ttft']:>14.2f} ms | {ttft_diff:>+7.1f}%")
    print(f"{'Decode P50 ITL (Median)':<30} | {res_base['p50_itl']:>16.2f} ms | {res_green['p50_itl']:>14.2f} ms | {((res_green['p50_itl'] - res_base['p50_itl']) / res_base['p50_itl'] * 100.0):>+7.1f}%")
    print(f"{'Decode P99 ITL (Tail)':<30} | {res_base['p99_itl']:>16.2f} ms | {res_green['p99_itl']:>14.2f} ms | {p99_diff:>+7.1f}%")
    print(f"{'Total Throughput':<30} | {res_base['throughput']:>13.2f} tok/s | {res_green['throughput']:>11.2f} tok/s | {tput_diff:>+7.1f}%")
    print("-" * 70)
    print(f"Total Tokens Processed: Baseline = {res_base['total_tokens']} tok, Green = {res_green['total_tokens']} tok")
    print(f"Total Elapsed Time:     Baseline = {res_base['total_time']:.2f} s, Green = {res_green['total_time']:.2f} s")
    print("=" * 70)
    print("\n💡 Key Insights:")
    if res_green['p99_itl'] < res_base['p99_itl']:
        reduction = ((res_base['p99_itl'] - res_green['p99_itl']) / res_base['p99_itl']) * 100.0
        print(f"  * Decode P99 tail latency was reduced by {reduction:.1f}% under Green Contexts!")
    else:
        print("  * No significant P99 decode tail latency reduction observed in this run.")
        
    if res_green['ttft'] > res_base['ttft']:
        print(f"  * Prefill TTFT increased by {ttft_diff:.1f}% as expected (since prefill SM resources were limited to 32 SMs instead of 48 SMs).")


if __name__ == "__main__":
    main()


