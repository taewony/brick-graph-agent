import os
import time
import argparse
from random import randint, seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-cutile", action="store_true", help="Use cuTile backend")
    parser.add_argument("--enforce-eager", action="store_true", help="Disable CUDA Graph replay")
    parser.add_argument("--cutile-cudagraph", action="store_true", help="Opt into cuTile CUDA Graph decode replay")
    parser.add_argument("--cutile-prefill-strategy", choices=["hybrid", "direct", "padded"], default="hybrid", help="cuTile prefill strategy for target-PC comparisons")
    parser.add_argument("--num-seqs", type=int, default=256, help="Number of concurrent requests")
    parser.add_argument("--max-input-len", type=int, default=1024, help="Maximum input prompt length (tokens)")
    parser.add_argument("--max-output-len", type=int, default=1024, help="Maximum output length (tokens)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--model-path", default=os.path.expanduser("~/huggingface/Qwen3-0.6B/"), help="Model path")
    parser.add_argument("--max-model-len", type=int, default=4096, help="Max model length")
    parser.add_argument("--graph-mode", choices=["persistent", "copy"], default="persistent", help="CUDA Graph buffer mode (Tier 3a A/B: persistent=신, copy=구)")
    args = parser.parse_args()

    if args.use_cutile:
        os.environ["NANO_VLLM_USE_CUTILE"] = "1"
        os.environ["NANO_VLLM_CUTILE_PREFILL_STRATEGY"] = args.cutile_prefill_strategy
        print("🚀 Using cuTile attention backend")
        print(f"cuTile prefill strategy: {args.cutile_prefill_strategy}")
        if args.cutile_cudagraph:
            print("cuTile CUDA Graph decode: enabled by explicit opt-in")
        else:
            print("cuTile CUDA Graph decode: disabled by default")
    else:
        print("⚡ Using default (FlashAttention) backend")

    os.environ["NANO_VLLM_GRAPH_MODE"] = args.graph_mode
    print(f"CUDA Graph buffer mode: {args.graph_mode}")

    from nanovllm import LLM, SamplingParams

    seed(args.seed)
    num_seqs = args.num_seqs
    max_input_len = args.max_input_len
    max_ouput_len = args.max_output_len

    path = args.model_path
    enforce_eager = args.enforce_eager or (args.use_cutile and not args.cutile_cudagraph)
    llm = LLM(path, enforce_eager=enforce_eager, max_model_len=args.max_model_len)

    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)) for _ in range(num_seqs)]

    llm.generate(["Benchmark: "], SamplingParams())

    print("🚀 Starting benchmark generation loop...")
    t = time.time()
    last_print_time = t
    total_finished = 0
    total_expected = len(prompt_token_ids)
    total_tokens_generated = 0

    for prompt, sp in zip(prompt_token_ids, sampling_params):
        llm.add_request(prompt, sp)

    while not llm.is_finished():
        outputs, num_tokens = llm.step()
        total_finished += len(outputs)
        if num_tokens < 0:
            total_tokens_generated += (-num_tokens)

        current_time = time.time()
        if current_time - last_print_time >= 30.0:
            elapsed = current_time - t
            percent = (total_finished / total_expected) * 100
            current_throughput = total_tokens_generated / elapsed if elapsed > 0 else 0
            active_running = len(llm.scheduler.running)
            active_waiting = len(llm.scheduler.waiting)
            print(f"⏱️ [Progress] Elapsed: {elapsed:.1f}s | Finished: {total_finished}/{total_expected} ({percent:.1f}%) | Active: {active_running} running, {active_waiting} waiting | Generated: {total_tokens_generated} tokens | Decode Throughput: {current_throughput:.1f} tok/s")
            last_print_time = current_time

    t = (time.time() - t)
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")

    import json
    result = {
        "type": "throughput_result",
        "total_tokens": total_tokens,
        "time_s": round(t, 4),
        "throughput_tok_s": round(throughput, 4),
        "conditions": {
            "use_cutile": args.use_cutile,
            "cutile_prefill_strategy": args.cutile_prefill_strategy,
            "cutile_cudagraph": args.cutile_cudagraph,
            "enforce_eager": enforce_eager,
            "num_seqs": num_seqs,
            "max_input_len": max_input_len,
            "max_output_len": max_ouput_len,
            "seed": args.seed,
            "model_path": path,
            "max_model_len": args.max_model_len,
            "graph_mode": args.graph_mode,
        },
    }
    print("RESULT_JSON:" + json.dumps(result))


if __name__ == "__main__":
    main()
