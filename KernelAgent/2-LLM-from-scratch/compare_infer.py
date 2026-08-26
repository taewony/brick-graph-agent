import torch
import time
import argparse
import os
import sys
import numpy as np

# Ensure src directory is in the path so model imports resolve correctly
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from model import GPT as GPT_PyTorch, GPTConfig
from model_cutile import GPT as GPT_cuTile

@torch.no_grad()
def timed_generate(model, prompt, stoi, max_new_tokens=200):
    device = next(model.parameters()).device
    tokens = [stoi[c] for c in prompt if c in stoi]
    idx = torch.tensor([tokens], dtype=torch.long, device=device)

    model.eval()
    
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    
    # 1. Prefill phase (using KV cache)
    logits, past_key_values = model(idx, use_cache=True)
    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    
    torch.cuda.synchronize()
    t_first = time.perf_counter()
    
    generated_tokens = [next_token.item()]
    
    # 2. Decoding phase (feeding in one token at a time with KV Cache)
    for _ in range(max_new_tokens - 1):
        logits, past_key_values = model(next_token, past_key_values=past_key_values, use_cache=True)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_tokens.append(next_token.item())
        
    torch.cuda.synchronize()
    t_end = time.perf_counter()
    
    ttft_sec = t_first - t_start
    total_sec = t_end - t_start
    
    return ttft_sec, total_sec, generated_tokens

def benchmark(model, prompt, stoi, warmup=5, repeats=20, max_new_tokens=200):
    # GPU Warmup
    for _ in range(warmup):
        _ = timed_generate(model, prompt, stoi, max_new_tokens=max_new_tokens)
        
    ttfts = []
    totals = []
    last_tokens = None
    
    for _ in range(repeats):
        ttft, total, tokens = timed_generate(model, prompt, stoi, max_new_tokens=max_new_tokens)
        ttfts.append(ttft)
        totals.append(total)
        last_tokens = tokens
        
    return np.array(ttfts), np.array(totals), last_tokens

if __name__ == "__main__":
    # Dynamically locate default checkpoint (cwd or relative to script)
    default_checkpoint = "checkpoint_final.pt"
    if not os.path.exists(default_checkpoint):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        alt_path = os.path.join(script_dir, "src", "checkpoint_final.pt")
        if os.path.exists(alt_path):
            default_checkpoint = alt_path

    parser = argparse.ArgumentParser(description="Benchmark PyTorch vs cuTile attention inference performance")
    parser.add_argument("--checkpoint", default=default_checkpoint, help="Path to checkpoint file")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup runs")
    parser.add_argument("--repeats", type=int, default=20, help="Number of measurement runs")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Number of tokens to generate")
    args = parser.parse_args()

    # Define standard long prompt (>64 chars to trigger cuTile FMHA)
    long_prompt = (
        "O Romeo, Romeo! wherefore art thou Romeo? Deny thy father and refuse thy name; "
        "Or, if thou wilt not, be but sworn my love, and I'll no longer be a Capulet. "
        "'Tis but thy name that is my enemy; thou art thyself, though not a Montague. "
        "What's Montague? it is nor hand, nor foot, nor arm, nor face, nor any other part "
        "belonging to a man. O, be some other name! What's in a name? that which we call a rose "
        "by any other name would smell as sweet; so Romeo would, were he not Romeo call'd, "
        "retain that dear perfection which he owes without that title. Romeo, doff thy name, "
        "and for that name which is no part of thee take all myself."
    )
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file '{args.checkpoint}' not found. Please train the model first.")
        sys.exit(1)

    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, weights_only=False, map_location='cuda')
    config = checkpoint["config"]
    stoi = checkpoint["stoi"]
    itos = checkpoint["itos"]
    
    # Dynamic sequence length adjustment to prevent position embedding out-of-bounds
    max_total_len = config.block_size
    if len(long_prompt) + args.max_new_tokens >= max_total_len:
        args.max_new_tokens = min(args.max_new_tokens, max_total_len // 2)
        prompt_len = max_total_len - args.max_new_tokens - 16
        long_prompt = long_prompt[:prompt_len]
    
    # Load models
    model_pt = GPT_PyTorch(config)
    model_pt.load_state_dict(checkpoint["model_state_dict"])
    model_pt = model_pt.half().to('cuda')
    
    # Initialize cuTile model with identical config and weights
    model_cu = GPT_cuTile(config).half().to('cuda')
    model_cu.load_state_dict(model_pt.state_dict())
    
    print("\nStarting benchmark...")
    print(f"Prompt length: {len(long_prompt)} characters")
    print(f"Generating {args.max_new_tokens} tokens per run over {args.repeats} repeats (with {args.warmup} warmups)")
    print(f"Model Configuration: {config.n_layer} Layers | {config.n_head} Heads | {config.n_embd} Embedding Dim")
    
    # Benchmark PyTorch
    print("\nBenchmarking PyTorch Native Attention...")
    pt_ttfts, pt_totals, pt_tokens = benchmark(
        model_pt, long_prompt, stoi, warmup=args.warmup, repeats=args.repeats, max_new_tokens=args.max_new_tokens
    )
    
    # Benchmark cuTile
    print("Benchmarking cuTile-based Attention...")
    cu_ttfts, cu_totals, cu_tokens = benchmark(
        model_cu, long_prompt, stoi, warmup=args.warmup, repeats=args.repeats, max_new_tokens=args.max_new_tokens
    )
    
    # Calculate statistics (ms)
    pt_ttft_mean, pt_ttft_std = np.mean(pt_ttfts) * 1000, np.std(pt_ttfts) * 1000
    pt_total_mean, pt_total_std = np.mean(pt_totals) * 1000, np.std(pt_totals) * 1000
    # Dec speed (tokens / sec) = (max_new_tokens - 1) / (total_sec - ttft_sec)
    pt_dec_speed = (args.max_new_tokens - 1) / (pt_totals - pt_ttfts)
    pt_dec_mean, pt_dec_std = np.mean(pt_dec_speed), np.std(pt_dec_speed)
    
    cu_ttft_mean, cu_ttft_std = np.mean(cu_ttfts) * 1000, np.std(cu_ttfts) * 1000
    cu_total_mean, cu_total_std = np.mean(cu_totals) * 1000, np.std(cu_totals) * 1000
    cu_dec_speed = (args.max_new_tokens - 1) / (cu_totals - cu_ttfts)
    cu_dec_mean, cu_dec_std = np.mean(cu_dec_speed), np.std(cu_dec_speed)
    
    # Verification
    tokens_match = (pt_tokens == cu_tokens)
    
    print("\n" + "="*70)
    print("                     BENCHMARK RESULTS")
    print("="*70)
    print(f"Metrics             | Baseline PyTorch       | cuTile Attention       | Speedup")
    print("-"*90)
    print(f"TTFT (Prefill)      | {pt_ttft_mean:7.2f} ± {pt_ttft_std:5.2f} ms | {cu_ttft_mean:7.2f} ± {cu_ttft_std:5.2f} ms | {pt_ttft_mean/cu_ttft_mean:6.3f}x")
    print(f"Total Response Time | {pt_total_mean:7.2f} ± {pt_total_std:5.2f} ms | {cu_total_mean:7.2f} ± {cu_total_std:5.2f} ms | {pt_total_mean/cu_total_mean:6.3f}x")
    print(f"Decoding Speed      | {pt_dec_mean:7.2f} ± {pt_dec_std:5.2f} t/s| {cu_dec_mean:7.2f} ± {cu_dec_std:5.2f} t/s| {cu_dec_mean/pt_dec_mean:6.3f}x")
    print("="*70)
    print(f"Generated Tokens Match Check: {'PASSED' if tokens_match else 'FAILED'}")
    if not tokens_match:
         print(f"First 10 tokens: PyTorch={pt_tokens[:10]} | cuTile={cu_tokens[:10]}")
    else:
         # Decode first 50 chars for illustration
         decoded = "".join([itos[t] for t in pt_tokens[:50]])
         print(f"Generated sample (first 50 tokens): '{decoded.replace(chr(10), ' ')}...'")
