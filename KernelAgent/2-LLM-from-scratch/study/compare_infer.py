import torch
import time
import argparse
from src.model import GPT as GPT_PyTorch
from src.model_cutile import GPT as GPT_cuTile

@torch.no_grad()
def measure_performance(model, prompt, stoi, max_new_tokens=200):
    device = next(model.parameters()).device
    tokens = [stoi[c] for c in prompt if c in stoi]
    idx = torch.tensor([tokens], dtype=torch.long, device=device)

    # 1. GPU Warmup (메모리 할당 등 초기화 지연 방지)
    _ = model(idx)
    torch.cuda.synchronize()

    # 2. Measure TTFT (Time To First Token - Prefill Phase)
    start_time = time.perf_counter()
    logits, _ = model(idx)
    torch.cuda.synchronize()
    ttft_ms = (time.perf_counter() - start_time) * 1000

    # 첫 토큰 샘플링
    logits = logits[:, -1, :]
    probs = torch.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    idx = torch.cat([idx, next_token], dim=1)

    # 3. Measure Decoding Speed (Tokens / sec)
    start_decode = time.perf_counter()
    for _ in range(max_new_tokens - 1):
        idx_cond = idx[:, -model.config.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_token], dim=1)
    torch.cuda.synchronize()
    
    decode_time = time.perf_counter() - start_decode
    tok_per_sec = (max_new_tokens - 1) / decode_time

    return ttft_ms, tok_per_sec

if __name__ == "__main__":
    checkpoint_path = "checkpoint_final.pt"
    # cuTile(T>=64)가 작동하도록 프롬프트를 64자 이상으로 길게 설정 
    # long_prompt = "O Romeo, Romeo! wherefore art thou Romeo? Deny thy father and refuse thy name; Or, if thou wilt not, be but sworn my love,"
    # compare_infer.py의 long_prompt를 1024자 이상으로 확장
    long_prompt = (
        "O Romeo, Romeo! wherefore art thou Romeo? Deny thy father and refuse thy name; "
        "Or, if thou wilt not, be but sworn my love, and I'll no longer be a Capulet. "
        "'Tis but thy name that is my enemy."
    )
    
    print("Loading Checkpoint...")
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location='cuda')
    config, stoi = checkpoint["config"], checkpoint["stoi"]

    print("\n[1] Baseline PyTorch Model")
    model_pt = GPT_PyTorch(config).to('cuda')
    model_pt.load_state_dict(checkpoint["model_state_dict"])
    model_pt.eval()
    pt_ttft, pt_tok = measure_performance(model_pt, long_prompt, stoi)
    print(f" ➔ TTFT (Prefill)  : {pt_ttft:.2f} ms")
    print(f" ➔ Decoding Speed  : {pt_tok:.2f} tokens/sec")

    print("\n[2] cuTile_v1 (64x64 FMHA) Model")
    model_cu = GPT_cuTile(config).to('cuda')
    model_cu.load_state_dict(checkpoint["model_state_dict"])
    model_cu.eval()
    cu_ttft, cu_tok = measure_performance(model_cu, long_prompt, stoi)
    print(f" ➔ TTFT (Prefill)  : {cu_ttft:.2f} ms")
    print(f" ➔ Decoding Speed  : {cu_tok:.2f} tokens/sec")
    
    print("\n=== Analysis ===")
    print(f"Prefill Speedup  : {pt_ttft / cu_ttft:.3f}x")
    print(f"Decoding Speedup : {cu_tok / pt_tok:.3f}x")
