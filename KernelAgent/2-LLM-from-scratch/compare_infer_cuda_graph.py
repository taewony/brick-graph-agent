import torch
import torch.nn as nn
import time
import argparse
import os
import sys
import numpy as np
import math
import cuda.tile as ct
from cuda.tile import RoundingMode as RMd

# Ensure src directory is in path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from model import GPT as GPT_PyTorch, GPTConfig
from model_cutile import GPT as GPT_cuTile
from cutile_kernel import cutile_fmha

INV_LOG_2 = 1.0 / math.log(2)
ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]

# =========================================================================
# 1. cuTile Decoding Attention Kernel Definition
# =========================================================================
@ct.kernel(occupancy=2)
def fmha_decode_kernel(Q, K, V, Out,
                       qk_scale: float,
                       step_idx_tensor,
                       TILE_D: ConstInt,  # head_dim
                       H: ConstInt,       # heads
                       TILE_M: ConstInt,
                       TILE_N: ConstInt,
                       QUERY_GROUP_SIZE: ConstInt):
    # Map block IDs to batch and head indices
    bid_x = ct.bid(0)
    bid_y = ct.bid(1)
    batch_idx = bid_y // H
    head_idx = bid_y % H
    off_kv_h = head_idx // QUERY_GROUP_SIZE

    # Scale query-key scaling factor for exp2
    qk_scale = qk_scale * INV_LOG_2

    # Load current sequence position dynamically from GPU tensor
    step_tile = ct.load(step_idx_tensor, index=(0,), shape=(1,))
    step_val = ct.astype(step_tile, np.int32).reshape((1, 1))

    # Initialize offsets for current query tile (M-dimension)
    offs_m = ct.full((TILE_M, 1), 0, dtype=np.int32) + step_val

    # Initialize local offsets for key/value tile (N-dimension)
    offs_n_tile = ct.arange(TILE_N, dtype=np.int32)
    offs_n_tile = offs_n_tile[None, :]

    # Initialize online softmax accumulators in float32 for stability
    m_i = ct.full((TILE_M, 1), -np.inf, dtype=np.float32)
    l_i = ct.full((TILE_M, 1), 0.0, dtype=np.float32)
    acc = ct.full((TILE_M, TILE_D), 0.0, dtype=np.float32)

    # Load query tile directly
    q = ct.load(
        Q, index=(batch_idx, head_idx, bid_x, 0), shape=(1, 1, TILE_M, TILE_D),
        padding_mode=ct.PaddingMode.ZERO,
        latency=2
    ).reshape((TILE_M, TILE_D))
    
    k_seqlen = K.shape[2]
    Tc = ct.cdiv(k_seqlen, TILE_N)

    # Loop over K, V blocks (N-dimension chunks)
    for j in range(0, Tc):
        # Load K (hint latency=2 for pipelined prefetch)
        k = ct.load(
            K, index=(batch_idx, off_kv_h, 0, j), shape=(1, 1, TILE_D, TILE_N),
            order=(0, 1, 3, 2),
            padding_mode=ct.PaddingMode.ZERO,
            latency=2
        ).reshape((TILE_D, TILE_N))
        
        # Compute QK product
        qk = ct.full((TILE_M, TILE_N), 0., dtype=np.float32)
        qk = ct.mma(q, k, qk)

        # Apply Causal Masking (Dynamic bounds using step_val from GPU)
        offs_n = j * TILE_N + offs_n_tile
        mask = offs_m >= offs_n
        mask = ct.where(mask, 0.0, -np.inf)
        qk += mask

        # Online Softmax Update step
        m_ij = max(m_i, ct.max(qk, axis=-1, keepdims=True) * qk_scale)
        qk = qk * qk_scale - m_ij

        p = ct.exp2(qk, flush_to_zero=True)
        l_ij = ct.sum(p, axis=-1, keepdims=True)
        alpha = ct.exp2(m_i - m_ij, flush_to_zero=True)
        
        l_i = l_i * alpha + l_ij
        acc = acc * alpha

        # Load V (hint latency=4 to hide DRAM latency)
        v = ct.load(
            V, index=(batch_idx, off_kv_h, j, 0), shape=(1, 1, TILE_N, TILE_D),
            padding_mode=ct.PaddingMode.ZERO,
            latency=4
        ).reshape((TILE_N, TILE_D))

        p = p.astype(Q.dtype)
        acc = ct.mma(p, v, acc)
        m_i = m_ij

    # Final Normalization and Store
    acc = ct.truediv(acc, l_i, flush_to_zero=True, rounding_mode=RMd.APPROX)
    acc = acc.reshape((1, 1, TILE_M, TILE_D)).astype(Out.dtype)
    ct.store(Out, index=(batch_idx, head_idx, bid_x, 0), tile=acc)


# ---- Wrapper function to launch the FMHA Decode kernel ----
def cutile_fmha_decode(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                       step_idx_tensor: torch.Tensor,
                       qk_scale: float | None = None,
                       tile_m: int = 64,
                       tile_n: int = 64,
                       query_group_size: int = 1) -> torch.Tensor:
    
    Batch, Heads, SeqLen_Q, D_k = Q.shape
    _, KV_Heads, SeqLen_KV, D_v = V.shape

    if qk_scale is None:
        qk_scale = 1.0 / math.sqrt(D_k)

    Out = torch.empty((Batch, Heads, SeqLen_Q, D_v), dtype=Q.dtype, device=Q.device)

    grid_x = math.ceil(SeqLen_Q / tile_m)
    grid_y = Batch * Heads
    grid = (grid_x, grid_y, 1)

    ct.launch(torch.cuda.current_stream(), grid, fmha_decode_kernel, (
        Q, K, V, Out,
        qk_scale, step_idx_tensor, D_k, Heads,
        tile_m, tile_n, query_group_size
    ))

    return Out


# =========================================================================
# 2. Static Model Runner for CUDA Graph Capture
# =========================================================================
class StaticGPTRunner(nn.Module):
    """
    Wraps the cuTile GPT model with pre-allocated static KV cache buffers.
    This allows capturing a 100% static execution graph during the decoding loop,
    completely bypassing dynamic shape allocations and pointer mutations.
    """
    def __init__(self, model_cu, max_seq_len=1024):
        super().__init__()
        self.model = model_cu
        self.max_seq_len = max_seq_len
        self.config = model_cu.config
        
        # Pre-allocate static KV Cache buffers for each layer
        self.static_k = []
        self.static_v = []
        for _ in range(self.config.n_layer):
            # Shape: [Batch, Heads, MaxSeqLen, HeadDim]
            k_buf = torch.zeros((1, self.config.n_head, max_seq_len, self.config.n_embd // self.config.n_head),
                                dtype=torch.float16, device='cuda')
            v_buf = torch.zeros((1, self.config.n_head, max_seq_len, self.config.n_embd // self.config.n_head),
                                dtype=torch.float16, device='cuda')
            self.static_k.append(k_buf)
            self.static_v.append(v_buf)

        # Pre-allocate static inputs/outputs for the captured graph step
        self.static_input = torch.zeros((1, 1), dtype=torch.long, device='cuda')
        self.static_step = torch.zeros((1,), dtype=torch.long, device='cuda')
        self.static_scatter_index = torch.zeros((1, self.config.n_head, 1, self.config.n_embd // self.config.n_head),
                                                dtype=torch.long, device='cuda')
        self.static_logits = None

    def reset_cache(self, prompt_tokens):
        """
        Initializes the static cache with prompt keys/values.
        """
        # Compute prompt keys/values
        device = next(self.model.parameters()).device
        idx = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
        
        # We run the prefill forward pass once and write directly into static_k and static_v
        self.model.eval()
        with torch.no_grad():
            static_kv = list(zip(self.static_k, self.static_v))
            logits, _ = self.model(idx, use_cache=True, static_kv=static_kv)
            
        prefill_len = len(prompt_tokens)
        return logits, prefill_len

    def forward_step(self, next_token, step_idx):
        """
        Runs a single decoding step using the pre-allocated static buffers.
        """
        B, T = next_token.shape
        pos = step_idx
        
        # Token & Position embeddings
        tok_emb = self.model.transformer.wte(next_token)
        pos_emb = self.model.transformer.wpe(pos)
        x = tok_emb + pos_emb

        for i, block in enumerate(self.model.transformer.h):
            # Layer Normalization before Attention
            normed = block.ln_1(x)
            
            # Compute new Q, K, V
            qkv = block.attn.c_attn(normed)
            q, k, v = qkv.split(self.config.n_embd, dim=2)
            
            head_dim = self.config.n_embd // self.config.n_head
            q = q.view(B, T, self.config.n_head, head_dim).transpose(1, 2).contiguous()
            k = k.view(B, T, self.config.n_head, head_dim).transpose(1, 2).contiguous()
            v = v.view(B, T, self.config.n_head, head_dim).transpose(1, 2).contiguous()

            # Write the new key & value in-place using graph-compatible scatter_
            self.static_k[i].scatter_(2, self.static_scatter_index, k)
            self.static_v[i].scatter_(2, self.static_scatter_index, v)

            # Launch custom cuTile attention decode kernel over the static buffers
            y = cutile_fmha_decode(
                Q=q, K=self.static_k[i], V=self.static_v[i],
                step_idx_tensor=step_idx,
                tile_m=64, tile_n=64
            )

            # Reshape back and add residual connection
            y = y.transpose(1, 2).contiguous().view(B, T, self.config.n_embd)
            x = x + block.attn.c_proj(y)
            
            # MLP block
            x = x + block.mlp(block.ln_2(x))

        # Final LayerNorm & LM Head
        x = self.model.transformer.ln_f(x)
        logits = self.model.lm_head(x)
        return logits


# =========================================================================
# 3. Eager Model Run Generators
# =========================================================================
@torch.no_grad()
def timed_generate_pytorch(model, prompt, stoi, max_new_tokens=200):
    device = next(model.parameters()).device
    tokens = [stoi[c] for c in prompt if c in stoi]
    idx = torch.tensor([tokens], dtype=torch.long, device=device)

    model.eval()
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    
    # Prefill
    logits, past_key_values = model(idx, use_cache=True)
    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize()
    t_first = time.perf_counter()
    
    generated = [next_token.item()]
    
    # Decode
    for _ in range(max_new_tokens - 1):
        logits, past_key_values = model(next_token, past_key_values=past_key_values, use_cache=True)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_token.item())
        
    torch.cuda.synchronize()
    t_end = time.perf_counter()
    return t_first - t_start, t_end - t_start, generated


@torch.no_grad()
def timed_generate_cutile_raw(model, prompt, stoi, max_new_tokens=200):
    device = next(model.parameters()).device
    tokens = [stoi[c] for c in prompt if c in stoi]
    idx = torch.tensor([tokens], dtype=torch.long, device=device)

    model.eval()
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    
    # Prefill
    logits, past_key_values = model(idx, use_cache=True)
    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize()
    t_first = time.perf_counter()
    
    generated = [next_token.item()]
    
    # Decode
    for _ in range(max_new_tokens - 1):
        logits, past_key_values = model(next_token, past_key_values=past_key_values, use_cache=True)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_token.item())
        
    torch.cuda.synchronize()
    t_end = time.perf_counter()
    return t_first - t_start, t_end - t_start, generated


@torch.no_grad()
def timed_generate_cutile_graph(runner, prompt, stoi, max_new_tokens=200):
    tokens = [stoi[c] for c in prompt if c in stoi]
    
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    
    # 1. Prefill and initialize cache
    logits, prefill_len = runner.reset_cache(tokens)
    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize()
    t_first = time.perf_counter()
    
    generated = [next_token.item()]
    
    # 2. CUDA Graph Capture for the decode step
    g = torch.cuda.CUDAGraph()
    
    # Set initial values for graph capture
    runner.static_input.copy_(next_token)
    runner.static_step.copy_(torch.tensor([prefill_len], device='cuda'))
    runner.static_scatter_index.fill_(prefill_len)
    
    # Capture step
    with torch.cuda.graph(g):
        runner.static_logits = runner.forward_step(runner.static_input, runner.static_step)
        
    # 3. Decode Loop using the replayed CUDA Graph
    next_token_tensor = next_token.clone()
    for step in range(prefill_len, prefill_len + max_new_tokens - 1):
        # Update inputs in-place without changing memory addresses
        runner.static_input.copy_(next_token_tensor)
        runner.static_step.copy_(torch.tensor([step], device='cuda'))
        runner.static_scatter_index.fill_(step)
        
        # Replay the CUDA graph (bypasses all PyTorch dispatch & Python launch overheads!)
        g.replay()
        
        # Sample next token
        next_token_tensor = runner.static_logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_token_tensor.item())
        
    torch.cuda.synchronize()
    t_end = time.perf_counter()
    return t_first - t_start, t_end - t_start, generated


def benchmark(model_type, model_or_runner, prompt, stoi, warmup=5, repeats=20, max_new_tokens=200):
    # Warmup
    for _ in range(warmup):
        if model_type == 'pytorch':
            _ = timed_generate_pytorch(model_or_runner, prompt, stoi, max_new_tokens)
        elif model_type == 'cutile_raw':
            _ = timed_generate_cutile_raw(model_or_runner, prompt, stoi, max_new_tokens)
        elif model_type == 'cutile_graph':
            _ = timed_generate_cutile_graph(model_or_runner, prompt, stoi, max_new_tokens)
            
    ttfts = []
    totals = []
    last_tokens = None
    
    for _ in range(repeats):
        if model_type == 'pytorch':
            ttft, total, tokens = timed_generate_pytorch(model_or_runner, prompt, stoi, max_new_tokens)
        elif model_type == 'cutile_raw':
            ttft, total, tokens = timed_generate_cutile_raw(model_or_runner, prompt, stoi, max_new_tokens)
        elif model_type == 'cutile_graph':
            ttft, total, tokens = timed_generate_cutile_graph(model_or_runner, prompt, stoi, max_new_tokens)
            
        ttfts.append(ttft)
        totals.append(total)
        last_tokens = tokens
        
    return np.array(ttfts), np.array(totals), last_tokens


# =========================================================================
# 4. Main Benchmark
# =========================================================================
if __name__ == "__main__":
    # Dynamically locate default checkpoint (cwd or relative to script)
    default_checkpoint = "checkpoint_final.pt"
    if not os.path.exists(default_checkpoint):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        alt_path = os.path.join(script_dir, "src", "checkpoint_final.pt")
        if os.path.exists(alt_path):
            default_checkpoint = alt_path

    parser = argparse.ArgumentParser(description="Benchmark PyTorch vs raw cuTile vs cuTile + CUDA Graphs")
    parser.add_argument("--checkpoint", default=default_checkpoint, help="Path to checkpoint file")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup runs")
    parser.add_argument("--repeats", type=int, default=20, help="Number of measurement runs")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Number of tokens to generate")
    args = parser.parse_args()

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
    
    model_pt = GPT_PyTorch(config)
    model_pt.load_state_dict(checkpoint["model_state_dict"])
    model_pt = model_pt.half().to('cuda')
    
    # Initialize cuTile models
    model_cu = GPT_cuTile(config).half().to('cuda')
    model_cu.load_state_dict(model_pt.state_dict())
    
    runner_graph = StaticGPTRunner(model_cu, max_seq_len=config.block_size)

    
    print("\nStarting benchmark...")
    print(f"Prompt length: {len(long_prompt)} characters")
    print(f"Generating {args.max_new_tokens} tokens per run over {args.repeats} repeats (with {args.warmup} warmups)")
    print(f"Model Configuration: {config.n_layer} Layers | {config.n_head} Heads | {config.n_embd} Embedding Dim")
    
    # 1. Benchmark PyTorch
    print("\nBenchmarking PyTorch Native Attention...")
    pt_ttfts, pt_totals, pt_tokens = benchmark(
        'pytorch', model_pt, long_prompt, stoi, warmup=args.warmup, repeats=args.repeats, max_new_tokens=args.max_new_tokens
    )
    
    # 2. Benchmark cuTile Raw
    print("Benchmarking cuTile Attention (Raw)...")
    cu_ttfts, cu_totals, cu_tokens = benchmark(
        'cutile_raw', model_cu, long_prompt, stoi, warmup=args.warmup, repeats=args.repeats, max_new_tokens=args.max_new_tokens
    )
    
    # 3. Benchmark cuTile + CUDA Graphs
    print("Benchmarking cuTile Attention + CUDA Graphs...")
    graph_ttfts, graph_totals, graph_tokens = benchmark(
        'cutile_graph', runner_graph, long_prompt, stoi, warmup=args.warmup, repeats=args.repeats, max_new_tokens=args.max_new_tokens
    )
    
    # Calculate statistics (ms)
    pt_ttft_mean, pt_ttft_std = np.mean(pt_ttfts) * 1000, np.std(pt_ttfts) * 1000
    pt_total_mean, pt_total_std = np.mean(pt_totals) * 1000, np.std(pt_totals) * 1000
    pt_dec_speed = (args.max_new_tokens - 1) / (pt_totals - pt_ttfts)
    pt_dec_mean, pt_dec_std = np.mean(pt_dec_speed), np.std(pt_dec_speed)
    
    cu_ttft_mean, cu_ttft_std = np.mean(cu_ttfts) * 1000, np.std(cu_ttfts) * 1000
    cu_total_mean, cu_total_std = np.mean(cu_totals) * 1000, np.std(cu_totals) * 1000
    cu_dec_speed = (args.max_new_tokens - 1) / (cu_totals - cu_ttfts)
    cu_dec_mean, cu_dec_std = np.mean(cu_dec_speed), np.std(cu_dec_speed)
    
    graph_ttft_mean, graph_ttft_std = np.mean(graph_ttfts) * 1000, np.std(graph_ttfts) * 1000
    graph_total_mean, graph_total_std = np.mean(graph_totals) * 1000, np.std(graph_totals) * 1000
    graph_dec_speed = (args.max_new_tokens - 1) / (graph_totals - graph_ttfts)
    graph_dec_mean, graph_dec_std = np.mean(graph_dec_speed), np.std(graph_dec_speed)
    
    # Verification
    pt_cu_match = (pt_tokens == cu_tokens)
    cu_graph_match = (cu_tokens == graph_tokens)
    
    print("\n" + "="*95)
    print("                                      BENCHMARK RESULTS")
    print("="*95)
    print(f"Metrics             | Baseline PyTorch       | cuTile Attention (Raw) | cuTile + CUDA Graphs   | Graph Speedup")
    print("-"*95)
    print(f"TTFT (Prefill)      | {pt_ttft_mean:7.2f} ± {pt_ttft_std:5.2f} ms | {cu_ttft_mean:7.2f} ± {cu_ttft_std:5.2f} ms | {graph_ttft_mean:7.2f} ± {graph_ttft_std:5.2f} ms | {cu_ttft_mean/graph_ttft_mean:6.3f}x")
    print(f"Total Response Time | {pt_total_mean:7.2f} ± {pt_total_std:5.2f} ms | {cu_total_mean:7.2f} ± {cu_total_std:5.2f} ms | {graph_total_mean:7.2f} ± {graph_total_std:5.2f} ms | {cu_total_mean/graph_total_mean:6.3f}x")
    print(f"Decoding Speed      | {pt_dec_mean:7.2f} ± {pt_dec_std:5.2f} t/s| {cu_dec_mean:7.2f} ± {cu_dec_std:5.2f} t/s| {graph_dec_mean:7.2f} ± {graph_dec_std:5.2f} t/s| {graph_dec_mean/cu_dec_mean:6.3f}x")
    print("="*95)
    print(f"Accuracy Verifications:")
    print(f"  ➔ PyTorch vs cuTile (Raw) Match Check   : {'PASSED' if pt_cu_match else 'FAILED'}")
    if not pt_cu_match:
        print(f"     PyTorch: {pt_tokens[:20]}")
        print(f"     cuTile (Raw): {cu_tokens[:20]}")
    print(f"  ➔ cuTile (Raw) vs cuTile (Graph) Match : {'PASSED' if cu_graph_match else 'FAILED'}")
    if not cu_graph_match:
        print(f"     cuTile (Raw): {cu_tokens[:20]}")
        print(f"     cuTile (Graph): {graph_tokens[:20]}")
