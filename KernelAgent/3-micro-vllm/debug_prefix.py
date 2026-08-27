"""Prefix-cache prefill 디버그: golden(SDPA) vs target(cuTile) 실제 값 비교."""
import os
import sys
import math
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch.nn.functional as F
from nanovllm.layers.cutile_attention import cutile_fmha_prefill

torch.manual_seed(0)
D = 64
H = 8
block_size = 16
num_blocks = 10

cu_seqlens_q = torch.tensor([0, 16], dtype=torch.int32, device="cuda")
cu_seqlens_k = torch.tensor([0, 48], dtype=torch.int32, device="cuda")
k_cache = torch.randn(num_blocks, block_size, H, D, dtype=torch.float16, device="cuda")
v_cache = torch.randn(num_blocks, block_size, H, D, dtype=torch.float16, device="cuda")
block_table = torch.tensor([[1, 3, 5]], dtype=torch.int32, device="cuda")
q = torch.randn(16, H, D, dtype=torch.float16, device="cuda")
k_dummy = torch.randn(16, H, D, dtype=torch.float16, device="cuda")
v_dummy = torch.randn(16, H, D, dtype=torch.float16, device="cuda")

# Golden: reconstruct kb from cache, SDPA is_causal=True
qb = q.transpose(0, 1).unsqueeze(0)  # [1,H,16,D]
kb = torch.zeros(1, H, 48, D, dtype=torch.float16, device="cuda")
vb = torch.zeros(1, H, 48, D, dtype=torch.float16, device="cuda")
for i in range(48):
    lb = i // block_size
    off = i % block_size
    kb[0, :, i, :] = k_cache[block_table[0, lb].item(), off, :, :]
    vb[0, :, i, :] = v_cache[block_table[0, lb].item(), off, :, :]
golden = F.scaled_dot_product_attention(qb, kb, vb, is_causal=True).squeeze(0).transpose(0, 1)  # [16,H,D]

# Target
target = cutile_fmha_prefill(
    q, k_dummy, v_dummy,
    cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
    max_seqlen_q=16, max_seqlen_k=48,
    scale=1.0 / math.sqrt(D), causal=True,
    k_cache=k_cache, v_cache=v_cache, block_table=block_table,
)

diff = (golden - target).abs()
print(f"max diff: {diff.max().item():.6f}")
print(f"mean diff: {diff.mean().item():.6f}")
print(f"--- token 0, head 0 (prefix에 많이 attend, m=0) ---")
print(f"  golden[:6]: {[round(x,4) for x in golden[0,0,:6].tolist()]}")
print(f"  target[:6]: {[round(x,4) for x in target[0,0,:6].tolist()]}")
print(f"--- token 15, head 0 (거의 full attend, m=15) ---")
print(f"  golden[:6]: {[round(x,4) for x in golden[15,0,:6].tolist()]}")
print(f"  target[:6]: {[round(x,4) for x in target[15,0,:6].tolist()]}")

# PyTorch is_causal semantics 확인: 수동 bottom-right mask와 비교
print("--- PyTorch is_causal semantics check ---")
q_ones = torch.ones(1, 1, 2, D, device="cuda")
k_ones = torch.ones(1, 1, 4, D, device="cuda")
v_ref = torch.tensor([[[[0.0],[1.0],[2.0],[3.0]]]], device="cuda").expand(1, 1, 4, D)
out = F.scaled_dot_product_attention(q_ones, k_ones, v_ref, is_causal=True)
# q0 attends to k0..k2 (bottom-right: j<=i+2) => out ~ (0+1+2)/3 = 1.0
# q1 attends to k0..k3 => out ~ (0+1+2+3)/4 = 1.5
print(f"  SDPA out[0,0,0,0] (expect ~1.0 if bottom-right): {out[0,0,0,0].item():.4f}")
print(f"  SDPA out[0,0,1,0] (expect ~1.5 if bottom-right): {out[0,0,1,0].item():.4f}")
