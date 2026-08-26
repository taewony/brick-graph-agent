import torch
import torch.nn as nn
from dataclasses import dataclass
from torch.nn import functional as F
from cutile_kernel import cutile_fmha # 우리가 만든 커널 임포트

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x, past_kv=None, use_cache=False, static_kv=None):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # reshape for multi-head: (B, T, C) -> (B, n_head, T, head_dim)
        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2).contiguous()
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2).contiguous()
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2).contiguous()

        # If static cache is provided, write directly into it
        if static_kv is not None:
            static_k_i, static_v_i = static_kv
            static_k_i[:, :, :T, :].copy_(k)
            static_v_i[:, :, :T, :].copy_(v)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)

        present_kv = (k, v) if use_cache else None

        # =================================================================
        # [Smart Routing] Prefill vs Decoding with KV Cache
        # =================================================================
        if past_kv is None:
            if T < 64:
                # Pad Q, K, V along the sequence dimension to 64 to avoid falling back to PyTorch SDPA
                pad_len = 64 - T
                q_padded = F.pad(q, (0, 0, 0, pad_len))
                k_padded = F.pad(k, (0, 0, 0, pad_len))
                v_padded = F.pad(v, (0, 0, 0, pad_len))
                y_padded = cutile_fmha(
                    Q=q_padded, K=k_padded, V=v_padded,
                    tile_m=64, tile_n=64,
                    causal=True,
                    input_pos=0
                )
                y = y_padded[:, :, :T, :]
            else:
                y = cutile_fmha(
                    Q=q, K=k, V=v,
                    tile_m=64, tile_n=64, # Phase 1 proven best tile
                    causal=True,
                    input_pos=0
                )
        else:
            cache_len = k.shape[2]
            y = cutile_fmha(
                Q=q, K=k, V=v,
                tile_m=64, tile_n=64,
                causal=False,
                input_pos=cache_len - 1
            )

        # reshape back to (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        out = self.c_proj(y)
        
        if use_cache:
            return out, present_kv
        return out
        
@dataclass
class GPTConfig:
    vocab_size: int = 65       # character-level: 65 unique chars in Shakespeare
    block_size: int = 256      # max sequence length (context window)
    n_layer: int = 6           # number of transformer blocks
    n_head: int = 6            # number of attention heads
    n_embd: int = 384          # embedding dimension

class CausalSelfAttention_Original_Code(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)  # Q, K, V projections
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)       # output projection
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # reshape for multi-head: (B, T, C) → (B, n_head, T, head_dim)
        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        # attention with causal mask (each token can only attend to previous tokens)
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)       # project up: 384 → 1536
        x = self.gelu(x)       # non-linearity
        return self.c_proj(x)  # project back down: 1536 → 384

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, past_kv=None, use_cache=False, static_kv=None):
        if use_cache:
            attn_out, present_kv = self.attn(self.ln_1(x), past_kv=past_kv, use_cache=use_cache, static_kv=static_kv)
            x = x + attn_out
            x = x + self.mlp(self.ln_2(x))
            return x, present_kv
        else:
            x = x + self.attn(self.ln_1(x), static_kv=static_kv)   # attention with residual connection
            x = x + self.mlp(self.ln_2(x))    # MLP with residual connection
            return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),   # token embeddings
            wpe = nn.Embedding(config.block_size, config.n_embd),   # position embeddings
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # weight tying: the output projection shares weights with the token embeddings
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx, targets=None, past_key_values=None, use_cache=False, static_kv=None):
        B, T = idx.shape
        
        prev_T = 0
        if past_key_values is not None:
            k_temp = past_key_values[0][0]
            if k_temp.dim() == 4:
                prev_T = k_temp.shape[2]
            else:
                prev_T = k_temp.shape[1]
                
        pos = torch.arange(prev_T, prev_T + T, dtype=torch.long, device=idx.device)

        tok_emb = self.transformer.wte(idx)    # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)    # (T, n_embd)
        x = tok_emb + pos_emb                  # (B, T, n_embd) — broadcasting adds position info

        new_past_key_values = [] if use_cache else None
        for i, block in enumerate(self.transformer.h):
            past_kv = past_key_values[i] if past_key_values is not None else None
            layer_static_kv = static_kv[i] if static_kv is not None else None
            if use_cache:
                x, present_kv = block(x, past_kv=past_kv, use_cache=use_cache, static_kv=layer_static_kv)
                new_past_key_values.append(present_kv)
            else:
                x = block(x, static_kv=layer_static_kv)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)               # (B, T, vocab_size)

        if use_cache:
            return logits, new_past_key_values

        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
        return logits, loss
