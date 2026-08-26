import torch
from torch import nn


def optional_compile(fn):
    try:
        import triton
        return torch.compile(fn)
    except ImportError:
        return fn


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    @optional_compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens

