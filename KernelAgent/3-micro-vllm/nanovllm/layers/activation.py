import torch
from torch import nn
import torch.nn.functional as F


def optional_compile(fn):
    try:
        import triton
        return torch.compile(fn)
    except ImportError:
        return fn


class SiluAndMul(nn.Module):

    def __init__(self):
        super().__init__()

    @optional_compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y

