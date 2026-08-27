import torch
from torch import nn
import torch.nn.functional as F

# fallback 발생 횟수 (논문 정합성용: 0이면 fused F.rms_norm이 정상 사용됨)
_RMSNORM_FALLBACKS = 0
_RMSNORM_WARNED = False


def _report_fallback():
    global _RMSNORM_WARNED
    if not _RMSNORM_WARNED:
        _RMSNORM_WARNED = True
        print("WARNING: F.rms_norm failed; falling back to manual RMSNorm (fusion INACTIVE)")


def get_rmsnorm_fallback_count() -> int:
    return _RMSNORM_FALLBACKS


class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def _rms_norm_manual(self, x: torch.Tensor) -> torch.Tensor:
        # 원본 수동 구현 (fallback 용)
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        return x.to(orig_dtype).mul_(self.weight)

    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        global _RMSNORM_FALLBACKS
        # Tier 2b: 수동 RMSNorm(다수 커널) -> PyTorch fused rms_norm(단일 커널)
        try:
            return F.rms_norm(x, [x.shape[-1]], self.weight, self.eps)
        except Exception:
            _RMSNORM_FALLBACKS += 1
            _report_fallback()
            return self._rms_norm_manual(x)

    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        global _RMSNORM_FALLBACKS
        orig_dtype = x.dtype
        # residual 연결은 정밀도 보존을 위해 float32 유지 (원본 동작과 동일)
        x = x.float().add_(residual.float())
        residual = x.to(orig_dtype)
        try:
            # RMSNorm 본체만 fused 커널 사용 (float32 입력, weight는 이후 별도 적용)
            x = F.rms_norm(x, [x.shape[-1]], eps=self.eps)
        except Exception:
            _RMSNORM_FALLBACKS += 1
            _report_fallback()
            var = x.pow(2).mean(dim=-1, keepdim=True)
            x = x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
