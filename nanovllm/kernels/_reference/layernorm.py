"""Reference RMSNorm implementations matching the original torch.compile behavior.

These functions are NOT for runtime use in production (the layers' torch.compile
wrappers are still active for the triton backend). They exist so that the
TileLang kernels in tests/bench_*/test_* scripts can be compared bit-equal
against the original semantics.
"""
import torch


@torch.compile
def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    orig_dtype = x.dtype
    x = x.float()
    var = x.pow(2).mean(dim=-1, keepdim=True)
    x.mul_(torch.rsqrt(var + eps))
    x = x.to(orig_dtype).mul_(weight)
    return x


@torch.compile
def add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_dtype = x.dtype
    x = x.float().add_(residual.float())
    residual = x.to(orig_dtype)
    var = x.pow(2).mean(dim=-1, keepdim=True)
    x.mul_(torch.rsqrt(var + eps))
    x = x.to(orig_dtype).mul_(weight)
    return x, residual
