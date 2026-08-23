"""Reference SiluAndMul implementation."""
import torch
import torch.nn.functional as F


@torch.compile
def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    x, y = x.chunk(2, -1)
    return F.silu(x) * y
