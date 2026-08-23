"""Reference rotary embedding implementation.

The `torch.compile` decorator on the original `apply_rotary_emb` breaks in
torch 2.8 for shapes > 1 (Dynamo graph build error). We keep the compiled
version as the production path and add an eager fallback `_apply_rotary_emb`
that the unit tests can use reliably.
"""
import torch


@torch.compile
def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


def _apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Eager reference, used by tests to avoid torch.compile Dynamo issues."""
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)
