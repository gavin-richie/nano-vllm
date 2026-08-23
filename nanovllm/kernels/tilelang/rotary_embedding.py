"""TileLang rotary position embedding (RoPE).

Follows the slice-syntax pattern (no `T.arange`, no `mask=mask`):
TileLang 0.1.13 doesn't expose `T.arange` on `tilelang.language`,
so we use Python slice `start:stop` directly inside the kernel body.
"""
import torch

from ._compile_cache import get_compiled


def _build_apply_rotary_emb_kernel(M: int, H: int, D: int, dtype: str):
    """Build apply_rotary_emb kernel for input (M, H, D).

    Splits the last dim into two halves (x1, x2) of size D/2 each, and
    computes:
        y1 = x1 * cos - x2 * sin
        y2 = x2 * cos + x1 * sin
    Concatenates back to shape (M, H, D).
    """
    import tilelang
    from tilelang import language as T

    half = D // 2

    @T.prim_func
    def rope_kernel(
        x: T.Tensor[(M, H, D), dtype],
        cos: T.Tensor[(M, half), dtype],
        sin: T.Tensor[(M, half), dtype],
        y: T.Tensor[(M, H, D), dtype],
    ):
        with T.Kernel(M * H, threads=half) as bx:
            row = bx // H
            head = bx % H

            x1_wire = T.alloc_shared([half, ], dtype)
            x2_wire = T.alloc_shared([half, ], dtype)
            cos_wire = T.alloc_shared([half, ], dtype)
            sin_wire = T.alloc_shared([half, ], dtype)

            T.copy(x[row, head, 0:half], x1_wire)
            T.copy(x[row, head, half:D], x2_wire)
            T.copy(cos[row, 0:half], cos_wire)
            T.copy(sin[row, 0:half], sin_wire)

            y1_wire = T.alloc_shared([half, ], dtype)
            y2_wire = T.alloc_shared([half, ], dtype)
            for i in T.Parallel(half):
                x1 = T.cast(x1_wire[i], T.float32)
                x2 = T.cast(x2_wire[i], T.float32)
                cc = T.cast(cos_wire[i], T.float32)
                ss = T.cast(sin_wire[i], T.float32)
                y1_wire[i] = (x1 * cc - x2 * ss).astype(dtype)
                y2_wire[i] = (x2 * cc + x1 * ss).astype(dtype)
            T.copy(y1_wire, y[row, head, 0:half])
            T.copy(y2_wire, y[row, head, half:D])

    return tilelang.compile(rope_kernel, target="cuda")


def tilelang_apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary embedding to x: (M, H, D) using cos, sin of shape (M, D/2).

    Accepts cos/sin with the head-axis broadcast dim (e.g. derived from
    `cos_sin_cache[positions].chunk(2, -1)` which has shape `(M, 1, D/2)`);
    we squeeze the singleton head axis before the kernel call.
    """
    assert x.is_cuda and cos.is_cuda and sin.is_cuda
    M, H, D = x.shape
    assert D % 2 == 0
    # Squeeze the head-axis broadcast dim if present.
    if cos.dim() == 3 and cos.shape[1] == 1:
        cos = cos.squeeze(1).contiguous()
    else:
        cos = cos.contiguous()
    if sin.dim() == 3 and sin.shape[1] == 1:
        sin = sin.squeeze(1).contiguous()
    else:
        sin = sin.contiguous()
    key = ("apply_rotary_emb", x.dtype, M, H, D)
    compiled = get_compiled(
        key,
        lambda: _build_apply_rotary_emb_kernel(M, H, D, str(x.dtype).replace("torch.", "")),
    )
    y = torch.empty_like(x)
    compiled(x, cos, sin, y)
    return y
