"""TileLang RMSNorm kernels (single + fused add+residual).

Two flavours:

- `tilelang_rmsnorm(x, weight, eps)` — pure RMSNorm on the last dim.
- `tilelang_add_rmsnorm(x, residual, weight, eps)` — fused residual add + RMSNorm;
  returns `(y, new_residual)` so the decoder can pass the residual through.

Both match the reference behaviour exactly: accumulate in fp32, cast back to the
input dtype before scaling by `weight`. The reference RMSNorm uses fp32
internally because the variance is the bottleneck for precision.

Each kernel processes one row per block. Inside the block, threads cooperate
to load the row in chunks of `BLOCK_N`, do a parallel reduction over the sum
of squares, then write the normalized output back. `BLOCK_N` is a constexpr
chosen at compile time; the loop over chunks is static.
"""
from typing import Tuple

import torch

# Lazy tilelang imports inside each function so that the triton backend does
# not pay the import cost and tilelang is not required at import time.


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def _build_rmsnorm_kernel(M: int, N: int, BLOCK_N: int, dtype: str, eps: float):
    """Build the single RMSNorm kernel for given shape/dtype.

    Each block holds `BLOCK_N` threads, iterates over the row in `BLOCK_N`
    chunks, and reduces the sum of squares via a tree reduction.
    """
    import tilelang
    from tilelang import language as T

    @T.prim_func
    def rmsnorm_kernel(
        x: T.Tensor[(M, N), dtype],
        weight: T.Tensor[(N,), "float32"],
        y: T.Tensor[(M, N), dtype],
    ):
        with T.Kernel(M, threads=BLOCK_N) as bx:
            x_s = T.alloc_shared((BLOCK_N,), "float32")
            acc = T.alloc_fragment((BLOCK_N,), "float32")
            T.fill(acc, 0.0)

            for tile in T.serial(T.ceildiv(N, BLOCK_N)):
                # offsets = tile * BLOCK_N + T.arange(0, BLOCK_N)
                # mask = offsets < N
                # Load tile (auto-cast from dtype to fp32 for accumulation).
                # xl = T.alloc_shared((BLOCK_N,), "float32")
                # T.copy(x[bx, offsets], xl, mask=mask)
                # x_s = xl
                # acc += x_s * x_s
                T.copy(x[bx, tile*BLOCK_N:(tile+1)*BLOCK_N], x_s)
                for i in T.Parallel(BLOCK_N):
                    acc[i] += x_s[i] * x_s[i]
                    

            sq = T.alloc_fragment((1,), T.float32)
            T.reduce_sum(acc, sq, dim=0)
            rstd = T.rsqrt(sq[0] / N + eps)

            for tile in T.serial(T.ceildiv(N, BLOCK_N)):
                xl = T.alloc_shared([BLOCK_N, ], "float32")
                T.copy(x[bx, tile*BLOCK_N:(tile+1)*BLOCK_N], xl)
                out = T.alloc_shared([BLOCK_N, ], dtype)
                wl = T.alloc_shared([BLOCK_N, ], "float32")
                T.copy(weight[tile*BLOCK_N:(tile+1)*BLOCK_N], wl)
                for i in T.Parallel(BLOCK_N):
                    val = (xl[i]*rstd*wl[i]).astype(dtype)
                    out[i] = val
                T.copy(out, y[bx, tile*BLOCK_N:(tile+1)*BLOCK_N])

                # offsets = tile * BLOCK_N + T.arange(0, BLOCK_N)
                # mask = offsets < N
                # xl = T.alloc_shared((BLOCK_N,), "float32")
                # T.copy(x[bx, offsets], xl, mask=mask)
                # out = (xl * rstd).astype(dtype) * weight
                # y_wire = T.alloc_shared((BLOCK_N,), dtype)
                # T.copy(out, y_wire)
                # T.copy(y_wire, y[bx, offsets], mask=mask)

    return tilelang.compile(rmsnorm_kernel, target="cuda")


def tilelang_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm forward (single, no residual). x: (M, N), weight: (N,)."""
    assert x.is_cuda and weight.is_cuda
    M, N = x.shape
    weight32 = weight.float().contiguous()
    BLOCK_N = min(_next_pow2(N), 1024)
    x = x.contiguous()
    from ._compile_cache import get_compiled
    key = ("rmsnorm", x.dtype, M, N, BLOCK_N)
    compiled = get_compiled(
        key,
        lambda: _build_rmsnorm_kernel(M, N, BLOCK_N, str(x.dtype).replace("torch.", ""), eps),
    )
    y = torch.empty_like(x)
    compiled(x, weight32, y)
    return y


def _build_add_rmsnorm_kernel(M: int, N: int, BLOCK_N: int, dtype: str, eps: float):
    """Build fused residual-add + RMSNorm kernel.

    Mirrors the slice-syntax pattern from the working `rmsnorm_kernel`:
    TileLang 0.1.13 does not expose `T.arange`, so we use Python slice
    `tile*BLOCK_N:(tile+1)*BLOCK_N` and `T.Parallel(BLOCK_N)` for the
    per-thread loop instead of vectorized fragments.
    """
    import tilelang
    from tilelang import language as T

    @T.prim_func
    def add_rmsnorm_kernel(
        x: T.Tensor[(M, N), dtype],
        residual: T.Tensor[(M, N), dtype],
        weight: T.Tensor[(N,), "float32"],
        y: T.Tensor[(M, N), dtype],
        new_residual: T.Tensor[(M, N), dtype],
    ):
        with T.Kernel(M, threads=BLOCK_N) as bx:
            acc = T.alloc_fragment((BLOCK_N,), "float32")
            T.fill(acc, 0.0)

            # Pass 1: read x + residual, accumulate sum of squares.
            for tile in T.serial(T.ceildiv(N, BLOCK_N)):
                x_s = T.alloc_shared([BLOCK_N, ], "float32")
                r_s = T.alloc_shared([BLOCK_N, ], "float32")
                T.copy(x[bx, tile * BLOCK_N:(tile + 1) * BLOCK_N], x_s)
                T.copy(residual[bx, tile * BLOCK_N:(tile + 1) * BLOCK_N], r_s)
                for i in T.Parallel(BLOCK_N):
                    s = x_s[i] + r_s[i]
                    acc[i] += s * s

            sq = T.alloc_fragment((1,), "float32")
            T.reduce_sum(acc, sq, dim=0)
            rstd = T.rsqrt(sq[0] / N + eps)

            # Pass 2: write the residual in original dtype + the normalized output.
            for tile in T.serial(T.ceildiv(N, BLOCK_N)):
                x_s = T.alloc_shared([BLOCK_N, ], "float32")
                r_s = T.alloc_shared([BLOCK_N, ], "float32")
                T.copy(x[bx, tile * BLOCK_N:(tile + 1) * BLOCK_N], x_s)
                T.copy(residual[bx, tile * BLOCK_N:(tile + 1) * BLOCK_N], r_s)

                new_res_wire = T.alloc_shared([BLOCK_N, ], dtype)
                out_wire = T.alloc_shared([BLOCK_N, ], dtype)
                wl = T.alloc_shared([BLOCK_N, ], "float32")
                T.copy(weight[tile * BLOCK_N:(tile + 1) * BLOCK_N], wl)
                for i in T.Parallel(BLOCK_N):
                    s = x_s[i] + r_s[i]
                    new_res_wire[i] = s.astype(dtype)
                    out_wire[i] = (s * rstd).astype(dtype) * wl[i]
                T.copy(new_res_wire, new_residual[bx, tile * BLOCK_N:(tile + 1) * BLOCK_N])
                T.copy(out_wire, y[bx, tile * BLOCK_N:(tile + 1) * BLOCK_N])

    return tilelang.compile(add_rmsnorm_kernel, target="cuda")


def tilelang_add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused residual add + RMSNorm. Returns (y, new_residual)."""
    assert x.is_cuda and residual.is_cuda and weight.is_cuda
    M, N = x.shape
    weight32 = weight.float().contiguous()
    BLOCK_N = min(_next_pow2(N), 1024)
    x = x.contiguous()
    residual = residual.contiguous()
    from ._compile_cache import get_compiled
    key = ("add_rmsnorm", x.dtype, M, N, BLOCK_N)
    compiled = get_compiled(
        key,
        lambda: _build_add_rmsnorm_kernel(M, N, BLOCK_N, str(x.dtype).replace("torch.", ""), eps),
    )
    y = torch.empty_like(x)
    new_residual = torch.empty_like(residual)
    compiled(x, residual, weight32, y, new_residual)
    return y, new_residual
