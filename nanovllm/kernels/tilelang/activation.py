"""TileLang SiluAndMul (SwiGLU activation)."""
import torch
import tilelang
from tilelang import language as T

from ._compile_cache import get_compiled


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def _build_silu_and_mul_kernel(M: int, K: int, BLOCK_K: int, dtype: str):
    """Build silu_and_mul kernel for input (M, 2K) -> (M, K).

    Uses one block per row, BLOCK_K threads per block, tiles of BLOCK_K
    over the K dimension. BLOCK_K must be ≤ 1024 (CUDA max threads per
    block).
    """

    @T.prim_func
    def silu_and_mul_kernel(
        x: T.Tensor[(M, 2 * K), dtype],
        y: T.Tensor[(M, K), dtype],
    ):
        with T.Kernel(M, threads=BLOCK_K) as bx:
            for tile in T.serial(T.ceildiv(K, BLOCK_K)):
                a_wire = T.alloc_shared([BLOCK_K, ], dtype)
                b_wire = T.alloc_shared([BLOCK_K, ], dtype)
                y_wire = T.alloc_shared([BLOCK_K, ], dtype)

                T.copy(x[bx, tile * BLOCK_K:(tile + 1) * BLOCK_K], a_wire)
                T.copy(x[bx, K + tile * BLOCK_K:K + (tile + 1) * BLOCK_K], b_wire)

                # silu = a * sigmoid(a) = a / (1 + exp(-a)). Use fp32 for stability.
                for k in T.Parallel(BLOCK_K):
                    a32 = T.cast(a_wire[k], T.float32)
                    b32 = b_wire[k].astype(T.float32)
                    silu = a32 / (1.0 + T.exp(-a32))
                    y_wire[k] = T.cast((silu * b32), dtype)
                T.copy(y_wire, y[bx, tile * BLOCK_K:(tile + 1) * BLOCK_K])

    return tilelang.compile(silu_and_mul_kernel, target="cuda")


def tilelang_silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """SwiGLU activation: y = silu(x[..., :K]) * x[..., K:2K]."""
    assert x.is_cuda
    M, two_K = x.shape
    assert two_K % 2 == 0
    K = two_K // 2
    BLOCK_K = min(_next_pow2(K), 1024)
    key = ("silu_and_mul", x.dtype, M, K, BLOCK_K)
    compiled = get_compiled(
        key,
        lambda: _build_silu_and_mul_kernel(M, K, BLOCK_K, str(x.dtype).replace("torch.", "")),
    )
    y = torch.empty((M, K), dtype=x.dtype, device=x.device)
    compiled(x, y)
    return y
