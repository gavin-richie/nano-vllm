"""nanovllm operator kernels: dispatch via `backend` config.

Each layer module imports a factory function from this package and calls it
to obtain the kernel callable for the active backend. The factory does a
lazy import so the heavy TileLang toolchain is only loaded when needed.

Backend options:
- "triton"   — original torch.compile / @triton.jit / flash_attn path.
- "tilelang" — TileLang implementations (see nanovllm/kernels/tilelang/).

Adding a new backend is a matter of registering it under the dict below.
"""
from __future__ import annotations
from typing import Callable, Literal

Backend = Literal["triton", "tilelang"]


def _triton_backend():
    from nanovllm.kernels.triton import (
        rmsnorm,
        add_rmsnorm,
        silu_and_mul,
        apply_rotary_emb,
        store_kvcache,
        triton_flash_attn_varlen,
        triton_flash_attn_with_kvcache,
    )
    return {
        "rmsnorm": rmsnorm,
        "add_rmsnorm": add_rmsnorm,
        "silu_and_mul": silu_and_mul,
        "apply_rotary_emb": apply_rotary_emb,
        "store_kvcache": store_kvcache,
        "flash_attn_varlen": triton_flash_attn_varlen,
        "flash_attn_with_kvcache": triton_flash_attn_with_kvcache,
    }


def _tilelang_backend():
    from nanovllm.kernels.tilelang import (
        tilelang_rmsnorm,
        tilelang_add_rmsnorm,
        tilelang_silu_and_mul,
        tilelang_apply_rotary_emb,
        tilelang_store_kvcache,
        tilelang_flash_attn_varlen,
        tilelang_flash_attn_with_kvcache,
    )
    return {
        "rmsnorm": tilelang_rmsnorm,
        "add_rmsnorm": tilelang_add_rmsnorm,
        "silu_and_mul": tilelang_silu_and_mul,
        "apply_rotary_emb": tilelang_apply_rotary_emb,
        "store_kvcache": tilelang_store_kvcache,
        "flash_attn_varlen": tilelang_flash_attn_varlen,
        "flash_attn_with_kvcache": tilelang_flash_attn_with_kvcache,
    }


_BACKENDS = {
    "triton": _triton_backend,
    "tilelang": _tilelang_backend,
}


def get_kernel(op: str, backend: Backend) -> Callable:
    """Return the kernel callable for `op` on `backend`.

    Ops:
        "rmsnorm", "add_rmsnorm", "silu_and_mul", "apply_rotary_emb",
        "store_kvcache", "flash_attn_varlen", "flash_attn_with_kvcache".
    """
    if backend not in _BACKENDS:
        raise ValueError(f"Unknown backend: {backend!r}. Choose from {list(_BACKENDS)}")
    return _BACKENDS[backend]()[op]


def list_backends() -> list[str]:
    return list(_BACKENDS.keys())
