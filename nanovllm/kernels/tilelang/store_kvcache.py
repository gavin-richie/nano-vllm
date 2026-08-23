"""TileLang store_kvcache kernel (paged KV cache write).

Mirrors the existing @triton.jit kernel in
`nanovllm/kernels/_reference/store_kvcache.py`. The cache is laid out as
`(num_blocks, block_size, num_heads, head_dim)` and we view it as a flat
`(num_blocks * block_size * D,)` buffer for the kernel. The cache size is
baked into the static shape so the kernel is recompiled per cache shape.
"""
import torch

from ._compile_cache import get_compiled


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def _build_store_kvcache_kernel(N: int, D: int, BLOCK_D: int, cache_size: int, dtype: str):
    """Build the store_kvcache kernel for N tokens with D = num_heads * head_dim."""
    import tilelang
    from tilelang import language as T

    @T.prim_func
    def store_kvcache_kernel(
        key: T.Tensor[(N, D), dtype],
        value: T.Tensor[(N, D), dtype],
        k_cache: T.Tensor[(cache_size,), dtype],
        v_cache: T.Tensor[(cache_size,), dtype],
        slot_mapping: T.Tensor[(N,), "int32"],
    ):
        with T.Kernel(N, threads=BLOCK_D) as bx:
            slot_w = T.alloc_shared([1, ], "int32")
            T.copy(slot_mapping[bx:bx + 1], slot_w)
            offset = slot_w[0] * D

            k_wire = T.alloc_shared([BLOCK_D, ], dtype)
            v_wire = T.alloc_shared([BLOCK_D, ], dtype)
            T.copy(key[bx, 0:BLOCK_D], k_wire)
            T.copy(value[bx, 0:BLOCK_D], v_wire)
            for i in T.Parallel(BLOCK_D):
                if slot_w[0] >= 0:
                    k_cache[offset + i] = k_wire[i]
                    v_cache[offset + i] = v_wire[i]

    return tilelang.compile(store_kvcache_kernel, target="cuda")


def tilelang_store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    """TileLang equivalent of the original store_kvcache Triton kernel."""
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N

    key2 = key.reshape(N, D).contiguous()
    value2 = value.reshape(N, D).contiguous()
    slot_mapping = slot_mapping.contiguous()
    BLOCK_D = _next_pow2(D)
    dtype_str = str(key.dtype).replace("torch.", "")

    # Use the actual cache size as the static shape. The kernel is
    # recompiled per (cache_size, D, dtype) tuple, which is essentially
    # the same number of recompiles as the original Triton kernel.
    cache_size = k_cache.numel()

    cache_key = ("store_kvcache", key2.dtype, N, D, BLOCK_D, cache_size)
    compiled = get_compiled(
        cache_key,
        lambda: _build_store_kvcache_kernel(N, D, BLOCK_D, cache_size, dtype_str),
    )

    k_flat = k_cache.reshape(-1)
    v_flat = v_cache.reshape(-1)
    compiled(key2, value2, k_flat, v_flat, slot_mapping)
