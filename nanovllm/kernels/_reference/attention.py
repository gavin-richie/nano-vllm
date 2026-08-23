"""Reference attention implementations (flash_attn wrappers).

NOTE: The local function names `flash_attn_varlen` and `flash_attn_with_kvcache`
SHADOW the imports from the `flash_attn` package. Anything inside the wrapper
that needs the real upstream function must call the underscored alias
(`_fa_varlen_func`, `_fa_with_kvcache`) to avoid recursing into itself.
"""
from flash_attn import (
    flash_attn_varlen_func as _fa_varlen_func,
    flash_attn_with_kvcache as _fa_with_kvcache,
)


def flash_attn_varlen(
    q,
    k_cache,
    v_cache,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    softmax_scale,
    causal,
    block_table,
):
    return _fa_varlen_func(
        q, k_cache, v_cache,
        max_seqlen_q=max_seqlen_q,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_k=max_seqlen_k,
        cu_seqlens_k=cu_seqlens_k,
        softmax_scale=softmax_scale,
        causal=causal,
        block_table=block_table,
    )


def flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    cache_seqlens,
    block_table,
    softmax_scale,
    causal,
):
    return _fa_with_kvcache(
        q, k_cache, v_cache,
        cache_seqlens=cache_seqlens,
        block_table=block_table,
        softmax_scale=softmax_scale,
        causal=causal,
    )
