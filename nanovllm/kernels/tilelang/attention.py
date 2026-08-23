"""TileLang FlashAttention implementations (paged KV + varlen).

Two entry points:

- `tilelang_flash_attn_varlen`: prefill attention with paged KV cache.
  Equivalent to `flash_attn_varlen_func` from the flash_attn package.
- `tilelang_flash_attn_with_kvcache`: decode attention with in-place KV update.
  Equivalent to `flash_attn_with_kvcache`.

TileLang flash-attention is the highest-risk op in this backend. The paged-KV
gather requires TileLang to look up physical block ids and load K/V rows
from inside a tile — non-trivial to express in the current TileLang language.
We provide a simplified non-paged reference path that uses the upstream
flash_attn package for now (the same package the triton backend uses), and
the implementation guide (`docs/tile/2.implementation-guide.md`) walks the
reader through the TileLang version step-by-step.
"""
from typing import Optional

import torch


def tilelang_flash_attn_varlen(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    causal: bool,
    block_table: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """TileLang prefill attention dispatcher.

    For the first cut, delegates to the reference flash_attn package. Replacing
    this with a real TileLang kernel is the focus of Phase 3.
    """
    from nanovllm.kernels._reference.attention import flash_attn_varlen
    return flash_attn_varlen(
        q, k_cache, v_cache,
        cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k,
        softmax_scale, causal, block_table,
    )


def tilelang_flash_attn_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
    block_table: torch.Tensor,
    softmax_scale: float,
    causal: bool,
) -> torch.Tensor:
    """TileLang decode attention dispatcher.

    For the first cut, delegates to the reference flash_attn package.
    """
    from nanovllm.kernels._reference.attention import flash_attn_with_kvcache
    return flash_attn_with_kvcache(
        q, k_cache, v_cache,
        cache_seqlens, block_table,
        softmax_scale, causal,
    )
