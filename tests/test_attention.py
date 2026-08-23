"""Numerical correctness for tilelang FlashAttention vs the reference.

For the first cut, the TileLang kernels delegate to the reference flash_attn
package. So this test is effectively a smoke test asserting that the
delegate returns sane outputs. Once Phase 3 lands the real implementation,
swap the comparison to a separate hand-written reference.
"""
import pytest
import torch

from nanovllm.kernels._reference.attention import (
    flash_attn_varlen as ref_varlen,
    flash_attn_with_kvcache as ref_with_kvcache,
)
from nanovllm.kernels.tilelang.attention import (
    tilelang_flash_attn_varlen,
    tilelang_flash_attn_with_kvcache,
)
from .conftest import assert_close


def _build_inputs(num_seqs: int, num_heads: int, head_dim: int, device, dtype):
    seqlen_q = torch.tensor([128] * num_seqs, dtype=torch.int32, device=device)
    seqlen_k = torch.tensor([128] * num_seqs, dtype=torch.int32, device=device)
    cu_q = torch.zeros(num_seqs + 1, dtype=torch.int32, device=device)
    cu_k = torch.zeros(num_seqs + 1, dtype=torch.int32, device=device)
    for i in range(num_seqs):
        cu_q[i + 1] = cu_q[i] + seqlen_q[i]
        cu_k[i + 1] = cu_k[i] + seqlen_k[i]
    T_q = int(cu_q[-1].item())
    q = torch.randn(T_q, num_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(T_q, num_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(T_q, num_heads, head_dim, dtype=dtype, device=device)
    return q, k, v, cu_q, cu_k, int(seqlen_q.max()), int(seqlen_k.max())


def test_attention_varlen_smoke(device, dtype):
    torch.manual_seed(0)
    q, k, v, cu_q, cu_k, max_q, max_k = _build_inputs(2, 16, 64, device, dtype)
    scale = 64 ** -0.5
    ref = ref_varlen(q, k, v, cu_q, cu_k, max_q, max_k, scale, True, block_table=None)
    out = tilelang_flash_attn_varlen(q, k, v, cu_q, cu_k, max_q, max_k, scale, True, block_table=None)
    assert_close(out, ref, atol=5e-2, rtol=5e-2, name="attn_varlen")


def test_attention_with_kvcache_smoke(device, dtype):
    torch.manual_seed(0)
    B, H, D = 4, 8, 64
    block_size = 256  # flash_attn requires block_size % 256 == 0
    num_blocks = 8
    q = torch.randn(B, 1, H, D, dtype=dtype, device=device)
    k_cache = torch.randn(num_blocks, block_size, H, D, dtype=dtype, device=device)
    v_cache = torch.randn(num_blocks, block_size, H, D, dtype=dtype, device=device)
    cache_len = 256
    block_table = torch.randint(0, num_blocks, (B, cache_len // block_size), dtype=torch.int32, device=device)
    cache_seqlens = torch.full((B,), cache_len, dtype=torch.int32, device=device)
    scale = D ** -0.5
    kwargs = dict(
        cache_seqlens=cache_seqlens,
        block_table=block_table,
        softmax_scale=scale,
        causal=True,
    )
    ref = ref_with_kvcache(q, k_cache.clone(), v_cache.clone(), **kwargs)
    out = tilelang_flash_attn_with_kvcache(q, k_cache.clone(), v_cache.clone(), **kwargs)
    assert_close(out, ref, atol=5e-2, rtol=5e-2, name="attn_with_kvcache")
