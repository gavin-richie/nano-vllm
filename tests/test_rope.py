"""Numerical correctness for tilelang RoPE vs the reference."""
import pytest
import torch

from nanovllm.kernels._reference.rotary_embedding import _apply_rotary_emb as ref_rope
from nanovllm.kernels.tilelang.rotary_embedding import tilelang_apply_rotary_emb
from .conftest import assert_close


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("seqlen", [1, 128, 4096])
def test_rope(seqlen, head_dim, device, dtype):
    torch.manual_seed(0)
    H = 16
    x = torch.randn(seqlen, H, head_dim, dtype=dtype, device=device)
    # Match the broadcast shape used by the original RotaryEmbedding.forward:
    # cos_sin_cache[positions] is (M, 1, D), chunk(2, -1) -> (M, 1, D/2).
    cos = torch.randn(seqlen, 1, head_dim // 2, dtype=dtype, device=device)
    sin = torch.randn(seqlen, 1, head_dim // 2, dtype=dtype, device=device)
    ref = ref_rope(x, cos, sin)
    out = tilelang_apply_rotary_emb(x, cos, sin)
    assert_close(out, ref, atol=1e-2, rtol=1e-2, name="rope")
