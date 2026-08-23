"""Numerical correctness for tilelang SiluAndMul vs the reference."""
import pytest
import torch

from nanovllm.kernels._reference.activation import silu_and_mul
from nanovllm.kernels.tilelang.activation import tilelang_silu_and_mul
from .conftest import assert_close


@pytest.mark.parametrize("M", [1, 32, 1024])
@pytest.mark.parametrize("K", [4096, 14336])
def test_silu_and_mul(M, K, device, dtype):
    torch.manual_seed(0)
    x = torch.randn(M, 2 * K, dtype=dtype, device=device)
    ref = silu_and_mul(x)
    out = tilelang_silu_and_mul(x)
    assert_close(out, ref, atol=1e-2, rtol=1e-2, name="silu_and_mul")
