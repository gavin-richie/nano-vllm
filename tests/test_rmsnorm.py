"""Numerical correctness for tilelang RMSNorm vs the reference."""
import pytest
import torch

from nanovllm.kernels._reference.layernorm import rmsnorm, add_rmsnorm
from nanovllm.kernels.tilelang.layernorm import tilelang_rmsnorm, tilelang_add_rmsnorm


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("M", [1, 32, 4096])
def test_rmsnorm(M, head_dim, device, dtype):
    torch.manual_seed(0)
    x = torch.randn(M, head_dim, dtype=dtype, device=device)
    weight = torch.randn(head_dim, dtype=torch.float32, device=device)
    ref = rmsnorm(x, weight, eps=1e-6)
    out = tilelang_rmsnorm(x, weight, eps=1e-6)
    assert_close(out, ref, atol=1e-2, rtol=1e-2, name="rmsnorm")


@pytest.mark.parametrize("hidden_size", [4096])
@pytest.mark.parametrize("M", [1, 32, 4096])
def test_rmsnorm_hidden(M, hidden_size, device, dtype):
    torch.manual_seed(0)
    x = torch.randn(M, hidden_size, dtype=dtype, device=device)
    weight = torch.randn(hidden_size, dtype=torch.float32, device=device)
    ref = rmsnorm(x, weight, eps=1e-6)
    out = tilelang_rmsnorm(x, weight, eps=1e-6)
    assert_close(out, ref, atol=5e-2, rtol=5e-2, name="rmsnorm(hidden)")


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("M", [1, 32, 4096])
def test_add_rmsnorm(M, head_dim, device, dtype):
    torch.manual_seed(0)
    x = torch.randn(M, head_dim, dtype=dtype, device=device)
    residual = torch.randn(M, head_dim, dtype=dtype, device=device)
    weight = torch.randn(head_dim, dtype=torch.float32, device=device)
    y_ref, r_ref = add_rmsnorm(x, residual, weight, eps=1e-6)
    y, r = tilelang_add_rmsnorm(x, residual, weight, eps=1e-6)
    assert_close(y, y_ref, atol=1e-2, rtol=1e-2, name="add_rmsnorm(y)")
    assert_close(r, r_ref, atol=0, rtol=0, name="add_rmsnorm(residual)")  # exact


def assert_close(a, b, atol, rtol, name):
    from .conftest import assert_close as _ac
    _ac(a, b, atol, rtol, name)
