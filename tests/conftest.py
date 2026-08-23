"""Shared test fixtures and helpers.

Each test file is GPU-only. The `cuda` fixture skips the test if CUDA is
unavailable. We seed cuda + cpu generators before each test for deterministic
tolerance checks.
"""
from __future__ import annotations

import os
import pytest
import torch


def _cuda_available() -> bool:
    return torch.cuda.is_available()


@pytest.fixture(scope="session")
def device() -> str:
    if not _cuda_available():
        pytest.skip("CUDA not available")
    return "cuda"


@pytest.fixture(autouse=True)
def seed_rng():
    seed = int(os.environ.get("NANOVLLM_TEST_SEED", "0"))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    yield


@pytest.fixture(params=["float16", "bfloat16"])
def dtype(request):
    return getattr(torch, request.param)


def assert_close(a: torch.Tensor, b: torch.Tensor, atol: float, rtol: float, name: str = "tensor"):
    if not torch.allclose(a, b, atol=atol, rtol=rtol):
        diff = (a - b).abs()
        max_abs = diff.max().item()
        max_rel = (diff / (b.abs() + 1e-9)).max().item()
        pytest.fail(f"{name}: max abs={max_abs:.4e}, max rel={max_rel:.4e} (atol={atol}, rtol={rtol})")
