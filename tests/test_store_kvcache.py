"""Numerical correctness for tilelang store_kvcache vs the reference.

The reference is the original @triton.jit kernel in `nanovllm/kernels/_reference/store_kvcache.py`.
For deterministic equality both should produce identical writes.
"""
import pytest
import torch
# from torch.testing import assert_close

from nanovllm.kernels._reference.store_kvcache import store_kvcache as ref_store
from nanovllm.kernels.tilelang.store_kvcache import tilelang_store_kvcache
from .conftest import assert_close


@pytest.mark.parametrize("D", [128, 256])
@pytest.mark.parametrize("N", [1, 32, 1024])
@pytest.mark.parametrize("num_blocks", [4, 16])
def test_store_kvcache(N, D, num_blocks, device, dtype):
    torch.manual_seed(0)
    block_size = 16
    num_heads = D // 64
    head_dim = 64
    key = torch.randn(N, num_heads, head_dim, dtype=dtype, device=device)
    value = torch.randn(N, num_heads, head_dim, dtype=dtype, device=device)
    k_cache = torch.zeros(num_blocks, block_size, num_heads, head_dim, dtype=dtype, device=device)
    v_cache = torch.zeros(num_blocks, block_size, num_heads, head_dim, dtype=dtype, device=device)

    # Build a slot mapping that uses some real slots and some -1 sentinel slots.
    # The real slots must be unique (no two tokens write to the same cache row)
    # so the result is deterministic. Cap real_count at cache_capacity so the
    # any-N test stays valid for small caches, but always force at least one
    # real slot for N >= 1 (otherwise an N=1 case is degenerate: real_count=0,
    # every slot is -1, no writes happen, and assert_close passes vacuously
    # because zeros == zeros).
    slots = torch.full((N,), -1, dtype=torch.int32, device=device)
    cache_capacity = num_blocks * block_size
    real_count = max(1, min(N // 2, cache_capacity))
    real_count = min(real_count, N)
    slots[:real_count] = torch.randperm(cache_capacity, dtype=torch.int32, device=device)[:real_count]
    slot_mapping = slots

    # Reference (writes into a clone so the original k_cache stays zero).
    k_ref = k_cache.clone().contiguous()
    v_ref = v_cache.clone().contiguous()
    ref_store(key, value, k_ref, v_ref, slot_mapping)
    # TileLang (on separate copies).

    k_tl = k_cache.clone().contiguous()
    v_tl = v_cache.clone().contiguous()
    tilelang_store_kvcache(key, value, k_tl, v_tl, slot_mapping)
    # Sanity check: ref_store should have written at least one element.
    # If k_ref is still all zeros despite real_count > 0 slots, the kernel
    # silently no-op'd and the equality check below would pass for the wrong
    # reason (zeros == zeros). This guards against silent regressions.
    assert torch.count_nonzero(k_ref).item() > 0, (
        f"ref_store did not write anything (real_count={real_count}, "
        f"cache_capacity={cache_capacity}, slots={slot_mapping.tolist()[:8]}…); "
        f"the test setup is wrong or the kernel regressed"
    )
    assert torch.count_nonzero(k_tl).item() > 0, (
        "tilelang_store_kvcache did not write anything; check the kernel"
    )
    assert_close(k_tl, k_ref, atol=0, rtol=0, name="k_cache")
    assert_close(v_tl, v_ref, atol=0, rtol=0, name="v_cache")
