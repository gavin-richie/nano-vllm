"""Triton backend.

The triton backend re-exports the existing reference implementations. In
practice, the layer modules' torch.compile wrappers and the original
@triton.jit `store_kvcache_kernel` continue to run on this code path; we
re-export them unified so the factory can return a single callable per op.
"""
from nanovllm.kernels._reference.layernorm import rmsnorm, add_rmsnorm
from nanovllm.kernels._reference.activation import silu_and_mul
from nanovllm.kernels._reference.rotary_embedding import apply_rotary_emb
from nanovllm.kernels._reference.store_kvcache import store_kvcache
from nanovllm.kernels._reference.attention import (
    flash_attn_varlen as triton_flash_attn_varlen,
    flash_attn_with_kvcache as triton_flash_attn_with_kvcache,
)

__all__ = [
    "rmsnorm",
    "add_rmsnorm",
    "silu_and_mul",
    "apply_rotary_emb",
    "store_kvcache",
    "triton_flash_attn_varlen",
    "triton_flash_attn_with_kvcache",
]
