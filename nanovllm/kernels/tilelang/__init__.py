"""TileLang kernels.

The functions here are NOT imported eagerly at module load time — they are
imported lazily by the factory in `nanovllm.kernels` so that simply having
`tilelang` listed as an optional dep doesn't break the triton backend if the
package isn't installed.
"""
from .layernorm import tilelang_rmsnorm, tilelang_add_rmsnorm
from .activation import tilelang_silu_and_mul
from .rotary_embedding import tilelang_apply_rotary_emb
from .store_kvcache import tilelang_store_kvcache
from .attention import tilelang_flash_attn_varlen, tilelang_flash_attn_with_kvcache

__all__ = [
    "tilelang_rmsnorm",
    "tilelang_add_rmsnorm",
    "tilelang_silu_and_mul",
    "tilelang_apply_rotary_emb",
    "tilelang_store_kvcache",
    "tilelang_flash_attn_varlen",
    "tilelang_flash_attn_with_kvcache",
]
