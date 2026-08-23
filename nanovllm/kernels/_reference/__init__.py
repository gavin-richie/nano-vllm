"""Reference implementations: torch.compile / triton / flash_attn originals.

These exist so unit tests can compare TileLang output against the original
semantics bit-for-bit. The runtime path for the triton backend is the
torch.compile / triton / flash_attn machinery in nanovllm/layers/.
"""
from .layernorm import rmsnorm, add_rmsnorm
from .activation import silu_and_mul
from .rotary_embedding import apply_rotary_emb, _apply_rotary_emb
from .store_kvcache import store_kvcache
from .attention import flash_attn_varlen, flash_attn_with_kvcache

__all__ = [
    "rmsnorm",
    "add_rmsnorm",
    "silu_and_mul",
    "apply_rotary_emb",
    "_apply_rotary_emb",
    "store_kvcache",
    "flash_attn_varlen",
    "flash_attn_with_kvcache",
]
