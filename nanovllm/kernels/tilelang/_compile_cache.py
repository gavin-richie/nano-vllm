"""Compile cache for TileLang kernels.

TileLang's `compile` step produces a CUDA module. We cache compiled artifacts
keyed by (op_name, dtype, *static_dims) so repeated calls with the same shape
don't re-trigger the JIT pipeline. This is essential for CUDA-graph capture
where the kernel must be precompiled.
"""
from typing import Any, Callable

_CACHE: dict[tuple, Any] = {}


def get_compiled(key: tuple, builder: Callable[[], Any]) -> Any:
    """Return a previously compiled kernel for `key`, building it on miss."""
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    artifact = builder()
    _CACHE[key] = artifact
    return artifact


def clear_cache() -> None:
    _CACHE.clear()
