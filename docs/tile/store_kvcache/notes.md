# store_kvcache — design notes

## Why a 1-D flat cache

The original Triton kernel takes `k_cache` as a 2-D buffer of
shape `(N, D)` because the kernel treats it as a flat row-store.
We do the same: pass `k_cache.reshape(-1)` so the AST matches a
1-D typed buffer of arbitrary length.

The first cut uses `(1 << 30,)` as the static shape — that's
enough slots for any reasonable model. The actual cache is smaller
but TileLang doesn't care about the static size so long as runtime
indexing stays in bounds.

## Why per-token programs

Store_kvcache is memory-bound. Each token writes exactly `D` bytes
to K and `D` bytes to V. Per-token programs give clean parallelism
and don't require cross-token reduction. One block per token with
`threads=D` threads lets each thread write one element.

## Slot == -1 masking

The Triton kernel uses `if slot == -1: return` which compiles to
a conditional skip. We don't have that in TileLang, so we use a
masked write:

```python
mask = (slot >= 0) & (T.arange(0, BLOCK_D) < D)
T.copy(k_wire, k_cache[offset], mask=mask)
```

When `slot == -1`, the offset is `(-1) * D = -D`, which is out of
bounds. The mask prevents the write. (If the runtime address is
actually out of bounds even with the mask, this could fault on
some arches; in practice the `slot >= 0` mask is fine because
TileLang lowers to predicated stores.)
