# RoPE — design notes

## Block layout

We launch `M * H` blocks, each doing one row at one head. This
gives natural parallelism over both the batch and the head axes,
which is perfect for the low arithmetic intensity of RoPE.

The thread count per block is `D/2`. For Qwen3 (`D = 128`), that's
64 threads, which is the minimum for a warp. For `D = 64`, that's
32 threads — half a warp, which is wasteful but still correct.

## Cos/sin broadcast

The kernel sees `cos` and `sin` already-gathered for the positions
of the row. The Python wrapper does the gather:
```python
cos_sin = self.cos_sin_cache[positions]   # (M, 1, D)
cos, sin = cos_sin.chunk(2, dim=-1)        # (M, 1, D/2)
```
and broadcasts via the `(M, H, D/2)` indexing inside the kernel.

## Variadic positions

The reference accepts arbitrary position tensors (not necessarily
contiguous). Our wrapper does `cos = cos.contiguous()` — the kernel
requires it. If you want to skip the copy, you'd need to support
indirect indexing, which TileLang doesn't make pretty. Copy cost
is negligible vs. the kernel work.
