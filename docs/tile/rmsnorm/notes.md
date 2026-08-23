# RMSNorm — design notes

## Why fp32 internal

The reference `torch.compile` RMSNorm casts `x` to fp32, computes the
variance, normalizes, then casts back. The TileLang version matches
this exactly: `T.copy(x, fp32_buffer)` then `acc += x_fp32 * x_fp32`.

If we tried to do the variance in fp16, the squared values would
underflow for inputs with magnitude < 1e-3, producing a noisy
estimate. The cost of the fp32 detour is one extra register pass
and a single `T.copy` per tile.

## `BLOCK_N` choice

We benchmarked 128, 256, 512, 1024 and 2048 on a 4096×4096 tensor
(reference GPU: RTX 4090). For `BLOCK_N < 1024`, the loop trips
multiple times and the `T.copy` overhead dominates. For `BLOCK_N =
2048`, we exceed shared memory budget on Ampere. 1024 is the sweet
spot for `hidden_size=4096`.

For `head_dim=128` we use `BLOCK_N=128` (one trip).

## Reducing compile count

The cache key is `(rmsnorm, dtype, M, N, BLOCK_N)`. The decoder
invariantly uses `M = batch_size * seqlen` and `N = 4096`, so we
compile once per `(M, N)` per dtype. For the per-head Q/K norm,
`N = 128` and `M` varies — we recompile per shape, but the cache
amortizes it across the 28 attention layers.

## CUDA-graph compatibility

`TileLang.compiled(grid, inputs)` is captured under CUDA graphs as
long as the input shape does not change between capture and replay.
We bootstrap the cache before `ModelRunner.capture_cudagraph` runs
in `ModelRunner.__init__` body, so by the time the capture loop
traverses the `graph_bs` buckets, the compiled kernel for the
largest bucket is already cached.

```python

def _build_rmsnorm_kernel(M: int, N: int, BLOCK_N: int, dtype: str, eps: float):
    """Build the single RMSNorm kernel for given shape/dtype.

    Each block holds `BLOCK_N` threads, iterates over the row in `BLOCK_N`
    chunks, and reduces the sum of squares via a tree reduction.
    """
    import tilelang
    from tilelang import language as T

    @T.prim_func
    def rmsnorm_kernel(
        x: T.Tensor[(M, N), dtype],
        weight: T.Tensor[(N,), "float32"],
        y: T.Tensor[(M, N), dtype],
    ):
        with T.Kernel(M, threads=BLOCK_N) as bx:
            x_s = T.alloc_shared((BLOCK_N,), "float32")
            acc = T.alloc_fragment((BLOCK_N,), "float32")
            T.fill(acc, 0.0)

            for tile in T.serial(T.ceildiv(N, BLOCK_N)):
                offsets = tile * BLOCK_N + T.arange(0, BLOCK_N)
                mask = offsets < N
                # Load tile (auto-cast from dtype to fp32 for accumulation).
                xl = T.alloc_shared((BLOCK_N,), "float32")
                T.copy(x[bx, offsets], xl, mask=mask)
                x_s = xl
                acc += x_s * x_s

            sq = T.reduce_sum(acc, dim=0)
            rstd = T.rsqrt(sq / N + eps)

            for tile in T.serial(T.ceildiv(N, BLOCK_N)):
                offsets = tile * BLOCK_N + T.arange(0, BLOCK_N)
                mask = offsets < N
                xl = T.alloc_shared((BLOCK_N,), "float32")
                T.copy(x[bx, offsets], xl, mask=mask)
                out = (xl * rstd).astype(dtype) * weight
                y_wire = T.alloc_shared((BLOCK_N,), dtype)
                T.copy(out, y_wire)
                T.copy(y_wire, y[bx, offsets], mask=mask)

    return tilelang.compile(rmsnorm_kernel, out_idx=[2], target="cuda")

```
