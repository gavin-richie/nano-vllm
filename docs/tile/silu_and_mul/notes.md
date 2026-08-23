# SiluAndMul — design notes

## Why a separate kernel

The reference uses `torch.compile` to fuse `chunk + F.silu + mul`.
TileLang expresses this as a single explicit kernel: one block per
row, two `T.copy` calls (one for each half), then a fused
`silu * y` write. The TileLang version has tighter control over
register usage and avoids the `torch.compile` overhead.

## Tile size

`BLOCK_K = 1024` is the upper bound for shared memory on Ampere for
a single register tile. Qwen3's intermediate size 14336 requires
14 chunks; the loop trip count is small (14 cycles) and the
`silu` latency is hidden behind the chunked memory traffic.

## Numerical equivalence

The reference computes `F.silu(x) * y` where `F.silu(x) = x *
sigmoid(x)`. We use the equivalent `x / (1 + exp(-x))` form. Both
formulas are bit-equivalent within fp16 ulp.

## Edge cases

- `M = 1`: the decode path uses batch size 1 sometimes. The kernel
  handles this correctly (1 block, full row).
- `K = 1` (degenerate): the kernel still works; the `threads=K`
  launches 1 thread.
