# Rotary Position Embedding (RoPE)

`apply_rotary_emb(x, cos, sin)` rotates each pair of consecutive
features in `x` by the angles in `cos` / `sin`. For Qwen3, `D` is
either 64 or 128 (head_dim).

## Kernel signature

| Tensor | Shape | Dtype |
| --- | --- | --- |
| `x` (input) | `(M, H, D)` | `float16` |
| `cos` (input) | `(M, D/2)` | `float16` |
| `sin` (input) | `(M, D/2)` | `float16` |
| `y` (output) | `(M, H, D)` | same as `x` |

## Wrapper signature

```python
def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor: ...
```

## How to run

```bash
python -m pytest tests/test_rope.py -v
python -m tests.bench_rope --backend both
```

## Pitfalls

- **Cos/sin shape**: `cos` and `sin` are `(M, D/2)`, not `(M, D)`.
  The Python `RotaryEmbedding.forward` slices `cos_sin_cache[positions]`
  and `chunk(2, -1)` to produce them. The kernel takes them as-is.
- **Per-row, per-head parallelism**: each block handles one row at
  all heads (head is folded into the block index). Thread count is
  `D/2` per block, not `H * D/2`.
- **fp32 math**: the four multiplications and two subtractions are
  in fp32; cast back to dtype before write.
- **`D` power of 2**: required for `T.arange(0, D/2)` to be a
  valid compile-time constexpr.

## See also

- `notes.md`.
- `../../2.implementation-guide.md`.
