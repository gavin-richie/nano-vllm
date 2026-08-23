# SiluAndMul (SwiGLU)

`silu_and_mul(x) = silu(x[..., :K]) * x[..., K:2K]`

Takes a `(M, 2K)` input and returns `(M, K)`. Used after the
`gate_up_proj` linear in the MLP. K is the intermediate size
(typically 14336 for Qwen3).

## Kernel signature

| Tensor | Shape | Dtype |
| --- | --- | --- |
| `x` (input) | `(M, 2K)` | `float16` |
| `y` (output) | `(M, K)` | same as `x` |

## Wrapper signature

```python
def silu_and_mul(x: torch.Tensor) -> torch.Tensor: ...
```

## How to run

```bash
python -m pytest tests/test_silu_and_mul.py -v
python -m tests.bench_silu_and_mul --backend both
```

## Pitfalls

- **Last-dim split**: input shape is `(M, 2K)`, output is `(M, K)`.
  The kernel splits the last dim logically — no `T.split` needed.
- **silu in fp32**: `silu(x) = x / (1 + exp(-x))`. The `T.exp`
  is fp32-stable only on recent arches; match the reference by
  computing in fp32 then casting back.
- **K alignment**: K must be a power of 2 for `BLOCK_K` to tile
  cleanly. Qwen3 uses 14336 = 2^11 × 7, so we use `BLOCK_K = 1024`
  and loop over 14 chunks.

## See also

- `notes.md`.
- `../../2.implementation-guide.md`.
