# RMSNorm (single + add+residual)

`rmsnorm` and `add_rmsnorm` are two related operators:

- `rmsnorm(x, weight, eps)` — pure RMSNorm over the last dim.
- `add_rmsnorm(x, residual, weight, eps)` — fused residual-add + RMSNorm;
  returns `(y, new_residual)`.

`add_rmsnorm` runs twice per decoder layer (input_layernorm and
post_attention_layernorm), so it's the higher-value target.

## Kernel signature

| Tensor | Shape | Dtype |
| --- | --- | --- |
| `x` (input) | `(M, N)` | `float16` / `bfloat16` |
| `weight` (input) | `(N,)` | `float32` |
| `residual` (input, add_rmsnorm only) | `(M, N)` | same as `x` |
| `y` (output) | `(M, N)` | same as `x` |
| `new_residual` (output, add_rmsnorm only) | `(M, N)` | same as `x` |

`M` is the number of tokens; `N` is `hidden_size` (4096 for the
decoder) or `head_dim` (64/128 for Q/K per-head norm).

## Wrapper signature

```python
def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor: ...

def add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]: ...
```

## How to run

```bash
# Numerical correctness
python -m pytest tests/test_rmsnorm.py -v

# Microbenchmark
python -m tests.bench_rmsnorm --backend both
```

## Pitfalls

- **fp32 accumulation**: the variance reduction must run in fp32.
  TileLang's `T.alloc_fragment(..., "float32")` accumulator and
  `T.reduce_sum(..., dim=0)` give you this for free.
- **Residual output dtype**: the fused add+RMS variant must round
  the residual back to the input dtype, like the reference does.
  Writing fp32 directly to a fp16 output buffer will diverge.
- **`BLOCK_N` choice**: 1024 is the upper bound for shared memory
  on Ampere. For `hidden_size=4096` we tile 4 chunks of 1024.
- **Per-head vs decoder**: the same kernel handles both. The
  second is just a larger `N`.

## See also

- `notes.md` — design rationale.
- `../../2.implementation-guide.md` — the skeleton template.
