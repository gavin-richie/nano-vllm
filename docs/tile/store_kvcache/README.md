# store_kvcache (paged KV cache write)

Mirrors the existing `@triton.jit` kernel in
`nanovllm/kernels/_reference/store_kvcache.py`. One program per token,
copies `key` and `value` into the paged `k_cache` / `v_cache` at the
slot indices in `slot_mapping`. Slots of `-1` are skipped.

## Kernel signature

| Tensor | Shape | Dtype |
| --- | --- | --- |
| `key` (input) | `(N, num_heads, head_dim)` | `float16` |
| `value` (input) | `(N, num_heads, head_dim)` | `float16` |
| `k_cache` (input/output) | `(num_blocks, block_size, D)` | `float16` |
| `v_cache` (input/output) | `(num_blocks, block_size, D)` | `float16` |
| `slot_mapping` (input) | `(N,)` | `int32` |

`D = num_heads * head_dim`. The kernel treats the cache as a flat
`(num_blocks * block_size * D,)` buffer.

## Wrapper signature

```python
def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None: ...
```

## How to run

```bash
python -m pytest tests/test_store_kvcache.py -v
python -m tests.bench_store_kvcache --backend both
```

## Pitfalls

- **Slot == -1 skip**: the original Triton kernel uses
  `if slot == -1: return`. TileLang doesn't have a clean runtime
  conditional branch; we use a masked write where the mask is
  `slot >= 0`. Both behave identically.
- **`k_cache` stride**: the cache must be contiguous in its last
  dim. See the assertion in the wrapper.
- **`-1` padding slots**: produced by `ModelRunner.run_model` for
  CUDA-graph padding. They must not write anywhere or they'll
  corrupt adjacent slots.

## See also

- `notes.md`.
- `../../2.implementation-guide.md`.
