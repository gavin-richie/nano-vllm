# FlashAttention (paged KV + varlen)

> Note: the file name `attn-tro.md` follows the project convention used
> in the request (`docs/titile/attention/attn-tro.md`). The reference
> to "tro" likely stands for "Triton-output" — i.e. the canonical
> comparison reference for the TileLang implementation.

This is the highest-risk op in the backend. The TileLang port is
in progress — see Phase 3 in `1.overview.md`.

## What the op does

Two flavors:

- **`tilelang_flash_attn_varlen`**: prefill attention with paged KV
  cache and varlen sequences. Equivalent to
  `flash_attn.flash_attn_varlen_func`.
- **`tilelang_flash_attn_with_kvcache`**: decode attention with
  in-place KV update. Equivalent to
  `flash_attn.flash_attn_with_kvcache`.

## Current state

- The first cut delegates to the upstream `flash_attn` package —
  the same package the triton backend uses. The TileLang kernel
  itself is not yet implemented.
- The half-baked skeleton in the first commit shows the structure
  (`T.Kernel(M_blocks, H, threads=BLOCK_M)` with online softmax
  over KV blocks). See `2.implementation-guide.md` for the
  rewrite checklist.

## Wrapper signature

```python
def tilelang_flash_attn_varlen(
    q: torch.Tensor,                   # (T_q, H, D)
    k_cache: torch.Tensor,              # (num_blocks, block_size, H_kv, D)
    v_cache: torch.Tensor,              # same
    cu_seqlens_q: torch.Tensor,         # (B+1,) int32
    cu_seqlens_k: torch.Tensor,         # (B+1,) int32
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    causal: bool,
    block_table: Optional[torch.Tensor],  # (B, max_blocks) int32, or None
) -> torch.Tensor: ...
```

## See also

- `notes.md` — Phase 3 work plan.
- `../../2.implementation-guide.md` — the skeleton template.
- The upstream TileLang `examples/flash_attention/` directory,
  especially `example_gqa_fwd_bshd_wgmma_pipelined.py` for Hopper.
