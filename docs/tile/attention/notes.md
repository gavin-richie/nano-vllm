# FlashAttention — design notes (Phase 3 work)

## Why this is the hardest op

- **Paged KV**: each query block needs to walk the `block_table` to
  find the physical block IDs for its KV sequence. The gather is
  data-dependent on the block table row, which TileLang supports
  via indirect indexing but is fiddly to express.
- **Varlen**: the actual sequence boundaries are in `cu_seqlens_q` /
  `cu_seqlens_k`. The kernel needs to read these on the host side
  to compute the "which sequence does this block belong to" map.
- **In-place KV update**: the decode path writes the new K/V row
  into the cache, then reads it back in the same kernel. Memory
  ordering matters — write-before-read intra-block is fine, but
  cross-block would need a `__threadfence`.
- **Causal masking**: per-block AND per-sequence.
- **Grouped-query attention**: Q has H heads, K/V have H_kv <
  H. Within a block, the K/V tile is shared by `H/H_kv` Q tiles.

## Suggested implementation order

1. **Non-paged, varlen prefill**. Single contiguous K, V. No
   `block_table`. This is the simplest path and lets you validate
   the basic FA2 structure (online softmax, accumulated P/V product).
2. **Paged prefill**. Add `block_table` lookup. Use the upstream
   TileLang `examples/flash_attention/example_gqa_fwd_paged.py` as a
   starting reference.
3. **Decode with kvcache**. Take the paged prefill and add the
   in-place KV write at the start of the kernel. Run the existing
   `store_kvcache` first and decode over the cached KV (decoupled)
   to validate the correctness; then fuse the write.

## Reference implementations

| Path | What it covers |
| --- | --- |
| `tile-ai/tilelang/examples/flash_attention/example_gqa_fwd_bshd_wgmma_pipelined.py` | Hopper WGMMA pipelined FA2 forward (GQA) |
| `tile-ai/tilelang/examples/flash_attention/example_gqa_bwd.py` | Backward for GQA |
| `tile-ai/tilelang/examples/flash_attention/example_gqa_fwd_paged.py` | Paged KV forward |

These are the templates to base your TileLang kernel on. The
paged version is the closest to what nanovllm needs.

## CUDA-graph fallback

If the paged/kvcache path proves too hard to express in TileLang,
the cleanest fallback is:

- Keep `tilelang_store_kvcache` for the KV write.
- Keep `tilelang_flash_attn_varlen` for the attention (delegated
  to the reference flash_attn package).
- Note the limitation in `1.overview.md`.

This is what the current first-cut does. It costs nothing for
correctness and gives you a clean migration path to a real TileLang
kernel later.
