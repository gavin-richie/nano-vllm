"""Microbenchmark: tilelang FlashAttention vs flash_attn reference.

For the first cut, the TileLang kernels delegate to the same flash_attn
package, so the comparison may show identical numbers. After Phase 3 lands
the real implementation, the numbers will diverge.
"""
import argparse
import statistics

import torch

from nanovllm.kernels._reference.attention import (
    flash_attn_varlen as ref_varlen,
    flash_attn_with_kvcache as ref_with_kvcache,
)
from nanovllm.kernels.tilelang.attention import (
    tilelang_flash_attn_varlen,
    tilelang_flash_attn_with_kvcache,
)


def bench(fn, args, warmup: int = 10, iters: int = 100) -> float:
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn(*args)
        e.record()
        e.synchronize()
        times.append(s.elapsed_time(e))
    return statistics.median(times)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["triton", "tilelang", "both"], default="both")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=100)
    args = p.parse_args()

    device = "cuda"
    torch.manual_seed(0)
    H, D = 16, 64
    seqlen = 1024
    q = torch.randn(seqlen, H, D, dtype=torch.float16, device=device)
    k = torch.randn(seqlen, H, D, dtype=torch.float16, device=device)
    v = torch.randn(seqlen, H, D, dtype=torch.float16, device=device)
    cu_q = torch.tensor([0, seqlen], dtype=torch.int32, device=device)
    cu_k = torch.tensor([0, seqlen], dtype=torch.int32, device=device)
    scale = D ** -0.5

    print("=" * 70)
    cases = [
        ("fa_varlen S=1024 H=16 D=64", (
            ref_varlen, tilelang_flash_attn_varlen,
            (q, k, v, cu_q, cu_k, seqlen, seqlen, scale, True, None),
        )),
    ]

    print(f"Op name                          | triton (us) | tilelang (us) | speedup")
    print("-" * 70)
    for name, (ref_fn, tl_fn, inputs) in cases:
        t_triton = float("nan")
        t_tl = float("nan")
        if args.backend in ("triton", "both"):
            t_triton = bench(ref_fn, inputs, args.warmup, args.iters)
        if args.backend in ("tilelang", "both"):
            t_tl = bench(tl_fn, inputs, args.warmup, args.iters)
        speedup = (t_triton / t_tl) if (t_tl == t_tl and t_triton == t_triton) else float("nan")
        print(f"{name:32} | {t_triton*1000:11.1f} | {t_tl*1000:13.1f} | {speedup:.2f}x")
    print("=" * 70)


if __name__ == "__main__":
    main()
