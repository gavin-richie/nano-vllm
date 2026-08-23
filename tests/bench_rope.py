"""Microbenchmark: tilelang RoPE vs torch.compile reference."""
import argparse
import statistics

import torch

from nanovllm.kernels._reference.rotary_embedding import apply_rotary_emb as ref_rope
from nanovllm.kernels.tilelang.rotary_embedding import tilelang_apply_rotary_emb


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
    # cos/sin must have a singleton head axis (M, 1, D/2) to broadcast with
    # the chunked x1/x2 of shape (M, H, D/2) inside apply_rotary_emb. Without
    # it, torch 2.8 Dynamo rejects the broadcast in fake-tensor tracing.
    # This shape matches what nanovllm/layers/rotary_embedding.py passes.
    cases = [
        ("rope seqlen=128 head_dim=128 H=16", (torch.randn(128, 16, 128, dtype=torch.float16, device=device),
                                              torch.randn(128, 1, 64, dtype=torch.float16, device=device),
                                              torch.randn(128, 1, 64, dtype=torch.float16, device=device))),
        ("rope seqlen=4096 head_dim=128 H=16", (torch.randn(4096, 16, 128, dtype=torch.float16, device=device),
                                                 torch.randn(4096, 1, 64, dtype=torch.float16, device=device),
                                                 torch.randn(4096, 1, 64, dtype=torch.float16, device=device))),
    ]

    print("=" * 70)
    print(f"Op name                                 | triton (us) | tilelang (us) | speedup")
    print("-" * 70)
    for name, inputs in cases:
        t_triton = float("nan")
        t_tl = float("nan")
        if args.backend in ("triton", "both"):
            t_triton = bench(ref_rope, inputs, args.warmup, args.iters)
        if args.backend in ("tilelang", "both"):
            t_tl = bench(tilelang_apply_rotary_emb, inputs, args.warmup, args.iters)
        speedup = (t_triton / t_tl) if (t_tl == t_tl and t_triton == t_triton) else float("nan")
        print(f"{name:39} | {t_triton*1000:11.1f} | {t_tl*1000:13.1f} | {speedup:.2f}x")
    print("=" * 70)


if __name__ == "__main__":
    main()
