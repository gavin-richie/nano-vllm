"""Microbenchmark: tilelang SiluAndMul vs torch.compile reference."""
import argparse
import statistics

import torch

from nanovllm.kernels._reference.activation import silu_and_mul
from nanovllm.kernels.tilelang.activation import tilelang_silu_and_mul


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
    cases = [
        ("silu_and_mul M=128 K=4096", (torch.randn(128, 2 * 4096, dtype=torch.float16, device=device),)),
        ("silu_and_mul M=128 K=14336", (torch.randn(128, 2 * 14336, dtype=torch.float16, device=device),)),
        ("silu_and_mul M=1 K=4096", (torch.randn(1, 2 * 4096, dtype=torch.float16, device=device),)),
    ]

    print("=" * 70)
    print(f"Op name                          | triton (us) | tilelang (us) | speedup")
    print("-" * 70)
    for name, inputs in cases:
        t_triton = float("nan")
        t_tl = float("nan")
        if args.backend in ("triton", "both"):
            t_triton = bench(silu_and_mul, inputs, args.warmup, args.iters)
        if args.backend in ("tilelang", "both"):
            t_tl = bench(tilelang_silu_and_mul, inputs, args.warmup, args.iters)
        speedup = (t_triton / t_tl) if (t_tl == t_tl and t_triton == t_triton) else float("nan")
        print(f"{name:32} | {t_triton*1000:11.1f} | {t_tl*1000:13.1f} | {speedup:.2f}x")
    print("=" * 70)


if __name__ == "__main__":
    main()
