"""Microbenchmark: tilelang RMSNorm vs triton/torch.compile reference.

Run via:
    python -m tests.bench_rmsnorm
    python -m tests.bench_rmsnorm --backend tilelang     # only one backend
    python -m tests.bench_rmsnorm --backend triton      # only the reference
"""
import argparse
import statistics
import time

import torch

from nanovllm.kernels._reference.layernorm import rmsnorm, add_rmsnorm
from nanovllm.kernels.tilelang.layernorm import tilelang_rmsnorm, tilelang_add_rmsnorm


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


def run_one(name: str, fn, args, backend: str) -> float:
    try:
        t = bench(fn, args)
    except Exception as exc:
        print(f"  [{backend}] {name}: FAILED -> {type(exc).__name__}: {exc}")
        return float("nan")
    print(f"  [{backend}] {name}: {t * 1000:.1f} us")
    return t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["triton", "tilelang", "both"], default="both")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=100)
    args = p.parse_args()

    device = "cuda"
    torch.manual_seed(0)
    cases = [
        ("rmsnorm M=32 head_dim=128", rmsnorm, tilelang_rmsnorm,
         (torch.randn(32, 128, dtype=torch.float16, device=device),
          torch.randn(128, dtype=torch.float32, device=device), 1e-6)),
        ("rmsnorm M=128 head_dim=128", rmsnorm, tilelang_rmsnorm,
         (torch.randn(128, 128, dtype=torch.float16, device=device),
          torch.randn(128, dtype=torch.float32, device=device), 1e-6)),
        ("rmsnorm M=4096 hidden=4096", rmsnorm, tilelang_rmsnorm,
         (torch.randn(4096, 4096, dtype=torch.float16, device=device),
          torch.randn(4096, dtype=torch.float32, device=device), 1e-6)),
        ("add_rmsnorm M=128 hidden=4096", add_rmsnorm, tilelang_add_rmsnorm,
         (torch.randn(128, 4096, dtype=torch.float16, device=device),
          torch.randn(128, 4096, dtype=torch.float16, device=device),
          torch.randn(4096, dtype=torch.float32, device=device), 1e-6)),
    ]

    print("=" * 70)
    print(f"Op name                          | triton (us) | tilelang (us) | speedup")
    print("-" * 70)
    for name, ref_fn, tl_fn, inputs in cases:
        t_triton = float("nan")
        t_tl = float("nan")
        if args.backend in ("triton", "both"):
            t_triton = run_one(name, ref_fn, inputs, "triton")
        if args.backend in ("tilelang", "both"):
            t_tl = run_one(name, tl_fn, inputs, "tilelang")
        speedup = (t_triton / t_tl) if (t_tl == t_tl and t_triton == t_triton) else float("nan")
        print(f"{name:32} | {t_triton*1000:11.1f} | {t_tl*1000:13.1f} | {speedup:.2f}x")
    print("=" * 70)


if __name__ == "__main__":
    main()
