"""Microbenchmark: tilelang store_kvcache vs triton kernel."""
import argparse
import statistics

import torch

from nanovllm.kernels._reference.store_kvcache import store_kvcache as ref_store
from nanovllm.kernels.tilelang.store_kvcache import tilelang_store_kvcache


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
    H, D_h = 8, 64
    num_blocks, block_size = 256, 16
    cases = [
        ("store_kvcache N=1 batch", (torch.randn(1, H, D_h, dtype=torch.float16, device=device),
                                    torch.randn(1, H, D_h, dtype=torch.float16, device=device),
                                    torch.zeros(num_blocks, block_size, H, D_h, dtype=torch.float16, device=device),
                                    torch.zeros(num_blocks, block_size, H, D_h, dtype=torch.float16, device=device),
                                    torch.randint(0, num_blocks * block_size, (1,), dtype=torch.int32, device=device))),
        ("store_kvcache N=64 batch", (torch.randn(64, H, D_h, dtype=torch.float16, device=device),
                                        torch.randn(64, H, D_h, dtype=torch.float16, device=device),
                                        torch.zeros(num_blocks, block_size, H, D_h, dtype=torch.float16, device=device),
                                        torch.zeros(num_blocks, block_size, H, D_h, dtype=torch.float16, device=device),
                                        torch.randint(0, num_blocks * block_size, (64,), dtype=torch.int32, device=device))),
        ("store_kvcache N=128 batch", (torch.randn(128, H, D_h, dtype=torch.float16, device=device),
                                       torch.randn(128, H, D_h, dtype=torch.float16, device=device),
                                       torch.zeros(num_blocks, block_size, H, D_h, dtype=torch.float16, device=device),
                                       torch.zeros(num_blocks, block_size, H, D_h, dtype=torch.float16, device=device),
                                       torch.randint(0, num_blocks * block_size, (128,), dtype=torch.int32, device=device))),
        ("store_kvcache N=1024 batch", (torch.randn(1024, H, D_h, dtype=torch.float16, device=device),
                                         torch.randn(1024, H, D_h, dtype=torch.float16, device=device),
                                         torch.zeros(num_blocks, block_size, H, D_h, dtype=torch.float16, device=device),
                                         torch.zeros(num_blocks, block_size, H, D_h, dtype=torch.float16, device=device),
                                         torch.randint(0, num_blocks * block_size, (1024,), dtype=torch.int32, device=device))),
        ("store_kvcache N=2048 batch", (torch.randn(2048, H, D_h, dtype=torch.float16, device=device),
                                        torch.randn(2048, H, D_h, dtype=torch.float16, device=device),
                                        torch.zeros(num_blocks, block_size, H, D_h, dtype=torch.float16, device=device),
                                        torch.zeros(num_blocks, block_size, H, D_h, dtype=torch.float16, device=device),
                                        torch.randint(0, num_blocks * block_size, (2048,), dtype=torch.int32, device=device))),
        ("store_kvcache N=4096 batch", (torch.randn(4096, H, D_h, dtype=torch.float16, device=device),
                                        torch.randn(4096, H, D_h, dtype=torch.float16, device=device),
                                        torch.zeros(num_blocks, block_size, H, D_h, dtype=torch.float16, device=device),
                                        torch.zeros(num_blocks, block_size, H, D_h, dtype=torch.float16, device=device),
                                        torch.randint(0, num_blocks * block_size, (4096,), dtype=torch.int32, device=device))),
    ]

    print("=" * 70)
    print(f"Op name                          | triton (us) | tilelang (us) | speedup")
    print("-" * 70)
    for name, inputs in cases:
        t_triton = float("nan")
        t_tl = float("nan")
        if args.backend in ("triton", "both"):
            t_triton = bench(ref_store, inputs, args.warmup, args.iters)
        if args.backend in ("tilelang", "both"):
            t_tl = bench(tilelang_store_kvcache, inputs, args.warmup, args.iters)
        speedup = (t_triton / t_tl) if (t_tl == t_tl and t_triton == t_triton) else float("nan")
        print(f"{name:32} | {t_triton*1000:11.1f} | {t_tl*1000:13.1f} | {speedup:.2f}x")
    print("=" * 70)


if __name__ == "__main__":
    main()
