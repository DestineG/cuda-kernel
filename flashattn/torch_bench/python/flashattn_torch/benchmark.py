import argparse

import torch

from .operators import OPERATORS
from .diagnose import reference as fp32_reference


def measure(fn, q, k, v, warmup, iterations):
    for _ in range(warmup):
        fn(q, k, v)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn(q, k, v)
    stop.record()
    stop.synchronize()
    return start.elapsed_time(stop) / iterations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, nargs="+", default=[1024])
    parser.add_argument("--head-dim", type=int, nargs="+", default=[64])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--br", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--kernels", nargs="+", choices=list(OPERATORS), default=list(OPERATORS))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(1234)
    failed = False

    for n in args.n:
        for d in args.head_dim:
            q = torch.randn((args.batch, args.heads, n, d), device="cuda", dtype=torch.bfloat16)
            k = torch.randn_like(q)
            v = torch.randn_like(q)
            reference = fp32_reference(q, k, v)
            print(f"B={args.batch} H={args.heads} N={n} D={d} Br={args.br}")
            for name in args.kernels:
                fn = lambda q, k, v, op=OPERATORS[name]: op(q, k, v, args.br) if name in {"naive", "fa1", "fa2"} else op(q, k, v)
                output = fn(q, k, v)
                torch.cuda.synchronize()
                error = (output.float() - reference).abs()
                maximum, mean = error.max().item(), error.mean().item()
                passed = torch.isfinite(output).all().item() and maximum <= .05 and mean <= .005
                failed |= not passed
                ms = measure(fn, q, k, v, args.warmup, args.iterations)
                flops = 4 * args.batch * args.heads * n * n * d / (ms * 1e9)
                print(f"  {name:12s} time_ms={ms:.6f} tflops={flops:.3f} max_abs={maximum:.6e} mean_abs={mean:.6e} {'PASS' if passed else 'FAIL'}")
    raise SystemExit(int(failed))


if __name__ == "__main__":
    main()
