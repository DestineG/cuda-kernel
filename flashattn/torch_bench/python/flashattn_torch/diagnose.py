import argparse

import torch

from .operators import fa1, fa2, torch_math


def reference(q, k, v):
    # Chunk only query rows to bound the FP32 math backend's score workspace.
    result = torch.empty_like(q, dtype=torch.float32)
    for b in range(q.size(0)):
        for h in range(q.size(1)):
            for start in range(0, q.size(2), 128):
                result[b:b+1, h:h+1, start:start+128] = torch_math(
                    q[b:b+1, h:h+1, start:start+128].float(),
                    k[b:b+1, h:h+1].float(),
                    v[b:b+1, h:h+1].float(),
                )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=4096)
    parser.add_argument('--head-dim', type=int, default=64)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--br', type=int, default=64)
    args = parser.parse_args()
    torch.manual_seed(1234)
    tensors = [torch.randn((args.batch, args.heads, args.n, args.head_dim),
                           device='cuda', dtype=torch.bfloat16) for _ in range(3)]
    ref = reference(*tensors)
    failed = False
    for name, op in [('fa1', fa1), ('fa2', fa2)]:
        out = op(*tensors, args.br)
        repeat = op(*tensors, args.br)
        separate = torch.empty_like(out)
        for b in range(args.batch):
            for h in range(args.heads):
                separate[b:b+1, h:h+1] = op(
                    *(t[b:b+1, h:h+1] for t in tensors), args.br)
        err = (out - ref).abs()
        isolated = (separate - ref).abs().max().item()
        layout = (out - separate).abs().max().item()
        nondeterminism = (out - repeat).abs().max().item()
        maximum, mean = err.max().item(), err.mean().item()
        passed = torch.isfinite(out).all().item() and maximum <= .05 and mean <= .005
        failed |= not passed
        print(f'{name} max_abs={maximum:.6e} mean_abs={mean:.6e} '
              f'isolated_max={isolated:.6e} combined_vs_isolated={layout:.6e} '
              f'repeat_diff={nondeterminism:.6e} {"PASS" if passed else "FAIL"}', flush=True)
    raise SystemExit(int(failed))


if __name__ == '__main__':
    main()
