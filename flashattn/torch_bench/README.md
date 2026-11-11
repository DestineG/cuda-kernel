# PyTorch attention comparison

This directory is independent of the existing raw CUDA benchmark. It builds a
PyTorch extension around the three local CUDA implementations and compares them
with PyTorch SDPA.

## Environment and Build

Use the repository's existing uv environment: `uv sync`. The extension needs
CUDA toolkit (`nvcc`), a compatible C++ compiler, Python development headers
(on Ubuntu with Python 3.12: `sudo apt-get install python3.12-dev`), and Ninja
(included in the project dependencies). PyTorch 2.14 requires C++20.

The first custom-operator call builds the extension. Later runs reuse the
Ninja build cache; changes to the extension or included CUDA sources trigger
recompilation. Build products are ignored by Git. CUDA compilation targets
the visible GPU by default; optionally set `TORCH_CUDA_ARCH_LIST=8.9` for
the RTX 4090. Existing standalone benchmark files are not moved.

## Benchmark

Run from the repository root:

```bash
PYTHONPATH=flashattn/torch_bench/python \
uv run python -m flashattn_torch.benchmark \
  --n 128 256 1024 \
  --head-dim 64 128 \
  --br 64 \
  --warmup 10 --iterations 100
```

The first run compiles the extension into `flashattn/torch_bench/build/`.
The wrapper accepts BF16 CUDA tensors with arbitrary batch/head counts,
non-causal attention, and `HEAD_DIM` 64 or 128. Use `--batch` and `--heads`
to select batch and head counts. Outputs of custom kernels are FP32.
Inputs must have identical shapes with positive dimensions, N divisible by
64 and Br, and Br equal to 16, 32, or 64. Causal attention and GQA are not
exposed by this wrapper.

For long sequences, select the fused implementations to avoid the naive and
math operators' quadratic intermediate buffers:

```bash
PYTHONPATH=flashattn/torch_bench/python \
uv run python -m flashattn_torch.benchmark \
  --n 4096 8192 16384 32768 --head-dim 64 128 \
  --batch 1 --heads 1 --br 64 --warmup 5 --iterations 30 \
  --kernels fa1 fa2 torch torch_flash

PYTHONPATH=flashattn/torch_bench/python \
uv run python -m flashattn_torch.benchmark \
  --n 1024 4096 8192 --head-dim 64 128 \
  --batch 2 --heads 8 --br 64 --warmup 5 --iterations 30 \
  --kernels fa1 fa2 torch torch_flash
```

`--kernels` accepts `naive`, `fa1`, `fa2`, `torch`, `torch_flash`, and
`torch_math`. `torch` uses default SDPA dispatch; the other two PyTorch
operators force FLASH_ATTENTION and MATH respectively. All are forward-only
with dropout disabled. Defaults are B=1, H=1, N=1024, D=64, Br=64,
warmup=10, iterations=100, and all operators selected.

Correctness uses an FP32 PyTorch math reference on the same BF16 inputs,
with query rows chunked to bound reference workspace. Inputs use seed 1234.
The absolute/mean error thresholds are 0.05/0.005; failures exit nonzero.
Timing includes wrapper allocations and layout conversion, so it measures
the PyTorch-facing call rather than only the custom CUDA kernel.
Reported TFLOPS uses `4 * B * H * N * N * D`. PyTorch BF16 outputs and
custom FP32 outputs have different storage precision; both are compared
against the same FP32 reference. Small-call timing also includes host launch
gaps and Python backend-selection overhead, so use large shapes to assess
GPU throughput. This tool does not generate NCU reports.

## Diagnostics

To compare combined execution against independent execution of each head:

```bash
PYTHONPATH=flashattn/torch_bench/python \
uv run python -m flashattn_torch.diagnose \
  --batch 2 --heads 8 --n 4096 --head-dim 64 --br 64
```

FA2 had shared-memory K/V reuse races between warps. Block barriers before
buffer reuse fix the issue; it was not specific to multiple attention heads.

## Verified Results

Retested on 2026-10-06 with RTX 4090, PyTorch 2.14.1+cu130, BF16 inputs,
Br=64, warmup=5, iterations=30. All 80 operator checks across 18 shape
configurations passed the FP32 reference checks. Small cases include naive
and PyTorch math; long cases select FA1/FA2/default SDPA/Flash SDPA.
Times below are milliseconds per PyTorch-facing call.

| B | H | N | D | FA1 | FA2 | PyTorch Default | PyTorch Flash |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 4096 | 64 | 0.207667 | 0.119600 | 0.042667 | 0.046421 |
| 1 | 1 | 8192 | 128 | 0.726219 | 0.343285 | 0.253884 | 0.254327 |
| 1 | 1 | 16384 | 128 | 2.630458 | 1.008845 | 0.937711 | 0.911394 |
| 1 | 1 | 32768 | 64 | 3.712068 | 2.256384 | 1.726089 | 1.688505 |
| 1 | 1 | 32768 | 128 | 10.478627 | 4.048759 | 3.410364 | 3.381585 |
| 2 | 8 | 4096 | 64 | 0.996727 | 0.588425 | 0.464725 | 0.465161 |
| 2 | 8 | 8192 | 128 | 11.605299 | 4.321045 | 3.361269 | 3.334144 |

FA2 max absolute error was 8.22e-5 at B=1,H=1,N=32768,D=128 and
1.69e-4 at B=2,H=8,N=8192,D=128. Measurements vary with device load;
rerun the commands above to reproduce on your GPU.
