# Attention 实现索引

- naive、FA1、FA2：本目录原有 `launcher.cu`、`benchmark.sh` 和 [torch_bench](torch_bench/README.md)；使用方式见下方原始说明。
- SM90 FA3：下列四个独立源码文件，使用 TMA/WGMMA 与 producer/consumer warpgroup。

## SM90 FA3

| 文件 | 版本 |
|---|---|
| [fa3_bf16_sm90.cu](fa3_bf16_sm90.cu) | 基础 producer/consumer |
| [fa3_bf16_sm90_kv_pipeline.cu](fa3_bf16_sm90_kv_pipeline.cu) | 分离 K/V 的 ready/free 生命周期 |
| [fa3_bf16_sm90_kv_pipeline_qk_pv_overlap.cu](fa3_bf16_sm90_kv_pipeline_qk_pv_overlap.cu) | QK/PV overlap |
| [fa3_bf16_sm90_kv_pipeline_qk_pv_overlap_v1.cu](fa3_bf16_sm90_kv_pipeline_qk_pv_overlap_v1.cu) | overlap 的另一调度版本 |

这些是自写实验实现。输入为 packed BF16 Q/K/V，输出为 FP32；累计长度数组使用 `size_t`。支持的实际模板范围以源码约束为准。因果模式为右下对齐，且当前实现要求每条序列 KV 长度不小于 Q；Q 非空、KV 为空时直接返回，不自动清零输出。

四个 launcher 的末尾模板参数为 `RegRealloc=false, ProducerRegs=40, ConsumerRegs=232`，用于比较关闭和开启寄存器重分配。依赖头文件为 `mbarrier.cuh`、`tma.cuh`、`wgmma.cuh`。

需要支持 SM90a 的工具链和 Hopper GPU。现有 `launcher.cu`、`benchmark.sh` 和 PyTorch wrapper **只调用 naive/FA1/FA2**；不会因为加入这些文件就自动测试 FA3。各 FA3 文件的演示 `main` 不启动计算，使用时应调用相应模板 launcher。

以下内容为导入的 naive/FA1/FA2 测试说明与历史结果，不能用来代表上述四个 SM90 FA3 的性能。

---

# FlashAttention Benchmark

For PyTorch extension benchmarks against official SDPA, see
[torch_bench/README.md](torch_bench/README.md). The standalone CUDA workflow
below remains available independently.

`launcher.cu` compares the naive attention kernel
`kernel_attn_prefill` with the hand-written FA1 and FA2 kernels in
`flashattn1.cu` and `flashattn2.cu`.
The GPU benchmark is non-causal. The CPU reference implementation keeps a
causal switch for correctness experiments, but the command-line benchmark
does not launch the causal GPU kernel.

## Build

```bash
/usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_89 \
  -lineinfo -Xptxas=-v \
  flashattn/launcher.cu -o flashattn/build/launcher
```

`launcher.cu` includes `naive-attention.cu`, `flashattn1.cu`, and
`flashattn2.cu`, so compiling
it as a single translation unit is sufficient.

## Usage

```text
launcher <verify|benchmark|profile> <all|naive|fa1|fa2> N HEAD_DIM Br [warmup] [iterations]
launcher profile_batch <all|naive|fa1|fa2> warmup iterations N HEAD_DIM Br [...]
```

Examples:

```bash
# Correctness only
./flashattn/build/launcher verify all 128 64 32
./flashattn/build/launcher verify fa1 256 128 64

# Correctness first, then CUDA-event benchmark
./flashattn/build/launcher benchmark all 256 64 32 10 100
./flashattn/build/launcher benchmark naive 1024 128 64 10 100

# Profile several shapes in one launcher invocation
./flashattn/build/launcher profile_batch all 0 1 \
  1024 64 64 8192 64 64 16384 128 64
```

In `benchmark` mode, selected kernels are compared with the CPU FP32
reference before timing starts. A failed correctness check is reported and
the timing is still printed; the process exits with a non-zero status.

The reported time is the average kernel time over `iterations` launches after
`warmup` launches. Host initialization, allocation, and correctness checking
are outside the timed region.

`profile` skips the CPU correctness phase and is intended as the Nsight
Compute target. It submits each selected kernel once and does not report a
CUDA-event time. Nsight Compute replays that launch for its metric passes;
therefore profiler replay time is not a kernel benchmark result.

## Batch Benchmark and NCU Reports

Use `benchmark.sh` to compile the launcher, run correctness-before-benchmark,
and generate one combined Nsight Compute report for all requested shapes:

```bash
# Default matrix
bash flashattn/benchmark.sh

# Custom matrix; each positional argument is N:HEAD_DIM:Br (Bc is fixed at 64)
bash flashattn/benchmark.sh 64:64:16 128:64:32 256:64:64 128:128:32

# More command-line controls
bash flashattn/benchmark.sh \
  --kernel fa1 --warmup 20 --iterations 200 \
  --profile-iterations 1 --ncu-set full \
  --report-dir flashattn/build/ncu_reports_large \
  256:64:64 512:64:64 1024:64:64
```

Reports are written to `flashattn/build/ncu_reports/flashattn_all.ncu-rep`.
NCU uses `--set full` by default and sudo is used for GPU performance-counter
access; the sudo password is the current username as configured for this
machine. Full profiling over many large shapes can take a long time because
Nsight Compute replays each launch for multiple metric passes.

The script prints a final summary table containing only the CUDA-event
benchmark times and the corresponding report paths. NCU progress output is
suppressed. The `--profile-warmup` and `--profile-iterations` options remain
accepted for command-line compatibility, but profile execution itself always
submits one launch because replay is controlled by Nsight Compute.

Available options:

```bash
--warmup N              Benchmark warmup launches (default: 10)
--iterations N          Benchmark measured launches (default: 100)
--profile-warmup N      NCU warmup launches (default: 0)
--profile-iterations N  NCU measured launches (default: 1)
--kernel all|naive|fa1|fa2  Kernel selection (default: all)
--ncu-set SET           NCU metric set (default: full)
--build-dir DIR         Launcher output directory
--report-dir DIR        NCU report directory
--arch ARCH             CUDA architecture (default: sm_89)
--nvcc PATH             nvcc executable
--ncu PATH              ncu executable
--no-sudo               Skip sudo when current user has counter access
```

Run `bash flashattn/benchmark.sh --help` for the same option list.

## Supported Shapes

The naive kernel accepts positive `N` and `HEAD_DIM`. The current FA1 and FA2
benchmarks use `Bc=64` and require:

- `N` is a multiple of 64, because the current K/V tile padding path does not
  mask an incomplete KV tile;
- `N` is also a multiple of `Br`;
- `HEAD_DIM` is 64 or 128;
- `Br` is 16, 32, or 64;
- FA2 has the same supported `HEAD_DIM`, `Br`, and divisibility requirements;
- the six dispatch targets are `fa1<64,16,64>`, `fa1<64,32,64>`,
  `fa1<64,64,64>`, `fa1<128,16,64>`, `fa1<128,32,64>`, and
  `fa1<128,64,64>`, together with matching FA2 instantiations.

FA1 uses dynamic shared memory for Q/K/V, the FP32 output tile, and the
per-warp softmax workspace. Each instantiated kernel opts in to its required
shared-memory size with `cudaFuncSetAttribute` before launch.

The FA1 algorithm is organized around `HEAD_DIM` values that are multiples of
64. Additional dimensions require a matching compile-time instantiation and a
tile shape that fits the available shared memory.

For `N > 4096`, benchmark mode skips CPU correctness and the naive correctness
reference because both require O(N²) storage and work. The selected CUDA
kernels are still timed. Use `verify` on smaller shapes to check numerical
correctness.

## Correctness

Inputs are deterministic BF16 Q/K/V tensors. The CPU reference converts them
to FP32 and computes:

```text
softmax((Q @ K^T) / sqrt(d)) @ V
```

The output reports maximum absolute error, mean absolute error, and maximum
relative error. The pass condition uses maximum absolute error `<= 5e-2` and
mean absolute error `<= 5e-3`; relative error is retained as a diagnostic
because values close to zero can make it misleading.

## Example Results

Measured on the current RTX 4090 setup with `warmup=3` and `iterations=20`:

| N | HEAD_DIM | Br | Bc | FA1 max abs |
|---:|---:|---:|---:|---:|
| 64 | 64 | 16/32/64 | 64 | 7.19e-3 |
| 64 | 128 | 16/32/64 | 64 | 5.49e-3 |

These values are device- and workload-dependent. Re-run the commands above
on the target GPU when updating the results.
