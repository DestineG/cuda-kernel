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

## SM90 FA3 v1 对照版本

原始 `fa3_bf16_sm90_kv_pipeline_qk_pv_overlap_v1.cu` 保留不变，新增三份独立源码：

| 文件后缀（均接在 `fa3_bf16_sm90_kv_pipeline_qk_pv_overlap_v1` 后） | 改动 |
|---|---|
| [`_compile_opt.cu`](fa3_bf16_sm90_kv_pipeline_qk_pv_overlap_v1_compile_opt.cu) | 首轮移到循环外；因果形状检查改为条件 trap，避免 __assertfail 调用 |
| [`_mask_opt.cu`](fa3_bf16_sm90_kv_pipeline_qk_pv_overlap_v1_mask_opt.cu) | 完全合法的 KV 块跳过逐元素 causal mask；producer/consumer 同时裁掉全非法 KV 块 |
| [`_compile_mask_opt.cu`](fa3_bf16_sm90_kv_pipeline_qk_pv_overlap_v1_compile_mask_opt.cu) | 同时应用上述两组改动 |

新文件的 kernel 和 launcher 名称均追加对应后缀，例如 `fa3_bf16_forward_kv_pipeline_qk_pv_overlap_v1_compile_mask_opt_launcher`；模板参数顺序沿用原始 v1。每份文件独立编译，原 naive/FA1/FA2 launcher 不自动调用这些版本。

组合版保留 v1 的调度：当前 softmax 的 exp/rowsum 在等待上一轮 PV 完成前执行，新 P 的打包及 O 的缩放在等待之后执行。首轮只等待一次 QK，没有上一轮 PV，也不需要初始化 P 来消除警告。

mask 使用序列局部坐标，右下对齐语义不变。`kv_limit = kv_lens - q_lens + q_row_begin + num_qtile_row`，两侧循环共用该上界；有效尾块也按这个上界屏蔽。TMA 对最后部分块仍加载完整 box，越界/无贡献的元素由 mask 排除。

验证范围：CUDA12.4.131 编译28个实例（64x64、D64/128、因果/非因果、重分配关闭/40+232，以及组合版128x128 D128）；约208万次CPU mask可见性比较通过。64x64的消警告版和组合版没有C7510/C7513；仅mask版保留原警告。组合版128x128 D128仍有C7511寄存器资源不足导致的串行化提示。主机GCC13使用allow-unsupported-compiler；后续已完成H800全量验证，见下方“工程化FA3测试”和独立实验说明。

## v1 组合版的独立双缓冲实验

以下三份文件基于 `fa3_bf16_sm90_kv_pipeline_qk_pv_overlap_v1_compile_mask_opt.cu`，原组合版保持不变。完整文件名在该前缀后追加：

| 后缀 | K槽数 | V槽数 |
|---|---:|---:|
| [`_k_dbuffer.cu`](fa3_bf16_sm90_kv_pipeline_qk_pv_overlap_v1_compile_mask_opt_k_dbuffer.cu) | 2 | 1 |
| [`_v_dbuffer.cu`](fa3_bf16_sm90_kv_pipeline_qk_pv_overlap_v1_compile_mask_opt_v_dbuffer.cu) | 1 | 2 |
| [`_kv_dbuffer.cu`](fa3_bf16_sm90_kv_pipeline_qk_pv_overlap_v1_compile_mask_opt_kv_dbuffer.cu) | 2 | 2 |

kernel/launcher名称也追加对应后缀，模板参数顺序保持一致。每份独立编译，不自动并入已有benchmark入口。

K/V按各自tile编号取模选槽，各槽具有独立ready/free barrier和phase。consumer在QK完成后释放K，在PV完成后释放V；producer只在重用槽时等待free，发出当前K后可等待对应V槽。Q与输出复用一组独立握手，producer在输出阶段前排空所有尚未确认的K消费。最后一块V无需再释放，因为没有下一次槽位重用，consumer等待最后PV完成后才写回输出。

Br=Bc=64、D128时，双K/双V分别占65536B数据+64B barrier；双KV占81920B数据+80B barrier。动态共享内存申请已按实际槽数更新。保留组合版的首轮分离、trap、因果裁剪、v1 softmax/PV重叠与寄存器重分配参数。

本次CUDA12.4编译30个实例通过：三版本64x64、D64/128、因果/非因果、rr关闭/40+232，以及128x128 D128、TMA box行数128。64x64无C751x提示；128x128 D128仍有C7511。CPU抽象协议模型覆盖1～33块、完整/部分输出与19800种交错调度；这不能代替GPU正确性或同步检测。后续已完成H800全量GPU测试，默认和入选配置通过数值及四类sanitizer检查，详见下方。

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

## 工程化FA3测试

[tests/README.md](tests/README.md)提供全部10个FA3版本的统一测试工具。源码/配置快照、编译、独立参数搜索、CPU正确性、sanitizer门槛、性能矩阵和报告生成在独立run中记录，支持恢复中断任务和归档。

```bash
# 从flashattn目录执行
python tests/run.py inspect
python tests/run.py all --config tests/configs/full.json --name all-fa3 --run-root /path/to/experiment-disk/fa3-runs
python tests/run.py resume --run /path/to/run
python tests/run.py report --run /path/to/run
python tests/run.py archive --run /path/to/run
```

报告为report.html：左侧输入条件，中间kernel/超参数，右侧实际测试结果；支持筛选、排序和当前CSV导出，不显示相对值。工具和配置进入源码仓库，runs/、缓存及实验产物默认不跟踪。

## BF16 输出与双 consumer 实验

[experiments/README.md](experiments/README.md)收录独立的double_V输出类型对照和V/KV BF16 single、dual、ping-pong实验，包括六份CUDA源码、测试框架、参数与结果摘录。原FP32实现保留。

全部10版本的H800运行覆盖157组正确性、268组性能输入和12组反序确认；6777条性能记录。基础版4个consumer=128候选在sanitizer下出现非有限值，已淘汰；容量超限单列。double_V输出实验2014条性能记录；V/KV双consumer实验4690条，修正CTA寄存器预算后完整重测通过。各实验适用范围和失败诊断见各自README及结果摘录。
