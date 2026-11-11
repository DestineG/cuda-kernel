# CUDA Kernel Practice

手写 CUDA 算子练习：Softmax、LayerNorm、RMSNorm、GEMM，以及 naive attention、FA1、FA2 和 SM90 FA3。

## 实现索引

| 算子 | 实现与计算路径 | 源码与使用说明 |
|---|---|---|
| Softmax | 多种归约和在线 Softmax 实现 | [softmax/README.md](softmax/README.md) |
| LayerNorm | 自写归约与归一化实现 | [layernorm/README.md](layernorm/README.md) |
| RMSNorm | 自写归约与归一化实现 | [rmsnorm/README.md](rmsnorm/README.md) |
| GEMM | BF16，cp.async + ldmatrix + mma.sync | [gemm/gemm_custom.cu](gemm/gemm_custom.cu) |
| GEMM SM90 | BF16，TMA + WGMMA，单缓冲与双缓冲 | [单缓冲](gemm/gemm_bf16_sm90.cu)、[双缓冲](gemm/gemm_bf16_sm90_dbuffer.cu) |
| GEMM SM120 | BF16，单缓冲与双缓冲 | [单缓冲](gemm/gemm_bf16_sm120.cu)、[双缓冲](gemm/gemm_bf16_sm120_dbuffer.cu) |
| Attention | naive、FA1、FA2，含 standalone 和 PyTorch SDPA 对比入口 | [flashattn/README.md](flashattn/README.md) |
| Attention SM90 | 四个 FA3 实验版本，packed varlen、GQA、可选因果掩码 | [SM90 源码索引](flashattn/README.md#sm90-fa3) |

GEMM 的编译和现有结果见 [gemm/README.md](gemm/README.md)。各目录 README 中的结果应结合对应 GPU、输入范围和计时方法阅读，不能将不同硬件的历史结果直接比较。

## 编译与运行

各算子保持独立的构建和测试入口，没有要求一次编译全部 CUDA 源码。

```bash
# 从仓库根目录执行；默认配置面向 RTX 4090 / SM89。
bash gemm/compile_custom.sh
bash gemm/verify_custom.sh
bash gemm/benchmark_custom.sh

# naive / FA1 / FA2；该脚本默认还会运行 Nsight Compute。
# 可先查看选项，再按机器权限和需求运行。
bash flashattn/benchmark.sh --help
```

Softmax、LayerNorm、RMSNorm 的 `compile.sh`、`verify.sh` 或对应检查方式见各自 README；它们的对比程序还需要 cuDNN。`gemm_custom.cu` 的测试程序需要 cuBLAS。

SM90 WGMMA 源码需要支持 `sm_90a` 的 CUDA 工具链，运行需要 Hopper GPU（如 H800/H20）。SM120 源码需要支持 `sm_120a` 的工具链和相应 GPU。具体编译目标以源码使用的指令为准；SM89 GPU 不能运行 SM90/SM120 专用 kernel。

SM90 FA3 的 `.cu` 文件提供模板 kernel 和 launcher，其演示 `main` 仅返回；直接编译并运行这个 `main` 不等于执行正确性或性能测试。SM90/SM120 GEMM 同样需要调用相应 launcher，不能把导入的 `gemm_custom` 测试脚本当成它们的测试入口。

## Python / PyTorch

[flashattn/torch_bench](flashattn/torch_bench/README.md) 是 naive/FA1/FA2 的 PyTorch 接口，尚未接入 SM90 FA3。

`pyproject.toml`、`uv.lock` 和 `src/practice` 保留自原 `practice` 项目，以兼容已有 Python 调用。该依赖配置要求 Python >=3.12、PyTorch >=2.14.1；不要将它当成全部历史实验的统一环境。standalone CUDA 构建不依赖这个 Python 包。

## SM90 FA3 寄存器重分配

四个 launcher 的模板参数末尾均包含：

```cpp
bool RegRealloc = false,
uint32_t ProducerRegs = 40,
uint32_t ConsumerRegs = 232
```

默认关闭。开启时，producer warpgroup 调用 `warpgroup_reg_dealloc`，consumer warpgroup 调用 `warpgroup_reg_alloc`；函数封装在 [wgmma.cuh](flashattn/wgmma.cuh)。目标数量为每线程 32 位寄存器数，必须为 8 的倍数且位于 [24,256]；它们与 ptxas 的静态寄存器报告不是同一指标。开启不保证性能提高。

## 测试归档与来源

- `build/`、各目录 `docs/` 继续作为本地生成文件及实验归档目录，默认不进入 Git；融合不会删除已有本地文件。
- 完整测试 ZIP 单独保存，源码仓库不包含全部历史实验的二进制和原始日志。
- 本仓库融合了 [DestineG/practice](https://github.com/DestineG/practice) 在提交 `37fa5c43f11c1e4b133da4aa0596517ea121ac05` 的内容，并保留其 Git 提交历史；当前仓库原有 SM90/SM120 GEMM 和 SM90 FA3 实现保留在原路径。
