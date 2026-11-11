# CUDA RMSNorm 优化

本目录实现 FP32 row-wise RMSNorm。输入是连续的 `[N, M]` 二维张量，每一行独立计算：

```text
mean_square = sum(x * x) / M
y = x * rsqrt(mean_square + epsilon) * gamma
```

代码使用 cuDNN RMSNorm 作为 baseline，并比较 shared memory 和 register cache 两种自定义实现。RMSNorm 不计算均值，也没有 beta。

## Kernel 优化过程

### 1. cuDNN Baseline

baseline 使用 cuDNN 9 Backend Graph 的 `CUDNN_RMS_NORM`。Graph、执行计划、workspace 和接口要求的零 bias 都在计时前准备完成。CUDA Event 只记录执行阶段，不包含初始化开销。

### 2. Block + Shared Memory

`rmsnorm_shared` 使用一个 block 处理一行，每个线程以 `blockDim.x` 为步长读取若干元素。读取输入时同时累加平方和，并将数据保存在 shared memory 中。

平方和采用两级规约：先通过 `__shfl_down_sync` 完成 warp 内规约，再由第一个 warp 合并各 warp 的结果。得到 `rsqrt(mean_square + epsilon)` 后，从 shared memory 读取输入，乘以归一化系数和 gamma，再写回 global memory。输入只从 global memory 读取一次，动态 shared memory 大小约为：

```text
(M + block_size / 32) * sizeof(float)
```

### 3. Block + Register Cache

`rmsnorm_register` 保持一个 block 处理一行，但将每个线程负责的输入保存在寄存器数组中。shared memory 只保存各 warp 的局部平方和，避免整行数据的 shared-memory 读写。

归一化维度通过 `128、256、512、1024、2048、4096` 六个 bucket 在编译期实例化，使寄存器数组长度固定、循环可以展开。代码同时测试 128、256 和 512 三种 block size，用来权衡并行度、规约开销和寄存器占用。

与 LayerNorm 相比，RMSNorm 不需要计算均值和中心化结果，只需进行一次平方和规约，因此规约和同步开销更低。

## 正确性验证

正确性使用 CPU double reference。输入和 gamma 由固定序列生成，每次运行一致。判定条件为：

```text
abs_error <= 2e-5 + 2e-4 * abs(reference)
```

测试矩阵覆盖小尺寸、非整齐尺寸以及每个 register bucket 的前后边界：

| 参数 | 测试值 |
| --- | --- |
| `N` | 1, 31, 512 |
| `M` | 1, 31, 32, 33, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025, 2047, 2048, 2049, 4095, 4096 |

共 63 个形状，cuDNN 和六个自定义配置全部通过。

## 性能测试方法

性能矩阵包含 36 个形状：

| 参数 | 测试值 |
| --- | --- |
| `N` | 512, 1024, 2048, 4096, 8192, 16384 |
| `M` | 128, 256, 512, 1024, 2048, 4096 |

每个实现先执行 10 次 warmup。每个样本连续执行 100 次 kernel，共采集 10 个样本，以下分析使用单次 kernel latency 的 median。计时不包含输入生成、内存分配、H2D/D2H 和 cuDNN Graph 创建。

测试环境为 NVIDIA GeForce RTX 4090、CUDA 13.0、cuDNN 9.26。加速比定义为：

```text
speedup = cuDNN median latency / custom median latency
```

大于 `1.0x` 表示自定义 kernel 更快。跨多个形状汇总时使用几何平均。

## 测试矩阵性能分析

### 按 M 汇总

下表对每个 `M` 下的六个 `N` 取几何平均。延迟单位为微秒，括号内为相对 cuDNN 的加速比。

| `M` | cuDNN | 最佳 shared | 最佳 register | 推荐配置 | 推荐配置加速比 |
| ---: | ---: | --- | --- | --- | ---: |
| 128 | 4.534 | shared 128: 3.843 (`1.180x`) | register 128: 3.759 (`1.206x`) | register 128 | `1.206x` |
| 256 | 6.175 | shared 128: 4.613 (`1.339x`) | register 128: 4.127 (`1.496x`) | register 128 | `1.496x` |
| 512 | 7.391 | shared 128: 6.996 (`1.056x`) | register 128: 6.095 (`1.213x`) | register 128 | `1.213x` |
| 1024 | 12.966 | shared 256: 13.263 (`0.978x`) | register 256: 11.871 (`1.092x`) | register 256 | `1.092x` |
| 2048 | 26.604 | shared 512: 28.525 (`0.933x`) | register 512: 25.450 (`1.045x`) | register 512 | `1.045x` |
| 4096 | 59.942 | shared 512: 63.204 (`0.948x`) | register 512: 58.435 (`1.026x`) | register 512 | `1.026x` |

### 逐形状最佳配置

每个单元格给出该形状下最快的自定义配置及其相对 cuDNN 加速比。`R` 表示 register，`S` 表示 shared，数字表示 block size。

| `M \ N` | 512 | 1024 | 2048 | 4096 | 8192 | 16384 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | R128 `1.506x` | R128 `1.413x` | R128 `1.179x` | R128 `1.029x` | R128 `1.075x` | R128 `1.108x` |
| 256 | R128 `1.366x` | R128 `1.227x` | R128 `1.289x` | R128 `1.519x` | R128 `1.747x` | R128 `1.959x` |
| 512 | R256 `1.195x` | R128 `1.117x` | R128 `1.175x` | R256 `1.252x` | R128 `1.296x` | R128 `1.339x` |
| 1024 | R256 `1.015x` | R512 `1.146x` | R256 `1.129x` | R256 `1.116x` | R256 `1.126x` | S512 `1.035x` |
| 2048 | R512 `1.042x` | R512 `1.099x` | R512 `1.044x` | R512 `1.030x` | R512 `1.031x` | R256 `1.029x` |
| 4096 | R512 `1.030x` | R512 `1.043x` | R512 `1.024x` | R256 `1.019x` | R512 `1.020x` | R512 `1.018x` |

### 固定配置的整体表现

| 配置 | 36 个形状几何平均 | 快于 cuDNN 的形状数 |
| --- | ---: | ---: |
| shared 128 | `1.004x` | 21/36 |
| shared 256 | `0.989x` | 18/36 |
| shared 512 | `0.836x` | 10/36 |
| register 128 | `1.129x` | 28/36 |
| register 256 | `1.077x` | 30/36 |
| register 512 | `0.902x` | 20/36 |

register 版本在全部 `M` 上都优于对应的最佳 shared 版本。收益同样集中在较小的归一化维度，其中 `M = 256` 的几何平均加速达到 `1.496x`，在 `[16384, 256]` 上达到本次矩阵的最大单点加速 `1.959x`。

随着 `M` 增大，最优 block size 从 128 逐步增加到 256 和 512。更大的 block 能减少每线程保存的元素数量和寄存器压力。即使在 `M = 4096` 时，register 512 仍取得 `1.026x` 的几何平均加速，但优势已经缩小到接近测量波动的范围。

如果只能使用一个固定配置，register 128 的整体几何平均最高，为 `1.129x`；register 256 的覆盖更稳定，在 30/36 个形状上快于 cuDNN。如果按照 `M` 选择上表中的推荐配置，整体几何平均为 `1.170x`，36/36 个形状全部快于 cuDNN。

本次矩阵中 RMSNorm 的收益比 LayerNorm 更稳定。从实现路径看，RMSNorm 只进行一次平方和规约，不需要额外的均值规约和中心化步骤，自定义 kernel 的同步与计算负担更小。

### 单个形状示例

`[N, M] = [4096, 1024]` 的原始结果如下，单位为微秒：

| Kernel | Median | Min | P90 |
| --- | ---: | ---: | ---: |
| cuDNN | 10.967 | 10.947 | 10.984 |
| shared 128 | 11.652 | 11.640 | 11.663 |
| shared 256 | 11.469 | 11.448 | 11.479 |
| shared 512 | 12.736 | 12.725 | 12.758 |
| register 128 | 10.219 | 10.209 | 10.237 |
| register 256 | 9.820 | 9.799 | 9.821 |
| register 512 | 10.435 | 10.414 | 10.452 |

该形状上 register 256 相对 shared 256 提升 `1.17x`，相对 cuDNN 提升 `1.12x`。

这些结果来自重复访问同一组 device buffer 的 kernel benchmark。缓存状态、GPU 频率和其他负载都会影响结果，复现时应以本机数据为准。

## 结果复现

编译并运行完整正确性矩阵：

```bash
bash rmsnorm/compile.sh
bash rmsnorm/verify.sh
```

运行完整性能矩阵：

```bash
bash rmsnorm/benchmark.sh
```

测试单个形状、单个 kernel，或者覆盖默认计时参数：

```bash
./rmsnorm/build/rmsnorm verify register_256 4096 1024
./rmsnorm/build/rmsnorm benchmark all 4096 1024
./rmsnorm/build/rmsnorm benchmark register_256 4096 1024 20 200 20
```

使用 Nsight Compute 采集单个 kernel：

```bash
sudo /usr/local/cuda/bin/ncu --set full ./rmsnorm/build/rmsnorm profile register_256 4096 1024
```
