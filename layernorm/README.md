# CUDA LayerNorm 优化

本目录实现 FP32 row-wise LayerNorm。输入是连续的 `[N, M]` 二维张量，每一行独立计算：

```text
mean = sum(x) / M
variance = sum((x - mean)^2) / M
y = (x - mean) * rsqrt(variance + epsilon) * gamma + beta
```

代码使用 cuDNN LayerNorm 作为 baseline，并比较 shared memory 和 register cache 两种自定义实现。

## Kernel 优化过程

### 1. cuDNN Baseline

baseline 使用 cuDNN 9 Backend Graph 的 `CUDNN_LAYER_NORM`。使用 training phase，让 cuDNN 根据当前输入计算每行均值和方差；inference phase 需要调用方提前提供统计量，不等价于这里的 LayerNorm kernel。

Graph、执行计划和 workspace 都在计时前创建。CUDA Event 只记录执行阶段，不包含初始化开销。

### 2. Block + Shared Memory

`layernorm_shared` 使用一个 block 处理一行，每个线程以 `blockDim.x` 为步长处理若干元素。

第一次读取 global memory 时，将输入保存到 shared memory 并计算线程局部和。均值采用两级规约：先使用 `__shfl_down_sync` 完成 warp 内规约，再由第一个 warp 合并各 warp 的结果。

得到均值后，从 shared memory 读取输入，计算中心化结果和平方和。方差使用相同的两级规约，最后应用 gamma 和 beta 并写回 global memory。输入只从 global memory 读取一次，动态 shared memory 大小约为：

```text
(M + block_size / 32) * sizeof(float)
```

### 3. Block + Register Cache

`layernorm_register` 保持一个 block 处理一行，但将每个线程负责的输入保存在寄存器数组中。shared memory 只保存各 warp 的局部规约结果，避免整行数据的 shared-memory 读写。

归一化维度通过 `128、256、512、1024、2048、4096` 六个 bucket 在编译期实例化，使寄存器数组长度固定、循环可以展开。代码同时测试 128、256 和 512 三种 block size，用来权衡并行度、规约开销和寄存器占用。

LayerNorm 需要分别对均值和方差执行两次完整的 block 规约。随着 `M` 增大，每个线程缓存的数据增多，register 版本的寄存器压力也会上升。

## 正确性验证

正确性使用 CPU double reference。输入、gamma 和 beta 由固定序列生成，每次运行一致。判定条件为：

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
| 128 | 5.134 | shared 128: 4.977 (`1.031x`) | register 128: 4.692 (`1.094x`) | register 128 | `1.094x` |
| 256 | 6.817 | shared 128: 5.801 (`1.175x`) | register 128: 5.005 (`1.362x`) | register 128 | `1.362x` |
| 512 | 7.967 | shared 128: 8.269 (`0.963x`) | register 128: 6.530 (`1.220x`) | register 128 | `1.220x` |
| 1024 | 12.883 | shared 256: 16.001 (`0.805x`) | register 256: 12.661 (`1.017x`) | register 256 | `1.017x` |
| 2048 | 26.215 | shared 256: 32.286 (`0.812x`) | register 256: 27.062 (`0.969x`) | register 256 | `0.969x` |
| 4096 | 59.450 | shared 512: 70.087 (`0.848x`) | register 512: 59.829 (`0.994x`) | register 512 | `0.994x` |

### 逐形状最佳配置

每个单元格给出该形状下最快的自定义配置及其相对 cuDNN 加速比。`R` 表示 register，`S` 表示 shared，数字表示 block size。

| `M \ N` | 512 | 1024 | 2048 | 4096 | 8192 | 16384 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | R128 `1.266x` | R128 `1.225x` | R128 `0.984x` | R128 `1.029x` | R128 `1.048x` | R128 `1.043x` |
| 256 | R128 `1.258x` | R128 `1.159x` | R128 `1.232x` | R128 `1.388x` | R128 `1.547x` | R128 `1.655x` |
| 512 | R256 `1.117x` | R128 `1.066x` | R128 `1.116x` | R128 `1.261x` | R128 `1.379x` | R128 `1.475x` |
| 1024 | R256 `0.956x` | R128 `0.947x` | R256 `1.023x` | R256 `1.067x` | R256 `1.099x` | S512 `1.034x` |
| 2048 | R256 `0.929x` | R512 `0.952x` | R256 `0.954x` | R256 `1.001x` | R512 `1.031x` | R256 `1.027x` |
| 4096 | R512 `0.937x` | R512 `0.969x` | R512 `0.998x` | R512 `1.022x` | R512 `1.020x` | R512 `1.019x` |

### 固定配置的整体表现

| 配置 | 36 个形状几何平均 | 快于 cuDNN 的形状数 |
| --- | ---: | ---: |
| shared 128 | `0.893x` | 12/36 |
| shared 256 | `0.837x` | 8/36 |
| shared 512 | `0.669x` | 6/36 |
| register 128 | `1.069x` | 24/36 |
| register 256 | `0.966x` | 19/36 |
| register 512 | `0.750x` | 6/36 |

register 版本在全部 `M` 上都优于对应的最佳 shared 版本，说明将行数据留在寄存器中能够有效减少 shared-memory 访问。收益主要集中在 `M = 128、256、512`，其中 `M = 256` 的几何平均加速达到 `1.362x`，最大单点加速为 `1.655x`。

当 `M >= 2048` 时，寄存器压力、两次规约和同步的成本变得明显，几何平均结果与 cuDNN 基本持平或略慢。较大的 `N` 能摊薄固定开销并提高 GPU 利用率，因此这些大 `M` 形状通常在 `N >= 4096` 后才开始略微超过 cuDNN。

如果只能使用一个固定配置，register 128 的整体几何平均为 `1.069x`，在 24/36 个形状上更快。如果按照 `M` 选择上表中的推荐配置，整体几何平均为 `1.101x`，在 27/36 个形状上更快。

### 单个形状示例

`[N, M] = [4096, 1024]` 的原始结果如下，单位为微秒：

| Kernel | Median | Min | P90 |
| --- | ---: | ---: | ---: |
| cuDNN | 12.263 | 12.234 | 12.278 |
| shared 128 | 15.678 | 15.666 | 15.695 |
| shared 256 | 16.128 | 16.106 | 16.148 |
| shared 512 | 19.688 | 19.671 | 19.692 |
| register 128 | 12.133 | 12.121 | 12.153 |
| register 256 | 11.500 | 11.479 | 11.510 |
| register 512 | 15.964 | 15.954 | 15.974 |

该形状上 register 256 相对 shared 256 提升 `1.40x`，相对 cuDNN 提升 `1.07x`。

这些结果来自重复访问同一组 device buffer 的 kernel benchmark。缓存状态、GPU 频率和其他负载都会影响结果，复现时应以本机数据为准。

## 结果复现

编译并运行完整正确性矩阵：

```bash
bash layernorm/compile.sh
bash layernorm/verify.sh
```

运行完整性能矩阵：

```bash
bash layernorm/benchmark.sh
```

测试单个形状、单个 kernel，或者覆盖默认计时参数：

```bash
./layernorm/build/layernorm verify register_256 4096 1024
./layernorm/build/layernorm benchmark all 4096 1024
./layernorm/build/layernorm benchmark register_256 4096 1024 20 200 20
```

使用 Nsight Compute 采集单个 kernel：

```bash
sudo /usr/local/cuda/bin/ncu --set full ./layernorm/build/layernorm profile register_256 4096 1024
```
