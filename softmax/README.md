# CUDA Softmax 优化

本目录实现 FP32 row-wise Softmax。输入是连续的 `[batch, num_classes]` 二维张量，每一行独立计算：

```text
max_value = max(x)
sum = sum(exp(x - max_value))
y = exp(x - max_value) / sum
```

减去最大值可以避免 `exp` 溢出。代码使用 cuDNN Softmax 作为 baseline，从串行实现开始，逐步优化线程映射、访存、规约和中间结果复用。

## Kernel 优化过程

### 1. cuDNN Baseline

baseline 使用 `cudnnSoftmaxForward`，算法为 `CUDNN_SOFTMAX_ACCURATE`，模式为 `CUDNN_SOFTMAX_MODE_CHANNEL`。cuDNN handle 和 tensor descriptor 在计时前创建，CUDA Event 只记录执行阶段。

### 2. Naive

`softmax_naive` 使用一个线程处理一整行，依次完成 max、sum 和归一化。它的结构最简单，但一行内部完全串行；同一个输入最多读取三次；一个 warp 中的线程分别访问不同行，访存地址间隔为 `num_classes`，无法形成理想的合并访存。

### 3. Vectorized

`softmax_vectorized` 保持一个线程处理一行，只将标量读写改为 `float4`。每条访存指令处理四个 float，减少访存指令和循环次数，但没有解决行内串行和跨行访存问题。该实现要求 `num_classes` 是 4 的倍数。

### 4. Warp

`softmax_warp` 改为一个 warp 处理一行。每个 lane 负责一部分元素，warp 内使用 `__shfl_down_sync` 完成 max 和 sum 规约。一行由 32 个线程并行处理，相邻 lane 也会访问相邻元素。相比单纯使用 `float4`，改变线程映射带来的提升更明显。

### 5. Warp + Vectorized

`softmax_warp_vectorized` 在一个 warp 处理一行的基础上使用 `float4`，同时利用行内并行、合并访存和向量化，进一步减少 load、store 和循环控制指令。

### 6. Online Softmax

`softmax_warp_vectorized_online` 使用 `(max, sum)` 状态合并 max 和 sum 的计算：

```text
m = max(m1, m2)
s = s1 * exp(m1 - m) + s2 * exp(m2 - m)
```

因此只需一次输入遍历就能得到最大值和指数和，整个 kernel 对输入的读取由三次减少为两次。代价是状态合并需要额外的指数运算，因此收益会随 `num_classes` 和硬件特征变化。

### 7. Block + Shared Memory

`softmax_block_vectorized_shared` 使用一个 block 处理一行，为较大的 `num_classes` 提供更多行内并行度。输入通过 `float4` 加载到 shared memory；max 和 sum 先完成 warp 内规约，再通过 shared memory 完成 warp 间规约；`exp(x - max)` 暂存在 shared memory，归一化阶段直接复用。

这个版本避免重复读取输入和重复计算指数，代价是 block 同步以及随 `num_classes` 增长的 shared memory 占用。

### 8. Block + Register Cache

`softmax_block_vectorized_register` 将每个线程负责的 `float4` 保存在寄存器数组中，shared memory 只保存各 warp 的局部规约结果，从而去掉整行 shared-memory 缓存及其读写。

模板参数给出类别数上界，使循环可以在编译期展开。类别数增大时，每线程寄存器占用也会上升，因此代码同时保留 128 和 256 线程版本。

### 9. Latest Register Version

`softmax_block_vectorized_register_latest` 在 register 版本上继续减少局部指令：加载输入时同时计算线程内最大值；使用树形方式规约 `float4`；使用 `__expf`；提前计算 `1 / sum`，归一化阶段使用乘法。

## 正确性验证

正确性使用 CPU double reference，并额外检查每行输出之和是否接近 1。判定条件为：

```text
abs_error <= 2e-6 + 1e-4 * abs(reference)
row_sum_error <= 1e-4
```

向量化和 block kernel 要求 `num_classes` 是 4 的倍数，block kernel 当前支持到 4096。运行所有支持的实现：

```bash
./softmax/build/softmax verify all 4096 1024
```

在 `[4096, 1024]` 上，cuDNN 和 11 个自定义 kernel 全部通过；最大绝对误差为 `9.020e-09`，最大行和误差为 `8.865e-07`。

## 性能测试方法

性能矩阵包含 30 个形状：

| 参数 | 测试值 |
| --- | --- |
| `batch` | 512, 1024, 2048, 4096, 8192, 16384 |
| `num_classes` | 256, 512, 1024, 2048, 4096 |

每个实现先执行 10 次 warmup。每个样本连续执行 100 次 kernel，共采集 10 个样本，以下分析使用单次 kernel latency 的 median。计时不包含输入生成、内存分配、H2D/D2H 和 cuDNN descriptor 创建。

测试环境为 NVIDIA GeForce RTX 4090、CUDA 13.0、cuDNN 9.26。加速比定义为：

```text
speedup = cuDNN median latency / custom median latency
```

大于 `1.0x` 表示自定义 kernel 更快。跨多个形状汇总时使用几何平均。naive 和 vectorized 保留在单点优化过程分析中，但由于它们是一行一线程的串行版本，不参与 30 形状的最终配置统计。

## 测试矩阵性能分析

### 按 num_classes 汇总

下表对每个 `num_classes` 下的六个 batch 取几何平均。延迟单位为微秒，括号内为相对 cuDNN 的加速比。

| `num_classes` | cuDNN | 最佳 warp | 最佳 shared | 最佳 register | 最佳 latest | 最佳自定义配置 |
| ---: | ---: | --- | --- | --- | --- | --- |
| 256 | 3.801 | online: 3.953 (`0.961x`) | shared 128: 4.935 (`0.770x`) | register 128: 4.705 (`0.808x`) | latest 128: 4.624 (`0.822x`) | online (`0.961x`) |
| 512 | 5.747 | online: 6.501 (`0.884x`) | shared 128: 6.159 (`0.933x`) | register 128: 5.854 (`0.982x`) | latest 128: 5.770 (`0.996x`) | latest 128 (`0.996x`) |
| 1024 | 18.146 | online: 13.836 (`1.312x`) | shared 256: 12.028 (`1.509x`) | register 128: 11.317 (`1.603x`) | latest 128: 11.142 (`1.629x`) | latest 128 (`1.629x`) |
| 2048 | 35.372 | online: 33.209 (`1.065x`) | shared 256: 26.145 (`1.353x`) | register 256: 24.062 (`1.470x`) | latest 256: 23.841 (`1.484x`) | latest 256 (`1.484x`) |
| 4096 | 79.762 | online: 87.002 (`0.917x`) | shared 256: 60.980 (`1.308x`) | register 256: 56.325 (`1.416x`) | latest 256: 56.140 (`1.421x`) | latest 256 (`1.421x`) |

### 逐形状最佳配置

每个单元格给出该形状下最快的自定义配置及其相对 cuDNN 加速比。`WO` 表示 warp vectorized online，`R` 表示 register，`L` 表示 latest register，数字表示 block size。

| `num_classes \ batch` | 512 | 1024 | 2048 | 4096 | 8192 | 16384 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | L128 `0.997x` | WO `0.937x` | WO `0.942x` | WO `0.995x` | WO `0.966x` | WO `0.938x` |
| 512 | L128 `0.977x` | L128 `0.971x` | WO `0.907x` | L128 `0.994x` | L128 `1.085x` | L128 `1.107x` |
| 1024 | L256 `1.496x` | L128 `1.734x` | L128 `1.840x` | L128 `1.946x` | L128 `2.003x` | L128 `1.004x` |
| 2048 | L256 `1.516x` | L256 `1.762x` | L256 `1.952x` | L256 `2.032x` | L128 `1.006x` | R128 `1.003x` |
| 4096 | L256 `1.905x` | L256 `2.042x` | L256 `2.096x` | R256 `1.006x` | R256 `1.002x` | R256 `1.001x` |

### 固定配置的整体表现

| 配置 | 30 个形状几何平均 | 快于 cuDNN 的形状数 |
| --- | ---: | ---: |
| warp | `0.855x` | 7/30 |
| warp vectorized | `0.949x` | 10/30 |
| warp vectorized online | `1.017x` | 10/30 |
| block shared 128 | `1.111x` | 14/30 |
| block shared 256 | `1.017x` | 13/30 |
| block register 128 | `1.195x` | 18/30 |
| block register 256 | `1.102x` | 18/30 |
| block register latest 128 | `1.214x` | 18/30 |
| block register latest 256 | `1.116x` | 18/30 |

`num_classes = 256` 时一行数据较少，cuDNN 的映射和固定开销控制得更好，所有自定义配置的几何平均都落后。`num_classes = 512` 时 latest 128 已基本追平，并在较大 batch 上取得约 `1.1x` 加速。

`num_classes = 1024、2048、4096` 时，block register 能提供足够的行内并行度，又避免整行 shared-memory 访问，明显优于 warp 和 shared 版本。最佳 block size 也随类别数从 128 增加到 256，以降低每线程缓存的数据量和寄存器压力。

矩阵中存在明显的缓存拐点：`[8192, 1024]`、`[4096, 2048]`、`[2048, 4096]` 的输入与输出合计约 64 MiB，可以放入 RTX 4090 的 72 MiB L2，此时加速比约为 `2.0x`；将 batch 再翻倍后，工作集超过 L2，加速比立即收敛到约 `1.0x`。这说明重复访问同一组 buffer 时，中间结果复用和指令优化主要在 L2 驻留区间体现；进入 DRAM 带宽受限区间后，自定义实现与 cuDNN 的差距很小。

如果只能使用一个固定配置，latest 128 的整体几何平均最高，为 `1.214x`。如果按 `num_classes` 选择最佳自定义配置，整体几何平均为 `1.269x`，在 20/30 个形状上更快；若在 256 和 512 上保留 cuDNN、在 1024 以上选择 latest 配置，整体几何平均为 `1.280x`。逐形状在 cuDNN 和自定义实现中择优时，理论上可达到 `1.288x`。

### 优化过程单点分析

`[batch, num_classes] = [4096, 1024]` 的完整优化链结果如下，单位为微秒：

| Kernel | Median | Min | P90 |
| --- | ---: | ---: | ---: |
| cuDNN | 19.630 | 19.610 | 19.648 |
| naive | 414.405 | 414.321 | 456.265 |
| vectorized | 119.806 | 119.654 | 119.856 |
| warp | 14.571 | 14.531 | 14.592 |
| warp vectorized | 13.310 | 13.292 | 13.320 |
| warp vectorized online | 12.217 | 12.205 | 12.227 |
| block shared 128 | 10.141 | 10.124 | 10.155 |
| block shared 256 | 9.912 | 9.889 | 9.923 |
| block register 128 | 9.436 | 9.380 | 9.461 |
| block register 256 | 9.420 | 9.411 | 9.431 |
| block register latest 128 | 9.326 | 9.295 | 9.329 |
| block register latest 256 | 9.440 | 9.421 | 9.452 |

只做 `float4` 向量化时，vectorized 相对 naive 提升 `3.46x`；改成 warp 处理一行后，相对 vectorized 再提升 `8.22x`，说明线程映射是最关键的一步。online 版本相对普通 warp vectorized 提升 `1.09x`，block shared 256 相对 online 提升 `1.23x`，register 256 相对 shared 256 提升 `1.05x`，latest 128 再提升约 `1.01x`。

最终 latest 128 相对 naive 提升 `44.44x`，相对 cuDNN 提升 `2.10x`。最后几项局部优化的收益已经很小，主要性能提升来自行内并行、block 级并行度和中间结果缓存位置的改变。

## 结果复现

编译并验证：

```bash
bash softmax/compile.sh
./softmax/build/softmax verify all 4096 1024
```

运行单个形状，或者覆盖默认计时参数：

```bash
./softmax/build/softmax benchmark all 4096 1024
./softmax/build/softmax benchmark block_register_latest_128 4096 1024 20 200 20
```

运行完整 latency 矩阵：

```bash
for batch in 512 1024 2048 4096 8192 16384; do
    for num_classes in 256 512 1024 2048 4096; do
        ./softmax/build/softmax benchmark all "$batch" "$num_classes"
    done
done
```

使用 Nsight Compute 采集单个 kernel：

```bash
sudo /usr/local/cuda/bin/ncu --set full ./softmax/build/softmax profile block_register_latest_128 4096 1024
```

`benchmark.sh` 用于批量生成 30 个形状的 NCU 报告，`summary.sh` 将报告转换为 Markdown 汇总：

```bash
bash softmax/benchmark.sh
bash softmax/summary.sh brief
bash softmax/summary.sh detail
```
