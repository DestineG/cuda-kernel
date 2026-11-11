# GEMM 实现索引

| 文件 | 实现 | 使用入口 |
|---|---|---|
| [gemm_custom.cu](gemm_custom.cu) | BF16，cp.async + ldmatrix + mma.sync；原 practice 实现 | 本文下方的 compile/verify/benchmark_custom.sh |
| [gemm_bf16_sm90.cu](gemm_bf16_sm90.cu) | SM90 TMA/WGMMA，单缓冲 | 源码中的 launcher |
| [gemm_bf16_sm90_dbuffer.cu](gemm_bf16_sm90_dbuffer.cu) | SM90 TMA/WGMMA，双缓冲 | 源码中的 launcher |
| [gemm_bf16_sm120.cu](gemm_bf16_sm120.cu) | SM120 单缓冲 | 源码中的 launcher |
| [gemm_bf16_sm120_dbuffer.cu](gemm_bf16_sm120_dbuffer.cu) | SM120 双缓冲 | 源码中的 launcher |

下面的配置、限制、结果和脚本仅对应 `gemm_custom.cu`。不同架构源码分别构建，不能用这些历史数字代表 SM90/SM120 实现。原有本地 `docs/` 归档保持不变，默认不进入 Git。

---

# CUDA BF16 Tensor Core GEMM

custom GEMM 实现。输入 `A`、`B` 为 row-major BF16，输出 `C` 为 row-major FP32，累加器为 FP32：

```text
C[M, N] = A[M, K] * B[K, N]
```

当前 aligned kernel 限制：

```text
M % 128 == 0
N % 128 == 0
K % 64 == 0
```

核心计算不使用 WMMA、CUTLASS 或 cuBLAS；cuBLAS 只用于 correctness 和性能对比。

## 实现结构

配置为：

```text
BM=128, BN=128, BK=64
WM=32,  WN=64
256 threads/block
2 shared-memory stages
65536 bytes dynamic shared memory/block
```

每个 warp 计算 `32x64` 的 C tile，每个 K tile 展开为 16 次：

```text
mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
```

数据流为：

```text
Global -> cp.async 16B -> swizzled Shared
       -> ldmatrix -> BF16 MMA -> FP32 accumulator -> Global FP32
```

Shared layout 以每个 128B segment 为单位做 8 个 16B chunk 的 cyclic shift：

```text
physical_chunk = (logical_chunk + (row % 8)) % 8
```

B tile 的 256B 行会拆成两个独立 128B segment，不会跨 segment 或跨行移动。

## 正确性

cuBLAS 使用 BF16 输入和 FP32 accumulate 作为 reference。验证条件为：

```text
max_absolute <= 1e-3
relative_l2 <= 1e-5
```

custom kernel 已通过：

```text
128x128x64
1024x1024x1024
2048x2048x2048
```

并通过 `compute-sanitizer --tool memcheck`，无内存错误。

## Benchmark

测试使用 CUDA events，默认 warmup 100 次；1024/2048 和矩形尺寸执行 1000 次，4096 执行 100 次。完整测试矩阵、结果和分析直接记录在本文档中。

| M | N | K | Kernel | Time (ms) | TFLOPS |
| ---: | ---: | ---: | --- | ---: | ---: |
| 1024 | 1024 | 1024 | cuBLAS | 0.0177 | 121.281 |
| 1024 | 1024 | 1024 | naive | 0.4168 | 5.153 |
| 1024 | 1024 | 1024 | custom | 0.0355 | 60.440 |
| 2048 | 2048 | 2048 | cuBLAS | 0.1013 | 169.554 |
| 2048 | 2048 | 2048 | naive | 3.2146 | 5.344 |
| 2048 | 2048 | 2048 | custom | 0.1362 | 126.171 |
| 4096 | 4096 | 4096 | cuBLAS | 0.9106 | 150.939 |
| 4096 | 4096 | 4096 | custom | 1.0406 | 132.074 |
| 2048 | 1024 | 2048 | cuBLAS | 0.0529 | 162.265 |
| 2048 | 1024 | 2048 | custom | 0.0691 | 124.329 |
| 1024 | 2048 | 2048 | cuBLAS | 0.0529 | 162.284 |
| 1024 | 2048 | 2048 | custom | 0.0692 | 124.197 |

custom 相对 cuBLAS 的吞吐率约为：

```text
1024^3:             49.8%
2048^3:             74.4%
4096^3:             87.5%
2048x1024x2048:     76.6%
1024x2048x2048:     76.5%
```

随着矩阵增大，固定的 launch、shared-memory 和流水线开销被摊薄，custom kernel 的吞吐率逐渐接近 cuBLAS。当前主要差距来自寄存器占用较高、每个 CTA 使用 64KB shared memory 导致 occupancy 受限，以及尚未针对 Ada 架构做更激进的访存和 MMA pipeline 调度。

编译时 ptxas 报告：

```text
custom registers/thread: 162
register spills: 0
dynamic shared memory/block: 65536 bytes
```

## 复现

编译：

```bash
bash gemm/compile_custom.sh
```

correctness：

```bash
bash gemm/verify_custom.sh
```

benchmark：

```bash
bash gemm/benchmark_custom.sh
```

自定义测试参数：

```bash
WARMUP=100 ITERATIONS=1000 LARGE_ITERATIONS=100 bash gemm/benchmark_custom.sh
```

单独运行某个 kernel：

```bash
./gemm/build/gemm_custom verify custom 2048 2048 2048
./gemm/build/gemm_custom benchmark custom 2048 2048 2048 100 1000
./gemm/build/gemm_custom profile custom 2048 2048 2048
```
