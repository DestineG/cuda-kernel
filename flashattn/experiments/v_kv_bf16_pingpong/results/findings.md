# V / KV BF16 双 consumer 与 ping-pong

六份独立 CUDA 文件已经在 H800 完成编译、正确性、sanitizer、独立参数搜索和完整性能测试。四号机仓库未修改。V-double 保持 BF16 输出；KV-double 先改为 BF16，再分别添加双 consumer 与 ping-pong。

single：1 producer WG + 1 consumer WG，256线程。dual：1 producer + 2 consumers，384线程，共享K/V，无交替turn限制。pingpong：与dual同样划分，增加QK-ready交接。所有版本使用FP32累加、BF16输出。每个consumer负责自己的Q行、softmax状态和O累加器；两者共同释放K/V槽。因果模式遍历两个Q块的KV范围并集，会增加较早Q行的masked工作。

该调度在QK结果可读、开始mask/max/exp/sum前交给另一WG；稳态自己的上一PV仍可能在飞行中，最后PV也参与一次turn。这是基于现有overlap代码的实验调度，不是官方FA3的逐行复刻。数值与耗时测试不等同于硬件执行时间线证明，本次没有声称通过profiling验证了具体重叠比例。

## 同参数实测

B=1，Hq=Hkv=16，D=128。每WG Br=64、Bc=64、关闭寄存器重分配，五轮CUDA graph中位数，单位μs。single CTA负责64行；dual/pingpong CTA负责128行。dual与pingpong间CTA形状相同，可以观察turn调度影响；single与dual还包含CTA划分、K/V复用、寄存器和shared占用变化。

| Q=KV | causal | V single | V dual | V pingpong | KV single | KV dual | KV pingpong | 官方FA3 |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| 2048 | False | 189.201 | 179.560 | 194.990 | 183.028 | 180.828 | 178.402 | 99.794 |
| 2048 | True | 120.908 | 115.469 | 125.604 | 122.613 | 118.426 | 117.189 | 53.745 |
| 8192 | False | 2882.656 | 2223.780 | 2475.252 | 2537.556 | 2145.348 | 2168.088 | 1127.836 |
| 8192 | True | 1351.656 | 1216.400 | 1314.272 | 1433.076 | 1305.840 | 1330.520 | 614.320 |

这个样例中V普通dual优于V pingpong，不能把“增加两个consumer”与“交替调度”都归因为ping-pong收益。KV两者更接近，不能只看一个小差值就认定稳定收益。2048因果输入反序确认：V single/dual/pingpong=120.111/114.172/124.494μs，KV=121.687/117.289/117.221μs。更多尺寸、寄存器40/232共同对照和独立调优结果见HTML与CSV。独立调优按每D/causal三个代表场景选择，不保证每个输入都是该输入的最优配置。

## 检查与诊断

修正后run状态COMPLETE：157组正确性输入；12个默认关闭/开启重分配实例及最终入选配置通过memcheck/racecheck/synccheck/initcheck；268组性能输入、12组反序确认，4690条性能记录全部PASS。268个编译配置。120条搜索记录因共享内存容量标为UNSUPPORTED，44条寄存器组合排除记录（原始尝试次数，可能含重复配置）单列。零KV不写回是独立观测。

初始run在寄存器目标32/240停滞。单独诊断复现32/240和48/232超时、40/232通过；前两种要求65536个寄存器，384线程编译初始分配168/线程，共64512。setmaxnreg不能把CTA目标当成整SM可自由取得的65536。双consumer源码现在静态限制128*(P+2*C)<=64512，搜索提前排除，重新创建完整源码快照从头测试。最终全部已编译重分配实例的目标又根据各自ptxas资源记录核对初始CTA池，均通过。保留了初始失败状态、诊断、编译日志，不将其混入成功性能结果。

V和KV两个pingpong副本都经过重新编译、两组因果ring输入复现。归档内保留完整源码、原版副本及哈希、diff、配置、编译命令、原始结果、工具、日志与精确官方FA3包。排除可重建自写二进制、参考缓存和工具链。

[可筛选报告](report.html) · [逐输入并排CSV](comparison.csv) · [完整记录CSV](results.csv) · [入选参数](selected.csv)

H800实验目录：`/root/autodl-tmp/fa3-v-kv-pingpong/`。源码：`flashattn/fa3_{v,kv}_bf16_{single,dual,pingpong}.cu`。有效run：`runs/20261112-083844-v-kv-bf16-pingpong-poolfix/`。
