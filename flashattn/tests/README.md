# FA3 tests

统一测试仓库内全部10个FA3实现。测试工具不改写kernel计算，原launcher和预编码TMA入口分别测量。运行环境需要Linux、CUDA12.4、Hopper GPU、PyTorch、NumPy及官方flash_attn_3。

```bash
# 从flashattn/执行，或使用测试入口绝对路径。
python tests/run.py inspect
python tests/run.py all --config tests/configs/full.json --name all-fa3
python tests/run.py resume --run /path/to/runs/<run-id>
python tests/run.py report --run /path/to/runs/<run-id>
python tests/run.py archive --run /path/to/runs/<run-id>
```

`--run-root /实验盘/fa3-runs` 将大型产物放到实验盘；默认是`tests/runs/`。所有源码、配置和测试工具在创建run时快照；恢复时检查源码/配置哈希。不要编辑已运行快照或修改配置后继续使用原run。

## 文件职责

- `configs/kernels.json`：10个kernel的文件名、符号、K/V槽数、barrier大小。
- `configs/full.json`：搜索范围、计时、阈值、超时、编译参数。
- `cases/`：固定正确性/性能输入，包含packed、GQA、因果未对齐和缓冲回绕。
- `adapters/`：不修改原CUDA源码的调用包装；新增ring版本按实际槽数分配共享内存。
- `references/`：CPU FP64、显式GPU FP32、官方FA3、PyTorch私有ATen varlen、公开SDPA逐序列Flash。
- `runners/`：编译、正确性、搜索、sanitizer门槛、性能执行和状态恢复。
- `reporting/`：只从结构化原始结果生成CSV和HTML，不执行GPU测试。

## 运行规则

每次独立run包含manifest、源码/工具快照、完整配置、编译日志/命令/资源、任务状态、每次尝试的原始记录和日志。成功且输入哈希一致的任务恢复时跳过；失败任务重新尝试使用新文件，旧结果不会覆盖。每个GPU测试序列持有主机锁；编译可并发，验证和计时串行。

默认20个关闭/40+232实例先完整正确性和同步检查。数值、guard、超时失败停止执行；普通搜索候选超共享内存上限记录UNSUPPORTED。搜索入选者重新跑完整CPU检查和四种sanitizer，失败候选记录REJECTED并选择下一名；不得关闭WGMMA检查来得到“通过”。共同对照失败则停止处理。

分阶段搜索每个kernel、每个D/causal独立进行：六种tile与关闭/默认重分配；胜出tile上的producer/consumer网格；前两种tile、较好寄存器参数和可行Q/KV/O box组合。不是全笛卡尔积穷举，实际覆盖在导出的配置表中。D和head数是输入条件，不作为优化参数。

主报告`report.html`：左为输入条件，中为kernel/参数，右为实际状态、GPU中位数/范围/TFLOP/s及launcher wall time。无加速比或相对耗时。支持多选、输入/参数/状态联动筛选、固定/调优/搜索视图、排序和当前筛选CSV导出。官方没有自写模板参数，使用空值显示。

归档保留复现所需源码、配置、测试工具、命令、原始数据、日志和精确安装的FA3包；排除可重建的自写二进制、参考缓存、下载工具链。HTML/CSV来自统一记录，任何失败/UNSUPPORTED均可查看。

## 比较条件

自写输出FP32，官方/PyTorch输出BF16。官方FA3采用`num_splits=0`自动选择，`pack_gqa=None`采用默认选择。GPU时间是热缓存CUDA graph重复调用，launcher wall time单列且包含Python与构造开销。PyTorch ATen varlen是版本特定私有API；逐序列SDPA含多次调用及packed输出组装，单独标识。右下对齐因果；KV零不写输出单列，不计为attention正确性通过。完整结果不代表所有输入或所有硬件路径正确。
