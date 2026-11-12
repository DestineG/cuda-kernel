# BF16 输出与双 consumer 实验

独立实验副本，不替换 flashattn 根目录的 FP32 输出实现。

- [double_v_output](double_v_output/README.md)：double_V 的 FP32/BF16 输出对照；[结果摘录](double_v_output/results/findings.md)。
- [v_kv_bf16_pingpong](v_kv_bf16_pingpong/README.md)：V/KV 两条线的 BF16 single、dual、ping-pong 六个实现；[结果摘录](v_kv_bf16_pingpong/results/findings.md)。

每个实验自带 tests/、固定输入、源码 diff 与复现配置。results/ 仅跟踪摘要、入选参数和归档校验记录；完整原始数据、HTML、CSV、日志及官方包保留在实验 ZIP，不将大文件加入源码仓库。结果摘录中的完整报告链接需要从 ZIP 解压目录查看。

ping-pong 为现有 overlap 上的 QK-ready 交接实验，并非官方 FA3 的逐行复刻。两个 consumer 共享 K/V、独立计算 Q 行，CTA 行数为每 WG Br 的两倍。因果模式遍历共同 KV 范围。双 consumer 寄存器目标必须符合源码中的 CTA 初始寄存器池限制；不能仅按整 SM 的 65536 估算。已保留最初超预算停滞的诊断说明。
