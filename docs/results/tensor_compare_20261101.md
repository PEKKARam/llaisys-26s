# Qwen2 三个中间检查点对照：2026-11-01

本次只增加默认关闭的诊断导出，未改变算子公式、dtype 分支或 kernel 启动参数。已重新编译并安装 NVIDIA release 共享库。

## 复现

```bash
xmake -y
xmake install -y
.venv/bin/python benchmark/compare_tensors.py
```

固定输入为 WikiText-2 `row_3_len_64`（64 tokens）。模型为本地 DeepSeek-R1-Distill-Qwen-1.5B，batch=1，BF16，HF eager attention，执行一次 prefill，读取一个新 token。HF 和 native 模型依次加载。三个检查点按 `[token, hidden]` 对齐后比较完整张量，而不是只比较前几个打印值。

HF 原始三个输出均为 BF16，hook 去掉 batch 维后转换到 CPU float32 保存。Native 原始 BF16 字节经同步/D2H 复制后写入二进制文件；shape/dtype/byte order 单独记录，Python 按元信息解析。误差统计以保存的数值为基础，不恢复原计算已损失的精度。

## 实测结果

| checkpoint | shape | 相同元素比例 | 最大绝对误差 | 平均绝对误差 | RMSE | 非有限数 HF/native |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| embedding | [64, 1536] | 100.00% | 0 | 0 | 0 | 0/0 |
| layer0.input_norm | [64, 1536] | 73.70% | 0.015625 | 0.0002541443 | 0.0008134123 | 0/0 |
| layer0.output | [64, 1536] | 1.41% | 3.53125 | 0.17498963 | 0.26661478 | 0/0 |

三个检查点均无 NaN/Inf。HF 首 token 为 26705，native 为 5055，与此前该样本的 token 对照一致。

Embedding 的元素数值完全相同，支持该输入的 embedding 查表与数据加载一致；这不代表其他权重均已验证。第 0 层 RMSNorm 是这三个观察点中首次出现非零差异的位置；第 0 层结束时误差已明显增大。

仅有三个检查点，不能证明全部差异由 RMSNorm 导致，更不能直接认定是哪行代码有 bug。下一步可用完全相同的 Embedding 与 Norm 权重做隔离 RMSNorm 对照，区分 FP32 累加顺序、中间 BF16 舍入与权重/参数问题；随后增加 QKV、RoPE、attention 等检查点，定位层内额外偏差。此报告不提供性能结论。

完整产物在本机 `cache/tensor_debug/row_3_len_64_z5w3_3mj/`，该缓存目录不提交 Git。`summary.json` 包含最大/平均绝对误差、RMSE、相同元素比例和首个差异坐标及数值。
