# Embedding：直接复制原始位，不做多余数值转换

日期：2026-11-02。源码：[src/ops/embedding/nvidia/embedding_nvidia.cu](../../src/ops/embedding/nvidia/embedding_nvidia.cu)。这是实际实验记录，含未获益和未采用方案。环境、计时限制及整机版本见 [实验总表](README.md)；原始样本见 [operators.jsonl](results/operators.jsonl)。

## 算什么，原来为什么慢

input_ids[t] 决定取权重矩阵第几行，把 weight[id,:] 复制成 hidden[t,:]。

Embedding 没有乘加数学运算，只要把选中的行搬过去。直接复制位可以省下类型转换和部分索引指令。

## 版本与测试

### V0：任务开始的基础CUDA版

一个线程复制一个元素，但先按 dtype 读成 float，再按 dtype 写回；每个元素还要算所在行与列。

原始源码另外保存在 `baselines/v0/`。使用V0只表示历史基线，不代表其BF16中间计算顺序完全等于当前参考。

### V1：直观的数值对照版

沿用 V0，没有改变数学语义。

### V2：当前优化版

每个 token 一块，按 int4 一次搬 16 字节原始数据，不经过 float 转换。行字节数与指针满足 16 字节对齐才使用；否则保留标量实现。int4 在这里是搬运载体，不是把模型量化成 INT4。

共同测试：BF16/FP16/FP32 测 T=1/128、宽度1536，以及宽度155的回退；边界检查使用非对齐权重和输出，要求复制结果精确一致。token ID 必须有效，模型 API 会检查词表范围。

V0/V1对每个下表BF16形状都执行正确性检查再计时；V2还对同一组形状执行FP16和FP32检查。注意：随机输入下容差通过，不代表所有真实模型激活逐位相同；整模型结果另行验证。

## 设备微基准：每个版本的真实数值

计时单位**微秒**，取多次CUDA Graph设备事件计时的中位数；这是开发基准中的graph replay，不代表引擎本身用了CUDA Graph。相同GPU/dtype/shape/seed；各原始记录含所有样本与library SHA。

| shape | V0 μs | V1 μs | V2 μs | V1/V2 |
| --- | ---: | ---: | ---: | ---: |
| [1, 1536] | 0.965 | 0.961 | 1.269 | 0.76× |
| [128, 1536] | 2.700 | 2.683 | 1.328 | 2.02× |
| [128, 155] | 1.056 | 1.058 | 3.930 | 0.27× |

V2正确性范围（每行包含上面全部形状）：

| dtype | 通过/总数 | 本组最大绝对误差 | atol / rtol |
| --- | ---: | ---: | --- |
| bf16 | 3/3 | 0 | 0.02 / 0.02 |
| f16 | 3/3 | 0 | 0.003 / 0.003 |
| f32 | 3/3 | 0 | 0.0001 / 0.0001 |

小于几微秒的kernel容易受时钟、launch等波动影响。V0/V1如果走同一个kernel，两组数不一样不能当成一次优化。这里没有给出与PyTorch的算子加速比：原始记录中的PyTorch计时包含Python边界，与native graph计时范围不同。

## 优化后整机怎样

全版本阶段表见 [总表](README.md)：所有V0、所有V1、所有V2，以及引擎工作区改动都有实际整机测量。下面再做单算子回退：保持其他算子V2、引擎E2，只把 embedding 切回V1。BF16、batch=1、prompt64、8个输出位置、固定token回放、warmup2/repeat15，真实权重。

| 设置 | TTFT中位数 ms | 每请求TPOT中位数 ms |
| --- | ---: | ---: |
| 全V2 | 13.211 | 15.059 |
| 仅本算子回退V1 | 12.840 | 11.210 |

原始整机记录见 [ablation.jsonl](results/ablation.jsonl)。这是跨进程探索性消融；主机干扰、未锁GPU时钟导致全V2在不同批次TPOT约11～15ms浮动。本批次甚至出现小算子回退后的TPOT低于全V2，**不据此声称回退提速或优化版整机退化**，也不计算小差异的因果收益。Linear/Attention这类明显量级差异有参考价值；小差异仍需交错配对与更严格环境控制。回放固定输入保证计算长度一致，但不代表HF生成质量已全部通过。

## 依据结果怎样判断，下一步是什么

长 prompt 的规则行有改善；单 token 只有几 KB 数据，原版已经很短，不能保证新版更快。宽度155实际回到同一个旧 kernel，表里的差异不能解读成算法收益。Nsight 中整个请求的 Embedding 约占0.01%，不是热点。

保留对齐回退即可。与其继续优化一次3KB复制，不如先处理 Linear 或 launch 开销；量化权重的 Embedding 需要另设计格式，当前没有做。

## 自己复现

```bash
# 参数在进程首次调用算子前确定；切版本重新启动进程
.venv/bin/python benchmark/cuda_experiments.py --op embedding --version 0 --dtype bf16
.venv/bin/python benchmark/cuda_experiments.py --op embedding --version 1 --dtype bf16
.venv/bin/python benchmark/cuda_experiments.py --op embedding --version 2 --dtype bf16
LLAISYS_CUDA_EMBEDDING_VERSION=1 .venv/bin/python benchmark/benchmark_infer.py \
  --prompt-tokens 64 --output-tokens 8 --mode replay
```

单独执行 `--op` 时随机数生成顺序不同于`--op all`，要完全复用总表输入生成顺序，请使用 `run_optimization_suite.py --phase ops`。进程退出前完成计时，不要一边安装动态库一边运行测试。
