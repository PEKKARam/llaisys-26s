# Add：减少搬运指令，接受小尺寸收益有限

日期：2026-11-02。源码：[src/ops/add/nvidia/add_nvidia.cu](../../src/ops/add/nvidia/add_nvidia.cu)。这是实际实验记录，含未获益和未采用方案。环境、计时限制及整机版本见 [实验总表](README.md)；原始样本见 [operators.jsonl](results/operators.jsonl)。

## 算什么，原来为什么慢

逐元素计算 out[i] = a[i] + b[i]。模型在 Attention 和 MLP 后各做一次残差相加。

这个算子只加一次，却要读两个数组再写一个数组，计算很少。成对读写主要减少读写和地址计算指令，不能减少必须搬运的数据字节数。

## 版本与测试

### V0：任务开始的基础CUDA版

一个线程处理一个元素，BF16/FP16 每次读一个 16-bit 数并转成 float，相加后写回。

原始源码另外保存在 `baselines/v0/`。使用V0只表示历史基线，不代表其BF16中间计算顺序完全等于当前参考。

### V1：直观的数值对照版

沿用 V0；Add 没有需要单独修正的中间舍入步骤。V0/V1 是同一条 kernel 路径，测量差别来自运行状态。

### V2：当前优化版

对齐时每线程处理两个相邻元素：FP32 用 float2，BF16/FP16 用一次 32-bit 读写打包两个数，计算仍用 float。地址不对齐走标量回退；奇数长度单独处理最后一个元素。

共同测试：真实尺寸 1536、128×1536、1024×1024；另外在三种 dtype 下验证偏移一个元素的非对齐指针和 3×157 奇数尾部，要求结果精确一致。

V0/V1对每个下表BF16形状都执行正确性检查再计时；V2还对同一组形状执行FP16和FP32检查。注意：随机输入下容差通过，不代表所有真实模型激活逐位相同；整模型结果另行验证。

## 设备微基准：每个版本的真实数值

计时单位**微秒**，取多次CUDA Graph设备事件计时的中位数；这是开发基准中的graph replay，不代表引擎本身用了CUDA Graph。相同GPU/dtype/shape/seed；各原始记录含所有样本与library SHA。

| shape | V0 μs | V1 μs | V2 μs | V1/V2 |
| --- | ---: | ---: | ---: | ---: |
| [1, 1536] | 1.024 | 3.411 | 2.456 | 1.39× |
| [128, 1536] | 2.601 | 3.091 | 1.647 | 1.88× |
| [1024, 1024] | 7.375 | 7.430 | 5.287 | 1.41× |

V2正确性范围（每行包含上面全部形状）：

| dtype | 通过/总数 | 本组最大绝对误差 | atol / rtol |
| --- | ---: | ---: | --- |
| bf16 | 3/3 | 0 | 0.02 / 0.02 |
| f16 | 3/3 | 0 | 0.003 / 0.003 |
| f32 | 3/3 | 0 | 0.0001 / 0.0001 |

小于几微秒的kernel容易受时钟、launch等波动影响。V0/V1如果走同一个kernel，两组数不一样不能当成一次优化。这里没有给出与PyTorch的算子加速比：原始记录中的PyTorch计时包含Python边界，与native graph计时范围不同。

## 优化后整机怎样

全版本阶段表见 [总表](README.md)：所有V0、所有V1、所有V2，以及引擎工作区改动都有实际整机测量。下面再做单算子回退：保持其他算子V2、引擎E2，只把 add 切回V1。BF16、batch=1、prompt64、8个输出位置、固定token回放、warmup2/repeat15，真实权重。

| 设置 | TTFT中位数 ms | 每请求TPOT中位数 ms |
| --- | ---: | ---: |
| 全V2 | 13.211 | 15.059 |
| 仅本算子回退V1 | 12.432 | 11.027 |

原始整机记录见 [ablation.jsonl](results/ablation.jsonl)。这是跨进程探索性消融；主机干扰、未锁GPU时钟导致全V2在不同批次TPOT约11～15ms浮动。本批次甚至出现小算子回退后的TPOT低于全V2，**不据此声称回退提速或优化版整机退化**，也不计算小差异的因果收益。Linear/Attention这类明显量级差异有参考价值；小差异仍需交错配对与更严格环境控制。回放固定输入保证计算长度一致，但不代表HF生成质量已全部通过。

## 依据结果怎样判断，下一步是什么

Nsight 中残差 Add 合计只占约 0.4% kernel 时间，单个平均约 0.66 微秒。微基准的大数组有改善，小数组有波动，单算子不会决定整个模型速度。

如果继续做，应根据 trace 尝试 Add+RMSNorm 融合以减少 launch 和中间写回；这需要新增融合接口并检查 BF16 舍入。当前没有实现，也不写成已完成成果。

## 自己复现

```bash
# 参数在进程首次调用算子前确定；切版本重新启动进程
.venv/bin/python benchmark/cuda_experiments.py --op add --version 0 --dtype bf16
.venv/bin/python benchmark/cuda_experiments.py --op add --version 1 --dtype bf16
.venv/bin/python benchmark/cuda_experiments.py --op add --version 2 --dtype bf16
LLAISYS_CUDA_ADD_VERSION=1 .venv/bin/python benchmark/benchmark_infer.py \
  --prompt-tokens 64 --output-tokens 8 --mode replay
```

单独执行 `--op` 时随机数生成顺序不同于`--op all`，要完全复用总表输入生成顺序，请使用 `run_optimization_suite.py --phase ops`。进程退出前完成计时，不要一边安装动态库一边运行测试。
