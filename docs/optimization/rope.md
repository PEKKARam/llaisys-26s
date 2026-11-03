# RoPE：一对通道共用一次 sin/cos

日期：2026-11-02。源码：[src/ops/rope/nvidia/rope_nvidia.cu](../../src/ops/rope/nvidia/rope_nvidia.cu)。这是实际实验记录，含未获益和未采用方案。环境、计时限制及整机版本见 [实验总表](README.md)；原始样本见 [operators.jsonl](results/operators.jsonl)。

## 算什么，原来为什么慢

对同一头前半/后半对应的 a、b：out_a=a*cosθ-b*sinθ，out_b=b*cosθ+a*sinθ；θ=position×base^(-2j/D)。是前后半配对，不是相邻两个元素配对。

同一对通道的角度一样，重复算 sin/cos 没有意义。让线程一次负责两个输出，可以省一半成对角度计算，也省掉重复读取。

## 版本与测试

### V0：任务开始的基础CUDA版

每个输出元素分别读取自己和另一半、计算频率和 sin/cos；同一对通道做两遍相同的角度计算。中间乘法不显式按输入 dtype 舍入。

原始源码另外保存在 `baselines/v0/`。使用V0只表示历史基线，不代表其BF16中间计算顺序完全等于当前参考。

### V1：直观的数值对照版

仍逐输出计算，但把 cos/sin 及乘积按低精度语义显式舍入，最后再加减，靠近HF eager运算顺序。

### V2：当前优化版

每线程处理一对前后半通道，调用一次 sincosf，读 a/b 后同时写两个输出。角度计算减少，但仍保留 dtype 舍入要求；不是把RoPE融合进Attention。

共同测试：T=1/128、12个头、head_dim128；非规则形状[5,2,10]；三种dtype；位置从19开始以覆盖非零历史偏移；独立测试位置0要求输出精确等于输入。旧测试覆盖其他位置与shape。

V0/V1对每个下表BF16形状都执行正确性检查再计时；V2还对同一组形状执行FP16和FP32检查。注意：随机输入下容差通过，不代表所有真实模型激活逐位相同；整模型结果另行验证。

## 设备微基准：每个版本的真实数值

计时单位**微秒**，取多次CUDA Graph设备事件计时的中位数；这是开发基准中的graph replay，不代表引擎本身用了CUDA Graph。相同GPU/dtype/shape/seed；各原始记录含所有样本与library SHA。

| shape | V0 μs | V1 μs | V2 μs | V1/V2 |
| --- | ---: | ---: | ---: | ---: |
| [1, 12, 128] | 3.445 | 2.076 | 1.682 | 1.23× |
| [128, 12, 128] | 6.246 | 7.569 | 4.349 | 1.74× |
| [5, 2, 10] | 1.460 | 2.680 | 1.439 | 1.86× |

V2正确性范围（每行包含上面全部形状）：

| dtype | 通过/总数 | 本组最大绝对误差 | atol / rtol |
| --- | ---: | ---: | --- |
| bf16 | 3/3 | 0 | 0.02 / 0.02 |
| f16 | 3/3 | 0 | 0.003 / 0.003 |
| f32 | 3/3 | 5.9604645e-08 | 0.0001 / 0.0001 |

小于几微秒的kernel容易受时钟、launch等波动影响。V0/V1如果走同一个kernel，两组数不一样不能当成一次优化。这里没有给出与PyTorch的算子加速比：原始记录中的PyTorch计时包含Python边界，与native graph计时范围不同。

## 优化后整机怎样

全版本阶段表见 [总表](README.md)：所有V0、所有V1、所有V2，以及引擎工作区改动都有实际整机测量。下面再做单算子回退：保持其他算子V2、引擎E2，只把 rope 切回V1。BF16、batch=1、prompt64、8个输出位置、固定token回放、warmup2/repeat15，真实权重。

| 设置 | TTFT中位数 ms | 每请求TPOT中位数 ms |
| --- | ---: | ---: |
| 全V2 | 13.211 | 15.059 |
| 仅本算子回退V1 | 13.222 | 13.163 |

原始整机记录见 [ablation.jsonl](results/ablation.jsonl)。这是跨进程探索性消融；主机干扰、未锁GPU时钟导致全V2在不同批次TPOT约11～15ms浮动。本批次甚至出现小算子回退后的TPOT低于全V2，**不据此声称回退提速或优化版整机退化**，也不计算小差异的因果收益。Linear/Attention这类明显量级差异有参考价值；小差异仍需交错配对与更严格环境控制。回放固定输入保证计算长度一致，但不代表HF生成质量已全部通过。

## 依据结果怎样判断，下一步是什么

Nsight 中448次RoPE合计约0.9%的kernel时间，平均约1.4微秒。单算子prefill有改善，整机小幅差异容易被主机调度和时钟波动覆盖，不给出可靠的独立整机加速倍数。

真实热点若转向launch开销，可考虑把RoPE与KV写入进一步融合。本轮已让K的RoPE直接输出缓存，省掉D2D；尚未实现QKV投影与RoPE的单kernel融合。

## 自己复现

```bash
# 参数在进程首次调用算子前确定；切版本重新启动进程
.venv/bin/python benchmark/cuda_experiments.py --op rope --version 0 --dtype bf16
.venv/bin/python benchmark/cuda_experiments.py --op rope --version 1 --dtype bf16
.venv/bin/python benchmark/cuda_experiments.py --op rope --version 2 --dtype bf16
LLAISYS_CUDA_ROPE_VERSION=1 .venv/bin/python benchmark/benchmark_infer.py \
  --prompt-tokens 64 --output-tokens 8 --mode replay
```

单独执行 `--op` 时随机数生成顺序不同于`--op all`，要完全复用总表输入生成顺序，请使用 `run_optimization_suite.py --phase ops`。进程退出前完成计时，不要一边安装动态库一边运行测试。
