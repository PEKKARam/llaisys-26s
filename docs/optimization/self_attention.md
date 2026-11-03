# Self-Attention：同一份 QK 和 Softmax 不要为每个输出通道重算

文档更新：2026-11-03；实验测量：2026-11-02，本次没有新增实验。源码：[src/ops/self_attention/nvidia/self_attention_nvidia.cu](../../src/ops/self_attention/nvidia/self_attention_nvidia.cu)。这是实际实验记录，含未获益和未采用方案。环境、计时限制及整机版本见 [实验总表](README.md)；原始样本见 [operators.jsonl](results/operators.jsonl)。

## 算什么，原来为什么慢

对每个Q头找到共享KV头，计算 softmax(QKᵀ/√D+causal mask)V。QL是新token数，KL包含历史；第qp行可见数量为 KL-QL+qp+1。

最主要的收益来自消除重复计算，而不是微调一个load指令。归约让线程分摊点积与softmax；缓存中间结果让多个输出通道共用它们。

## 版本与测试

### V0：任务开始的基础CUDA版

每个输出通道线程各自算整条QK，再为softmax重算一遍QK；同一个头的128个输出通道重复做基本相同的打分工作，复杂度中多出不必要的D因子。所有中间分数/概率留在float。

原始源码另外保存在 `baselines/v0/`。使用V0只表示历史基线，不代表其BF16中间计算顺序完全等于当前参考。

### V1：直观的数值对照版

保持串行直观算法，显式处理打分、缩放、softmax概率的低精度舍入，支持V的head_dim与Q/K不同。为了单独量化概率再乘V，有三轮打分循环，因此可能比V0更慢；这不是失败的数据要隐藏。

### V2：当前优化版

拆成3个kernel：Warp协作求每个Q/K点积；每行block归约做稳定softmax；PV按输出通道并行。一次算好分数和概率写入可复用FP32 scratch，避免为每个输出通道重做QK。因果mask、GQA头映射和缓存偏移均保留。

共同测试：QL/KL覆盖1/128、32/32、128/128、1/1024、5/11，Q头/KV头12/2或4/2；三种dtype；额外检查Dqk=8、Dv=3，并修改未来K/V验证前面的query不受影响。旧self_attention脚本也通过。

V0/V1对每个下表BF16形状都执行正确性检查再计时；V2还对同一组形状执行FP16和FP32检查。注意：随机输入下容差通过，不代表所有真实模型激活逐位相同；整模型结果另行验证。

## 设备微基准：每个版本的真实数值

计时单位**微秒**，取多次CUDA Graph设备事件计时的中位数；这是开发基准中的graph replay，不代表引擎本身用了CUDA Graph。相同GPU/dtype/shape/seed；各原始记录含所有样本与library SHA。

| shape | V0 μs | V1 μs | V2 μs | V1/V2 |
| --- | ---: | ---: | ---: | ---: |
| [1, 128, 12, 2, 128] | 689.888 | 930.784 | 9.881 | 94.20× |
| [32, 32, 12, 2, 128] | 359.776 | 603.712 | 16.058 | 37.59× |
| [128, 128, 12, 2, 128] | 4705.888 | 6966.016 | 171.123 | 40.71× |
| [1, 1024, 12, 2, 128] | 6392.608 | 9234.848 | 53.820 | 171.59× |
| [5, 11, 4, 2, 8] | 11.874 | 11.750 | 4.241 | 2.77× |

V2正确性范围（每行包含上面全部形状）：

| dtype | 通过/总数 | 本组最大绝对误差 | atol / rtol |
| --- | ---: | ---: | --- |
| bf16 | 5/5 | 0.00012207031 | 0.02 / 0.02 |
| f16 | 5/5 | 3.0517578e-05 | 0.003 / 0.003 |
| f32 | 5/5 | 5.9604645e-08 | 0.0001 / 0.0001 |

小于几微秒的kernel容易受时钟、launch等波动影响。V0/V1如果走同一个kernel，两组数不一样不能当成一次优化。这里没有给出与PyTorch的算子加速比：原始记录中的PyTorch计时包含Python边界，与native graph计时范围不同。

## 优化后整机怎样

全版本阶段表见 [总表](README.md)：所有V0、所有V1、所有V2，以及引擎工作区改动都有实际整机测量。下面再做单算子回退：保持其他算子V2、引擎E2，只把 self_attention 切回V1。BF16、batch=1、prompt64、8个输出位置、固定token回放、warmup2/repeat15，真实权重。

| 设置 | TTFT中位数 ms | 每请求TPOT中位数 ms |
| --- | ---: | ---: |
| 全V2 | 13.211 | 15.059 |
| 仅本算子回退V1 | 62.631 | 24.407 |

原始整机记录见 [ablation.jsonl](results/ablation.jsonl)。这是跨进程探索性消融；主机干扰、未锁GPU时钟导致全V2在不同批次TPOT约11～15ms浮动。本批次甚至出现小算子回退后的TPOT低于全V2，**不据此声称回退提速或优化版整机退化**，也不计算小差异的因果收益。Linear/Attention这类明显量级差异有参考价值；小差异仍需交错配对与更严格环境控制。回放固定输入保证计算长度一致，但不代表HF生成质量已全部通过。

## 依据结果怎样判断，下一步是什么

QL=128、KL=128时，V1到V2的设备时间改善非常明显，长历史decode同样改善。优化后本次trace中QK、softmax、PV合计约3.6%kernel时间；继续优化应看更长上下文是否重新成为热点。

目前仍显式保存scores，空间O(QL×HQ×KL)，没有FlashAttention的分块online softmax，也没有PagedAttention。后续可在数学验证充分后尝试小范围融合或online softmax；长上下文需另测，不能用128长度的结果保证131072长度可用。

## 面试讲解与自测

可以先讲：“基础版为每个输出通道重复算QK和softmax。当前把一次Attention拆成QK、softmax、PV三个kernel，让所有输出通道复用同一行概率。”这里的三个kernel依次提交到同一个stream，不是融合Attention，也没有online softmax。

**直观小例子：** 假设一个query对两个key的softmax概率已经算出为`[0.25,0.75]`，两个value是`[2,4]`和`[6,8]`，输出就是`[5,7]`。计算输出第0、1通道都使用同一对概率，无需各自重算一次QK与softmax；通道变成128个时，仍然只有这一份概率分布。这里展示的是复用关系，数值用于手算，不是新增benchmark。

| 阶段 | 线程与block映射 | 数据流及数值 |
| --- | --- | --- |
| QK | 每warp负责一个(query位置,Q头,key位置)点积；每block256线程，即8个点积 | lane沿head_dim以32为步长读取，相邻lane访问相邻通道；FP32归约，点积及缩放后各按输入dtype舍入，结果写入FP32 scores |
| softmax | 每个(query位置,Q头)一行，一个256线程block | 对KL个分数做max与sum归约，使用减最大值的稳定计算；概率按输入dtype舍入后原地覆盖scores |
| PV | 每行一个128线程block，线程沿Dv分工 | 各线程遍历可见历史位置；同一位置的相邻线程读取相邻V通道，复用已算概率，FP32累加后写目标dtype |

FP32 scratch的存储类型不表示分数和概率一直保留完整FP32精度：代码用`rounded<MODE>`显式控制中间舍入。当前仍分配`QL×HQ×KL×4`字节scores，包含被mask的格子；没有借三角mask省掉这部分存储。

scores只是这次Attention的临时结果，不是跨token保留历史的KV Cache。当前scratch按线程/设备保留容量，后续调用顺序复用，softmax在原地址把分数改写为概率；它既不是每层永久保存一份完整分数矩阵，也不是算完就必定释放显存。长度增大时仍可能扩容，相关分配不计入模型的Tensor allocator统计。

**画图自测：** 取QL=2、KL=5、HQ=4、HK=2，历史长度为3。画2×5可见矩阵：第0个新query可看key 0～3，第1个可看0～4；Q头0/1共享KV头0，Q头2/3共享KV头1。scores共有2×4×5=40个float，即160字节；QK启动5个256线程block，softmax与PV各8个block。再把Dv改成3，输出应为[2,4,3]，可见范围和GQA映射都不变。

**典型追问：** 为什么比原版快很多，却不叫FlashAttention？收益主要来自消除每个输出通道的重复打分；当前仍显式写出和读取完整scores，长上下文的存储与访存成本仍在。PV也仍沿历史串行累加，不能宣称整个Attention都已充分并行。需要根据更长上下文的测量再决定分块或融合方案。

代码入口是[Attention源码](../../src/ops/self_attention/nvidia/self_attention_nvidia.cu)中的`attn_run`及三个`*_parallel_v2` kernel；GQA、历史偏移和Dv不同的检查见[CUDA边界检查](../../test/test_cuda_edges.py)。项目层面的优化归因见[主问答](RESUME_AND_INTERVIEW.md)。

## 自己复现

```bash
# 参数在进程首次调用算子前确定；切版本重新启动进程
.venv/bin/python benchmark/cuda_experiments.py --op self_attention --version 0 --dtype bf16
.venv/bin/python benchmark/cuda_experiments.py --op self_attention --version 1 --dtype bf16
.venv/bin/python benchmark/cuda_experiments.py --op self_attention --version 2 --dtype bf16
LLAISYS_CUDA_SELF_ATTENTION_VERSION=1 .venv/bin/python benchmark/benchmark_infer.py \
  --prompt-tokens 64 --output-tokens 8 --mode replay
```

单独执行 `--op` 时随机数生成顺序不同于`--op all`，要完全复用总表输入生成顺序，请使用 `run_optimization_suite.py --phase ops`。进程退出前完成计时，不要一边安装动态库一边运行测试。
