# RMSNorm：让一整块线程一起算一行

文档更新：2026-11-03；实验测量：2026-11-02，本次没有新增实验。源码：[src/ops/rms_norm/nvidia/rms_norm_nvidia.cu](../../src/ops/rms_norm/nvidia/rms_norm_nvidia.cu)。这是实际实验记录，含未获益和未采用方案。环境、计时限制及整机版本见 [实验总表](README.md)；原始样本见 [operators.jsonl](results/operators.jsonl)。

## 算什么，原来为什么慢

y = (x / sqrt(mean(x²)+eps)) × weight。这里不减均值，不是 LayerNorm。归约与 eps 计算用 FP32。

原版让一个线程连续算1536个数，很多 GPU 单元闲着。改成整块协作后，通道访问合并，长串行循环变成短循环+树形归约。

## 版本与测试

### V0：任务开始的基础CUDA版

一个线程负责整行，自己遍历全部通道求平方和，再遍历整行写输出；低精度路径把 x、weight、缩放因子一次相乘后才舍入。

原始源码另外保存在 `baselines/v0/`。使用V0只表示历史基线，不代表其BF16中间计算顺序完全等于当前参考。

### V1：直观的数值对照版

仍用串行行循环，但先把归一化后的 x 舍入回输入 dtype，再乘 weight，更接近 HF eager 的计算顺序。这一步主要修正数值路径，不承诺提速。

### V2：当前优化版

一行分配256线程，各自累加一部分通道；先做 Warp shuffle 归约，再用8个共享内存槽合并8个 Warp。得到缩放因子后各线程写自己的通道，保留 V1 中间舍入。平方使用显式 float 乘法；并行求和顺序与串行不同，容差验证仍必需。

共同测试：真实hidden=1536、T=1/128，以及宽度157的非整齐形状；三种 dtype；独立测试覆盖零向量、常量向量和 eps 防除零，旧测试还覆盖512×4096。第0层真实模型输入Norm与当前HF检查点相同元素100%，这不是所有层都逐位一致的声明。

V0/V1对每个下表BF16形状都执行正确性检查再计时；V2还对同一组形状执行FP16和FP32检查。注意：随机输入下容差通过，不代表所有真实模型激活逐位相同；整模型结果另行验证。

## 设备微基准：每个版本的真实数值

计时单位**微秒**，取多次CUDA Graph设备事件计时的中位数；这是开发基准中的graph replay，不代表引擎本身用了CUDA Graph。相同GPU/dtype/shape/seed；各原始记录含所有样本与library SHA。

| shape | V0 μs | V1 μs | V2 μs | V1/V2 |
| --- | ---: | ---: | ---: | ---: |
| [1, 1536] | 96.800 | 124.491 | 3.102 | 40.13× |
| [128, 1536] | 387.808 | 388.624 | 12.837 | 30.27× |
| [5, 157] | 14.062 | 16.357 | 12.581 | 1.30× |

V2正确性范围（每行包含上面全部形状）：

| dtype | 通过/总数 | 本组最大绝对误差 | atol / rtol |
| --- | ---: | ---: | --- |
| bf16 | 3/3 | 0 | 0.02 / 0.02 |
| f16 | 3/3 | 6.1035156e-05 | 0.003 / 0.003 |
| f32 | 3/3 | 2.3841858e-07 | 0.0001 / 0.0001 |

小于几微秒的kernel容易受时钟、launch等波动影响。V0/V1如果走同一个kernel，两组数不一样不能当成一次优化。这里没有给出与PyTorch的算子加速比：原始记录中的PyTorch计时包含Python边界，与native graph计时范围不同。

## 优化后整机怎样

全版本阶段表见 [总表](README.md)：所有V0、所有V1、所有V2，以及引擎工作区改动都有实际整机测量。下面再做单算子回退：保持其他算子V2、引擎E2，只把 rms_norm 切回V1。BF16、batch=1、prompt64、8个输出位置、固定token回放、warmup2/repeat15，真实权重。

| 设置 | TTFT中位数 ms | 每请求TPOT中位数 ms |
| --- | ---: | ---: |
| 全V2 | 13.211 | 15.059 |
| 仅本算子回退V1 | 27.946 | 18.655 |

原始整机记录见 [ablation.jsonl](results/ablation.jsonl)。这是跨进程探索性消融；主机干扰、未锁GPU时钟导致全V2在不同批次TPOT约11～15ms浮动。本批次甚至出现小算子回退后的TPOT低于全V2，**不据此声称回退提速或优化版整机退化**，也不计算小差异的因果收益。Linear/Attention这类明显量级差异有参考价值；小差异仍需交错配对与更严格环境控制。回放固定输入保证计算长度一致，但不代表HF生成质量已全部通过。

## 依据结果怎样判断，下一步是什么

整机回退实验能看到明显变慢，说明原版串行行循环确实拖累模型；优化后在 Nsight 中只占约1.9%的kernel时间，单次平均约3微秒。最终微基准有几个十微秒级波动，表中照录，不用最好的一次替代全部结果。

进一步应考虑和残差相加融合，减少跨 kernel 写回。仅继续调 block 大小可能只有小收益，需在目标长度上测，而不是看到归约就一律用1024线程。

## 面试讲解与自测

可以先讲：“原版一个线程处理整行；当前每行一个256线程block，线程分摊通道，先在8个warp内归约，再用8个共享内存槽合并平方和。”输入是X[M,K]、weight[K]，M个block彼此独立。对K=1536，线程t处理`t、t+256、…、t+1280`六个元素；同一轮相邻线程读取相邻通道。共享内存只放warp部分和与最终缩放因子，没有缓存整行X，输出阶段会再次读取X和weight。

平方通过`__fmul_rn`以FP32计算，求和、除K、加eps和`rsqrtf`也在FP32中完成。低精度路径先将`x*inv`舍入回输入dtype，再乘weight并写回目标dtype；它与“所有乘法都在float里做完，最后只舍入一次”不等价。V2保留V1的舍入位置，但树形归约改变了加法顺序，因此仍要比较容差，不能承诺所有输入逐位一致。

**画图自测：** 画出“256线程→8个warp和→1个总和→广播inv→并行输出”的过程，并标出两次`__syncthreads()`：第一次保证8个warp和已写入，第二次保证第一个warp算出的inv已发布。把K改成157时，线程157～255的初始部分和为0，仍需参加归约和同步，不能直接退出。手算非零向量`x=[1,-1,1,-1]`、`weight=[1,2,3,4]`且为便于手算令eps=0，输出是`[1,-2,3,-4]`；全零输入则要保留正eps，避免`0×inf`。

**典型追问：** 为什么不用一个warp或1024线程？线程数影响每线程工作量、跨warp归约和资源占用，当前256线程是实现选择，需按实际K测量，不能靠线程越多越好判断。为什么不能在两个阶段之间仅依赖warp shuffle？shuffle只解决warp内通信，8个warp之间仍要共享内存和block同步。

当前实现的`partial[8]`和第二阶段读取8个warp结果，与256线程配套。不能只把launch中的256改成1024做调优，否则共享内存槽数不足；改成128却仍合并8个槽，也会读到没有写入的部分和。练习修改线程数时，应同时调整warp数量、共享内存与合并范围，再覆盖K小于block大小和非整齐尾部的用例。

**再追问：RMSNorm输出的均值为0、RMS为1吗？** 都不能这样保证。它不减均值；忽略eps且weight全为1时，非零输入归一化后的RMS才是1。乘上逐通道weight后通常不再为1。上面的手算输出`[1,-2,3,-4]`均值为-0.5，平方均值为7.5，正好可以说明这两个区别。

代码入口是[RMSNorm源码](../../src/ops/rms_norm/nvidia/rms_norm_nvidia.cu)中的`rms_parallel_v2`、`rms_run`；`warpSum`与`rounded`见[common.cuh](../../src/ops/nvidia/common.cuh)。数值排查如何连接到完整模型，见[主问答](RESUME_AND_INTERVIEW.md)。

## 自己复现

```bash
# 参数在进程首次调用算子前确定；切版本重新启动进程
.venv/bin/python benchmark/cuda_experiments.py --op rms_norm --version 0 --dtype bf16
.venv/bin/python benchmark/cuda_experiments.py --op rms_norm --version 1 --dtype bf16
.venv/bin/python benchmark/cuda_experiments.py --op rms_norm --version 2 --dtype bf16
LLAISYS_CUDA_RMS_NORM_VERSION=1 .venv/bin/python benchmark/benchmark_infer.py \
  --prompt-tokens 64 --output-tokens 8 --mode replay
```

单独执行 `--op` 时随机数生成顺序不同于`--op all`，要完全复用总表输入生成顺序，请使用 `run_optimization_suite.py --phase ops`。进程退出前完成计时，不要一边安装动态库一边运行测试。
