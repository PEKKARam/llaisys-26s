# Argmax：从一条长循环到两阶段归约

文档更新：2026-11-03；实验测量：2026-11-02，本次没有新增实验。源码：[src/ops/argmax/nvidia/argmax_nvidia.cu](../../src/ops/argmax/nvidia/argmax_nvidia.cu)。这是实际实验记录，含未获益和未采用方案。环境、计时限制及整机版本见 [实验总表](README.md)；原始样本见 [operators.jsonl](results/operators.jsonl)。

## 算什么，原来为什么慢

每行找最大值与最先出现的索引；模型最后一行词表长度151936。只支持greedy，不执行temperature/top-p随机采样。

第一轮优化只解决“线程内串行”，第二轮解决“只有一个block”。大词表让许多SM分摊扫描，再只合并少量候选；不是让一个block把线程数从256无限加大。

## 版本与测试

### V0：任务开始的基础CUDA版

一个线程扫描完整一行；大词表时基本只有一个线程在串行读取151936个数。

原始源码另外保存在 `baselines/v0/`。使用V0只表示历史基线，不代表其BF16中间计算顺序完全等于当前参考。

### V1：直观的数值对照版

沿用V0。正常有限值、并列最大值按最小索引处理。

### V2：当前优化版

V2a：每行一个256线程block做局部最大值与warp/shared归约，已比串行快，但大词表只有一个block仍用不满GPU。实测约101.75微秒后继续改为V2b（当前）：K>4096时每1024元素一块，第一阶段多block得到(value,index)，第二阶段合并；小行保留单block，避免多一次launch。规则统一处理并列最大值、NaN和全负无穷。

共同测试：真实词表151936、5×157，三种dtype，返回值和I64索引与PyTorch一致。额外验证NaN取首个NaN、全-inf取索引0、跨warp并列、151937尾元素为最大，覆盖两个归约阶段及非整齐尾部。

V0/V1对每个下表BF16形状都执行正确性检查再计时；V2还对同一组形状执行FP16和FP32检查。注意：随机输入下容差通过，不代表所有真实模型激活逐位相同；整模型结果另行验证。

## 设备微基准：每个版本的真实数值

计时单位**微秒**，取多次CUDA Graph设备事件计时的中位数；这是开发基准中的graph replay，不代表引擎本身用了CUDA Graph。相同GPU/dtype/shape/seed；各原始记录含所有样本与library SHA。

| shape | V0 μs | V1 μs | V2 μs | V1/V2 |
| --- | ---: | ---: | ---: | ---: |
| [1, 151936] | 1990.976 | 1989.728 | 4.040 | 492.47× |
| [5, 157] | 3.074 | 3.050 | 1.577 | 1.93× |

V2正确性范围（每行包含上面全部形状）：

| dtype | 通过/总数 | 本组最大绝对误差 | atol / rtol |
| --- | ---: | ---: | --- |
| bf16 | 2/2 | 0 | 0.02 / 0.02 |
| f16 | 2/2 | 0 | 0.003 / 0.003 |
| f32 | 2/2 | 0 | 0.0001 / 0.0001 |

小于几微秒的kernel容易受时钟、launch等波动影响。V0/V1如果走同一个kernel，两组数不一样不能当成一次优化。这里没有给出与PyTorch的算子加速比：原始记录中的PyTorch计时包含Python边界，与native graph计时范围不同。

## 优化后整机怎样

全版本阶段表见 [总表](README.md)：所有V0、所有V1、所有V2，以及引擎工作区改动都有实际整机测量。下面再做单算子回退：保持其他算子V2、引擎E2，只把 argmax 切回V1。BF16、batch=1、prompt64、8个输出位置、固定token回放、warmup2/repeat15，真实权重。

| 设置 | TTFT中位数 ms | 每请求TPOT中位数 ms |
| --- | ---: | ---: |
| 全V2 | 13.211 | 15.059 |
| 仅本算子回退V1 | 14.414 | 13.103 |

原始整机记录见 [ablation.jsonl](results/ablation.jsonl)。这是跨进程探索性消融；主机干扰、未锁GPU时钟导致全V2在不同批次TPOT约11～15ms浮动。本批次甚至出现小算子回退后的TPOT低于全V2，**不据此声称回退提速或优化版整机退化**，也不计算小差异的因果收益。Linear/Attention这类明显量级差异有参考价值；小差异仍需交错配对与更严格环境控制。回放固定输入保证计算长度一致，但不代表HF生成质量已全部通过。

## 依据结果怎样判断，下一步是什么

V2a到V2b约101.75→3.99微秒，原始记录在operator_pilots；最终复测约4.04微秒。原版约1.99毫秒，每生成一个token只调用一次，所以单算子数百倍不能等同整个引擎数百倍。Nsight当前两阶段合计只占很小比例。

继续调归约参数的整机收益已经很小。若实现随机采样，不能直接沿用Argmax的比较规则；top-k/top-p需要新算法与验证。当前不扩大到这些功能。

最后还用最终安装库单独做了3次样本的release smoke，大词表测得约12.02μs、结果正确，见[release_smoke.jsonl](results/release_smoke.jsonl)。完整suite的4.04μs与这次不同，说明孤立运行/时钟/少量采样会影响微秒级数据；不保证每次启动都得到4μs，也不把12μs隐藏。整机热身后的Nsight两阶段合计约3.6μs，性能结论始终按各自条件解释。

## 面试讲解与自测

可以先讲：“初版每行只有一个线程，第一次并行化改成256线程但仍只有一个block。当前大词表分成1024元素一块，让多个SM承担扫描，再用一个block合并候选；小行保留单block，省掉第二次launch。”对于151936元素，第一阶段每行149个block，最后一块384元素；每个block的256线程以步长256扫描，相邻线程读取相邻元素。8个warp先各自归约，再通过共享内存合并(value,index)，第二阶段只读取149个候选。它没有重新扫描整个词表，也不需要全局atomic。

比较的对象始终是“值和原索引”。V2的`better`规定：NaN优先于非NaN，多个NaN选最小索引；普通值先比大小，值相等选最小索引。初始候选为`(-inf, 最大无符号索引)`，所以全负无穷也能得到索引0。V0/V1只用串行严格大于比较，不能把V2的完整NaN规则说成所有版本都具备。

**画图自测：** 画一行4100元素，按1024分成5块，最后一块只有4个元素。若索引1023和1024都为最大值9，其余值更小，两阶段都必须保留原索引，最终返回1023；若再将3000和4099改为NaN，则最终返回3000。说明第二阶段若只保留“第几个块”会丢掉什么，以及尾部无数据线程为何不能制造候选。

**典型追问：** 为什么不对所有行都分块？当前`K<=4096`直接走一个block，额外launch与scratch读写可能抵消小行收益；4096是当前实现选择，不是证明过的通用最优阈值。算子从约2ms降到几微秒也不等于整机同倍数提速，需回到每token调用次数和整机时间占比。

**再追问：两阶段怎样保证正确？** 每一块保留按同一套`better`规则选出的最佳候选，再从所有候选中选最佳，就等于直接在整行选择；值相等时携带原索引，才能保证分块前后选择一致。第一阶段不同block写各自的scratch槽，第二阶段与第一阶段在同一stream依次启动，靠kernel之间的顺序保证候选已经写好；`__syncthreads()`只能同步一个block，不能用来等待所有第一阶段block。这里没有把多个block塞进一个kernel后自行全局同步。

代码入口是[Argmax源码](../../src/ops/argmax/nvidia/argmax_nvidia.cu)中的`argmax_run`、`argmax_chunks`和`better`；特殊值用例见[CUDA边界检查](../../test/test_cuda_edges.py)。再回到[主问答](RESUME_AND_INTERVIEW.md)练习将选择、验证和结果连起来讲。

## 自己复现

```bash
# 参数在进程首次调用算子前确定；切版本重新启动进程
.venv/bin/python benchmark/cuda_experiments.py --op argmax --version 0 --dtype bf16
.venv/bin/python benchmark/cuda_experiments.py --op argmax --version 1 --dtype bf16
.venv/bin/python benchmark/cuda_experiments.py --op argmax --version 2 --dtype bf16
LLAISYS_CUDA_ARGMAX_VERSION=1 .venv/bin/python benchmark/benchmark_infer.py \
  --prompt-tokens 64 --output-tokens 8 --mode replay
```

单独执行 `--op` 时随机数生成顺序不同于`--op all`，要完全复用总表输入生成顺序，请使用 `run_optimization_suite.py --phase ops`。进程退出前完成计时，不要一边安装动态库一边运行测试。
