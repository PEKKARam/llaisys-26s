# Linear：把串行点积换成 cuBLAS

文档更新：2026-11-03；实验测量：2026-11-02，本次没有新增实验。源码：[src/ops/linear/nvidia/linear_nvidia.cu](../../src/ops/linear/nvidia/linear_nvidia.cu)。这是实际实验记录，含未获益和未采用方案。环境、计时限制及整机版本见 [实验总表](README.md)；原始样本见 [operators.jsonl](results/operators.jsonl)。

## 算什么，原来为什么慢

X[M,K] × W[N,K]^T + bias[N] → Y[M,N]。prefill 的 M 是多个 token，decode 的 M=1，后者更接近 GEMV。

调用厂商库替换了极低效的线程布局和串行算法。prefill 可以利用高效 GEMM；decode 主要仍要读取大量权重。这里的贡献是后端集成、dtype/布局/数值控制和实验，不是自己写出了 cuBLAS kernel。

## 版本与测试

### V0：任务开始的基础CUDA版

一个线程负责一个输出，沿 K 串行累加；相邻线程读取不同权重行，同一步读取地址相隔 K，访存与并行利用都很差。bias 存在时先加到 FP32 累加器，再开始累加乘积。

原始源码另外保存在 `baselines/v0/`。使用V0只表示历史基线，不代表其BF16中间计算顺序完全等于当前参考。

### V1：直观的数值对照版

沿用 V0。它是直观基线，不是优化版 SGEMM，也没有 Tensor Core 分块。

### V2：当前优化版

接入 cublasGemmEx：BF16/FP16 输入、FP32 累加，禁止低精度 partial reduction；FP32 使用 PEDANTIC，不偷偷启用 TF32。通过列主序解释计算 Y^T=W×X^T，不额外转置权重。无 bias 直接输出目标 dtype；有 bias 时先输出 FP32 scratch，再加 bias 并转回目标 dtype。每线程/设备复用 handle 和 grow-only scratch。

共同测试：测 Q/MLP/down/词表投影的6种实际尺寸，每种含 bias，三种 dtype 通过 PyTorch 容差检查；独立边界测试还覆盖 M=3,N=7,K=13、有/无 bias，旧测试包含512×4096大矩阵。真正模型主路径大多数投影无 bias，整机 benchmark 另外覆盖。

V0/V1对每个下表BF16形状都执行正确性检查再计时；V2还对同一组形状执行FP16和FP32检查。注意：随机输入下容差通过，不代表所有真实模型激活逐位相同；整模型结果另行验证。

## 设备微基准：每个版本的真实数值

计时单位**微秒**，取多次CUDA Graph设备事件计时的中位数；这是开发基准中的graph replay，不代表引擎本身用了CUDA Graph。相同GPU/dtype/shape/seed；各原始记录含所有样本与library SHA。

| shape | V0 μs | V1 μs | V2 μs | V1/V2 |
| --- | ---: | ---: | ---: | ---: |
| [1, 1536, 1536] | 154.216 | 152.928 | 8.041 | 19.02× |
| [32, 1536, 1536] | 922.304 | 914.592 | 10.562 | 86.59× |
| [1, 8960, 1536] | 153.760 | 153.557 | 28.902 | 5.31× |
| [32, 8960, 1536] | 4943.904 | 4923.968 | 38.848 | 126.75× |
| [1, 1536, 8960] | 907.584 | 925.344 | 69.301 | 13.35× |
| [1, 151936, 1536] | 3816.384 | 4030.592 | 1209.152 | 3.33× |

V2正确性范围（每行包含上面全部形状）：

| dtype | 通过/总数 | 本组最大绝对误差 | atol / rtol |
| --- | ---: | ---: | --- |
| bf16 | 6/6 | 0 | 0.02 / 0.02 |
| f16 | 6/6 | 0 | 0.003 / 0.003 |
| f32 | 6/6 | 1.1444092e-05 | 0.0001 / 0.0001 |

小于几微秒的kernel容易受时钟、launch等波动影响。V0/V1如果走同一个kernel，两组数不一样不能当成一次优化。这里没有给出与PyTorch的算子加速比：原始记录中的PyTorch计时包含Python边界，与native graph计时范围不同。

## 优化后整机怎样

全版本阶段表见 [总表](README.md)：所有V0、所有V1、所有V2，以及引擎工作区改动都有实际整机测量。下面再做单算子回退：保持其他算子V2、引擎E2，只把 linear 切回V1。BF16、batch=1、prompt64、8个输出位置、固定token回放、warmup2/repeat15，真实权重。

| 设置 | TTFT中位数 ms | 每请求TPOT中位数 ms |
| --- | ---: | ---: |
| 全V2 | 13.211 | 15.059 |
| 仅本算子回退V1 | 954.552 | 62.844 |

原始整机记录见 [ablation.jsonl](results/ablation.jsonl)。这是跨进程探索性消融；主机干扰、未锁GPU时钟导致全V2在不同批次TPOT约11～15ms浮动。本批次甚至出现小算子回退后的TPOT低于全V2，**不据此声称回退提速或优化版整机退化**，也不计算小差异的因果收益。Linear/Attention这类明显量级差异有参考价值；小差异仍需交错配对与更严格环境控制。回放固定输入保证计算长度一致，但不代表HF生成质量已全部通过。

## 依据结果怎样判断，下一步是什么

实测 Nsight Systems 中厂商矩阵运算合计约92%的 kernel 时间。单独对 BF16 [1,1536]×[8960,1536]^T 的无 bias decode kernel 运行 NCU：75.26微秒，DRAM throughput 89.76%，SM throughput 27.74%，achieved occupancy 23.71%。说明这一个尺寸明显偏访存受限；不能把低 occupancy 单独当作问题，也不能把这一个尺寸代表所有 GEMM。

不建议马上重写厂商 GEMV。更有价值的候选是减少权重字节数的量化或更高 batch 下复用权重，前者可以接你的量化学习线；当前不支持量化，也不扩展 batch。QKV 合并投影也是后续候选，需要重排权重并重新验证。

## 面试讲解与自测

可以先讲：“基础版一线程负责Y的一个元素，沿K串行累加；同一warp的线程读不同W行，在同一个k处地址相隔K个元素。当前改用cuBLAS，我负责把现有row-major张量映射到库接口，并处理dtype、bias和资源生命周期。”cuBLAS内部的线程块配置由库选择，不能把它说成自己设计的tiling；自己写的bias kernel仍是每线程一个输出、每block256线程。

**画图自测：** 取M=2、N=4、K=3，先手算下面结果，再按内存地址解释库调用：

```text
X = [[1,2,3], [4,5,6]]
W = [[1,0,0], [0,1,0], [0,0,1], [1,1,1]]
Y = X Wᵀ = [[1,2,3,6], [4,5,6,15]]
```

W的row-major[N,K]内存直接解释为column-major[K,N]，因此A指向W并使用`transA=T`；X解释为column-major[K,M]，B指向X并使用`transB=N`。输出column-major[N,M]与所需row-major[M,N]共享同一布局，所以`GemmEx(m=N,n=M,k=K,lda=K,ldb=K,ldc=N)`；本例是`m=4,n=2,k=3,lda=3,ldb=3,ldc=4`。转置标志改变解释方式，没有额外生成转置权重副本。

地址可以直接验算：原W的`W[n,k]`偏移为`n*K+k`，重解释后的列主序A在`[k,n]`处也是`k+n*K`；输出Y的`Y[m,n]`偏移为`m*N+n`，列主序输出在`[n,m]`处也是`n+m*N`。`lda/ldb/ldc`表示列主序相邻列起点之间的元素数，不是字节数，也不是看到转置标志就交换它们。

**典型追问：** 为什么有bias时还保留FP32 scratch和一个kernel？当前先用FP32累加并输出FP32乘积，再加bias，最后转目标dtype；若先把乘积舍入到BF16再加bias，数值路径就变了。cuBLASLt融合bias曾在真实激活验证中出现更大误差，因此保留独立bias路径。prefill的较大M可复用权重，decode的M=1复用较少；NCU的约90%是已测一个GEMV尺寸的DRAM吞吐指标，既不是算力利用率，也不代表所有Linear形状。

**再追问：为什么M=1容易受带宽限制？** 做一个只计权重的估算：BF16的N×K权重读一遍约需`2NK`字节，M=1时乘加约`2NK`次浮点运算，比例约为1 FLOP/byte；M行共同使用这份权重时，理想复用下运算增至`2MNK`。这说明更大M有提高权重复用的机会，但估算忽略输入、输出、缓存命中与实际读写次数，不能替代NCU测量，也不能承诺batch增大就线性加速。

代码入口是[Linear源码](../../src/ops/linear/nvidia/linear_nvidia.cu)中的`linear_blas`、`linear_bias`；handle、禁用低精度partial reduction和grow-only scratch见[resources.cuh](../../src/ops/nvidia/resources.cuh)。无bias时直接输出目标dtype，有bias时才用Linear的FP32 scratch。选择与失败候选的完整讲法见[主问答](RESUME_AND_INTERVIEW.md)。

## 自己复现

```bash
# 参数在进程首次调用算子前确定；切版本重新启动进程
.venv/bin/python benchmark/cuda_experiments.py --op linear --version 0 --dtype bf16
.venv/bin/python benchmark/cuda_experiments.py --op linear --version 1 --dtype bf16
.venv/bin/python benchmark/cuda_experiments.py --op linear --version 2 --dtype bf16
LLAISYS_CUDA_LINEAR_VERSION=1 .venv/bin/python benchmark/benchmark_infer.py \
  --prompt-tokens 64 --output-tokens 8 --mode replay
```

单独执行 `--op` 时随机数生成顺序不同于`--op all`，要完全复用总表输入生成顺序，请使用 `run_optimization_suite.py --phase ops`。进程退出前完成计时，不要一边安装动态库一边运行测试。

未采用的融合bias候选及数值证据见 [Lt实验](experiments/lt_bias/README.md)。这是一次未通过真实激活验证的尝试，没有合入，也没有用它的速度作为结果。
