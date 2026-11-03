# SwiGLU：保留 SiLU 舍入，再做成对读写

日期：2026-11-02。源码：[src/ops/swiglu/nvidia/swiglu_nvidia.cu](../../src/ops/swiglu/nvidia/swiglu_nvidia.cu)。这是实际实验记录，含未获益和未采用方案。环境、计时限制及整机版本见 [实验总表](README.md)；原始样本见 [operators.jsonl](results/operators.jsonl)。

## 算什么，原来为什么慢

out = SiLU(gate) × up，SiLU(g)=g/(1+exp(-g))。gate/up来自两个不同的Linear。

减少一部分索引和存储指令，不能减少必须算的两个sigmoid/exp。它本来已经是融合的SiLU×up kernel；这次不是从两个独立CUDA kernel才首次做融合。

## 版本与测试

### V0：任务开始的基础CUDA版

一个线程处理一个元素，把 up*gate/(1+exp(-gate)) 连着计算后才写回。低精度乘法顺序与HF单独调用SiLU再乘up不同。

原始源码另外保存在 `baselines/v0/`。使用V0只表示历史基线，不代表其BF16中间计算顺序完全等于当前参考。

### V1：直观的数值对照版

先计算 SiLU 并舍入回输入 dtype，再乘 up，仍每线程一个元素。中间舍入是为了说明和约束数值语义，不是提速技巧。

### V2：当前优化版

和Add类似，每线程读写两个相邻元素，BF16/FP16打包32bit、FP32用float2；各元素仍用FP32计算exp和乘法。检查对齐并处理奇数尾部。没有使用Tensor Core，也没有声称采用half2算术。

共同测试：MLP实际宽度8960，T=1/128；[5,157]奇数长度；三种dtype；额外验证非对齐切片、gate=-100/-20/0/20/100时的饱和行为。旧测试也通过。

V0/V1对每个下表BF16形状都执行正确性检查再计时；V2还对同一组形状执行FP16和FP32检查。注意：随机输入下容差通过，不代表所有真实模型激活逐位相同；整模型结果另行验证。

## 设备微基准：每个版本的真实数值

计时单位**微秒**，取多次CUDA Graph设备事件计时的中位数；这是开发基准中的graph replay，不代表引擎本身用了CUDA Graph。相同GPU/dtype/shape/seed；各原始记录含所有样本与library SHA。

| shape | V0 μs | V1 μs | V2 μs | V1/V2 |
| --- | ---: | ---: | ---: | ---: |
| [1, 8960] | 4.826 | 0.925 | 1.076 | 0.86× |
| [128, 8960] | 8.950 | 11.082 | 7.626 | 1.45× |
| [5, 157] | 12.384 | 2.585 | 1.726 | 1.50× |

V2正确性范围（每行包含上面全部形状）：

| dtype | 通过/总数 | 本组最大绝对误差 | atol / rtol |
| --- | ---: | ---: | --- |
| bf16 | 3/3 | 0 | 0.02 / 0.02 |
| f16 | 3/3 | 0 | 0.003 / 0.003 |
| f32 | 3/3 | 0 | 0.0001 / 0.0001 |

小于几微秒的kernel容易受时钟、launch等波动影响。V0/V1如果走同一个kernel，两组数不一样不能当成一次优化。这里没有给出与PyTorch的算子加速比：原始记录中的PyTorch计时包含Python边界，与native graph计时范围不同。

## 优化后整机怎样

全版本阶段表见 [总表](README.md)：所有V0、所有V1、所有V2，以及引擎工作区改动都有实际整机测量。下面再做单算子回退：保持其他算子V2、引擎E2，只把 swiglu 切回V1。BF16、batch=1、prompt64、8个输出位置、固定token回放、warmup2/repeat15，真实权重。

| 设置 | TTFT中位数 ms | 每请求TPOT中位数 ms |
| --- | ---: | ---: |
| 全V2 | 13.211 | 15.059 |
| 仅本算子回退V1 | 13.293 | 11.558 |

原始整机记录见 [ablation.jsonl](results/ablation.jsonl)。这是跨进程探索性消融；主机干扰、未锁GPU时钟导致全V2在不同批次TPOT约11～15ms浮动。本批次甚至出现小算子回退后的TPOT低于全V2，**不据此声称回退提速或优化版整机退化**，也不计算小差异的因果收益。Linear/Attention这类明显量级差异有参考价值；小差异仍需交错配对与更严格环境控制。回放固定输入保证计算长度一致，但不代表HF生成质量已全部通过。

## 依据结果怎样判断，下一步是什么

Nsight中约0.4%的kernel时间，平均约1.18微秒。长数组的设备时间有改善，单tokenV1/V2相近；整机回退没有显示可信的独立收益，不能把微小波动写成推理提速成果。

若要继续，先看是否值得与gate/up投影的epilogue融合；这会限制GEMM算法选择并改变低精度舍入，必须重新验证。当前优先保留清晰、可调试的独立算子。

## 自己复现

```bash
# 参数在进程首次调用算子前确定；切版本重新启动进程
.venv/bin/python benchmark/cuda_experiments.py --op swiglu --version 0 --dtype bf16
.venv/bin/python benchmark/cuda_experiments.py --op swiglu --version 1 --dtype bf16
.venv/bin/python benchmark/cuda_experiments.py --op swiglu --version 2 --dtype bf16
LLAISYS_CUDA_SWIGLU_VERSION=1 .venv/bin/python benchmark/benchmark_infer.py \
  --prompt-tokens 64 --output-tokens 8 --mode replay
```

单独执行 `--op` 时随机数生成顺序不同于`--op all`，要完全复用总表输入生成顺序，请使用 `run_optimization_suite.py --phase ops`。进程退出前完成计时，不要一边安装动态库一边运行测试。
