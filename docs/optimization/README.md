# Qwen2 单卡 CUDA 优化与实验记录

文档更新：2026-11-03；实验测量：2026-11-02，本次只整理讲解与证据，没有新增实验。目标是让这个教学工程具备可解释的计算、可复现的实验与有依据的简历描述。范围固定为 **Linux / NVIDIA 单卡 / batch=1 / 单请求 / BF16 模型 / greedy**；不扩展第二GPU平台、服务端、多batch或量化执行。

## 先读哪几份

1. [KV Cache 与前向过程](KV_CACHE_AND_FORWARD.md)：回答缓存在哪里、decode怎样读写、为什么最后缓存长度不是输出总长度。
2. [Argmax记录](argmax.md)：最适合先上手，一条长循环 → 单block → 两阶段归约，有实测反馈。
3. [Linear记录](linear.md)、[Attention记录](self_attention.md)、[RMSNorm记录](rms_norm.md)：结合线程映射、形状和小算例解释优化；四份重点算子文档均有“面试讲解与自测”。
4. [三条简历文案与32个面试问答](RESUME_AND_INTERVIEW.md)：按项目介绍、实现与优化、验证和测量组织讲解，统一指标、贡献边界和当前验证限制。先用主问答建立完整叙述，被追问具体实现时再进入算子记录。

每个算子都有独立记录：[Add](add.md)、[Embedding](embedding.md)、[Linear](linear.md)、[RMSNorm](rms_norm.md)、[RoPE](rope.md)、[Self-Attention](self_attention.md)、[SwiGLU](swiglu.md)、[Argmax](argmax.md)。[执行时间线](WORK_LOG.md)保留版本推进、未采用实验和环境问题。

本文维护实测事实和复现条件；[主问答](RESUME_AND_INTERVIEW.md)维护投递文案、短答、追问与证据索引；[短期计划](../../RESUME_AND_30_DAY_PLAN.md)安排个人练习。准备面试时先建立项目叙述，再回到本文核查每个数字，不需要按顺序背完整份实验报告。

## 已经改了什么

- CUDA算子保留V0/V1/V2分支，默认V2；V0是基础CUDA算法，V1先约束低精度中间计算顺序，V2优化并行和存储。V1不是“整模型正确性已全部完成”的代名词。
- Linear接入cuBLAS及FP32累加；RMSNorm并行归约；Attention复用QK/softmax，避免逐输出通道重复打分；Argmax经两轮改进采用大词表两阶段归约；其余算子改进成对处理或直接复制。
- 引擎E2复用工作区、每次infer只上传一份positions、最终只对最后token计算词表logits；V直接投影进cache，K的RoPE直接输出cache切片，省去逐层同步D2D。
- Python暴露prefill/decode/reset/state/close；cache_capacity与模型位置上限分离，默认2048；C接口捕获模型异常，越界不会让异常直接跨ctypes边界终止Python。
- 增加Tensor allocator统计，修复Context切设备时复制runtime表造成的资源问题。统计范围见下文，不冒充完整进程显存。
- 重写模型计时，从同一请求内分别测prefill和各次decode；新增CUDA Graph设备微基准、串行实验套件、边界回归和NVTX/profiler采样入口。

## 正确性结论：通过什么，没通过什么

| 检查 | 实测结果 | 能说明什么 |
| --- | --- | --- |
| 算子微基准 | 140/140通过 | BF16 V0/V1/V2各28形状；V2 FP16/FP32各28形状，按记录容差对照PyTorch |
| 专门的CUDA边界检查 | 57/57通过 | 奇数尾部、非对齐、因果mask、GQA/Dv差异、NaN/并列归约等 |
| 原有8个GPU算子脚本 | 全部通过 | 含FP32/FP16/BF16旧用例，兼容已有测试 |
| 小模型回归 | GPU36/36、CPU-only构建36/36 | HF prefill/teacher-forced decode 的预测 ID、分块/逐token cache、reset、越界、FFI错误、工作区复用、tied weights |
| Runtime与Tensor | CPU/NVIDIA extended、CPU-only Runtime、原Tensor测试通过 | 同步/异步复制、流、切片等相关基础链路 |
| 真实模型9条WikiText首token | 9/9一致 | 仅这些固定prefill的argmax对照 |
| 同9条、最多生成8个token | **5/9序列完全一致** | 仍有4条后续生成分歧，严格HF生成回归没有全部通过 |

小模型回归的 HF 与完整前缀/缓存对照主要检查返回的 argmax ID；其余检查覆盖状态和接口行为。这 36 项不是整份 logits 的逐元素数值一致性检查，不能把预测 ID 相同推成所有候选分数一致。

真实模型不能宣布“与HF完全对齐”。当前记录：[每条生成结果](results/generation_compare.jsonl)、[完整张量指标](results/tensor_comparison.jsonl)。`row_3_len_64`最终logits最大绝对差0.375、平均绝对差约0.06794、cosine约0.99993，top5一致；这只是一个prefill，不能用它代替全数据质量评估。第0层K/V projection已存在舍入差异，其他阶段还会累积误差。未完成所有分歧的逐步归因，不能把剩余差异全部武断归为“正常误差”。

参考固定为PyTorch 2.13.0+cu130 / Transformers 5.15.0，BF16、HF eager、FP32 GEMM reduction、TF32关闭，最终生成对照`logits_to_keep=1`。历史11月1日0/9使用的数值设置与实现不同，保留为历史证据；不能与11月2日结果当作同配置A/B。HF低精度partial reduction是合法的速度/精度取舍，关闭它是为了明确对照口径，不是在宣称HF默认有bug。

### 参考后端控制实验（仅用于定位）

额外只设置HF环境变量`DISABLE_ADDMM_CUDA_LT=1`，native代码与输入不变，再跑同9条/8token，对照变为**8/9序列一致**；剩余`row_11_len_64`在第6个预测处不同。[完整记录](results/reference_no_lt.jsonl)将环境与原记录一起保存。复现：`DISABLE_ADDMM_CUDA_LT=1 .venv/bin/python benchmark/compare_prompt.py --all --max-new-tokens 8`。

这提供了实际证据：仅参考端BLAS调用选择变化，就会改变BF16后续argmax，说明库路径/舍入影响很大。**不把8/9替换默认设置的5/9，也没有用它重新包装HF性能基线。** 尚余一条受控参考分歧，完整生成正确性仍未闭环；不能单凭这组结果宣布所有差异无害。源码参考：[PyTorch CUDA BLAS分发](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/Blas.cpp)，实际运行以本机安装版本为准。

## 环境与计时口径

GPU为RTX 5060 Ti 16GB，driver621.01，CUDA编译器与精确版本见[环境文件](results/environment.txt)。未锁GPU时钟，也未隔离桌面/主机调度干扰，因此原始波动全部保留。最终安装库的[独立smoke](results/release_smoke.jsonl)也保留：大词表Argmax少量采样约12μs，而完整suite约4μs，不能保证每次孤立启动都复现最佳热身状态。当前Git HEAD仍是开发前提交，**不能只靠HEAD复现未提交代码**；记录含库SHA，最终源码/库校验见[release.sha256](results/release.sha256)，原始V0源码保存在baselines/v0；本地配置、tokenizer和权重SHA见[model.sha256](results/model.sha256)。

模型为本地DeepSeek-R1-Distill-Qwen-1.5B。输入来自WikiText-2 raw test，固定原顺序3条正文的16/64/128前缀，共9条，commit `b08601e04326c79dfdd32d625aee71d232d685c3`；[冻结输入](results/prompts.jsonl)可直接复用，不套chat template。来源：[WikiText数据集](https://huggingface.co/datasets/Salesforce/wikitext)。这是小型smoke集合，不是完整性能/语言质量基准。

**微基准**：C++专用入口在独立stream捕获多次算子调用，再用CUDA events测graph replay的设备时间，排除Python逐次派发间隙。模型正常执行不使用CUDA Graph。原始记录里PyTorch参考仍按Python边界events计时，因此本报告不拿两种计时结果算算子加速比。包括scratch/handle的首次分配和graph创建均在计时外。

**整机**：perf_counter+设备同步，在同一次请求测TTFT和每次decode。范围是“CPU token IDs → 模型API → CPU预测ID”，包括API检查、必要复制与同步；不包括权重加载、tokenization、chat template、网络、排队或打印。这里的TTFT是模型调用层首token延迟，不是在线服务总延迟。

- 请求N个输出：prefill产生第1个，后面最多N−1次decode。
- TPOT：每请求decode step平均，再报告这些请求的中位数。
- decode token/s：所有实际decode步数 / decode总时间，等于平均step耗时倒数；**不一定等于1000/TPOT中位数**。
- `replay`：每步都输入相同的预定token，忽略EOS，保证所有实现经历相同输入前缀与长度；它用于消融，**不是greedy文本生成**。
- `greedy`：模型把自己预测的ID作为下一步输入，保留EOS停止。当前HF序列尚未全对齐，不能把下表的时延比直接当作“等质量HF加速倍数”。

## 整机分阶段结果

统一BF16、batch=1、cache容量2048、WikiText `row_3_len_64`、P=64、N=8，固定token回放，warmup1/repeat5。E1保留旧执行策略：每层临时分配、逐层positions与同步D2D、全prompt词表投影。E2是本轮优化执行策略。所有版本都在同一份已修复runtime/新增统计的基础设施下比较，不是拿初始checkout原封不动运行。

| CUDA算子 / 引擎 | TTFT中位数 ms | TPOT中位数 ms | decode token/s（均值口径） | 测量区Tensor设备分配次数 |
| --- | ---: | ---: | ---: | ---: |
| V0 / E1 | 1298.094 | 96.750 | 10.41 | 18160 |
| V1 / E1 | 1315.775 | 100.841 | 9.88 | 18160 |
| V2 / E1 | 78.710 | 28.595 | 34.49 | 18160 |
| V2 / E2 | 12.604 | 14.232 | 66.93 | 0 |

原始记录：[engine_stages.jsonl](results/engine_stages.jsonl)。V0有不同的低精度中间语义，历史性能仅作参考；优化收益优先以V1/E1对V2/E2描述。算子版本切换不是逐位等价证明。

**简历里的总收益包含算子优化和引擎执行策略优化。** 在上表同一轮分阶段记录中，V1/E1→V2/E1保持旧执行策略，换算子后TTFT为1315.8→78.7ms、TPOT为100.8→28.6ms；V2/E1→V2/E2再加入工作区复用、K/V直接写缓存和末token词表投影，变为78.7→12.6ms、28.6→14.2ms。后一组改动不能归到某个CUDA kernel名下；这里也没有分别隔离三项执行策略各自的贡献。

为看波动，另外对最终V2/E2运行warmup3/repeat21，TTFT中位数 **14.130 ms**，TPOT中位数 **12.868 ms**，decode **73.66 token/s**；各请求TPOT范围11.49～19.35ms，见[21次原始数据](results/engine_repeat21.jsonl)。简历采用基础V1/E1的warmup1/repeat5与这次最终复测的中位数，前后热身及重复次数不同，不能称为同一轮配对测量。相对V1/E1，TTFT约93.1倍、TPOT约7.8倍；这是DeepSeek-R1-Distill-Qwen-1.5B在上述固定回放条件下，相对项目朴素CUDA算法及旧执行策略的综合结果。它不表示相对HF/llama.cpp的收益，也不保证所有prompt长度有同样倍数。简历正文与面试表述见[主问答](RESUME_AND_INTERVIEW.md)。

E2热身后的测量区Tensor设备存储分配为0，workspace按需要扩容并复用；这不代表没有C++ view元数据分配或CUDA库内部活动。旧策略每请求约3632次Tensor分配，repeat5共18160次。

## 单算子回退实验

保持E2、其他算子V2，只把一项退回V1；同P64/N8回放，warmup2/repeat15。完整数据[ablation.jsonl](results/ablation.jsonl)。

| 回退对象 | TTFT中位数 ms | TPOT中位数 ms |
| --- | ---: | ---: |
| 不回退，全V2 | 13.211 | 15.059 |
| add | 12.432 | 11.027 |
| embedding | 12.840 | 11.210 |
| linear | 954.552 | 62.844 |
| rms_norm | 27.946 | 18.655 |
| rope | 13.222 | 13.163 |
| self_attention | 62.631 | 24.407 |
| swiglu | 13.293 | 11.558 |
| argmax | 14.414 | 13.103 |

**不要把小差异硬解释成优化收益。** 全V2跨批次TPOT约11～15ms浮动，而部分小算子回退后恰好更快，说明跨进程实验的主机/时钟噪声仍明显。本次可以有依据地说Linear和Attention替换大幅降低基础实现开销，RMSNorm也有明显影响；Add/Embedding/RoPE/SwiGLU的小幅整机贡献未被可靠分离，不宣称它们各自带来多少百分比。若继续研究，应做交错配对、CPU亲和/背景负载控制和更多长度，而不是只挑最好的一次。

## 当前greedy模型延迟与HF背景数据

每种P复用同一条WikiText前缀，最多N=16，warmup2/repeat5，实际均输出16token。HF使用eager、dynamic cache、同样`logits_to_keep=1`及参考精度设置；native是连续预分配cache。两者独立加载运行。

| 后端 | 输入token | TTFT ms | TPOT ms | decode token/s |
| --- | ---: | ---: | ---: | ---: |
| llaisys | 16 | 11.702 | 11.090 | 84.70 |
| llaisys | 64 | 13.843 | 11.765 | 80.66 |
| llaisys | 128 | 18.773 | 11.415 | 84.21 |
| hf | 16 | 29.580 | 26.428 | 30.92 |
| hf | 64 | 27.186 | 25.552 | 39.08 |
| hf | 128 | 24.183 | 26.618 | 38.17 |

这是背景观测，原始记录[engine_greedy.jsonl](results/engine_greedy.jsonl)。它不是HF最佳实现，没测SDPA/FlashAttention、torch.compile或llama.cpp，且native/HF后续预测可能不同，**简历不采用“比HF快X倍”**。先完成严格生成/数值归因与更广质量验证，再做同质量比较。

## 显存与Cache结果

本模型权重Tensor合计3,554,176,000字节；容量2048的分层BF16 K/V合计58,720,256字节=56MiB。按模型位置上限131072预留则是3,758,096,384字节=3584MiB。容量131072也已实际创建并测量，见[cache_capacity.jsonl](results/cache_capacity.jsonl)，model.state返回的cache字节数与公式一致。

这是运行容量配置变化，不是压缩KV值的算法，也不是在同样最大可处理长度下无代价减少内存。

`model.state`可查询cached_len、capacity、cache/weight/workspace字节数。Tensor allocator peak仅覆盖框架设备Tensor存储，不包含原始cudaMalloc的算子scratch、cuBLAS workspace、CUDA context、其他进程。HF的PyTorch allocator与它口径不同，因此不把两者峰值直接比较成“整体显存降低”。同容量的阶段实验也保留了Tensor allocator峰值及作用域标签。

## 真正的profiling证据

Nsight Systems只捕获热身后的一次P64/N8回放，带prefill/decode NVTX标记，避开权重加载：

- 1次prefill、7次decode；NVTX范围分别约12.93ms和78.42ms。
- 厂商GEMM/GEMV/相关归约约占92%的GPU kernel时间。Add/Embedding/SwiGLU等已不是主要热点。
- CUDA API区间只剩24次cudaMemcpy，对应16次H2D和8次D2H；没有D2D，沒有热身后cudaMalloc/cudaFree。真实拷贝设备时间合计约15.5微秒。
- cudaMemcpy API耗时看起来约61.7ms，主要是同步调用在等待前面的GPU计算，**不是几百字节跨PCIe花了60ms**。不能按API表的第一名直接决定“先优化PCIe”。
- launch API数量仍多，CUDA Graph可能是未来方向；本轮只在微基准使用Graph，正常引擎未使用。

Nsight Compute对无bias的decode Linear X[1,1536]、W[8960,1536]单独采样：75.26μs、DRAM吞吐89.76%、SM吞吐27.74%、achieved occupancy23.71%、每线程164寄存器。这里约90%指DRAM吞吐指标，不是90%算力利用率；证据支持这个代表性GEMV尺寸偏访存受限，不能推广成所有decode Linear或整机均达到该比例，也不能看到低occupancy就机械地减寄存器。NCU分析会重放kernel，其耗时不能混进普通benchmark表。

原始摘要：[GPU kernel CSV](results/nsys_final_cuda_gpu_kern_sum.csv)、[CUDA API CSV](results/nsys_final_cuda_api_sum.csv)、[NVTX CSV](results/nsys_final_nvtx_sum.csv)、[传输 CSV](results/nsys_final_cuda_gpu_mem_time_sum.csv)、[NCU明细](results/ncu_decode_linear.txt)。二进制trace在本机`cache/profiles/`，不提交大文件。

## 怎样复现，不用先重读所有代码

从仓库根目录执行；不要同时跑多个GPU benchmark，也不要在进程使用.so时覆盖安装。

```bash
xmake f -y --nv-gpu=true -m release
xmake -y
xmake install -y
# 避免Python导入site-packages里旧复制的.so
uv pip install --python .venv/bin/python --no-deps -e ./python

.venv/bin/python test/test_engine_regression.py --device nvidia
.venv/bin/python test/test_cuda_edges.py
.venv/bin/python benchmark/compare_prompt.py --all --max-new-tokens 8
# 当前预期：5/9序列一致，因此这个严格对照返回非零；不要吞掉它并宣布全部通过

.venv/bin/python benchmark/cuda_experiments.py --op argmax --version 2
.venv/bin/python benchmark/benchmark_infer.py   --prompt-tokens 64 --output-tokens 8 --mode replay --warmup 3 --repeat 21

# 完整实验：数分钟，串行运行；结果追加，先指定新目录避免混入旧轮次
.venv/bin/python benchmark/run_optimization_suite.py --phase all   --output-dir benchmark/results/my_new_run
```

算子版本`LLAISYS_CUDA_VERSION=0/1/2`，单算子覆盖例如`LLAISYS_CUDA_RMS_NORM_VERSION=1`；第一次调用后进程内缓存版本，切换须新进程。引擎策略`LLAISYS_ENGINE_VERSION=1/2`，benchmark对应`--engine-version`。单独脚本会设置全局版本参数，但保留单算子覆盖；完整套件会清理继承的覆盖，确保实验设定可控。

复用发布的输入：加`--prompts docs/optimization/results/prompts.jsonl`。默认读取本机cache中的同名集合；换机器需准备本地权重与依赖，不能仅下载源码就复现模型性能。完整Python环境见environment.txt；本地Python.h补丁路径见benchmark/README。

Profiling命令：

```bash
mkdir -p cache/profiles
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none   --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=true   -o cache/profiles/engine_final .venv/bin/python benchmark/benchmark_infer.py   --prompt-tokens 64 --output-tokens 8 --mode replay --warmup 1 --repeat 1 --nvtx --profile
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum   cache/profiles/engine_final.nsys-rep
ncu --set basic --launch-count 1 --force-overwrite   -o cache/profiles/decode_linear .venv/bin/python benchmark/profile_decode.py
```

## 接下来由你完成的学习与验收

本轮代码、测量和记录已落实。你不用从头背底层，先独立复现Argmax和P64/N8的结果，再把Cache时间表讲出来，接着解释Linear从串行点积到厂商库的变化与NCU证据。能自己改一次合法参数、重新测试并解释结果，再把优化细节写成“掌握”；阅读过代码本身还不足以支撑面试里的独立设计表述。

仍待研究的明确事项：4条BF16生成分歧的逐步归因和更广质量验证；小算子收益的严格配对测量；长上下文；服务化/多batch/量化均不在本轮实现范围。当前可以用阶段性项目投递，但保留上述验证边界，不写成生产级推理框架。

技术参考：[cuBLAS官方文档](https://docs.nvidia.com/cuda/cublas/index.html)、[PyTorch数值精度说明](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)、[Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)、[Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)。硬件瓶颈结论以本机测得CSV/NCU为依据，不把文档中的一般原则当作实测。
