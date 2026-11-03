# 三条简历描述与面试问答

文档更新：2026-11-03；实验数据引用 2026-11-02 的记录。本次统一文案并补充讲解，没有新增性能实验，也不表示个人已经掌握全部实现。详细数字见[实验报告](README.md)，前向机制见[KV Cache 讲解](KV_CACHE_AND_FORWARD.md)。

本文是三条简历、32 个问答和指标解释的维护基准；[一个月计划](../../RESUME_AND_30_DAY_PLAN.md)同步展示简历，[学习地图](../../LEARNING_PLAN.md)用于查数学与代码。先练短答，再用例子解释；代码和原始记录用于查证，不要求把下面每段背下来。

## 1. 可以直接整理进简历的三条

项目名：**基于 C++/CUDA 的 Qwen2 单卡推理引擎**

- 基于 LLAISYS 教学框架完善 Qwen2 推理后端，打通 BF16 权重加载与 C++/CUDA 前向计算，支持 GQA、RoPE、分层 KV Cache，以及单请求 prefill 和逐 token decode。
- 针对串行归约与重复 Attention 打分，实现 RMSNorm 并行归约、Argmax 两阶段归约及 Attention 中间结果复用，集成 cuBLAS Linear；结合工作区复用、KV 直接写入与末 token 词表投影，减少计算、分配和复制开销。
- 在 RTX 5060 Ti 上，以 DeepSeek-R1-Distill-Qwen-1.5B、BF16、batch=1、输入64／输出8的固定 token 回放测试，相对项目朴素 CUDA 基线，将模型调用层 TTFT 中位数从 **1315.8 ms 降至 14.1 ms**，TPOT 中位数从 **100.8 ms 降至 12.9 ms**。

三条分别回答“做了什么、为什么能提速、在什么条件下测到了什么结果”。第三条是第二条中**算子与执行策略共同优化**后的成绩。Nsight 分析、边界检查、状态查询可以在追问时展开，不必把所有细节挤进简历。

### 指标要能解释到哪一层

| 版本 | 具体改变 | TTFT 中位数 ms | TPOT 中位数 ms | 采样 |
| --- | --- | ---: | ---: | --- |
| V1/E1 | 朴素 CUDA 算子、旧执行策略 | 1315.775 | 100.841 | warmup 1 / repeat 5 |
| V2/E1 | 替换算子，保留旧执行策略 | 78.710 | 28.595 | warmup 1 / repeat 5 |
| V2/E2 | 再加入工作区复用、KV 直接写入等执行优化 | 12.604 | 14.232 | warmup 1 / repeat 5 |
| V2/E2 最终复测 | 最终实现重复采样，简历采用这一组 | 14.130 | 12.868 | warmup 3 / repeat 21 |

前三行来自[阶段记录](results/engine_stages.jsonl)，最后一行来自[21 次复测](results/engine_repeat21.jsonl)。各组 GPU、模型、BF16、batch=1、cache=2048 和 P64/N8 回放输入一致；前后采样次数不同，也未锁 GPU 时钟，不能说所有实验条件完全相同。V1 对部分算子约束了低精度中间语义，但没有因此证明整个模型与参考逐位一致。约 93 倍／7.8 倍是相对很朴素基线的这一组观测，不代表相对成熟引擎或所有长度的收益。

**验证边界：** 默认参考配置下，9 条固定输入首 token 一致；最多生成 8 个 token 时，仅 **5/9** 序列完全一致，4 条后续分歧尚未完成归因。不能写“完全对齐 HF”或“等质量比 HF 快 X 倍”。序列不同本身不能直接证明实现有错，也不能未经定位就认定全部是无害的舍入差异。

## 2. 面试开场与展开顺序

### 约一分钟的开场

> 这是基于 LLAISYS 教学框架完善的 Qwen2 单卡推理项目，实测模型是 DeepSeek-R1-Distill-Qwen-1.5B。Python 负责权重加载和生成循环，C++/CUDA 负责模型前向、算子和分层 KV Cache，目前支持单请求、batch=1、greedy。
>
> 优化分两层：算子层处理串行归约和重复 Attention 打分，Linear 接入 cuBLAS；执行层复用工作区，让 K/V 直接写缓存，并只给最后一个 token 做词表投影。固定回放下，相对项目朴素基线，首 token 延迟从约 1.32 秒降到 14.1 毫秒，后续每 token 从约 100.8 降到 12.9 毫秒。
>
> 当前算子和小模型回归已有记录，真实模型首 token 9/9 一致，但 8-token 序列仍有 4 条分歧。接下来优先做相同前缀下的数值定位，再扩展验证。

这是项目事实的讲述模板。个人部分另外补一句“我亲自完成／复现／修改的是……”，只填真实经历；不要把 AI 执行过的步骤自动改成自己的排错故事。

### 被要求“挑一个优化详细讲”时

优先选能亲自复现的 Argmax：**一个线程扫描词表 → 单 block 仍只用一个 SM → 多 block 产生局部候选 → 第二阶段合并 → 检查索引、NaN 和尾部 → 解释微基准与整机收益不同。** 对应 Q16、Q24、Q29 和 [Argmax 记录](argmax.md)。

若面试更偏推理引擎，选择 Cache 与工作区：**画一次请求 → 指出新行目标地址 → 解释同 stream 的依赖 → 说明哪些数据可复用 → 用分配计数与 trace 验证。** 对应 Q8～Q12、Q26、Q28 和 [Cache 讲解](KV_CACHE_AND_FORWARD.md)。

| 简历对应内容 | 先练的问答 | 被追问时需要拿出的东西 |
| --- | --- | --- |
| 推理链路与 Cache | Q2～Q5、Q8～Q13 | 形状表、P+N−1 时间表、cache slice 地址 |
| 算子与执行优化 | Q14～Q18、Q28 | 旧版多做了什么、新版线程如何分工、额外开销是什么 |
| 性能结果 | Q22～Q24、Q29～Q30 | 基线版本、固定回放含义、计时范围、原始样本 |
| 正确性与工程判断 | Q19～Q21、Q25～Q27、Q31～Q32 | 同前缀定位方法、profiling 分母、失败候选与当前限制 |

## 3. 贡献边界怎么说

| 部分 | 合适的描述 | 需要区分的事实 |
| --- | --- | --- |
| 框架基础抽象、原 C API、CPU Runtime | 基于既有框架扩展，理解职责并在需要时修复 | 不把整套 Tensor/Runtime 设计归为个人从零实现 |
| 作业阶段 Tensor 方法、七个 CPU 算子 | 按个人实际完成的版本解释 | Git 提交只能证明变更，不能独立证明代码如何产生 |
| Qwen2/CUDA 初版及本轮优化 | AI 辅助完成，再说明自己验证、修改的具体部分 | 阅读理解、亲自复现、独立设计是不同层次 |
| cuBLAS Linear | 集成厂商库，处理布局、dtype、bias、资源复用和验证 | cuBLAS kernel 本身由厂商提供 |
| Attention | 并行 QK/Softmax/PV，共用中间结果 | 仍显式保存 scores，没有实现 FlashAttention |
| Cache | 单请求、连续预分配、分层增量读写 | 没有分页、前缀共享或多请求调度 |

来源核查见[原贡献表](../../RESUME_AND_30_DAY_PLAN.md)。面试被问 AI 使用时，可以说明哪些部分借助了工具，再展示自己能解释、复现和修改的实际内容。无需编造个人试错过程。

## 4. 常见提问与大白话答案

### Q1：你参考的是 llama.cpp 吗？

**短答：** 主要基于 LLAISYS 教学框架，按 Qwen2 结构完成前向，并用本地 HF 实现做参考验证。当前没有做 llama.cpp 移植或性能对照。

**展开：** 先区分框架提供的抽象、自己补齐的功能、参考验证对象和性能基线。这里四者不同：HF 是参考实现，简历性能基线则是项目自己的 V1/E1。仅仅同样使用 C++，不能推断借鉴了某个引擎。

**追问：** “你具体负责什么？”用第 3 节和个人实际改动回答，定位到函数，避免只说“做了整个推理框架”。

### Q2：Python 和 C++ 各负责什么？

**短答：** Python 读 config/safetensors、映射权重并组织生成循环，通过 ctypes/C API 调用后端；C++ 组织前向和 Cache，算子再分发到 CPU 或 CUDA。

**展开：** 输入是 CPU token IDs，前向中把 IDs/positions 上传设备，最后把 Argmax 的一个 ID 取回 CPU。权重加载在模型初始化时完成，不是每个 token 重新加载。这里 HF 用于对照，没有代替 native 模型做前向。

**追问：** “调用链在哪里？”从 [Python 模型](../../python/llaisys/models/qwen2.py) 的 `prefill/decode`，到 [C API](../../src/llaisys/models/qwen2.cc)，再到 [Model::infer](../../src/models/qwen2/model.cpp)。模型 C API 捕获异常，Python 根据返回值读取错误；这不等于整个工程每个接口都具备同样错误协议。

### Q3：一次 infer 为什么只返回一个 ID？为什么只投影最后一行？

**短答：** 当前接口目标是预测下一个 token，greedy 只需要最后位置的 logits 及其 Argmax。优化版只对最后一行做最终 RMSNorm 和词表投影。

**展开：** P=64 时，旧 LM head 产生 `64×151936` 个数，最后仅使用一行；现在输出 `[1,151936]`。裁剪发生在**全部 Transformer 层之后**，prefill 中间各层仍处理 64 行并保存 K/V。最终 Norm 和 Linear 按行独立，因此可以删掉不用的输出行；不能说整个模型计算量少了 64 倍。

**追问：** “训练和困惑度也能这样做吗？”它们可能需要每个位置的 logits，不能直接沿用当前接口。改变 GEMM 形状还可能改变浮点计算路径，因此数学上可裁剪仍需数值回归。代码看 `output_rows/final_input`。

### Q4：prefill 与 decode 有什么不同？

**短答：** prefill 处理整段 prompt；decode 每次只处理一个新 token，并读取已有 K/V。Linear 的 M 从 prompt 长度变成 1，矩阵运算的形状和权重复用机会随之变化。

**展开：** prefill 的多个输入行能复用同一份权重；单 token decode 的 Linear 接近 GEMV，要为较少计算读取大量权重。但 Attention 的 KV 长度仍随生成增加，decode 不会变成固定工作量。哪部分受计算或带宽限制，应看具体 shape 和采样。

**追问：** “首个输出属于哪一步？”prefill 已产生首个预测。8 个输出通常只需要 7 次 decode；不能把 8 次都计为 decode。

### Q5：Q/K/V 在本模型里是什么形状？

**短答：** 对 T 个新 token，hidden 为 `[T,1536]`，Q 为 `[T,12,128]`，新 K/V 各为 `[T,2,128]`；Attention 读取历史加当前的 S 行 K/V，输出 `[T,12,128]`。

**展开：** prompt 有 A/B/C，第一次 decode 输入 D，此时 `T=1,S=4`。Q 只有 D 的一行，但 K/V 有 A～D 四行。Attention 输出展平为 `[T,1536]` 再做 O projection；T 表示本次序列长度，不是 batch。

**追问：** “为什么每层要单独缓存？”每层权重和隐藏状态不同，K/V 也不同。28 层不能共用一份历史值。完整图见 [Cache 讲解](KV_CACHE_AND_FORWARD.md)。

### Q6：GQA 怎样映射头，节省什么？

**短答：** 12 个 Q 头共享 2 个 KV 头，每连续 6 个 Q 头读同一个 KV 头，索引为 `kh=h/(HQ/HK)`。

**展开：** 整数除法下，Q 头 0～5 对应 KV 头 0，6～11 对应 KV 头 1；例如头 7 读 KV 头 1。当前 kernel 根据索引读取，没有先复制成 12 份 K/V。相同长度、层数、head_dim 和 dtype 下，KV 头存储是 12 个 KV 头方案的 1/6。

**追问：** “整机显存也少 6 倍吗？”这个比例只描述 KV 部分，权重和其他存储不随之缩小。Cache 公式见 Q12。

### Q7：K 为什么缓存旋转后的值？V 为什么不做 RoPE？

**短答：** 当前模型用 RoPE 改变 Q/K 的位置相关打分；历史 K 的位置确定后，旋转结果可以重复使用。V 按该模型结构保存内容表示，不做 RoPE。

**展开：** 一个二维配对的旋转是 `(a cosθ−b sinθ, b cosθ+a sinθ)`。本实现按前后半维配对，不应默认是相邻两个通道。新 token 的位置为 `old_len+i`；decode 不能每次都从位置 0 开始，否则形状正确也会改变注意力打分。

**追问：** “为什么 Q 不缓存？”未来步骤要用的是新 Q 与历史 K/V，旧 Q 不参与当前生成路径。代码及舍入处理见 [RoPE](rope.md)。

### Q8：缓存怎样追加？会复制全部历史吗？

**短答：** 每层 K/V 预分配 `[capacity,KV_heads,head_dim]`，新数据写入 `[old_len,cache_end)` 视图，Attention 读取 `[0,cache_end)`。

**展开：** 本模型一行 K 含 `2×128` 个 BF16 元素，占 512 字节。old_len=3 时，新行地址是基址加 1536 字节。`slice` 共享 Storage，只改变元数据与偏移；模型仍持有缓存，销毁临时视图不会把它释放。

**追问：** “直接写入省掉什么？”V Linear 的最终输出和 K RoPE 的输出直接落在 cache slice，省去模型级中转和追加 D2D。含 bias 的 V Linear 仍可能使用 FP32 scratch；K 也仍有未旋转的临时投影。不是没有任何中间存储。代码看 `k_dst/v_dst/v_flat/k_rot`。

### Q9：decode 里谁调度 Cache？需要每层同步吗？

**短答：** Python 发起调用，C++ 逐层提交计算。当前 kernel 与 cuBLAS 在同一 stream 上按依赖顺序执行，不需要在每层额外做设备同步。

**展开：** 同一 stream 中，写 K/V 排在读 K/V 之前；CPU 提交完成不代表 GPU 已完成。最终同步 D2H 取回预测 ID 后，才更新 `_cached_len`，表示本轮所有层共同有效的长度。

**追问：** “能同时处理两个请求吗？”同一个 Model 的 Cache 和工作区会被共同修改，当前不支持并发请求。换成多 stream 需要重新设计依赖与资源隔离；长度最后更新也不等于任意 GPU 故障都能事务回滚。

### Q10：生成 3 个 token，为什么 prefill 后 Cache 只增加 2 行？

**短答：** prefill 已预测出第一个 token；为了预测第二、第三个，只再把前两个喂回去。最后返回的预测还没有做过前向。

**展开：** `[A,B,C] → D` 后长度为 3，输入 D 预测 E 后为 4，输入 E 预测 F 后为 5。输出列表已有 6 个 token，但 F 不在 Cache。对非空输入、N≥1 且未提前结束，最终长度是 `P+N−1`；P64/N8 对应 71 行和 7 次 decode。

**追问：** “EOS 或 N=0 呢？”EOS 可以让实际调用减少，但 `generate` 开始前按请求上限检查容量。N=0 提前返回原输入，不做前向，也不 reset 已有 Cache；不要代入 P+N−1。

### Q11：reset 为何不清空 GPU 内存？prefill 和 infer 有什么区别？

**短答：** reset 把有效长度清零，新请求从头覆盖，Attention 只读取有效前缀，因此无需每次清零整块缓存。

**展开：** 这依赖长度和 slice 边界正确，物理旧数据仍在显存里。`prefill` 验证输入后 reset，再开始新请求；`infer` 则把本次新 token 追加到已有历史。分块输入第二块时，应调用追加接口，不能再次 prefill 把历史清掉。

**追问：** “如何验证不会串上下文？”用重复 prefill 与 HF 参考的预测 ID 对照，并比较分块/逐 token 输入与完整前缀重算，同时检查 reset 状态和容量边界。[小模型回归](../../test/test_engine_regression.py)有这些检查；接口检查与任意 GPU 故障恢复要分开讨论。

### Q12：KV Cache 占多少？默认 2048 是模型最大长度吗？

**短答：** KV 字节数为 `2×层数×capacity×KV头数×head_dim×dtype字节数`。本模型 BF16 每位置 28 KiB，2048 容量共 56 MiB。

**展开：** 代入 `2×28×2048×2×128×2=58,720,256` 字节。按 131072 预留为 3584 MiB，即 3.5 GiB。运行时容量与模型配置的位置上限分开；默认少分配意味着请求长度也受较小容量限制。

**追问：** “是不是把显存压缩了 64 倍？”改变的是预留位置数量，没有压缩每个 KV 值，也没有在同一容量下得到这项节省。模型权重、workspace 和 CUDA context 等仍需另算。实测见 [cache_capacity](results/cache_capacity.jsonl)。

### Q13：Attention 的 mask 如何处理历史偏移？

**短答：** `KL` 包含历史，`QL` 是本次新输入，所以旧长度是 `KL−QL`。第 qp 个新 query 可见 key 数量为 `KL−QL+qp+1`。

**展开：** 历史 4 个，再输入 3 个，QL=3、KL=7，三个 query 分别可看 5、6、7 个 key。若仍用无历史的普通下三角，会错误屏蔽已有上下文。decode 的 QL=1，可见全部有效 K/V。

**追问：** “怎么检查没读未来？”修改未来位置的 K/V，较早 query 的输出应保持不变，同时覆盖有历史偏移的 case。[CUDA 边界测试](../../test/test_cuda_edges.py)包含相关检查。

### Q14：Attention 为什么快了？与 FlashAttention 差在哪？

**短答：** 旧实现让每个输出通道重复计算同一份 QK 打分和 Softmax；新实现算好后共用，分 QK、Softmax、PV 三个 kernel 完成。

**展开：** 固定一个 query/head，输出有 128 个通道，但它们使用同一组注意力概率；变化的是 V 的通道。V1 还为 max、分母和加权和多次重做打分。V2 由一个 warp 协作一个 Q/K 点积，每行 block 归约 Softmax，再按输出通道求 PV。消除重复点积比简单增加线程更关键。

**追问：** “代价是什么？”增加可复用 FP32 scores scratch 和中间读写，空间为 `QL×HQ×KL×4` 字节。它没有分块 online softmax，长上下文仍需另测。完整小例子与代码见 [Attention](self_attention.md)。

### Q15：RMSNorm 怎么并行？为什么需要共享内存？

**短答：** 每行一个 256 线程 block，各线程先求自己负责元素的平方和，warp 内归约，再通过共享内存合并 8 个 warp 的部分和。

**展开：** hidden=1536 时，每线程处理 6 个通道。先得到平方均值，再算 `inv=1/sqrt(mean(x²)+eps)`，最后各线程计算 `x×inv×weight`。共享内存只需保存少量部分和与 inv；warp shuffle 不能直接跨 warp 交换数据，因此还需要 block 级协作与同步。

**追问：** “和 LayerNorm 一样吗？”这里没有减均值。当前低精度路径还在归一化后显式舍入，再乘 weight；并行求和也可能改变舍入次序。公式、小例子与两次同步位置见 [RMSNorm](rms_norm.md)。

### Q16：Argmax 为什么要两阶段？

**短答：** 词表 151936 个元素时，一行一个 block 只能让一个 SM 承担扫描。拆成每块 1024 个元素，先并行产生局部 `(value,index)`，再合并 149 个候选。

**展开：** `ceil(151936/1024)=149`，每块 256 个线程最多各读 4 个元素。全局最大值必然在某一块的局部候选中，前提是两阶段使用相同的比较规则。有限值相同时取最小原索引；当前 V2 对 NaN 选择第一个 NaN，全为负无穷时返回索引 0。索引必须跟着值一起归约。

**追问：** “为什么小行不也这么做？”K≤4096 时保留单 block，避免额外 launch 和 scratch。149 个 block 只是提供更多可调度任务，不保证同时常驻。历史单 block 约 101.75 μs，最终 suite 约 4.04 μs；独立低采样 smoke 约 12 μs，均见 [Argmax 记录](argmax.md)，不承诺每次启动固定 4 μs。

### Q17：为什么 Linear 接 cuBLAS，不用另一个 SGEMM 项目？

**短答：** FP32 SGEMM 的优化经验不能直接证明 BF16、尤其 M=1 的实际模型形状也更快。这里先集成厂商实现，再在真实形状下验证性能和数值。

**展开：** 自己需要负责 Y=XWᵀ 的布局解释、输入输出 dtype、FP32 累加、bias 路径，以及 handle/scratch 复用。带 bias 时，当前先把乘积保留在 FP32 scratch，再加 bias 并转回目标 dtype，避免先转低精度再加 bias 的不同舍入路径。

**追问：** “调用库有什么工作量？”可解释布局错误、bias 舍入差异和真实激活验证；不把厂商 GEMM kernel 算作自研。如果要替换成自己实现，需要同 shape、精度和计时范围的证据。代码与实验见 [Linear](linear.md)。

### Q18：不转置权重也能调用列主序 cuBLAS？

**短答：** 同一段 row-major 内存可以解释为转置矩阵的 column-major 存储。把 `Y=XWᵀ` 改写为 `Yᵀ=W Xᵀ`，不需要实际搬一份转置权重。

**展开：** X 为 row-major `[M,K]`，W 为 `[N,K]`。W 的内存被 cuBLAS 解释为 column-major `[K,N]`，所以对 A 使用转置；X 的内存天然是 column-major `[K,M]`，B 不转置。最终 C 的 column-major `[N,M]` 与所需 row-major `[M,N]` 地址一致。

**追问：** “参数怎么对应？”`m=N,n=M,k=K,transA=T,transB=N,lda=K,ldb=K,ldc=N`。例如 row-major `[[1,2,3],[4,5,6]]` 的连续内存，按 column-major 3×2 看就是 `[[1,4],[2,5],[3,6]]`。这些参数适用于当前连续 Tensor，不能无检查推广到任意 stride。见 [Linear 地址算例](linear.md)。

### Q19：为什么同为 BF16 还会出现差异？

**短答：** 存储类型相同，不代表累加顺序、内部计算精度和转回 BF16 的位置相同；这些差异可能改变接近并列的 logits 排名。

**展开：** 先将 SiLU 结果舍入到 BF16 再乘 up，与全部 FP32 算完再舍入，可能不同。串行和树形归约的加法顺序也不同。即使整体误差很小，也可能在前两名分数接近的位置改变排序；这是说明可能性，不是已测得的本项目根因。

只在 HF 参考端禁用 addmm 的 CUDA Lt 路径，native 不变，序列一致数由 5/9 变为 8/9，见[控制实验](results/reference_no_lt.jsonl)。它支持“参考调用路径会影响结果”，不能证明所有剩余差异无害，也不替代默认参考结果。

**追问：** “怎么判断数值差异是否可以接受？”在相同前缀下检查误差、top-k 与 top1/top2 margin，再结合更广质量验证。当前尚未完成这些分歧的归因，不能仅凭 cosine 高或文字通顺下结论。

### Q20：你怎样定位正确性差异？teacher-forced 是什么？

**短答：** 固定权重、token IDs、dtype、参考计算设置和生成规则，先找到首个分歧，再让两边输入同一条预定前缀，逐阶段比较。

**展开：** teacher-forced 在这里表示“喂入共同选定的 token，不使用各自上一轮预测作为新输入”。例如两边都比较 `[A,B,C,D]` 后的预测，先共同 prefill A/B/C，再共同 decode D。否则一边喂 D、一边喂 X 后，后面的张量已经不属于相同输入，难以归因。

若 prefill 就不同，按 Embedding→Norm→QKV→RoPE→Attention→MLP 找第一个偏离阶段，把同一份阶段输入和权重隔离对照；若 decode 才不同，还要检查 old_len、位置与 Cache 有效范围，并与完整前缀重算比较。

**追问：** “当前工具做到哪了？”[compare_tensors.py](../../benchmark/compare_tensors.py)支持 prefill 的 Embedding、第 0 层和最终 logits 检查；native debug 开关只在 `_cached_len==0` 时导出，后续 decode 不自动覆盖这些快照。小模型已有 teacher-forced ID 检查，真实模型逐步 decode 张量导出和根因闭环仍待补齐。

### Q21：现在正确性到底通过了吗？

**短答：** 已有算子容差和边界检查、小模型回归通过的记录；真实模型首 token 9/9 一致，8-token greedy 序列 5/9 一致，严格生成对齐尚未全部通过。

**展开：** 算子实验覆盖 140 个版本/类型/形状组合，CUDA 边界 57 项，小模型 CPU/GPU 各 36 项。小模型的 HF/完整前缀比较主要检查返回的 Argmax ID，同时验证 Cache 状态和错误处理；不能说逐元素 logits 全部相同。真实模型这 9 条也只是 smoke 集合。

**追问：** “是否能用于生产？”当前支持和验证范围有限，尚缺更广质量、长上下文及多请求等验证。准确讲出已测范围，比用“所有测试通过”概括更有用。[验证记录](results/validation.jsonl)与[真实生成记录](results/generation_compare.jsonl)分别支持不同结论。

### Q22：TTFT、TPOT、token/s 怎样测？固定回放是什么？

**短答：** 在同一次请求中测 prefill 到首个 CPU 预测 ID 的时间，再分别测后续 decode。用主机时钟并同步设备，避免只量到异步提交。

**展开：** 计时范围是 CPU token IDs→模型 API→CPU 预测 ID，包括必要复制和同步，排除加载、分词、网络、排队和打印。每请求先平均其 decode 步耗时，再对请求均值取中位数得到 TPOT；decode token/s 则是全部 decode 步数除以总 decode 时间。只有 1 个输出时没有 decode，TPOT 不应编成 0。

本项目 replay 每步喂固定的 `prompt 最后一个 token ID`，忽略 EOS；不是把模型自己的预测喂回，也不是实际文本生成。这样各版本经历相同输入与 Cache 长度。greedy 才使用自身预测，并保留 EOS 停止。实现见 [benchmark_infer.py](../../benchmark/benchmark_infer.py) 的 `request` 和结果字段。

**追问：** “能用两次 generate 的差值算 TPOT 吗？”独立运行的热身、输出与干扰可能不同，本实现直接保存每次 decode 样本，不用差分替代。

### Q23：为什么 1000/TPOT 有时不等于表里的 token/s？

**短答：** TPOT 用请求均值的中位数，token/s 用所有 decode 步的总时间，两者统计量不同。

**展开：** 假设三条请求各只有一个 decode 步，耗时为 10、10、40 ms，则 TPOT 中位数是 10 ms，但吞吐是 `3/0.06=50 token/s`，不是 100。这个算例用于解释口径；实际请求若提前 EOS，decode 步数还可能不同。

**追问：** “实际数据是哪组？”最终回放的 TPOT 中位数约 12.868 ms，所有步平均约 13.576 ms，对应 73.66 token/s。记录同时保留请求均值和逐步样本，不能挑一个最好值配另一个统计量。

### Q24：单算子快很多，为何整机只快一点？你的总收益来自哪里？

**短答：** 优化只能减少它原本占用的时间；整模型还包含其他算子、分配、复制与 CPU 派发。简历成绩同时包含算子和执行策略改动。

**展开：** 假设某算子只占总时间 1%，加速 2 倍后总耗时仍为原来的 `99%+0.5%=99.5%`。这是讲解用算例，不能拿 GPU kernel 时间占比直接套成主机端到端占比。

本项目阶段表中 V1/E1→V2/E1 的 TTFT 为 1315.8→78.7 ms，随后 V2/E2 为 12.6 ms；最终另一轮复测为 14.1 ms。不能把 1315.8→14.1 全归给 Argmax，或把整个 E2 的收益全归给一次 Cache 写入改动。

**追问：** “做了消融吗？”有保持 E2 并逐项回退算子的[记录](results/ablation.jsonl)。Linear/Attention 的差异明显，小算子回退有时反而更快，说明跨进程主机/时钟波动仍大；小收益需要交错配对测量，不能强行分配百分比。

### Q25：Nsight Systems 和 Compute 各看什么？

**短答：** Systems 看整个请求的时间线和热点；Compute 看一个热点 kernel 的硬件指标。先确定时间花在哪里，再解释为什么。

**展开：** 本次热身后的 P64/N8 trace 中，厂商矩阵运算约占 GPU kernel 时间 92%；这个分母不含全部主机开销。另对一个无 bias decode Linear，X `[1,1536]`、W `[8960,1536]` 采样，NCU 报告 DRAM 吞吐约为峰值的 89.76%，支持这个尺寸偏访存受限。

**追问：** “整个 decode 都一定受带宽限制吗？”单 shape 的 NCU 不能证明全部 Linear 或长上下文的瓶颈。NCU 会采样/重放，报告里的 75.26 μs 也不能混进普通 benchmark 表。证据见 [Nsight 汇总](README.md)与[NCU 明细](results/ncu_decode_linear.txt)。

### Q26：cudaMemcpy 的 API 时间最长，为什么没继续优化 PCIe？

**短答：** 同步复制的 CPU 调用区间可能包含等待此前 GPU 任务的时间，不能等同于设备实际搬运数据的时间。

**展开：** 本次最终 trace 是 16 次小 H2D、8 次小 D2H，无 D2D，设备复制合计约 15.5 μs；API 区间却约 61.7 ms。8 次 infer 中，每次上传 IDs 和 positions、取回预测 ID，与这些次数相符。应对照时间线看等待在哪里，而不是看到 API 表第一名就认定 PCIe 带宽不足。

**追问：** “全部改成 async 就能消除吗？”正常 greedy 需要预测 ID 决定下一步输入。本次 replay 的下一步输入虽已预定，当前 infer API 仍同步取回 ID；能否重叠取决于接口与任务组织，不能只换复制 API 就保证收益。

### Q27：occupancy 低是不是说明 kernel 一定慢？

**短答：** occupancy 描述活跃 warp 相对容量的比例，不等于算力或带宽利用率，不能单独作为优化目标。

**展开：** 所测 GEMV 的 achieved occupancy 约 23.71%，但 DRAM 吞吐约 89.76%，SM 吞吐约 27.74%。这组证据更支持优先考虑数据读取；强行减寄存器可能造成 spill，增加额外访存。

**追问：** “接下来如何选方向？”先复核真实热点和具体 shape，再考虑减少权重字节数或增加权重复用。量化可能增加解包和误差，batch 增大也改变任务场景；两者都要另做实验，当前没有实现收益可报。

### Q28：热身后“0 次分配”意味着什么？工作区为什么能复用？

**短答：** 指框架设备 Tensor 存储的分配计数增量为 0。工作区按名字保存容量，相同模型配置下，行数够用就复用，不够才扩容。

**展开：** prefill 先分配 `normed[64,1536]`，decode 只取前一行；再次遇到 128 行才需要扩容。同 stream 中上一层的读取排在下一层覆盖前，因此部分临时量可跨层复用。但 gate/up 要同时给 SwiGLU 使用，必须保留各自存储，不能把所有临时变量指向同一地址。

**追问：** “整个进程完全不分配了吗？”CPU view/slice 元数据、原始 CUDA scratch 和厂商库活动不在这个 Tensor 统计器内。`reset()` 也不会清除工作区的预留容量或累计计数。代码看 `Model::temporary`，验证看同规模调用前后的 state 和 allocator 增量。

### Q29：算子微基准为何用 CUDA Graph，整机却没用？

**短答：** 微秒级算子若逐次通过 Python 提交，CUDA events 区间可能包含派发间隙。开发基准用 Graph replay 测设备执行，整机则保留真实 API 调用开销。

**展开：** scratch/handle 的首次准备和 graph 创建在微基准计时外；整机 warmup 后测一次次正常 prefill/decode。本项目正常模型路径没有 CUDA Graph，不能因为微基准用了它就写成实现了引擎 Graph 优化。

**追问：** “可以和 PyTorch 的 events 数字直接比吗？”当前原始 PyTorch 算子参考仍经 Python 边界提交，计时范围不同，报告不拿二者计算算子加速比。算法版本比较要使用同一种计时器，并同时检查结果。

### Q30：约 93 倍提速可信吗？你声称比 HF 快了吗？

**短答：** 这是相对项目朴素 V1/E1 的特定固定回放观测，原版有严重串行和重复计算；没有据此宣布优于 HF 或其他成熟引擎。

**展开：** 基线 warmup1/repeat5，最终复测 warmup3/repeat21，未锁 GPU 时钟。数据足以展示这一基础实现的显著改进，但不是严格配对的稳定产品指标。阶段记录用相同采样配置帮助区分改动，复测保留全部波动。

**追问：** “有 HF 对照吗？”有 eager、dynamic cache 的背景数据，但未测试其 SDPA/compile 等配置，且真实生成尚未全部对齐。Cache 策略也不同，因此不作等质量的 HF 加速结论；没有测 llama.cpp。实验条件详见[总表](README.md)。

### Q31：有没有失败的优化，怎么决定撤回？

**短答：** 尝试过 cuBLASLt 融合 bias，但真实激活的数值表现变差、首 token 改变，因此没有保留，也没有报告它的加速收益。

**展开：** 目标是少一次 FP32 product scratch 和单独 bias kernel。`row_3_len_64` 的候选 Q projection 相同元素比例约 65.95%，首 token 从参考 26705 变成 3822；正式路径在这处 Q projection 对照中相同元素为 100%，首 token 与参考一致。这些只描述该次局部实验，不表示正式模型全部生成已通过。

**追问：** “说明 Lt 不好吗？”它只说明当时算法选择和数值约束下的这个候选没有通过，原因尚未全部隔离。不能推广为所有 Lt 都不准确，也不能编造未测的性能提升。候选与证据见 [Lt 实验](experiments/lt_bias/README.md)。

### Q32：下一步优先什么？

**短答：** 先对真实模型 4 条后续分歧完成同前缀定位，再扩展输入与质量验证；性能方向根据实际热点做小范围实验。

**展开：** 选一条失败记录，确定第一个不一致预测的位置；固定到该步的公共输入，检查 cache 长度、位置和 logits，再逐层隔离。现有 prefill 导出不能直接代替 decode 定位，先补必要观察点，并确认关闭诊断后结果不变。

**追问：** “为什么不直接做量化或多 batch？”所测 GEMV 提示减少权重读取可能有价值，但前提是可解释的基线与验证。量化先做受控 Linear 实验，验证误差、解包成本和实际收益；多 batch、分页和服务化属于新增范围，不作为当前成果。

## 5. 个人练习与模拟面试

### 先完成五个练习

每项都按“闭卷短答 → 手算或画图 → 找函数 → 运行并记录 → 解释限制”进行。运行前确认本地权重、环境和动态库已准备好；[实验总表](README.md)保留构建与复现步骤。下面是供个人练习的命令，本次文档修改没有执行这些实验。

| 练习 | 动手任务 | 完成后应能解释 |
| --- | --- | --- |
| 一次请求 | 按 Cache 文档的 state 示例观察 prefill/decode/reset | 3→4→5→0；P64/N8 为何只做 7 次 decode、用 71 行 Cache |
| 一个算子 | 顺序跑 Argmax V1/V2，检查结果并画两阶段 | 149 个候选、原索引为何不能丢、小行为什么不用两阶段 |
| 一个模型指标 | 跑一次 P64/N8 replay，查看 JSONL 原始样本 | 测量边界、统计量、基线与最终版本改变了什么 |
| 一份 profile | 阅读保存的 Systems CSV 和 NCU 摘要 | 92% 与 89.76% 的分母不同；API 等待与设备复制不同 |
| 一个失败用例 | 跑严格生成对照，找第一处分歧；再按 Q20 设计同前缀定位 | 5/9 的范围；已有导出能查什么、仍需补什么 |

在仓库根目录顺序执行；不要同时跑多个 GPU benchmark。使用独立练习目录保存追加输出，避免混入发布数据。开始前确认没有遗留 `LLAISYS_CUDA_<OP>_VERSION` 单算子覆盖、`LLAISYS_QWEN2_DEBUG_DIR` 或参考端 `DISABLE_ADDMM_CUDA_LT=1`；生成和张量对照使用的 `LLAISYS_CUDA_VERSION`、`LLAISYS_ENGINE_VERSION` 应未设置或均为 2。前面 benchmark 的版本选项只改变子进程环境，不会替你清理 shell 中的覆盖。

```bash
.venv/bin/python benchmark/cuda_experiments.py --op argmax --version 1 --dtype bf16 \
  --output benchmark/results/interview_practice/argmax_v1.jsonl
.venv/bin/python benchmark/cuda_experiments.py --op argmax --version 2 --dtype bf16 \
  --output benchmark/results/interview_practice/argmax_v2.jsonl
.venv/bin/python benchmark/benchmark_infer.py --cuda-version 2 --engine-version 2 \
  --prompts docs/optimization/results/prompts.jsonl --sample-id row_3_len_64 \
  --prompt-tokens 64 --output-tokens 8 --mode replay --warmup 3 --repeat 21 \
  --json benchmark/results/interview_practice/engine.jsonl
.venv/bin/python benchmark/compare_prompt.py --all --max-new-tokens 8 \
  --prompts docs/optimization/results/prompts.jsonl \
  --output benchmark/results/interview_practice/generation_compare.jsonl
```

历史默认严格比较是 5/9，脚本因此返回非零；新运行要照实记录结果，不把报错吞掉后写成全部通过。单独跑 `--op argmax` 的随机输入生成顺序与 `--op all` 不同，微秒数据也受热身和时钟影响，不要求逐项复现同一个小数。

若要先查看已实现的 prefill 检查点，可以执行下面一条。它只帮助学习第 0 层与最终 logits 对照，**不完成真实模型后续 decode 的根因定位**：

```bash
.venv/bin/python benchmark/compare_tensors.py --sample-id row_3_len_64 --full-layer \
  --prompts docs/optimization/results/prompts.jsonl \
  --output-dir cache/interview_tensor_debug
```

### 一轮 15 分钟模拟面试

| 时间 | 自问 | 合格回答的依据 |
| --- | --- | --- |
| 0～2 分钟 | 介绍项目和个人贡献 | 说清框架、支持范围、亲自做过的部分与当前限制 |
| 2～5 分钟 | 画一次 P64/N8 请求 | 正确画出 1 次 prefill、7 次 decode、71 行 Cache 与绝对位置 |
| 5～9 分钟 | 任选 Argmax 或 Attention 深挖 | 解释旧版浪费、线程映射、新开销和边界测试 |
| 9～12 分钟 | 质疑“93 倍”和“零分配” | 说清基线、综合收益、采样差异与统计作用域 |
| 12～15 分钟 | 质疑 5/9 和失败优化 | 分清已知证据、推断、未完成工作，给出可执行定位步骤 |

卡住时标注“概念不清 / 找不到代码 / 没亲自跑 / 证据不足”之一，回到对应材料补一项。不要用背更多术语代替这个缺口。

### 个人记录模板

```text
日期 / 主题：
我亲自做的动作与改动：
输入、版本、命令、输出记录路径：
观察到的结果（含失败和波动）：
我的解释，以及证据还不能证明什么：
下一次只验证哪个问题：
```

## 6. 代码与实验查证索引

| 要证明的内容 | 代码入口 | 已保存证据或讲解 |
| --- | --- | --- |
| 权重加载与 Python/C++ 调用 | [qwen2.py](../../python/llaisys/models/qwen2.py)、[C API](../../src/llaisys/models/qwen2.cc) | [贡献来源](../../RESUME_AND_30_DAY_PLAN.md) |
| Cache、工作区、最后一行投影 | [Model::infer/temporary](../../src/models/qwen2/model.cpp)、[Tensor::slice](../../src/tensor/tensor.cpp) | [Cache 讲解](KV_CACHE_AND_FORWARD.md)、[小模型检查](../../test/test_engine_regression.py) |
| Argmax 两阶段 | [argmax_run/argmax_chunks/better](../../src/ops/argmax/nvidia/argmax_nvidia.cu) | [Argmax](argmax.md)、[算子记录](results/operators.jsonl) |
| Linear 布局与 bias | [linear_blas](../../src/ops/linear/nvidia/linear_nvidia.cu) | [Linear](linear.md)、[失败候选](experiments/lt_bias/README.md) |
| Attention 与 RMSNorm | [Attention kernel](../../src/ops/self_attention/nvidia/self_attention_nvidia.cu)、[RMSNorm kernel](../../src/ops/rms_norm/nvidia/rms_norm_nvidia.cu) | [Attention](self_attention.md)、[RMSNorm](rms_norm.md) |
| 分阶段计时与综合收益 | [benchmark_infer.py](../../benchmark/benchmark_infer.py) | [阶段结果](results/engine_stages.jsonl)、[最终复测](results/engine_repeat21.jsonl)、[回退实验](results/ablation.jsonl) |
| 数值与生成验证边界 | [compare_prompt.py](../../benchmark/compare_prompt.py)、[compare_tensors.py](../../benchmark/compare_tensors.py) | [5/9 记录](results/generation_compare.jsonl)、[参考路径控制](results/reference_no_lt.jsonl) |
| 热点与带宽结论 | [profile_decode.py](../../benchmark/profile_decode.py) | [GPU kernel CSV](results/nsys_final_cuda_gpu_kern_sum.csv)、[传输 CSV](results/nsys_final_cuda_gpu_mem_time_sum.csv)、[NCU](results/ncu_decode_linear.txt) |

原始实验日志保持不变，后续若代码或验证结果变化，先补新记录，再同步本文与计划中的结论。更新文档日期不等于重新测过性能。
