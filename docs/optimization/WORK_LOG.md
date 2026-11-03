# 优化执行记录

开始日期：2026-11-02。范围：NVIDIA 单卡、单请求、batch=1；八个模型依赖算子、正确性、可复现性能与 KV Cache 解释。

## 约定

- V0：任务开始时的原始 CUDA 实现，源码保存在 baselines/v0/。它可生成文本，但此前九条 BF16 greedy 用例均未与 HF 完全一致，因此性能仅作为历史实现对照。
- V1：保持直观算法，先修正低精度计算顺序；作为优化版数值语义的对照。
- V2：在数值验证基础上优化并行、访存和厂商 BLAS 调用。
- 单算子版本号和引擎改动分别记录；没有实际测量的指标不填推测值。某项改动没有收益或不值得继续优化也照实记录。
- 用同一 GPU、dtype、输入、计时范围比较，保存原始样本、库 hash 与环境。微基准和模型指标分开，不把一个 kernel 的加速倍数当作整模型加速。

## 开始时的进度（后续完成情况见下文）

1. 已保存初始源码和未提交 diff；已扩展第 0 层张量观察点。
2. 已建立真实尺寸的算子实验入口 cuda_experiments.py，正在测 V0。
3. 后续顺序：数值路径验证 → 算子优化 → 模型计时/Cache 管理 → 回归/消融 → 每算子记录与简历问答。

## 本轮完成的版本链

### 1. 初始V0保存与诊断

保存任务开始时12份相关源码、工作区diff与旧benchmark，在`baselines/v0/`。先把layer0检查点扩展到Norm、QKV、RoPE、Attention、残差和MLP，最后补上final norm/logits。没有把“看上去能生成”当作质量通过。

### 2. V1数值顺序

独立输入实验发现：Norm先归一化并转回BF16再乘weight、SiLU先转回BF16再乘up、RoPE中间乘法和Attention概率舍入，都可能影响后续计算。保留直观串行V1作为对照；Attention V1因为多一轮评分，有些shape反而更慢，记录中照实保留。

HF默认允许部分低精度GEMM归约；仅把参考设置改成FP32 reduction就能明显改变Q projection误差。正式记录明确TF32/归约设置，旧默认记录保留为历史，不偷换成同配置提升。

### 3. V2a：八算子第一轮优化

Linear接GemmEx+FP32 bias scratch；Norm用block/warp归约；RoPE每线程处理前后半一对；Attention分QK、softmax、PV共用中间结果；Add/SwiGLU成对读写；Embedding按16字节复制；Argmax先做一行一个block。

初版Python/events微基准受派发间隙影响；试过C++ events连续循环后仍可能有主机提交间隙，最后改为私有stream CUDA Graph设备计时。最终主表只使用`timer=cuda_graph_device_events`，不把不同计时器的数据混起来算加速。

### 4. V2b：Argmax第二轮

真实词表151936时，V2a单block仍约101.75μs；不能因为用了warp shuffle就认为结束。改成1024元素一块的局部候选+第二阶段合并，约3.99μs，最终复测约4.04μs。V2a源码保留在`baselines/v2a/`，早期记录在`operator_pilots.jsonl`，正式V2默认采用V2b。

### 5. 引擎E2a/E2b

E1保留旧执行策略；E2a增加工作区复用、每infer一次positions、只对最后行做最终投影；E2b进一步把V projection和K RoPE直接写到对应cache切片。E2a源码保留在`baselines/engine_v2a/`，早期整机数据在`engine_v2a.jsonl`。本轮最终E2对应E2b，没有给“普通推理”虚加CUDA Graph实现。

缓存容量独立配置，默认2048；补prefill/decode/reset/state/close、模型C API错误协议、Tensor allocator current/peak/计数。修复Context未初始化指针和切设备复制runtime表的问题。越界、非法ID、closed、tied weights、cache/chunk/reset与工作区复用通过小模型实测。

### 6. 未采用：Lt融合bias

尝试过cublasLtMatmul bias epilogue候选，真实layer0误差变大、首token变化，因此撤回，未测速度也不主张收益。详见`experiments/lt_bias/README.md`，保留代码片段及trace。正式路径保持明确FP32 product+单独bias转换。

### 7. 验证与profiling

CPU-only release与NVIDIA release均构建安装成功，最终恢复NVIDIA。GPU/CPU tiny模型各36项、CUDA边界57项、八个旧GPU算子脚本、CPU/NVIDIA extended Runtime、Tensor和Qwen2生命周期检查通过。最终算子微基准140/140通过，三种dtype。

真实模型默认参考首token9/9、8-token序列5/9；另做“只禁用HF addmm Lt”的控制实验，native未改变，一致数8/9。它说明参考库调用路径影响显著，但仍有未闭环分歧，默认5/9没有被替换成通过。

Nsight Systems采样热身后P64/N8一次回放：只剩16H2D、8D2H，无D2D与热身后Tensor分配；厂商矩阵运算约92.1% kernel时间。NCU单组decode GEMV：DRAM89.76%、SM27.74%、75.26μs，支持该组尺寸偏访存受限。原始CSV与TXT在results/，大trace留cache/profiles。

### 8. 环境与报告收尾

发现`.venv`原来装的是复制包，旧测试直接import会读到过期.so，出现invalid device id。用`uv pip install --no-deps -e ./python`改成editable后旧测试通过；新benchmark本身也会优先当前源码。

更正旧学习文档的Cache算术：131072位置=3.5GiB，2048位置=56MiB；旧稿7GiB/112MiB多算一倍。另实际创建131072容量验证state字节数，见results/cache_capacity.jsonl。

所有版本的每算子记录、整机阶段、消融、计时边界和噪声限制已整理；小算子独立整机收益没有被可靠分离，报告不挑最好样本、不把它们各自说成百分比提升。已更新根学习/求职计划和benchmark入口说明。

## 尚未完成、必须如实讲的事项

- 完整BF16生成对齐：默认参考5/9序列通过，受控参考8/9；仍需teacher-forced逐步数值归因和更广质量验证。
- 小算子整机收益的严格配对测量、长上下文特征和总体进程峰值显存尚未完整测量。
- 多batch/服务化/第二GPU平台/量化执行不在本轮范围；本轮没有实现FlashAttention、PagedAttention或引擎CUDA Graph。

本轮已完成授权的算子优化、单请求cache管理与解释、性能实测和简历问答材料；不把上述范围外功能或尚未通过验证的结论包装成成果。

最终收尾：Python静态检查、语法编译、本地Markdown链接和diff检查通过，源码/库校验值一致。最终安装库的Argmax独立smoke与Qwen2生命周期检查通过；独立低采样Argmax出现约12μs结果，已在对应记录和总表说明时钟/采样波动，不隐藏不利测量。

## 2026-11-03：统一简历与面试讲解

本次仅整理文档，沿用 2026-11-02 的实验记录，没有重跑 GPU、构建或模型回归，也没有解决剩余生成分歧。此前的代码验证结论仍属于上面的历史实验。

- 以 [简历与面试主文档](RESUME_AND_INTERVIEW.md)统一三条文案，补模型、朴素基线和算子/执行策略的共同归因；保留不同采样配置与 9/9、5/9 的验证范围。
- 保留 32 个问答编号，补充短答、算例、代码入口、追问、五项动手练习、15 分钟模拟面试及个人记录模板。明确小模型回归主要检查预测 ID 和状态，prefill 张量导出不等于已完成 decode 逐层定位。
- 同步 Cache、Argmax、Linear、Attention、RMSNorm 的讲解，以及学习/投递计划、实验总表、中英文 README 和 benchmark 导航。原始结果文件不变，不把 AI 辅助完成或阅读过材料当作个人已经独立掌握。

文档检查：12 份材料的 207 个本地文件链接与代码围栏通过；两处简历逐字一致，32 题连续且均有短答/展开/追问；5 条个人练习命令的 shell 语法、脚本路径与 CLI 参数通过静态核对。阶段数据、最终中位数及生成一致数与原始 JSONL 一致。保留原 README 中已存在的空白，不为整理问答改动课程正文。练习命令本次未执行。
