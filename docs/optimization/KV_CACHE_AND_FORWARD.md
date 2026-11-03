# 前向推理与 KV Cache：从一条请求讲清楚

文档更新：2026-11-03。本次补充讲解并核对现有代码，没有重跑模型实验；验证与性能数字仍引用 2026-11-02 的记录。

范围是当前工程：Qwen2、单 GPU、单请求、batch=1、greedy。下面说的是实际代码，不是 vLLM 那类服务框架的通用设计。性能数据统一见 [实验总表](README.md)，简历与面试见 [问答](RESUME_AND_INTERVIEW.md)。

阅读时先回答三个问题：**这一轮真正输入了什么、哪些行的 K/V 已经有效、下一轮还需要保留什么。** 再把答案对应到代码：

| 想确认的行为 | 代码入口 |
| --- | --- |
| prefill、decode、generate 如何串起来 | [Python Qwen2](../../python/llaisys/models/qwen2.py)：`prefill/decode/generate` |
| 每层 K/V 如何分配、追加，临时空间如何复用 | [C++ Model](../../src/models/qwen2/model.cpp)：`allocateCache/temporary/infer` |
| slice/view 是否发生数据复制 | [Tensor](../../src/tensor/tensor.cpp)：`slice/view/data` |
| 缓存长度、reset、分块输入怎样检查 | [小模型回归](../../test/test_engine_regression.py) |

## 1. 一次 infer 到底干什么

入口在 `python/llaisys/models/qwen2.py`，计算主干在 `src/models/qwen2/model.cpp::Model::infer`。

`infer(ids)` 把这些**新输入 token**接在已有上下文后面，返回“下一个 token 的预测 ID”。它不会重新输入已经缓存的历史 token。`prefill(ids)` 先验证输入长度和 token 范围，通过后才 reset 并调用 infer；`decode(token)` 要求已经有缓存，再以一个新 token 调用 infer。`generate()` 用这几个接口组织自回归循环。

这些入口的状态语义不同：**新请求用 `prefill`，已有请求追加一段 token 用 `infer`，追加一个 token 可用 `decode`。** 例如先 `prefill([A,B])` 再 `infer([C,D])` 才是分块处理 `[A,B,C,D]`；若第二次调用 `prefill([C,D])`，有效上下文会被重置为 `[C,D]`。非法 `prefill` 在 reset 前拒绝，已有缓存长度保持不变，回归中也检查了这一点。

假设 prompt 是三个 token `[A,B,C]`：

| 调用 | 本次真正计算的输入 | 本次位置编号 | infer 结束后的 cached_len | 返回的预测 |
| --- | --- | --- | ---: | --- |
| prefill([A,B,C]) | A、B、C | 0、1、2 | 3 | D |
| decode(D) | D | 3 | 4 | E |
| decode(E) | E | 4 | 5 | F |

此时用户已经拿到 D、E、F 三个新 token，但缓存只到 E，长度是 5。**F 还没有作为输入做过前向，所以没有 F 的 K/V。** `generate()` 返回的是完整列表 `[A,B,C,D,E,F]`，不是只返回新增的三个 token；列表长度 6 与缓存有效长度 5 表示的是不同东西。

对非空输入、`N >= 1` 且未提前结束的生成，请求 P 个输入、N 个输出，需要 1 次 prefill 和 `N-1` 次 decode，最终缓存长度是 `P+N-1`。例如容量 5 可以完成上表的三个输出，再调用 `decode(F)` 才会超容量。若遇 EOS 提前结束，实际处理的 token 更少；但当前 `generate()` **会在开始前按 `P+max_new_tokens-1` 检查容量**，不会假定模型一定提前输出 EOS。`max_new_tokens=0` 则直接返回输入，不做前向、也不 reset 已有缓存，不能套用这个公式。空输入的 `generate([])` 同样直接返回空列表；它与会拒绝空输入的 `infer([])`、`prefill([])` 不同。

这里有两个“时间”不要混在一起：Transformer 一层层执行；生成循环则一个 token 接一个 token 执行。每生成一个新 token，都要经过全部 28 层。

## 2. 一层中的形状

本地 DeepSeek-R1-Distill-Qwen-1.5B 使用 Qwen2 架构：hidden size=1536，28 层，Q 头=12，KV 头=2，head dimension=128，MLP intermediate size=8960。T 是本次输入长度，S 是历史长度加本次长度。这里没有 batch 维。

```text
input_ids [T]
  → Embedding → hidden [T,1536]
  → 每一层：
      RMSNorm
      Q projection [T,1536] → [T,12,128]
      K projection [T,256]  → [T, 2,128]
      V projection [T,256]  → [T, 2,128]
      Q/K 做 RoPE
      把本次 K/V 追加到本层缓存
      Q 读取历史及本次 K/V → Attention [T,12,128]
      O projection + 残差
      RMSNorm → gate/up projection → SwiGLU → down projection + 残差
  → 最后一行的 RMSNorm [1,1536]
  → LM head [1,151936]
  → Argmax → 一个预测 ID
```

每一层都有不同的权重、隐藏状态，也就有不同的 K/V。不能把第 0 层的缓存给第 1 层用。Q 只参与本次计算，通常不缓存；未来 token 要读的是旧 K/V。

把上表的第一次 decode 代入：本次只有 D，`T=1`；历史 A/B/C 加上 D，`S=4`。Q 是 `[1,12,128]`，新 K/V 各为 `[1,2,128]`，Attention 读到的有效 K/V 各为 `[4,2,128]`。每个 Q 头有 4 个打分，而不是只有当前 D 的一个打分。Cache 避免重算历史投影和 MLP，但 Attention 仍要读取历史 K/V；因此 decode 的工作量并非与上下文长度无关。

本项目为了生成下一个 token，最终只需要最后一行的 logits；因此优化版在**全部 Transformer 层完成之后**，只对最后一行做最终 RMSNorm 和 LM head。中间各层在 prefill 时仍然处理全部 T 行。这不是把前面的 token 丢掉，它们已经通过因果 Attention 影响最后一行，并且 K/V 已写入缓存。

可以这样手算节省在哪里：P=64 时，旧路径最后的 LM head 输出 `64×151936` 个数，实际只用最后 `151936` 个；新路径只算这一行。这里减少的是最后投影的行数，不能说整个模型计算量下降 64 倍。最终 RMSNorm 按行独立，LM head 也按行投影，所以可以裁掉其余行；若要训练或计算整段 token 的损失，则可能需要所有位置 logits，不能照搬当前仅返回一个预测 ID 的接口。改变矩阵形状也可能改变厂商库的计算路径，数学上可裁剪仍需要数值回归。

## 3. 缓存由谁分配，放在哪里

`Model::allocateCache()` 给每层分别创建 K、V Tensor。它们与权重在同一个设备上，BF16 布局都是：

```text
K_cache[layer]: [cache_capacity, num_kv_heads, head_dim]
V_cache[layer]: [cache_capacity, num_kv_heads, head_dim]
```

分配一次，后面反复写入，不会每生成一个 token 就重新创建整个缓存。底层由 Tensor → Storage → Runtime 的设备分配接口申请显存；模型析构时依靠 Tensor 的引用计数释放。

- `cache_capacity`：这一条请求最多容纳多少个已经做过前向的 token。
- `_cached_len`：目前有多少行已经有效。
- `cache_end = old_len + ntoken`：本次追加后有效区间的终点。

原来直接按模型 `max_position_embeddings=131072` 预留。现在 Python 构造器默认 `min(2048, 模型位置上限)`，允许显式指定；超过容量会报错，不会偷偷丢弃上下文。**模型支持的位置上限与本次运行预留容量是两个概念。** 默认容量变小，也就主动限制了可处理的请求长度。

BF16 每元素 2 字节，缓存字节数：

```text
2(K和V) × 层数 × capacity × KV头数 × head_dim × 每元素字节数
= 2 × 28 × capacity × 2 × 128 × 2
```

所以每个位置占 28 KiB，2048 容量占 **56 MiB**，131072 容量占 **3584 MiB（3.5 GiB）**。这是 KV Cache 本身的预留量，**不是整个模型显存**。权重还有约 3.55 GB，另外还有临时空间、CUDA context、cuBLAS workspace 等。

GQA 下，每 6 个连续的 Q 头共享一个 KV 头：`kv_head = query_head / (12/2)`，这里是整数除法。Q 头 0～5 读 KV 头 0，Q 头 6～11 读 KV 头 1；例如 Q 头 7 对应 KV 头 1。缓存存 2 个 KV 头，而不是给 12 个 Q 头各存一份，本工程不通过 repeat_interleave 真正复制出 12 份 K/V。同样层数、长度、head dimension 和 dtype 下，KV 存储是 12 个 KV 头方案的 1/6；这个比例不适用于包含权重和工作区的总显存。

## 4. decode 时怎样读写

这里的“调度”很朴素：**Python 发起一次 decode，C++ 在同一条 CUDA 默认 stream 上按层顺序提交计算。** 没有单独的 KV 调度线程。

在每层里执行以下步骤：

1. 计算本次 token 的 Q/K/V，位置是 `old_len`；prefill 的多个位置则是 `old_len + i`。
2. 得到目标视图 `cache.slice(0, old_len, cache_end)`。slice 改的是形状和起始地址，底层仍然是那块缓存，**不会拷贝整个缓存**。
3. 优化版让 V 的 Linear 输出直接写进 V 的目标视图；让 K 的 RoPE 输出直接写进 K 的目标视图。缓存里的 K 已经带位置信息，历史 K 无需重新做 RoPE。
4. 用 `slice(0,0,cache_end)` 把所有有效历史行连同新行交给 Attention。有效前缀之外的预留行不参与计算。
5. 本层完成后进入下一层。全部层、LM head 和 Argmax 成功，并且预测 ID 传回主机后，才提交 `_cached_len=cache_end`。

以第一次 `decode(D)` 为例，`old_len=3`、`cache_end=4`，本层 K 的目标视图是 `[3,4)`。每行有 `2×128` 个 BF16 元素，因此它的起始地址就是 `K_cache_base + 3×2×128×2` 字节，即基址后 1536 字节。RoPE 将这一行写到最终位置，Attention 随后读 `[0,4)`；原来的前三行不需要搬动。Tensor 的 slice/view 共享 Storage，临时视图销毁不会释放仍由模型持有的缓存。

这里的“直接写入”指最终算子输出落在 cache 视图里。V Linear 含 bias 时，当前 cuBLAS 路径仍会先产生 FP32 scratch，再由 bias kernel 写入目标；K 也仍需要未旋转的投影临时值。优化消除的是额外的模型级 K/V 中转 Tensor 和追加拷贝，不意味着所有中间数据都消失。

同一 stream 内按提交顺序执行，后面的 Attention 能读到前面 kernel 写完的 K/V，因此这条路径不需要额外调用 `cudaDeviceSynchronize` 来保证每层的 K/V 可见。**CPU 已提交 kernel 不等于 GPU 已计算完成**；最后取预测 ID 的 D2H 是同步边界。当前手写 kernel 与 cuBLAS 使用同一个 `launchStream()`，正常推理为默认 stream；单算子开发基准临时切换的私有 stream 不代表模型已经支持多 stream 推理。若未来使用多 stream，就需要重新设计事件依赖。

旧路径会先写临时 K/V，再每层做两次同步 D2D；28 层下，优化路径每次 infer 省掉 56 次额外拷贝调用。最终性能和原始 Nsight 记录见实验总表。调试导出默认关闭，它会引入 D2H 和文件 I/O，不能开着它计时。

当前诊断开关 `LLAISYS_QWEN2_DEBUG_DIR` 只在 **`_cached_len==0` 的首次 infer** 生效，导出 Embedding、第 0 层中间结果及最终 Norm/logits。后续 decode 不导出，以免覆盖首次快照；新一次 reset/prefill 若复用同一目录，则会重新写入。现有 [张量对照脚本](../../benchmark/compare_tensors.py) 因此用于 prefill 检查，不能仅打开开关就声称已经获得逐步 decode 的诊断数据。真实模型后续分歧仍需要补同前缀、逐步导出的定位流程。

## 5. 因果 mask 的位置怎么算

Attention 接口没有传完整 mask 矩阵，而是根据形状计算每个 query 可以访问到哪里：

```text
old_len = kv_len - query_len
第 qp 个新 query 可以看到 key [0, old_len + qp]
代码中的数量 limit = kv_len - query_len + qp + 1
```

例如已有 4 个历史 token，再输入 3 个新 token，QL=3、KL=7，那么三个 query 分别可看到 5、6、7 个 key。普通 decode 是 QL=1，可以看全部已有和本次 token；prefill 多行则各有不同上界。

优化版显式生成 QK 分数、做稳定 Softmax、再乘 V。它仍然会保存分数矩阵，额外空间随 `QL × HQ × KL` 增长，**不是 FlashAttention**。长上下文和大 prefill 仍然会增加内存与计算负担。

## 6. reset、越界和临时空间

`reset()` 只把有效长度清零，不释放和清空显存。下一次请求从第 0 行开始覆盖，Attention 只读取本次有效前缀，所以正常模型调用不会读到上个请求的残留行。它没有把旧数据从物理内存擦除；避免串用上下文依赖正确的长度和 slice 边界。不同请求不能同时共用一个 Model 实例。

临时空间与 KV Cache 的角色不同：KV Cache 保存跨 token 的历史信息；workspace 保存本次计算过程中的隐藏状态、Q、gate 等。优化版按名字缓存这些工作 Tensor，同层形状相同就复用，跨层也复用；需要更长行数时才扩容。已经热身过相同或更大输入后，后续模型前向不再申请新的**设备 Tensor 存储**。这不代表 C++ 没有创建 slice/view 元数据，也不代表 CUDA 库内部永远不分配内存。

例如第一次 prefill 为 `normed` 预留 `[64,1536]`，后面的单 token decode 只取它的 `[0,1)` 行视图；下次 prefill 为 128 行时才需扩容。跨层能复用，是因为上一层对该临时量的读取已经排在下一层写入之前。同一层的 `gate` 与 `up` 却要同时交给 SwiGLU，所以是不同名字、不同存储；不能把所有临时变量随意指向同一块内存。`temporary()` 还会检查 dtype、维数与除第 0 维以外的形状，这里的“只扩容”限于模型配置不变时的行数变化。

入口会先检查容量和 token 范围，再开始写入；非法输入的拒绝可以在回归中验证。`_cached_len` 在预测 ID 返回后才更新，表示整次 infer 的有效前缀已提交；这不等于实现了任意 GPU 故障下的事务回滚。

`model.state` 返回有效长度、容量、Cache/权重/workspace 字节数和 workspace 累计分配次数。Runtime 另有 Tensor allocator 当前量、峰值和累计次数。这些数字不包含算子原始 cudaMalloc scratch、CUDA context、cuBLAS 内部分配；HF 的 PyTorch allocator 统计范围也不同，不能拿两者相减得出“整个引擎省显存多少”。

```python
import llaisys
with llaisys.models.Qwen2(
    'DeepSeek-R1-Distill-Qwen-1.5B',
    llaisys.DeviceType.NVIDIA, cache_capacity=2048,
) as model:
    token = model.prefill([1, 2, 3])
    print(model.state)               # cached_len=3
    token = model.decode(token)
    print(model.state)               # cached_len=4
    token = model.decode(token)
    print(model.state)               # cached_len=5，已取得第3个预测
    model.reset()
    print(model.state)               # cached_len=0，容量和预留内存保留
```

这个例子的 ID 只是演示接口；正式实验使用保存好的 WikiText token IDs。

## 7. 已验证与尚未支持

2026-11-02 的记录中，`test/test_engine_regression.py` 使用随机但固定种子的小型 Qwen2，不用下载权重，GPU/CPU 各通过 36 项检查：HF prefill 对照、HF teacher-forced decode、完整前缀与缓存增量一致、分块输入、reset、精确容量边界、非法 token/C API 错误处理、重复调用工作区不扩容、关闭后拒绝访问、tied embeddings 加载。这里的 HF 和完整前缀对照比较的是**下一 token 的 argmax ID**，不等于已经逐元素验证整份 logits 或所有层的 K/V。真实 1.5B 模型的 9 条固定输入首 token 一致，最多生成 8 个 token 时仅 5/9 序列完全一致，后续分歧尚未完成归因；不能拿小模型通过替代大模型质量评估。

当前没有多 batch、连续批处理、PagedAttention、分页块表、前缀共享、滑动窗口淘汰、并发请求队列和流式网络服务。也没有实现任意模型适配器。面试时可以说“实现单请求的连续预分配分层 KV Cache 和增量推理”，不宜说“实现了高并发 KV Cache 调度系统”。

## 8. 三组口述练习：先短答，再展开

### 练习一：Cache 到底省了什么？

**30 秒短答：** “每层保存历史 token 已经计算好的 K/V。decode 只处理新 token 的投影和 MLP，再用新 Q 读取历史加当前的 K/V；因此省去历史 token 的重复前向，但 Attention 仍随历史长度增加。”

**追问时展开：** 用 `[A,B,C] → D → E → F` 画上面的调用表，指出 F 尚未入 cache；再写出第一次 decode 的 Q `[1,12,128]` 和有效 K/V `[4,2,128]`。如果解释时把 S 写成 1，就遗漏了历史读取。

**自测：** 不看表，计算 P=64、N=8 的 decode 次数和最终 cache 长度。答案为 7 次、71 行；实际最大容量检查也用这 71 行。能同时说明 EOS 提前结束和 N=0 的边界才算讲完整。

### 练习二：直接写 Cache 和工作区复用如何避免出错？

**30 秒短答：** “Cache 先预分配，slice 给出新行的最终地址，V Linear 和 K RoPE 的输出直接落在新行。临时 Tensor 按用途保存并复用。当前都在同一 stream 上按依赖顺序执行，数据仍被读取时不会提前覆盖。”

**追问时展开：** 手算 old_len=3 时的 1536 字节偏移；解释 slice 共享 Storage，`gate/up` 为何必须分别保存，以及为什么最后 D2H 之前 CPU 提交完成不等于 GPU 完成。源码定位到 `k_dst/v_dst`、`temporary()` 和最后的 `memcpy_sync`。

**自测：** 用上面的 state 示例观察 3→4→5→0；对相同形状再次 prefill/decode，比较 `workspace_allocations` 是否保持不变。记录实验条件和实际结果，不把“读过实现”写成“亲自复现通过”。

### 练习三：怎样证明增量推理没读错 Cache？

**30 秒短答：** “固定同一条 token 前缀，比较从头计算和分块缓存计算的结果，再检查位置、长度、reset 与容量边界。比较两个实现时必须喂同样的 token，不能让它们自由生成分叉后还逐层直接比较。”

**追问时展开：** 例如都在比较 `[A,B,C,D]` 的下一个预测，一边完整 prefill 四个 token，另一边先 prefill 三个再 decode D。D 可以是预先固定的 token，不要求它恰好是各自上一轮预测。这样才能把输入分叉排除，专门检查缓存路径与数值差异。

**自测：** 到小模型回归里找到完整前缀/增量、分块输入、reset 和越界四类检查，解释每类能发现什么问题，并指出 ID 一致不能排除哪些数值偏差。真实模型仍有 4 条 greedy 序列分歧，后续同前缀逐步诊断属于待完成工作，不能用这份讲解替代实验结论。
