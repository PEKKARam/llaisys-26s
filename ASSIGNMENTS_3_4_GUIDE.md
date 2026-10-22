# 作业 3 与作业 4 实施指南

> 状态说明（2026-09-14）：本文保留作业要求和验收清单；早期“尚未实现”的状态描述已按当前代码更新。
> 面向学习和代码导读的主文档见 [`ASSIGNMENT_3学习路线与Qwen2实现导读.md`](ASSIGNMENT_3学习路线与Qwen2实现导读.md)。

本文根据 `README_ZN.md`、当前代码骨架和测试脚本整理。它既是开发任务清单，也可以作为最终 PR 复现报告的底稿。

## 1. 最终要交付什么

### 作业 3：大语言模型推理

在 CPU 上使用 LLAISYS 自己的 C/C++ 后端完成 `DeepSeek-R1-Distill-Qwen-1.5B` 推理。Python 只负责读取配置、读取 safetensors、调用 C API 和组织生成循环，不能用 PyTorch 等 Python 框架实现模型计算。

必须达到的结果：

- 能创建和销毁 Qwen2 模型。
- 能把 Hugging Face 模型配置和全部权重载入 C/C++ 后端。
- 能执行 Qwen2 前向计算并通过 argmax 逐 token 生成。
- 必须实现 KV Cache，避免每生成一个 token 都重算完整上下文。
- 在 `--test` 模式下，LLAISYS 生成的完整 token 序列与 Hugging Face/PyTorch 完全相同。
- `python test/test_infer.py --model <模型目录> --test` 通过。

### 作业 4：集成 CUDA/类 CUDA 平台

将 Runtime、作业 3 使用的全部算子和模型推理迁移到 GPU，并适配以下平台中的至少两款：

- NVIDIA
- 天数智芯（Iluvatar）
- 摩尔线程（Moore Threads）
- 沐曦（MetaX）

必须达到的结果：

- 每个平台都有独立、可开关的构建配置。
- 每个平台实现完整 Runtime API，包括设备、流、内存和拷贝管理。
- 每个平台实现并测试模型需要的 GPU 算子。
- 作业 3 的模型可以在对应 GPU 上加载权重并完成推理。
- Runtime 测试、算子测试和 GPU 推理测试通过。
- PR 中逐个平台记录环境、构建命令、测试结果、已知限制和支持状态。

> 注意：README 明确要求“至少两款平台”。当前仓库只定义了 `CPU` 和 `NVIDIA`，测试脚本的 `--device` 也只接受这两项。因此，仅完成 NVIDIA 不满足作业文字要求；第二个平台的枚举、Python 映射、构建脚本、Runtime、算子分发和测试入口都需要自行扩展。

## 2. 当前仓库状态与缺口

### 已有基础

- CPU Tensor、Runtime 和主要算子已经存在并通过原生测试。
- Qwen2 C API、C++ 模型对象、权重槽位、KV Cache 和完整前向已经实现。
- Python 已完成 config/safetensors 读取、权重映射、ctypes 调用和 argmax 生成循环。
- NVIDIA Runtime、CUDA 算子和 `--nv-gpu` 构建路径已经接入；本地 NVIDIA 端到端 token 对齐已验证。

### 作业 3 主要缺口

- 需要补充更系统的 CPU 真实模型验收、中间 Tensor 对齐和长序列边界测试。
- 需要区分并记录 prefill/decode 性能，当前 CPU 实现仍是教学型朴素算子。
- 需要进一步清理 Python 参考辅助路径，并完善 C API 错误处理语义。

### 作业 4 主要缺口

- 当前只支持 CPU 和 NVIDIA；课程要求的第二个 GPU/CUDA 类平台尚未适配。
- CUDA 算子目前主要是正确性基线，linear/attention 仍需使用专用库和 kernel 做性能优化。
- 需要修正并固化 GPU self-attention 测试脚本的参考 mask 设备问题。

## 3. 推荐实施顺序

按下面顺序推进，前一阶段通过测试后再进入下一阶段：

1. 确认作业 1、2 的 CPU Tensor 和算子测试全部通过。
2. 实现 Qwen2 C/C++ 模型对象和模型 C API。
3. 实现 ctypes 绑定、配置解析和权重加载。
4. 实现 CPU 前向、KV Cache 和 argmax 生成，通过作业 3。
5. 选定并申请两种 GPU 平台环境，先完成 NVIDIA Runtime。
6. 完成 NVIDIA 算子和模型推理，独立通过全部 GPU 测试。
7. 复制同一抽象边界适配第二个平台，避免在模型代码中写平台专用计算。
8. 做正确性、长序列、内存和性能验证，整理 PR 报告。

## 4. 作业 3 详细任务

### 4.1 定义模型后端与 C API

建议保留 `include/llaisys/models/qwen2.h` 中已有的四类基本接口：

- `llaisysQwen2ModelCreate`：接收模型元数据、设备类型和设备 ID，创建模型、权重 Tensor、临时工作区和 KV Cache。
- `llaisysQwen2ModelDestroy`：释放模型拥有的 Tensor、缓存和资源。
- `llaisysQwen2ModelWeights`：返回稳定的权重句柄集合，供 Python 将 safetensors 数据复制进去。
- `llaisysQwen2ModelInfer`：执行 prefill 或 decode，返回 argmax 得到的下一个 token ID。

需要新增：

- `src/llaisys/models/qwen2.cc`（或遵循仓库结构的同类路径）：导出 C API，只做参数转换和调用 C++ 模型。
- `src/models/qwen2/`：存放模型类、层结构、推理逻辑和缓存管理。
- 对模型生命周期、重复调用、空输入、超长序列和设备 ID 做明确校验。
- 将新增源文件和依赖目标加入 Xmake。

建议明确 API 语义：第一次调用传入完整 prompt，完成 prefill；后续调用只传入新 token，沿用 KV Cache。若每次传完整 token 列表，则后端必须记录已缓存长度并只计算新增后缀。新一轮对话需要有显式 reset API，或者定义“输入不再是现有前缀时自动清空缓存”的可靠规则。

### 4.2 解析 Hugging Face 配置

在 Python 构造函数中读取模型目录下的 `config.json`，填充 `LlaisysQwen2Meta`。至少映射：

| LLAISYS 字段 | Hugging Face 配置字段 |
| --- | --- |
| `dtype` | 权重 dtype/实现选择，目标模型通常为 BF16 |
| `nlayer` | `num_hidden_layers` |
| `hs` | `hidden_size` |
| `nh` | `num_attention_heads` |
| `nkvh` | `num_key_value_heads` |
| `dh` | `hidden_size / num_attention_heads` |
| `di` | `intermediate_size` |
| `maxseq` | `max_position_embeddings` |
| `voc` | `vocab_size` |
| `epsilon` | `rms_norm_eps` |
| `theta` | `rope_theta` |
| `end_token` | `eos_token_id` |

还要检查 `tie_word_embeddings`。若为真，输出头可以与输入 embedding 共享；若为假，必须单独加载 `lm_head.weight`。

### 4.3 建立 ctypes 绑定

建议新增 `python/llaisys/libllaisys/qwen2.py`：

- 用 `ctypes.Structure` 精确声明 `LlaisysQwen2Meta` 和 `LlaisysQwen2Weights`，字段顺序必须和 C 结构一致。
- 为四个 C API 设置完整的 `argtypes` 和 `restype`。
- 在 `python/llaisys/libllaisys/__init__.py` 中调用 `load_qwen2(LIB_LLAISYS)` 并导出所需类型。
- 模型句柄用不透明 `c_void_p` 表示；token 数组用连续的 `int64` 缓冲区传递。
- Python `Qwen2` 对象保存模型句柄，并保证析构只执行一次。

### 4.4 加载 safetensors 权重

遍历所有 `*.safetensors` 分片，把 Hugging Face 名称映射到 `LlaisysQwen2Weights`：

| Hugging Face 权重 | LLAISYS 权重 |
| --- | --- |
| `model.embed_tokens.weight` | `in_embed` |
| `lm_head.weight` | `out_embed` |
| `model.norm.weight` | `out_norm_w` |
| `model.layers.i.input_layernorm.weight` | `attn_norm_w[i]` |
| `model.layers.i.self_attn.q_proj.weight/bias` | `attn_q_w[i]` / `attn_q_b[i]` |
| `model.layers.i.self_attn.k_proj.weight/bias` | `attn_k_w[i]` / `attn_k_b[i]` |
| `model.layers.i.self_attn.v_proj.weight/bias` | `attn_v_w[i]` / `attn_v_b[i]` |
| `model.layers.i.self_attn.o_proj.weight` | `attn_o_w[i]` |
| `model.layers.i.post_attention_layernorm.weight` | `mlp_norm_w[i]` |
| `model.layers.i.mlp.gate_proj.weight` | `mlp_gate_w[i]` |
| `model.layers.i.mlp.up_proj.weight` | `mlp_up_w[i]` |
| `model.layers.i.mlp.down_proj.weight` | `mlp_down_w[i]` |

加载时必须：

- 验证名称、层号、shape、dtype 和总权重数，未知或缺失权重不能静默忽略。
- 使用 safetensors/NumPy 暴露的连续内存地址调用 `tensorLoad`；不要让 NumPy 数组在复制完成前被回收。
- 确认 BF16 的读取与位级复制路径。若 NumPy 无法直接表达 BF16，应按原始 16 位数据复制，不能先转成错误格式。
- GPU 模型仍从主机读取文件，再由 Runtime 的 H2D 拷贝进入目标设备。

### 4.5 实现 Qwen2 前向

单层 Transformer 的计算顺序应与 Hugging Face Qwen2 保持一致：

1. token embedding 得到 hidden states。
2. 对输入做 RMSNorm。
3. 分别做 Q/K/V Linear，其中 Q/K/V 带 bias。
4. reshape 为 `[seq, heads, head_dim]`，对 Q/K 应用 RoPE。
5. 把本次 K/V 写入该层 KV Cache。
6. 用 Q 和截至当前位置的缓存 K/V 做 causal grouped-query attention，缩放为 `1 / sqrt(head_dim)`。
7. attention 输出经过 O Projection，与残差相加。
8. 对 attention 残差结果做第二个 RMSNorm。
9. 分别计算 gate projection 和 up projection，执行 SwiGLU，再做 down projection。
10. 与第二条残差相加，进入下一层。
11. 最后一层输出做 RMSNorm 和 LM head Linear，只取最后位置 logits。
12. 对 logits 做 argmax，返回下一个 token ID。

重点核对：

- Qwen2 使用 GQA，`nh` 通常大于 `nkvh`，K/V 头要按组匹配 Q 头。
- causal mask 在 prefill 和 decode 时的对角线位置不同；已有 `self_attention` CPU 算子以 `kv_len >= query_len` 表达带历史缓存的情况。
- Linear 权重布局应与 safetensors 的 `[out_features, in_features]` 一致。
- RoPE 的 position ID 从当前缓存长度开始，不能在每次 decode 时重置为 0。
- 中间 Tensor 的 shape、stride、连续性和 dtype 必须符合现有算子检查。
- BF16 累加敏感操作（RMSNorm、softmax、点积）建议内部使用 FP32 累加，再转换回模型 dtype。

### 4.6 实现 KV Cache

每层至少维护 K Cache 和 V Cache，逻辑形状建议为：

```text
[max_sequence_length, num_key_value_heads, head_dim]
```

需要保存当前有效长度，并支持两种阶段：

- Prefill：一次处理 prompt 的全部 token，将产生的 K/V 写入 `[0:prompt_len]`。
- Decode：每次只处理一个新 token，把 K/V 写入当前位置，并让 attention 读取全部有效缓存。

必须检查：

- `cached_len + new_len <= maxseq`。
- 每层缓存长度一致。
- 新会话能清空缓存。
- 缓存位于模型所在设备，decode 过程中不在 CPU/GPU 间来回复制整块缓存。
- Tensor view/slice 的生命周期不会导致底层存储提前释放或重复释放。

### 4.7 实现 Python 生成循环

`Qwen2.generate()` 在 `top_k == 1` 时执行确定性 argmax：

- 保留输入 token 列表。
- 首次将 prompt 送入后端得到 next token。
- 逐步追加 next token，并仅让后端处理新增 token/KV Cache。
- 达到 `max_new_tokens` 或生成 `end_token` 时停止。
- 返回“原始输入 + 新生成 token”的完整列表，因为测试会与 Hugging Face 的完整输出直接比较。

作业测试只要求 argmax。`top_p`、`temperature` 和 `top_k > 1` 可以明确报 `NotImplementedError`，或另行实现采样；不要悄悄忽略参数并产生语义错误。

### 4.8 作业 3 调试与验收

先做小输入逐层对齐：embedding、每层 norm、Q/K/V、RoPE、attention、MLP、最终 logits 和 argmax。使用现有 Tensor `debug()` 与 Hugging Face hook/中间结果比较，优先定位第一个出现明显误差的张量。

验收命令：

```bash
xmake f -c -v
xmake
xmake install
python test/test_runtime.py --device cpu
python test/test_tensor.py
python test/test_ops.py
python test/test_infer.py --model <模型目录> --test
```

如仓库实际没有 `test/test_ops.py` 聚合脚本，则逐个执行 `test/ops/*.py`。至少再验证空 prompt/最短 prompt、不同 prompt、提前 EOS、达到最大生成长度和重复创建/销毁模型。

## 5. 作业 4 详细任务

### 5.1 先确定两个目标平台

最早需要确认：

- 两个平台的机器、驱动、编译器和 SDK 是否可用。
- CUDA/类 CUDA API、BLAS 库、编译器命令和架构参数。
- 平台是否能兼容 CUDA 源码，哪些头文件/API/库名需要条件编译。

推荐先把 NVIDIA 做成参考实现，再适配第二个平台。平台差异应封装在 `src/device/<platform>/`、`src/ops/*/<platform>/` 和对应 Xmake 文件中，模型层只依赖通用 Tensor/ops API。

### 5.2 扩展设备类型和构建开关

每个新增平台都要贯通：

- `include/llaisys.h`：增加连续的 `LLAISYS_DEVICE_*` 枚举并更新 `TYPE_COUNT`。
- `python/llaisys/libllaisys/llaisys_types.py`：增加数值完全一致的 `DeviceType`。
- `test/test_utils.py`：设备名到 LLAISYS/PyTorch 设备的映射。
- 测试脚本：增加命令行 `choices`。
- `src/device/runtime_api.hpp/.cpp`：增加宏保护的 `getRuntimeAPI()` 声明和分发。
- `xmake.lua`：增加独立开关，例如 `--nv-gpu` 和第二平台开关。
- `xmake/<platform>.lua`：定义编译器、源文件、include、链接库、架构参数和 target 依赖。

关闭某个平台开关时，该平台源码不应参与编译，CPU-only 构建仍必须成功。

### 5.3 实现 Runtime API

NVIDIA 版本应将通用 API 映射到 CUDA Runtime：

| LLAISYS Runtime API | NVIDIA CUDA 对应操作 |
| --- | --- |
| `get_device_count` | `cudaGetDeviceCount` |
| `set_device` | `cudaSetDevice` |
| `device_synchronize` | `cudaDeviceSynchronize` |
| `create_stream` | `cudaStreamCreate` |
| `destroy_stream` | `cudaStreamDestroy` |
| `stream_synchronize` | `cudaStreamSynchronize` |
| `malloc_device` | `cudaMalloc` |
| `free_device` | `cudaFree` |
| `malloc_host` | `cudaMallocHost` 或 `cudaHostAlloc` |
| `free_host` | `cudaFreeHost` |
| `memcpy_sync` | `cudaMemcpy` + copy-kind 映射 |
| `memcpy_async` | `cudaMemcpyAsync` + 指定 stream |

要求：

- 每次 CUDA/平台 API 调用都检查返回码，并给出包含操作名和错误字符串的异常。
- 正确映射 H2H、H2D、D2H、D2D。
- 零字节分配/拷贝、空指针和无设备环境的行为明确。
- `memcpyAsync` 的声明、定义和函数指针签名一致。
- 资源析构前使用正确设备上下文；流、BLAS handle 和已分配内存无泄漏。

Runtime 验收：

```bash
xmake f --nv-gpu=y -c -v
xmake
xmake install
python test/test_runtime.py --device nvidia
```

现有 Runtime 测试只覆盖同步 H2D、D2D、D2H，建议补充：设备数量/切换、host pinned memory、异步拷贝、stream synchronize、device synchronize、多个 stream 和多 GPU。

### 5.4 实现 GPU 算子

当前模型和测试要求至少覆盖：

- `add`
- `argmax`
- `embedding`
- `linear`（可用 cuBLAS 或平台 BLAS）
- `rms_norm`
- `rope`
- `self_attention`
- `swiglu`
- 模型实际需要时的 `rearrange`/连续化操作

每个算子需要：

- 在 `src/ops/<op>/<platform>/` 中提供声明和实现。
- 在 `src/ops/<op>/op.cpp` 中加入相应设备分发。
- 在平台 Xmake target 中编译 `.cu`/平台源文件并链接所需库。
- 支持模型实际使用的 BF16，同时覆盖现有测试要求的 dtype 和 shape。
- 使用当前 Context 的设备和 stream；如创建 BLAS handle，将其放入平台 Resource 并与 Runtime 生命周期绑定。
- kernel launch 后检查 launch error；在测试/调试阶段适当同步以暴露异步错误。

优化优先级建议：

1. 先保证逐算子数值正确。
2. Linear 使用厂商 BLAS，确认行列主序、转置和 BF16 compute type。
3. 对 RMSNorm、RoPE、SwiGLU、add、embedding、argmax 写简单 kernel。
4. 先实现正确的 causal GQA attention，再考虑 fused/Flash Attention。
5. 用 profiling 确认瓶颈后再融合算子，避免在正确性尚未稳定时扩大调试范围。

逐算子验收示例：

```bash
python test/ops/add.py --device nvidia
python test/ops/linear.py --device nvidia
python test/ops/embedding.py --device nvidia
python test/ops/rms_norm.py --device nvidia
python test/ops/rope.py --device nvidia
python test/ops/self_attention.py --device nvidia
python test/ops/swiglu.py --device nvidia
python test/ops/argmax.py --device nvidia
```

第二个平台执行同一组测试，只替换 `--device` 值。测试容差应与 dtype 相匹配，不能为了通过测试而无依据地放宽。

### 5.5 支持 GPU 模型推理

作业 3 模型代码应尽量无需改计算逻辑，只需：

- 创建权重、临时 Tensor 和 KV Cache 时使用所选设备及 device ID。
- 加载权重时通过对应 Runtime 做 H2D。
- 所有算子按 Tensor device 自动分发到对应 GPU 实现。
- argmax 结果只拷回一个 token，而不是把全部 logits 拷回主机。
- 避免 decode 循环中的重复分配；预分配工作区和 KV Cache。

NVIDIA 推理验收：

```bash
python test/test_infer.py --model <模型目录> --test --device nvidia
```

第二个平台也必须运行等价命令。除 token 完全一致外，应记录首 token 延迟、decode tokens/s、峰值显存和测试 prompt/长度。

## 6. 测试矩阵与完成定义

| 层级 | CPU | NVIDIA | 第二平台 |
| --- | --- | --- | --- |
| 构建与安装 | 必须通过 | 必须通过 | 必须通过 |
| Runtime | 必须通过 | 必须通过 | 必须通过 |
| Tensor 基础功能 | 必须通过 | 建议覆盖 | 建议覆盖 |
| 每个基础算子 | 必须回归 | 必须通过 | 必须通过 |
| Qwen2 argmax 推理 | 作业 3 必须通过 | 作业 4 必须通过 | 作业 4 必须通过 |
| 重复运行/资源释放 | 必须检查 | 必须检查 | 必须检查 |
| 性能与显存记录 | 建议记录 | 必须记录 | 必须记录 |

单个平台只有在以下条件全部满足时才能标为“支持”：

- 从干净环境可按文档成功构建和安装。
- Runtime 测试无跳过且通过。
- 所有模型依赖算子均通过参考对比。
- Qwen2 在该平台完成 argmax 推理且 token 与 PyTorch 一致。
- 没有已知崩溃、明显内存泄漏或每步全量重算等问题。

## 7. PR 与复现报告内容

README 的提交要求是：CI 通过，并提交简要复现报告，逐个平台说明状态。建议 PR 或报告包含：

1. 实现概述：Qwen2、KV Cache、Runtime、GPU 算子和两个平台。
2. 平台环境：机器/GPU、驱动、SDK/Toolkit、编译器、Xmake、Python 版本。
3. 构建步骤：每个平台从配置到安装的完整命令。
4. 模型准备：模型名称、revision/commit、下载或本地目录约定。
5. 测试步骤：Runtime、算子和推理命令。
6. 正确性结果：通过数量、生成 token 是否完全一致、数值误差范围。
7. 性能结果：prompt 长度、生成长度、首 token 延迟、tokens/s、峰值显存。
8. 平台支持表：`已支持`、`部分支持` 或 `未支持`，并解释限制。
9. 已知问题：不支持的 dtype、shape、采样方式、多 GPU 或长序列限制。

## 8. 可直接使用的开发检查清单

### 作业 3

- [ ] CPU Runtime、Tensor 和全部已有 CPU 算子测试通过。
- [ ] Qwen2 C++ 模型类和资源生命周期完成。
- [ ] 创建、销毁、权重访问、推理 C API 完成。
- [ ] Xmake 编译并导出模型符号。
- [ ] ctypes 元数据、权重结构和函数签名完成。
- [ ] `config.json` 字段映射与校验完成。
- [ ] 所有 safetensors 分片和权重映射完成。
- [ ] BF16 权重位级加载正确。
- [ ] embedding、Transformer layers、final norm、LM head 完成。
- [ ] causal GQA、RoPE position 和残差顺序与 Qwen2 对齐。
- [ ] KV Cache prefill/decode/reset/上限检查完成。
- [ ] Python argmax 生成、EOS 和最大长度处理完成。
- [ ] 中间张量逐层对齐完成。
- [ ] CPU `test/test_infer.py --test` token 完全一致。

### 作业 4

- [ ] 已确认两款目标平台并获得可用环境。
- [ ] 两个平台的设备枚举、Python 映射和测试入口完成。
- [ ] 两个平台的独立构建开关和 Xmake 文件完成。
- [ ] CPU-only 构建没有被 GPU 改动破坏。
- [ ] 两个平台的 12 项 Runtime API 完成并检查错误码。
- [ ] 同步/异步拷贝、stream、pinned host memory 和多设备测试完成。
- [ ] 两个平台的全部模型依赖算子完成。
- [ ] BF16 和现有测试 dtype/shape 正确。
- [ ] 两个平台逐算子测试全部通过。
- [ ] 权重、工作区和 KV Cache 正确放置在目标 GPU。
- [ ] 两个平台 Qwen2 argmax token 均与 PyTorch 完全一致。
- [ ] 无每 token 重复分配、整 logits 回传或整段上下文重算。
- [ ] 已记录性能、显存、环境和已知限制。
- [ ] CI 通过，PR 复现报告完整。

## 9. 两个需要尽早确认的问题

1. **第二平台选哪一个。** 这决定 SDK、构建工具、设备枚举、测试环境和可复用 CUDA 代码比例，不能留到 NVIDIA 完成后才申请环境。
2. **课程自动测试对第二平台的命名和接口约定。** 当前公开测试只接受 `cpu/nvidia`，但作业文本要求两个平台。应向导师确认第二平台期望的设备字符串、宏名、构建参数和 CI 运行方式，避免实现正确但接口命名不被测试识别。
