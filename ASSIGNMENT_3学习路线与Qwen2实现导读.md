# Assignment 3 学习路线与 Qwen2 推理实现导读

本文是当前仓库的主学习入口，面向希望把 LLAISYS 做成简历项目、并真正理解推理引擎实现的人。它以 Assignment 3（Qwen2 推理）为第一阶段目标，描述当前代码的真实状态、阅读顺序、知识补齐顺序、验证方法和后续工程化方向。

## 1. 先建立整体认识

这个项目不是“用 Python 调一个模型库”，而是一个分层的推理系统：

```text
模型文件(config.json / safetensors)
        │
        ▼
Python Qwen2 包装器：解析配置、映射权重、组织生成循环
        │ ctypes C API
        ▼
C++ Model：保存权重槽位、KV Cache，执行完整 Transformer 前向
        │
        ▼
算子分发层：按设备选择 CPU 或 NVIDIA 实现
        │
        ▼
Tensor / Runtime / Device：内存、dtype、shape、stride、拷贝和同步
```

理解这条边界非常重要：Python 可以读取配置和 safetensors，也可以调用 C API；模型计算本身应该在 C++/CUDA 后端完成。`Qwen2.generate()` 只负责把 prompt 变成 token、调用后端并循环生成。

当前仓库已经具备：

- CPU Tensor、Runtime 和八个基础算子；
- NVIDIA Runtime 和对应 CUDA 算子；
- C++ Qwen2 模型对象、权重槽位、KV Cache 和完整前向；
- Python safetensors 权重加载和 ctypes 绑定；
- 本地 `DeepSeek-R1-Distill-Qwen-1.5B` 的 NVIDIA 端到端 argmax 结果与 Hugging Face 对齐。

仍未达到课程最终全部要求的部分：第二个 GPU 平台尚未实现；CPU 大模型性能尚未优化；测试脚本和 benchmark 还需要继续工程化。

## 2. 代码应该按什么顺序阅读

不要一开始从 1.5B 模型的 28 层循环读起。建议沿着一次数据流逐层深入。

### 第一步：Tensor 和 Runtime

先阅读：

- `include/llaisys/tensor.h`
- `src/tensor/tensor.hpp`、`src/tensor/tensor.cpp`
- `src/core/storage/`
- `src/core/context/`
- `src/device/runtime_api.hpp`
- `src/device/cpu/cpu_runtime_api.cpp`
- `python/llaisys/tensor.py`

需要回答的问题：

1. Tensor 的 shape、stride、dtype、offset 和 storage 分别表示什么？
2. `view`、`permute`、`slice` 为什么通常不复制数据？
3. 什么情况下必须调用 `contiguous()`？
4. CPU 指针和 CUDA 指针为什么不能互相解引用？H2D、D2H、D2D 拷贝由谁负责？
5. `Context` 如何选择当前设备，Runtime API 如何被统一调用？

先运行：

```bash
xmake -y
xmake install -y
PYTHONPATH=python .venv/bin/python test/test_tensor.py
PYTHONPATH=python .venv/bin/python test/test_runtime.py --device cpu
```

### 第二步：先学懂八个算子

阅读每个算子的三个位置：

1. `include/llaisys/ops.h` 或对应公共声明；
2. `src/ops/<op>/op.cpp` 的设备分发；
3. `src/ops/<op>/cpu/` 的实际计算。

建议顺序：

```text
add → embedding → linear → rms_norm → rope → self_attention → swiglu → argmax
```

重点掌握的数学和 shape：

- `embedding`: `[token] -> [token, hidden]`；
- `linear`: `Y = X Wᵀ + b`，权重保存为 `[out_features, in_features]`；
- `rms_norm`: 沿最后一维计算均方根并乘以可学习权重；
- `rope`: 对 head dimension 的成对通道按 position 旋转；
- `self_attention`: Q/K/V 的点积、causal mask、softmax、GQA 的 KV head 复用；
- `swiglu`: `silu(gate) * up`；
- `argmax`: 最后一个位置的 vocab logits 取最大索引。

每学一个算子都要同时看测试输入的 shape、dtype 和参考 PyTorch 实现。不要只记 API 名字，要能手算一个很小的例子。

### 第三步：从 Python 包装器认识 Qwen2

阅读 `python/llaisys/models/qwen2.py`，重点是这些函数：

- `__init__`: 读取 `config.json`，创建 C++ 模型并取得权重句柄；
- `_load_weights` / `_assign_backend_weight`: 遍历 safetensors，按 Hugging Face 名字映射到后端槽位；
- `_make_ids_tensor`、`_make_positions_tensor`: 创建 token 和 position Tensor；
- `_layer_forward`: 一层 Transformer 的参考流程；
- `_forward_chunk`: embedding、全部层、final norm、LM head、argmax；
- `generate`: prefill、decode、EOS 和最大长度控制。

这个文件既是 Python API，也是阅读 C++ 实现时的“可执行规格”。其中旧的辅助 Tensor 前向代码可以用来理解算子组合，但真正的 `generate()` 已通过 C API 调用 C++ `Model::infer()`。

### 第四步：阅读 C API 和 C++ Model

对应文件：


- `include/llaisys/models/qwen2.h`
- `src/llaisys/models/qwen2.cc`
- `src/models/qwen2/model.hpp`
- `src/models/qwen2/model.cpp`
- `python/llaisys/libllaisys/qwen2.py`

构造函数依次做三件事：校验 meta、分配所有权重 Tensor、分配每层 K/V Cache，并把 Tensor 句柄数组暴露给 C API。

`Model::infer(token_ids, ntoken)` 的执行顺序是：

1. 创建 I64 token Tensor，并把 host token 拷入当前设备；
2. `embedding` 得到 `[ntoken, hs]` hidden；
3. 对每一层执行下方的 Transformer block；
4. final RMSNorm；
5. LM head 得到 `[ntoken, vocab]` logits；
6. 只取最后一行 logits，调用 `argmax`；
7. 将索引从设备拷回 host，作为下一个 token 返回。

一层 Transformer 的实际数据流是：

```text
hidden
  ├─ RMSNorm(input_layernorm)
  ├─ q_proj / k_proj / v_proj
  ├─ reshape 成 [token, head, head_dim]
  ├─ RoPE(q, position) / RoPE(k, position)
  ├─ 写入本层 K/V Cache
  ├─ causal GQA self-attention
  ├─ o_proj
  ├─ residual add
  ├─ RMSNorm(post_attention_layernorm)
  ├─ gate_proj / up_proj
  ├─ SwiGLU
  ├─ down_proj
  └─ residual add
```

## 3. 必须真正理解的 Qwen2 细节

### GQA 的 shape

Qwen2 的 query head 数是 `nh`，KV head 数是 `nkvh`，通常 `nh > nkvh`。每个 KV head 服务多个 query head，映射关系可以写成：

```text
kv_head = query_head // (nh / nkvh)
```

因此 Q 的 shape 是 `[T, nh, dh]`，K/V 的 shape 是 `[T, nkvh, dh]`。如果把 K/V 错误地当成和 Q 一样多的 head，结果会错且显存会浪费。

### RoPE 的 position

prefill 的 position 是 `0 ... prompt_len-1`。如果已经缓存了 `cached_len` 个 token，下一次 decode 的 position 必须从 `cached_len` 开始。position 重置为 0 是 KV Cache 最常见的 bug 之一。

### KV Cache 的时间顺序

假设先输入 3 个 token，再逐个生成 2 个 token：

```text
prefill: cache[0:3]，attention 读取 0:3
decode1: cache[3:4]，attention 读取 0:4
decode2: cache[4:5]，attention 读取 0:5
```

每次 decode 的 Q 只有一个位置，但 K/V 包含全部历史。`Model::_cached_len` 是状态变量，`reset()` 必须在新对话开始时调用。

### 权重映射

以 `model.layers.0.self_attn.q_proj.weight` 为例：Python 根据名字找到 `weights.attn_q_w[0]`，检查 dtype 和 shape，再调用 `tensorLoad` 把 host safetensors 数据复制到 C++ Tensor。这个过程是“文件格式名称”到“后端内存对象”的适配层，不应把文件指针直接当作模型权重长期保存。

## 4. 你需要补齐的知识，按优先级学习

### 必须先会

- C++17：类、引用、`unique_ptr`、`shared_ptr`、`vector`、RAII、异常和头文件边界；
- Python ctypes：指针、结构体、数组、`argtypes/restype`、生命周期；
- NumPy/PyTorch Tensor：shape、stride、contiguous、dtype 和 data pointer；
- Transformer 基础：embedding、线性层、RMSNorm、RoPE、causal attention、SwiGLU、residual；
- 自回归生成：logits、argmax、EOS、prefill 与 decode；
- 内存基础：host/device memory、同步/异步拷贝、stream。

### Assignment 3 之后再学

- CUDA kernel 的 grid/block/thread 和 dtype 转换；
- cuBLAS/GEMM、共享内存、coalesced access、occupancy；
- profiler、CUDA events、Nsight Systems/Compute；
- KV Cache 的分页、量化和 batch 解码；
- 多请求调度、continuous batching、服务化接口。

## 5. 建议的动手路线和完成标准

### 阶段 A：用小 Tensor 复现一层

创建一个极小 meta，例如 `nlayer=1, hs=8, nh=2, nkvh=1, dh=4, di=16, voc=32`，用随机 BF16 权重。分别在 Python 参考路径和 C++ 后端打印并比较：embedding、q/k/v、RoPE、attention、MLP、最终 logits。

完成标准：每一步 shape 相同，数值误差在 dtype 允许范围内。

### 阶段 B：验证 prefill 和 decode 分开正确

先只输入一段 prompt，记录返回 token；再调用 `infer()` 输入一个新 token，检查 position、cache 长度和输出。与“把完整前缀一次性重新计算”的参考实现逐 token 对比。

完成标准：至少连续生成 3 个 token 一致，`reset()` 后同一输入再次得到相同结果，超过 `maxseq` 会报错。

### 阶段 C：跑真实模型验收

```bash
PYTHONPATH=python .venv/bin/python test/test_infer.py \
  --model ./DeepSeek-R1-Distill-Qwen-1.5B \
  --device cpu --test --max_steps 1
```

如果使用 NVIDIA：

```bash
xmake f -c --nv-gpu=y -m release
xmake -y && xmake install -y
PYTHONPATH=python .venv/bin/python test/test_infer.py \
  --model ./DeepSeek-R1-Distill-Qwen-1.5B \
  --device nvidia --test --max_steps 3
```

先追求 token 一致，再测速度。参考实现和 LLAISYS 必须使用相同 prompt、dtype、采样策略（`top_k=1`、temperature 不影响 argmax）。

### 阶段 D：整理成简历项目

记录固定实验环境、模型配置、prompt 长度、生成长度、设备、dtype、TTFT、TPOT、吞吐和正确性结果。简历描述应基于已测量事实，例如：

> 使用 C++17/CUDA 实现 Qwen2 推理后端，通过 ctypes 接入 safetensors 权重，支持 GQA、RoPE、RMSNorm、SwiGLU、KV Cache 和 BF16；在固定 prompt 上与 Hugging Face 逐 token 对齐，并对 CPU/NVIDIA 算子进行独立正确性测试。

不要在没有 benchmark 数据时声称“高性能”或“实时”。当前实现的 CUDA kernel 主要是正确性基线，后续应使用 cuBLAS 和专门的 attention kernel 做性能优化。

## 6. 当前项目的下一批任务

按推荐顺序：

1. 把 `test/ops/self_attention.py` 的 CUDA mask 设备问题修正，并加入回归测试；
2. 增加一个小模型的 C++/Python 中间 Tensor 对齐测试；
3. 增加 Qwen2 的 reset、maxseq、连续 decode 和多 prompt 测试；
4. 分离 prefill/decode 计时，记录 JSONL benchmark；
5. 优化 CUDA linear（优先接 cuBLAS），再优化 attention；
6. 清理 Python 中仅用于参考的旧辅助路径，明确后端是唯一推理实现；
7. 选择第二个 GPU 平台，完成 Assignment 4 的平台适配。

每完成一个任务，都应同时提交：代码、测试命令、结果、已知限制和一段解释。这样最终得到的不只是“能运行的仓库”，而是一份可以在面试中从内存布局讲到模型输出的项目。

