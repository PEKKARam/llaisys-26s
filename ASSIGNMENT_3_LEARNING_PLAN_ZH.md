# 作业三学习与实现计划：从 Python 原型迁移到 C++ 推理后端

本文基于 2026-08-11 的仓库状态整理。目标不只是让测试偶然通过，而是完成一个边界清楚、可验证、可 benchmark、适合写入简历的 Qwen2 推理引擎项目。

## 1. 先说结论：当前还不能算完成作业三

当前 `python/llaisys/models/qwen2.py` 已经实现了权重加载、Qwen2 前向、KV Cache 和 argmax 生成循环，算法结构基本完整。但是 README 的要求是：

- 模型推理逻辑必须在 LLAISYS 的 C/C++ 后端实现。
- Python 只负责读取配置和 safetensors、调用 ctypes C API、组织对外接口。
- `include/llaisys/models/qwen2.h` 中的 create/destroy/weights/infer API 必须真正落地。

目前 `src/llaisys/models/qwen2.cc` 实际上还是空文件（只有 include guard），`python/llaisys/libllaisys/` 也没有 Qwen2 C API 绑定。因此当前版本应定义为：

> 一个可用于理解和验证算法的 Python orchestration prototype，还不是符合课程要求的 C++ 模型后端。

不要删除这个原型。迁移期间它是非常有价值的“可执行规格”，可以逐函数翻译到 C++，并用于比较中间 Tensor。

## 2. 用 C++ 思维理解 `qwen2.py`

这份脚本没有使用复杂的 Python 魔法。可以把 `Qwen2` 类理解为一个拥有模型元数据、权重 Tensor 和 KV Cache 的 C++ 类。

| Python 写法 | 对应的 C++ 概念 |
| --- | --- |
| `self.xxx` | 类成员 `this->xxx` |
| `list[Tensor]` | `std::vector<tensor_t>` |
| `Tensor | None` | 可空句柄或 `std::optional` |
| `Sequence[int]` | `std::span<const int64_t>` 或指针加长度 |
| `Path.glob("*.safetensors")` | Python 侧遍历权重文件 |
| `Ops.linear(...)` | 调用已有 C++ 算子分发层 |
| `raise ValueError` | 参数检查失败；C API 边界需定义错误策略 |
| `__del__` | RAII 析构或显式 destroy C API |

脚本可以按五个区域阅读：

1. `__init__`：读取 `config.json`，保存模型超参数，建立权重和 cache 容器。
2. `_load_weights` / `_assign_weight`：将 Hugging Face 权重名字映射到模型成员。
3. `_ensure_cache`：为每一层分配 `[max_seq, num_kv_heads, head_dim]` 的 K/V Cache。
4. `_layer_forward` / `_forward_chunk`：执行 Transformer 层和最终 LM head。
5. `generate`：第一次 prefill 整个 prompt，之后每次 decode 一个 token。

### 2.1 一层 Qwen2 到底做了什么

`_layer_forward` 对应下面的 C++ 伪代码：

```cpp
Tensor layer_forward(Tensor hidden, size_t layer, size_t chunk_len) {
    auto residual = hidden;
    auto normed = rms_norm(hidden, attn_norm_w[layer]);

    auto q = linear(normed, q_weight[layer], q_bias[layer]);
    auto k = linear(normed, k_weight[layer], k_bias[layer]);
    auto v = linear(normed, v_weight[layer], v_bias[layer]);

    q = rope(view_heads(q), positions);
    k = rope(view_kv_heads(k), positions);
    write_to_kv_cache(layer, k, v);

    auto attention = self_attention(q, cached_k(layer), cached_v(layer));
    auto post_attn = residual + linear(attention, o_weight[layer]);

    auto mlp_input = rms_norm(post_attn, mlp_norm_w[layer]);
    auto mlp = swiglu(linear(mlp_input, gate_w[layer]),
                      linear(mlp_input, up_w[layer]));
    return post_attn + linear(mlp, down_w[layer]);
}
```

最容易出错的不是 Python 语法，而是 shape 和状态：

- Q 是 `[chunk_len, num_heads, head_dim]`。
- K/V 是 `[chunk_len, num_kv_heads, head_dim]`，因为 Qwen2 使用 GQA。
- prefill 时 `chunk_len = prompt_len`；decode 时通常为 1。
- RoPE position 从 `cached_len` 开始，decode 时不能重置为 0。
- attention 读取的是本次 token 加历史缓存，而不是只有本次 K/V。
- LM head 只需要最后一个位置的 logits，再沿 vocab 维度 argmax。

### 2.2 KV Cache 的时间顺序

以 3 个 prompt token、生成 2 个 token 为例：

```text
prefill: input=[t0,t1,t2], positions=[0,1,2], cache 写入 [0:3]
         logits(t2) -> t3, cached_len 变为 3
decode:  input=[t3],       position=[3],     cache 写入 [3:4]
         logits(t3) -> t4, cached_len 变为 4
```

注意：新生成的 `t3` 在 prefill 后还没有进入 cache；它要在下一次 decode 中作为输入，产生自己的 K/V。当前 Python 原型正是按这个顺序更新 `_cached_len`。

## 3. 推荐的 C++ 目标结构

尽量沿用仓库现有边界，不要在 Python 中继续扩大模型实现：

```text
include/llaisys/models/qwen2.h       # 稳定的 C ABI
src/llaisys/models/qwen2.cc          # C ABI 到 C++ 类的薄适配
src/models/qwen2/model.hpp           # Qwen2Model 声明
src/models/qwen2/model.cpp           # 权重、cache、prefill/decode
python/llaisys/libllaisys/qwen2.py   # ctypes 结构和函数签名
python/llaisys/models/qwen2.py       # 配置/权重文件/生成接口
```

C++ 后端至少应拥有：

- `Qwen2Meta`：层数、hidden size、head 数、vocab、epsilon、theta 等。
- `Qwen2Weights`：所有权重 Tensor；用 RAII 管理生命周期。
- 每层一组 K/V Cache，以及全模型统一的 `cached_len`。
- `reset()`：开始一次新生成前清空逻辑 cache 长度。
- `infer(token_ids)`：第一次接收 prompt，之后接收单 token，返回 next token。

建议给 C API 补充显式 reset。只靠“猜测输入是否属于同一会话”会让 API 语义不稳定：

```c
void llaisysQwen2ModelReset(struct LlaisysQwen2Model *model);
```

## 4. 分阶段开发计划

### 阶段 A：建立可信基线

- 保持现有 Python 原型可运行。
- 固定一个短 prompt 和 `max_new_tokens=1`，记录 Hugging Face token。
- 跑通 build、Runtime、Tensor 和全部 CPU 算子测试。
- 修复测试环境与源码环境不一致的问题，确保测试导入的是本仓库代码。

完成标准：所有基础测试通过，Python 原型至少能产生与 HF 相同的第一个 token。

### 阶段 B：先打通最小 C API，不急着写完整前向

- 当前进度：create/destroy/weights/reset、C++ 权重槽位、ctypes ABI 和微型生命周期测试已经完成；真实 safetensors 加载尚未接入该后端。
- 实现 meta/weights 的 ctypes 结构，逐字段核对 size 和顺序。
- 实现 create/destroy/weights/reset，先只分配模型权重。
- 从 safetensors 加载权重到后端 Tensor。
- 增加缺失权重、错误 shape、错误 dtype 测试。

完成标准：模型能反复创建、加载、销毁；ASan/Valgrind 下无明显泄漏或 double free。

### 阶段 C：按可对照的小步骤迁移前向

迁移顺序建议为：embedding -> 一层 attention 前半段 -> KV 写入 -> attention -> MLP -> 全部层 -> final norm/lm head/argmax。每完成一步，就和 Python 原型或 Hugging Face 中间 Tensor 对齐。

每个检查点至少比较：

- shape、stride、dtype、device；
- 前若干元素；
- max absolute error、max relative error；
- min/max/mean/L2 norm。

不要一口气写完整 28 层后只比较最终文本。最终 token 不一致时，逐层回查会非常困难。

### 阶段 D：KV Cache 与生成状态

- prefill 一次处理完整 prompt。
- decode 每次只处理一个 token。
- 验证 prefill 最后位置 logits 与“无 cache 全量重算”一致。
- 验证连续两次 `generate()` 之间会 reset。
- 验证超过 `max_position_embeddings` 会失败且不越界。

完成标准：多个短 prompt 在 argmax 下与 Hugging Face 完整 token 序列一致。

### 阶段 E：benchmark 和性能工程

先正确，再测量，再优化。推荐分开记录：

- 模型加载时间；
- TTFT（包含 prefill）；
- TPOT（稳态 decode 每 token 延迟）；
- prefill tokens/s 与 decode tokens/s；
- 峰值内存；
- prompt/output 长度、dtype、设备、编译模式、commit SHA。

当前 `benchmark/benchmark_infer.py` 通过“一 token generate”和“完整 generate”相减来估算 TPOT。它可用于早期趋势，但不是精确 phase timing。C++ 后端完成后，应暴露 prefill/decode 或内部计时点，避免把两次独立运行的波动相减。

CPU 优化优先级：

1. 将 `linear` 替换为成熟 BLAS/oneDNN 路径。
2. 避免每层、每 token 重复创建临时 Tensor 和堆分配。
3. 只计算最后位置需要的 logits，或优化 LM head。
4. profile 后再考虑算子融合、线程并行和内存布局。

## 5. 测试与验收命令

仓库没有 `test/test_ops.py` 聚合脚本，应逐个运行算子：

```bash
xmake -y
xmake install
PYTHONPATH=python .venv/bin/python test/test_runtime.py --device cpu --extended
PYTHONPATH=python .venv/bin/python test/test_tensor.py
for test_file in test/ops/*.py; do
    case "$test_file" in *__init__.py) continue ;; esac
    PYTHONPATH=python .venv/bin/python "$test_file" --device cpu || exit 1
done
PYTHONPATH=python .venv/bin/python test/test_infer.py \
    --model DeepSeek-R1-Distill-Qwen-1.5B \
    --device cpu --test --max_steps 1
```

开发期先用 `max_steps=1` 对齐首 token，再增加到 4、16 和完整要求。朴素 CPU linear 在 1.5B 模型上很慢，不能用运行时间长来代替正确性结论。

2026-08-11 在当前工作区的首 token 基线如下（单次运行，只用于正确性和确定优化起点，不是正式 benchmark 报告）：

| implementation | prompt/output | generated token | end-to-end | output tok/s |
| --- | --- | ---: | ---: | ---: |
| Hugging Face | 9/1 | 91786 | 851.84 ms | 1.174 |
| 当前 LLAISYS Python 原型 | 9/1 | 91786 | 5426.41 ms | 0.184 |

两个实现的完整 token 序列一致。正式性能结论仍需 warmup、多次重复、固定 CPU 频率/线程数并记录硬件信息。

## 6. Benchmark 应如何做得能写进简历

只有“加了 benchmark 脚本”不够。简历项目应能回答：和谁比、怎么测、测了什么、优化了哪里、结果是否可复现。

建议形成一张版本对比表：

| revision | device | prompt/output | TTFT | TPOT | decode tok/s | peak memory | correctness |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| naive CPU | CPU 型号 | 128/32 | ... | ... | ... | ... | HF tokens equal |
| BLAS linear | 同上 | 128/32 | ... | ... | ... | ... | HF tokens equal |
| buffer reuse | 同上 | 128/32 | ... | ... | ... | ... | HF tokens equal |

每次性能结论至少报告 warmup、repeat、均值/中位数/P90/P99、编译模式和 commit。不要只挑最好的一次，也不要把 PyTorch CUDA event 与 LLAISYS host wall clock 混成同一种指标。

简历描述应在拿到真实数据后填写，例如：

> 使用 C++17 实现 Qwen2 1.5B BF16 自回归推理后端，完成 GQA、RoPE、RMSNorm、SwiGLU 与 KV Cache；通过 ctypes 接入 Python/Hugging Face 权重，在固定 prompt 集上与参考实现逐 token 对齐；建立 TTFT/TPOT/吞吐/尾延迟 JSONL benchmark，并通过 BLAS 与工作区复用将 decode 性能提升 X%。

这里的 `X%`、设备和测试矩阵必须来自提交到仓库的小型可复现报告，不能先写结论再找数据。

## 7. 当前仓库的已知问题清单

- 作业三核心逻辑仍在 Python，不满足 README 的 C/C++ 后端要求。
- `src/llaisys/models/qwen2.cc` 和 Qwen2 ctypes 绑定尚未实现。
- `ASSIGNMENTS_3_4_GUIDE.md` 的“当前状态”基于旧骨架，与现有 Python 原型不一致。
- 测试可能导入 `.venv/site-packages/llaisys`，修改源码后必须重新安装，或明确使用 editable install/PYTHONPATH。
- 当前 inference benchmark 的 TTFT/TPOT 是 public `generate()` API 下的估算值。
- CPU 推理仍大量分配临时 Tensor，naive linear 也不适合作为最终性能实现。
- `top_p` 和 `temperature` 参数在 argmax 模式没有作用；`top_k != 1` 会明确拒绝，这是符合当前作业范围的。
