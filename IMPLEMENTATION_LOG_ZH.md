# LLAISYS 实现记录（学习版）

本文记录本次实现的代码变化、设计理由和验证结果。它与作业指南配合使用：指南描述目标，这里描述当前工作区实际完成到的程度。

## 本次完成的内容

### 1. Tensor 元数据与存储变换

文件：`src/tensor/tensor.cpp`

- `Tensor::contiguous()`：连续张量现在返回共享底层存储的轻量视图；非连续 CPU 张量按照 stride 逐元素复制到新的连续存储。复制时把线性下标还原成多维下标，因此支持 `permute()`、`slice()` 等组合视图，而不是只处理一种固定转置。
- `Tensor::reshape()`：连续张量直接调用经过布局检查的 `view()`；非连续张量先连续化再 view。这样不会错误地改变一个不兼容 stride 的张量形状。
- `Tensor::to()`：通过 Runtime 的 `memcpy_sync` 执行 CPU、设备之间的拷贝，并自动选择 H2D、D2H 或 D2D 方向；非连续输入会先转为连续布局。

### 2. `rearrange` 算子

文件：`src/ops/rearrange/op.cpp`

增加了设备、dtype、shape 和输出连续性的检查。连续输入走一次整块复制，非连续输入复用 `Tensor::contiguous()` 的通用 stride 拷贝逻辑。该算子因此可以作为“把 view 变成可供矩阵算子使用的连续 Tensor”的明确边界。

### 3. C/Python API 补全

在 `include/llaisys/tensor.h`、`src/llaisys/tensor.cc` 和 Python ctypes 包装中新增 `tensorContiguous`、`tensorReshape`、`tensorTo`，并在 Python `Tensor` 类提供同名方法。这样学习者可以直接从 Python 构造非连续 view、物化布局、改变形状并测试设备拷贝。

## 代码阅读提示

1. `view()` 只修改 shape/stride/offset，不搬运数据；它会按连续块检查新旧布局是否兼容。
2. `permute()` 只交换 shape 和 stride，所以通常是零拷贝的。
3. `slice()` 通过增加字节 offset 实现半开区间 `[start, end)`，底层 storage 仍由共享指针保持生命周期。
4. `contiguous()` 是真正的布局物化操作；当前设备视图的逐元素转换尚未提供 CUDA kernel，因此会明确报错，而不是在主机上解引用设备指针。

## 当前尚未完成的部分

- `src/models/qwen2/model.cpp` 中 C++ `Model::infer()` 仍是生命周期骨架，返回 `-1`；可运行的 Qwen2 前向、KV Cache 和生成循环仍在 `python/llaisys/models/qwen2.py` 原型中。
- NVIDIA Runtime 和 NVIDIA 算子分支仍未实现。本次没有伪造 GPU 支持，也没有把 CPU 指针当成 CUDA 指针使用。
- Python safetensors 加载目前直接服务于 Python 原型，尚未把权重复制到 C++ Qwen2 模型的权重槽位。

## 验证记录

执行环境使用仓库 `.venv`：

```bash
xmake -y
xmake install -y
PYTHONPATH=python .venv/bin/python test/test_tensor.py
for f in test/ops/*.py; do PYTHONPATH=python .venv/bin/python "$f" --device cpu; done
PYTHONPATH=python .venv/bin/python test/test_qwen2_api.py
```

结果：Tensor 测试、add/argmax/embedding/linear/rms_norm/rope/self_attention/swiglu 全部通过，Qwen2 模型创建、权重句柄 shape 检查、reset/destroy 生命周期测试通过。

## 建议的后续学习顺序

先在 `Model::infer()` 中迁移 Python 原型的单层计算，并用一个很小的 meta 和随机权重逐层比较；随后加入每层 `[maxseq, nkvh, dh]` 的 K/V 缓存，最后再接入 Python ctypes 的权重加载。每一步都应先比较 Tensor 的 shape、stride、dtype 和误差，再比较最终 token，这样定位问题会容易得多。
