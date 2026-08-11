# LLAISYS 推理引擎：测试、Benchmark 与 Profiling 设计

本文说明如何把当前课程代码逐步升级为一个可调试、可测量、可优化的推理引擎项目。目标不是为了某一次作业临时添加计时打印，而是建立一套以后可以支持 CPU、NVIDIA 和其他 GPU backend 的正确性、性能、显存和回归验证体系。

## 1. 当前测试能做什么，不能做什么

### 已有能力

当前 `test/ops/*.py` 已经具备一个初步 benchmark 入口：

- 每个算子接受 `--profile`。
- `test/test_utils.py::benchmark()` 有 warmup 和 repeat。
- benchmark 前后调用 `device_synchronize()`，避免异步 GPU 工作尚未完成就读时间。
- 测试将 PyTorch 和 LLAISYS 的结果做数值比较。
- `test/test_infer.py` 会打印 Hugging Face 和 LLAISYS 的总耗时。

这是一个有用的起点，但它仍然是“测试脚本级计时”，不是完整 profiling 系统。

### 当前限制

1. `time.time()` 适合粗粒度端到端计时，不适合精确测量 CUDA kernel。GPU 上应使用 CUDA events 或 Nsight 时间线。
2. 现在的 benchmark 没有输出标准化 JSON/CSV，无法稳定比较不同 commit、shape、dtype 和设备。
3. 模型测试只有总耗时，没有区分模型加载、权重拷贝、prefill、decode、每层和每个算子。
4. 没有记录输入长度、生成长度、batch、dtype、峰值显存、设备名称和 git revision。
5. 没有 NVTX 标记，`nsys` 时间线中无法快速定位 embedding、attention、MLP 等阶段。
6. 没有单独的中间 Tensor dump/对比机制，只能靠 `debug()` 或最终 token 发现错误。
7. 没有回归基线，优化后即使变慢或显存增加，也没有自动警告。
8. CPU 和 GPU 的测试/benchmark 入口还没有统一的配置和结果格式。

## 2. 参考项目如何做 profiling

下面这些项目的价值不在于复制代码，而在于学习它们如何分层测量。

### llama.cpp / ggml

- 项目：[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
- Tensor/backend：[ggml-org/ggml](https://github.com/ggml-org/ggml)

建议阅读或搜索：`llama-bench`、`--timing`、`perf_context`、`ggml_backend`、`ggml-cuda`、`flash_attn`。

可以借鉴的做法：

- 单独统计 prompt processing（prefill）和 token generation（decode）。
- 输出 prompt tokens/s、generated tokens/s、总耗时和阶段耗时。
- benchmark 使用固定参数和多次重复，结果可以直接放进表格。
- backend 通过统一接口切换，模型层不关心 CUDA kernel 的具体实现。
- 低层算子可以独立 benchmark，模型级结果再作为端到端指标。

### vLLM

- 项目：[vllm-project/vllm](https://github.com/vllm-project/vllm)

建议搜索：`benchmarks/`、`benchmark_serving.py`、`benchmark_throughput.py`、`metrics`、`torch.profiler`。

可以借鉴的做法：

- 明确区分 TTFT（time to first token）和 TPOT/ITL（每个后续 token 的时间）。
- benchmark 参数包含 prompt 长度、输出长度、请求数、batch 和 dtype。
- 正确性 benchmark 与性能 benchmark 分开，避免 profiler 和参考模型干扰结果。
- 服务场景记录吞吐、并发和延迟分位数；单模型项目可以先实现单请求指标。

### TensorRT-LLM

- 项目：[NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)

建议搜索：`benchmarks/python`、`trtllm-bench`、`profiler`、`NVTX`、`--profiling-verbosity`。

可以借鉴的做法：

- 先 warmup，再采集稳定区间。
- 通过 NVTX 给模型阶段、layer、attention、GEMM 打标签。
- 使用 Nsight Systems 看全局时间线，使用 Nsight Compute 看单 kernel 的 occupancy、带宽和访存效率。
- 将性能数字与模型配置、GPU 型号、CUDA/TensorRT 版本一起保存。

### PyTorch Profiler

- 文档：[PyTorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)

虽然 LLAISYS 不能用 PyTorch 实现推理逻辑，但可以借鉴它的 trace 思路：

- 活动包括 CPU、CUDA、内存和 shape。
- 只对 warmup 后的一小段 step 采集，避免 trace 文件过大。
- 将 trace 导出到 TensorBoard 或 Chrome trace 查看。

### NVIDIA CUDA Samples

- 项目：[NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples)

重点看 `simpleStreams`、`asyncAPI`、`deviceQuery`、`matrixMul`。它们适合验证 Runtime、stream、异步拷贝、设备查询和 kernel 基准，而不是模型级性能。

## 3. 建议的 profiling 分层

不要把所有问题都交给 `nsys`。每一层回答的问题不同：

| 层级 | 回答的问题 | 推荐工具/接口 | 输出 |
| --- | --- | --- | --- |
| 正确性 | 数值和 token 是否正确 | PyTorch 对照、Tensor dump、debug | pass/fail、误差、首个不一致位置 |
| 算子 benchmark | 哪个算子慢 | `benchmark()`、CUDA events | 平均/中位数/p95、吞吐 |
| 模型 benchmark | prefill/decode 速度如何 | 独立 benchmark 脚本 | TTFT、TPOT、tokens/s、总时间 |
| Runtime 诊断 | 拷贝、分配、同步是否异常 | API 计数器、NVTX、日志 | malloc/copy/sync 次数和字节数 |
| 系统时间线 | CPU、GPU、stream 是否重叠 | Nsight Systems (`nsys`) | `.qdrep`/SQLite 时间线 |
| kernel 细节 | occupancy、带宽、指令效率 | Nsight Compute (`ncu`) | kernel report |
| CPU 细节 | 热点函数和线程利用率 | `perf`、VTune | call graph、热点 |

正确性必须先于性能。一个更快但 token 不一致的 kernel 不能进入性能基线。

## 4. 第一阶段：修正现有算子 benchmark

### 4.1 使用单调时钟和统计分布

将 `test/test_utils.py::benchmark()` 从 `time.time()` 改为 `time.perf_counter()`，并返回结构化结果，而不是只打印两个浮点数。至少记录：

```text
device, device_id, op, shape, dtype, warmup, repeat,
mean_ms, median_ms, p90_ms, p99_ms, min_ms, max_ms
```

CPU 和 GPU 都应在计时区域外完成输入创建、权重创建和输出初始化。否则测到的是“分配 + 算子”，而不是算子本身。

### 4.2 GPU 计时必须使用 CUDA events

GPU kernel 是异步的。`device_synchronize()` 可以避免明显错误，但会把整条设备队列混在一起，也无法准确测量一个 kernel。NVIDIA backend 建议增加一个计时工具：

```cpp
cudaEventRecord(start, stream);
call_operator();
cudaEventRecord(stop, stream);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&ms, start, stop);
```

应用层可以先继续使用 host timer；真正的单算子 GPU benchmark 应用 CUDA events。第二个平台应提供等价的 device event API，或者暂时在平台层封装同样的接口。

### 4.3 输出 JSONL

建议新增 `benchmark/results/`（加入 `.gitignore`）并让每次 benchmark 输出一行 JSON。例如：

```json
{"op":"linear","device":"nvidia","shape":[512,4096],"dtype":"bf16","mean_ms":0.84,"p99_ms":0.91,"repeat":100}
```

JSONL 比人工复制终端输出更适合作图、回归和 CI。另写一个简单脚本把 JSONL 转为 Markdown 表格或 CSV。

## 5. 第二阶段：增加模型级 benchmark

不要把现有 `test/test_infer.py --test` 直接当性能脚本。它同时加载 Hugging Face 模型、比较输出、打印文本，测量结果会受参考模型和 Python 逻辑影响。

建议新增：

```text
benchmark/benchmark_infer.py
benchmark/README.md
benchmark/results/.gitkeep
```

`benchmark_infer.py` 应支持：

```bash
python benchmark/benchmark_infer.py \
  --model <model_dir> \
  --device cpu \
  --prompt-tokens 128 \
  --output-tokens 128 \
  --warmup 2 \
  --repeat 10 \
  --json benchmark/results/cpu.jsonl
```

GPU 版本：

```bash
python benchmark/benchmark_infer.py \
  --model <model_dir> \
  --device nvidia \
  --prompt-tokens 128 \
  --output-tokens 128 \
  --warmup 5 \
  --repeat 30 \
  --json benchmark/results/nvidia.jsonl
```

### 必须分别测量的阶段

1. **Load**：模型对象创建、配置解析和权重加载。
2. **H2D**：GPU 权重和输入拷贝；CPU 可以记为 0 或不适用。
3. **Prefill**：一次处理完整 prompt 的时间。
4. **First token**：从 prompt 开始到第一个新 token 的时间，通常包含 prefill 和 argmax。
5. **Decode**：后续单 token 的平均时间。
6. **End-to-end**：从生成开始到输出完成的总时间。

推荐指标：

```text
TTFT_ms = first_token_time
TPOT_ms = (decode_end - decode_start) / decoded_token_count
prefill_tokens_per_s = prompt_tokens / prefill_seconds
decode_tokens_per_s = decoded_tokens / decode_seconds
peak_memory_bytes
```

至少测试以下矩阵：

| prompt tokens | output tokens |
| ---: | ---: |
| 1 | 32 |
| 128 | 32 |
| 512 | 128 |
| 2048 | 128 |

如果显存或 CPU 内存不够，缩小长度，但在报告中明确记录。

## 6. 第三阶段：加入中间 Tensor 调试和正确性诊断

### 6.1 Debug 不应该只打印屏幕

目前 Tensor 有 `debug()`，但模型调试需要可复现、可比较的输出。建议增加一个受环境变量控制的诊断机制：

```bash
LLAISYS_DEBUG_TENSORS=embedding,layer0.q,layer0.attn,logits \
LLAISYS_DEBUG_DIR=debug/run-001 \
python benchmark/compare_infer.py --model <model_dir>
```

C/C++ 后端对匹配的 Tensor 保存：

- 名称
- dtype
- shape
- stride
- device
- min/max/mean
- L2 norm
- 前 N 个元素
- 可选的完整二进制 dump

不要默认 dump 所有大 Tensor。默认保存摘要，只有显式指定名称时才保存完整数据。

### 6.2 与 PyTorch 对照的两种模式

**模式 A：逐层摘要对照**

- Python 参考模型通过 forward hook 输出相同名称的摘要。
- LLAISYS 输出对应摘要。
- 脚本比较 shape、dtype、绝对误差、相对误差和首个明显偏差。

**模式 B：小模型/短序列完整 dump**

- 只使用短 prompt 和少量层，保存完整 Tensor。
- 用 NumPy 比较每个元素。
- 找到第一个不一致的算子后，再回到该算子测试。

推荐在模型后端提供一个 debug hook 或阶段枚举，而不是把大量 `printf` 永久写进前向路径。

## 7. 第四阶段：集成 NVTX、Nsight Systems 和 Nsight Compute

### 7.1 NVTX 是低成本高收益的第一步

Nsight Systems 只能显示 kernel 和 API 名称。用 NVTX 给阶段命名后，时间线才容易阅读。建议给以下区域加 range：

```text
qwen2.load_weights
qwen2.prefill
qwen2.decode
layer[i]
embedding
attention_norm
qkv_projection
rope
self_attention
o_projection
mlp_norm
gate_up_projection
swiglu
down_projection
lm_head
argmax
```

CUDA C++ 可以使用 NVTX C API；如果不希望让 CPU-only 构建依赖 CUDA，应在 `#ifdef ENABLE_NVIDIA_API` 下封装一组 no-op 宏：

```cpp
LLAISYS_NVTX_RANGE("qwen2.decode");
```

CPU 构建中宏为空，NVIDIA 构建中映射到 `nvtxRangePushA`/`nvtxRangePop`。

### 7.2 Nsight Systems：先看全局时间线

用途：

- CPU 调用和 GPU kernel 是否重叠
- 是否有过多 `cudaMemcpy`
- 是否每个 token 都在分配显存
- 是否出现不必要的 device synchronize
- stream 是否真的在工作
- prefill/decode 的阶段边界

建议命令：

```bash
mkdir -p profile/nsys
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --output=profile/nsys/qwen2_decode \
  python benchmark/benchmark_infer.py \
    --model <model_dir> \
    --device nvidia \
    --prompt-tokens 128 \
    --output-tokens 32 \
    --warmup 2 \
    --repeat 1
```

第一次采集应使用很短的输入和输出，避免生成巨大的 trace。先确认 NVTX 标签、kernel 名称、拷贝次数和同步位置，再增加长度。

可用命令行摘要：

```bash
nsys stats profile/nsys/qwen2_decode.nsys-rep
```

不同版本可能输出 `.qdrep` 或 `.nsys-rep`；最终以安装的 `nsys profile --help` 为准。

### 7.3 Nsight Compute：再分析单个热点 kernel

用途：

- kernel occupancy
- achieved memory bandwidth
- global memory load/store efficiency
- warp stall 原因
- FLOP throughput
- tensor core 使用情况
- launch 配置

不要对完整模型和所有 kernel 直接运行 `ncu --set full`，结果会非常大且运行很慢。先从时间线上选择一个热点 kernel，再指定名称或一次 launch：

```bash
mkdir -p profile/ncu
ncu \
  --set basic \
  --target-processes all \
  --launch-skip 20 \
  --launch-count 1 \
  --export profile/ncu/linear_hotspot \
  python benchmark/benchmark_infer.py \
    --model <model_dir> \
    --device nvidia \
    --prompt-tokens 8 \
    --output-tokens 2 \
    --warmup 0 \
    --repeat 1
```

`--launch-skip` 和 `--launch-count` 只是示例，实际值要根据 kernel 顺序调整。也可以先用 `ncu --kernel-name-base demangled` 和过滤器锁定自定义 kernel。

### 7.4 常见误区

- `nsys` 用于全局时间线，不是精确评估单个 kernel 的最终吞吐。
- `ncu` 适合热点 kernel，不适合每次 commit 的常规回归。
- profiler 运行会显著改变性能，不能把 profiler 内的时间当作生产 benchmark。
- 先 warmup 再采集；第一次运行通常包含 lazy initialization、CUDA context 和缓存建立。
- 不要在每个 kernel 后无条件同步作为最终实现；同步只用于诊断和计时边界。
- 不要把 `.qdrep`、`.nsys-rep`、`.ncu-rep` 大文件直接提交 Git，保存命令、摘要和必要截图即可。

## 8. CPU profiling

即使最终目标是 GPU，CPU 也需要有基线：

```bash
perf stat -d \
  python benchmark/benchmark_infer.py \
    --model <model_dir> --device cpu \
    --prompt-tokens 128 --output-tokens 32 --repeat 5

perf record -g --call-graph dwarf \
  python benchmark/benchmark_infer.py \
    --model <model_dir> --device cpu \
    --prompt-tokens 128 --output-tokens 32 --repeat 1

perf report
```

重点看：

- Linear 是否占据绝大部分 CPU 时间。
- attention 是否重复分配或复制 Tensor。
- decode 是否每一步重算完整上下文。
- 线程是否真正并行。
- memcpy、rearrange 和 allocator 是否成为隐藏热点。

CPU profiler 不要求每次运行都采集；只在怀疑瓶颈时使用。

## 9. 建议新增的仓库结构

```text
benchmark/
  benchmark_infer.py       # 端到端性能
  benchmark_ops.py         # 统一算子性能入口
  compare_infer.py         # PyTorch/LLAISYS 中间结果对照
  report.py                # JSONL 转 CSV/Markdown
  README.md
  results/.gitkeep
profile/
  README.md                # nsys/ncu/perf 复现命令
  .gitignore
src/utils/
  profiling.hpp            # no-op/NVTX/计时封装
```

建议不要把 profiling 逻辑散落在每个模型函数中。统一封装的好处是：CPU-only 构建不会依赖 CUDA，平台 backend 可以提供各自的 event，模型代码只使用阶段名称。

## 10. 推荐的最小实现顺序

### 第一版：马上可以做

- [ ] 将 `time.time()` 改为 `time.perf_counter()`。
- [ ] `benchmark()` 返回 mean/median/p90/p99/min/max。
- [ ] 输出 JSONL，记录设备、shape、dtype、repeat 和 git revision。
- [ ] 从加载时间中排除 benchmark 算子本身的输入创建。
- [ ] 为 `test_infer.py` 增加 load/prefill/decode/total 四个计时点。
- [ ] 增加 prompt token 数、生成 token 数、TTFT、TPOT、tokens/s。

### 第二版：模型实现后做

- [ ] 增加 prefill/decode 独立 benchmark。
- [ ] 增加 KV Cache 开关，用于比较有无缓存的速度和内存。
- [ ] 增加阶段级 debug 摘要和 Tensor dump。
- [ ] 增加 `LLAISYS_DEBUG_TENSORS` 和 `LLAISYS_DEBUG_DIR`。
- [ ] 为模型层和关键算子添加 NVTX ranges。
- [ ] 增加 allocator/copy/synchronize 计数器。

### 第三版：NVIDIA backend 完成后做

- [ ] 用 CUDA events 提供精确算子计时。
- [ ] 用 Nsight Systems 检查时间线和同步。
- [ ] 用 Nsight Compute 分析 Linear、RMSNorm、attention 等热点 kernel。
- [ ] 将优化前后结果保存为同一格式。
- [ ] 增加性能回归阈值，例如变慢超过 10% 时 CI 警告。

## 11. 完整的性能实验模板

每次实验至少保存：

```text
日期
git commit
模型路径和 revision
平台/GPU 型号
驱动、CUDA、编译器和 Xmake 版本
dtype
prompt tokens
output tokens
warmup/repeat
TTFT
prefill tokens/s
TPOT
decode tokens/s
总时间
峰值显存
是否通过 token 正确性校验
```

性能优化必须同时报告正确性。例如：

| commit | backend | dtype | prompt | output | TTFT | decode tok/s | peak memory | token check |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | cpu | bf16 | 128 | 32 | ... | ... | ... | pass |
| cuda-v1 | nvidia | bf16 | 128 | 32 | ... | ... | ... | pass |

没有输入长度、dtype 和 warmup 的单个“快了 2 倍”数字不可复现，也不能作为可靠简历数据。

## 12. 项目化后的成果表达

完成上述基础设施后，这个仓库可以从“课程作业”升级为“可测量的推理引擎开发项目”：

- 正确性：逐算子、逐层、端到端 token 对照。
- 可观测性：Tensor 摘要、NVTX、Nsight Systems/Compute。
- 性能：prefill/decode 分离、TTFT/TPOT、tokens/s、峰值显存。
- 工程：统一 Runtime、平台 backend、Xmake 构建和 JSONL 结果。
- 可维护性：benchmark 与 correctness test 分离，性能结果可回归。

简历中只能写已经真实完成并能复现的部分。建议使用这种表述：

> 为 LLAISYS 构建模型级 benchmark 和 profiling 基础设施，区分 prefill/decode 阶段并记录 TTFT、TPOT、tokens/s 和峰值显存；增加 Tensor 中间结果对照、NVTX 标记及 Nsight Systems/Compute 复现脚本，支持定位算子、同步和内存拷贝瓶颈。

如果第二平台或 GPU kernel 尚未完成，应明确写“进行中”或列入后续计划，不要写成已经支持。

## 13. 本项目的实际优先级

你现在已经通过作业 1、2，因此最有价值的下一步是：

1. 先把现有 `--profile` 变成可复现的算子 benchmark。
2. 再实现 Qwen2，并从第一天就加入 prefill/decode 计时和 debug dump。
3. CPU 正确性稳定后，再接 CUDA events 和 NVTX。
4. 用 `nsys` 找全局问题，用 `ncu` 找单 kernel 问题。
5. 最后才做高级 kernel 优化和第二平台。

不要为了“使用 nsys/ncu”而在模型还不能正确生成 token 时提前优化。profiling 的第一原则是：先建立正确、稳定、可重复的 workload，再测量它。
