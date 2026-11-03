# Qwen2 greedy token 对照记录：2026-11-01

本次将 WikiText-2 的 9 条固定输入分别送入 Hugging Face 与 LLAISYS。两边使用本地 DeepSeek-R1-Distill-Qwen-1.5B 权重、BF16、NVIDIA GPU、同一份 input IDs 和配置中的 EOS，最多生成 8 个新 token。HF 使用 eager attention、eval/inference mode、带 KV Cache 的逐步 argmax；LLAISYS 使用现有 generate(top_k=1)。没有采样或聊天模板。

环境：RTX 5060 Ti，PyTorch 2.13.0+cu130，Transformers 5.15.0。数据集 commit 为 `b08601e04326c79dfdd32d625aee71d232d685c3`。当前有未提交源码修改，本报告对应当时实际安装的共享库与 Python 包装器；并未声称某个干净 commit 已通过测试。后续修复/基准应重新构建并记录版本。

## 复现

```bash
.venv/bin/python benchmark/compare_prompt.py --all --max-new-tokens 8
```

本机缺少系统 Python.h，已从 Ubuntu libpython3.12-dev 包在 cache/python-dev/ 解压开发头文件；脚本通过进程的 C_INCLUDE_PATH 提供给 Triton 编译器。未修改 PyTorch/HF 的算子实现，也未安装系统包。换机器时需要对应 Python 的开发头文件。

## 结果

0/9 生成序列完全一致。脚本保存结果后以状态 1 退出，表示对照失败。以下“首差异”从新增 token 的第 0 个开始编号，不是包含 prompt 的全序列下标。

| 样本 | 首差异 | HF token | LLAISYS token |
| --- | ---: | ---: | ---: |
| row_3_len_16 | 2 | 438 | 304 |
| row_3_len_64 | 0 | 26705 | 5055 |
| row_3_len_128 | 0 | 4389 | 3757 |
| row_4_len_16 | 1 | 5700 | 15305 |
| row_4_len_64 | 0 | 482 | 596 |
| row_4_len_128 | 5 | 304 | 389 |
| row_11_len_16 | 1 | 279 | 264 |
| row_11_len_64 | 0 | 16 | 17 |
| row_11_len_128 | 0 | 11 | 518 |

5 条样本从首个新 token 就不同，因此不能只归因于后续 decode 的 cache 追加。其余样本在后续位置分歧，也不能据此直接判定 KV Cache 有 bug。

## 证据边界与下一步

- 本次证明了所列条件下的 greedy 输出不一致；没有证明差异一定来自 LLAISYS 的某个算子，也没有证明是 BF16 误差。需同时核查参考配置、数值路径、权重和 native 实现。
- 短文本看起来连贯、逐算子教学测试通过，都不能替代整模型对照。
- 一旦分歧，后续自由生成的输入随之不同；应先定位首个差异，必要时用共同的历史 token 做固定序列回放。
- 下一步选择首 token 已不同的 row_3_len_64，只做 prefill；对照 embedding、第一层 RMSNorm/QKV/RoPE/attention 及最终 logits，逐步找出首个偏差。优先用小模型或受控的张量诊断，关闭诊断后再测性能。
- 记录是 token 对照，不包含所有 logits 的误差，也没有性能结论。

原始 JSONL 位于 benchmark/results/generation_compare.jsonl（Git 忽略）；此摘要保存关键条件与结果。
