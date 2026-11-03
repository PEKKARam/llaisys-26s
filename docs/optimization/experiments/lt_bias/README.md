# 未采用的 cuBLASLt bias epilogue 候选

这是一次没有合入默认路径的实验，保留代码片段 `linear_nvidia.cu` 与真实 `trace.txt`，便于说明为什么放弃。它不是当前可直接编译的独立目标：依赖当时试验性的 Lt handle、workspace 和链接设置，均已从正式代码移除。

目的：把 GEMM 和 bias 相加放到一次 Lt 调用里，减少临时 FP32 product 与额外 bias kernel。使用 CUBLAS_COMPUTE_32F、低精度输入输出、BIAS epilogue、最多 32 MiB workspace、第一次 heuristic 返回的算法；尝试同一实现处理不同 Q/K/V 形状。

实际检查：本地 BF16 模型，`row_3_len_64`，HF eager + FP32 reduction，采集第 0 层完整检查点。候选的 Q projection 与 HF 相同元素仅 65.95%，最大差 0.5；第 0 层输出平均绝对误差约 0.06316，首 token 从 HF 的 26705 变成 3822。作为对照，正式 GEMMEx+FP32 bias 路径的同一 Q projection 相同元素为 100%，第 0 层输出平均绝对误差约 0.01033，首 token 与 HF 相同。

决定：这套候选的数值表现变差，**没有继续进行性能宣传或测得加速结论**。候选使用的 Lt 算法选择/归约约束与正式路径不同；还没有把所有差异精确归因，不能单凭 compute type 声明就断定每一步舍入与 HF 完全相同。保留明确的 FP32 product+加 bias 路径，其额外 kernel 在整机 trace 中只占约 0.7% GPU kernel 时间，当前不值得为了融合强行引入更难解释的数值行为。

这也说明：调用同一家 BLAS 库、输入 dtype 相同，不代表最终 BF16 结果逐位相同。以后要重试，应先固定算法与数值约束，再分别验证 Q/K/V 的真实输入，而不是只测小幅随机数据。
