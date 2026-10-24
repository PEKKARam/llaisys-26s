# Profiling

For a first operator trace, how to read it, and current measurement limitations, see the
[learning plan](../LEARNING_PLAN.md). The native backend currently has no NVTX phase
markers; tracing `nvtx` does not add them automatically.

Benchmark outside profilers first. Profiler overhead changes timing, so Nsight output is for
diagnosis rather than performance baselines.

## Nsight Systems

Use a short workload for the first trace:

```bash
mkdir -p profile/nsys
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --output=profile/nsys/qwen2_decode \
  python benchmark/benchmark_infer.py \
    --model Models \
    --device nvidia \
    --prompt-tokens 128 \
    --output-tokens 32 \
    --warmup 2 \
    --repeat 1

nsys stats profile/nsys/qwen2_decode.nsys-rep
```

## Nsight Compute

Select a hotspot found in Nsight Systems. Do not run `--set full` over the entire model.

```bash
mkdir -p profile/ncu
ncu \
  --set basic \
  --target-processes all \
  --launch-skip 20 \
  --launch-count 1 \
  --export profile/ncu/linear_hotspot \
  python benchmark/benchmark_infer.py \
    --model Models \
    --device nvidia \
    --prompt-tokens 8 \
    --output-tokens 2 \
    --warmup 0 \
    --repeat 1
```

Adjust `--launch-skip`, `--launch-count`, or a kernel-name filter after inspecting the actual
kernel order.

## CPU perf

```bash
perf stat -d \
  python benchmark/benchmark_infer.py \
    --model Models --device cpu --prompt-tokens 128 --output-tokens 32 --repeat 5

perf record -g --call-graph dwarf \
  python benchmark/benchmark_infer.py \
    --model Models --device cpu --prompt-tokens 128 --output-tokens 32 --repeat 1

perf report
```
