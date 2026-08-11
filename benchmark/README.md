# Benchmark

Correctness tests and performance measurements are deliberately separate. Every operator
script checks its output against PyTorch before recording timings. JSONL files are appended
so that results from separate shapes and commits can be compared later.

For assignment 4, validate the Runtime before benchmarking operators:

```bash
python test/test_runtime.py --device cpu --extended
python test/test_runtime.py --device nvidia --extended
```

The extended test covers synchronous copies, pinned host allocations, asynchronous H2D/D2D/D2H
copies, device synchronization, and two streams. A missing requested backend is a failure unless
`--allow-missing` is explicitly supplied.

## Operator benchmark

Run all CPU operators:

```bash
python benchmark/benchmark_ops.py --device cpu --warmup 10 --repeat 100
```

Run selected NVIDIA operators:

```bash
python benchmark/benchmark_ops.py \
  --device nvidia \
  --ops linear rms_norm rope self_attention \
  --warmup 20 \
  --repeat 100 \
  --json benchmark/results/ops_nvidia.jsonl
```

CPU samples use `time.perf_counter()`. NVIDIA samples use CUDA events with a device
synchronization after each invocation. Inputs, weights, and outputs are allocated before
timing starts.

## Inference benchmark

Run one shape or the full length matrix from the profiling guide:

```bash
python benchmark/benchmark_infer.py \
  --model Models \
  --device cpu \
  --prompt-tokens 128 \
  --output-tokens 32 \
  --warmup 2 \
  --repeat 10

python benchmark/benchmark_infer.py \
  --model Models \
  --device nvidia \
  --matrix \
  --warmup 5 \
  --repeat 30
```

The current public `Qwen2.generate()` API does not expose separate prefill and decode calls.
The script measures TTFT with a one-token generation and estimates TPOT by subtracting that
measurement from a full generation. Records identify this as
`generate_api_ttft_and_tpot_estimate`; `prefill_ms`, `h2d_ms`, and native backend peak memory
remain `null` until the model/runtime exports exact phase and memory instrumentation.

Inference correctness remains the responsibility of:

```bash
python test/test_infer.py --model Models --device cpu --test
python test/test_infer.py --model Models --device nvidia --test
```

## Reports

```bash
python benchmark/report.py benchmark/results/ops_nvidia.jsonl \
  --implementation llaisys \
  --format markdown \
  --output benchmark/results/ops_nvidia.md

python benchmark/report.py benchmark/results/ops_nvidia.jsonl \
  --format csv \
  --output benchmark/results/ops_nvidia.csv
```

Commit benchmark commands and small reports when useful. Raw JSONL results, profiler traces,
and large reports are ignored by Git.
