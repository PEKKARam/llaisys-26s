# Benchmark


Current implementation/results: [CUDA optimization report](../docs/optimization/README.md), [per-operator logs and commands](../docs/optimization/README.md#先读哪几份), [Cache explanation](../docs/optimization/KV_CACHE_AND_FORWARD.md). Current strict BF16 HF comparison: 9/9 first tokens match, 5/9 full eight-token sequences match; remaining divergence is recorded, not treated as a pass.

For interview practice, use the [resume and 32-question guide](../docs/optimization/RESUME_AND_INTERVIEW.md) for measurement explanations, worked examples and commands that save separate practice records. The November 3, 2026 documentation update does not add new measurements; published results remain from November 2. Resume latency reductions combine CUDA operator and model execution changes against the project's naive baseline.

After building/installing the native library, use an editable Python install to avoid loading a stale copy from site-packages:

```bash
uv pip install --python .venv/bin/python --no-deps -e ./python
```

Without that, a plain `import llaisys` can load an older installed package even after `xmake install`. The new benchmark scripts prepend the workspace Python directory; legacy scripts need the editable install or `PYTHONPATH=python`.

## Prompt preparation dependencies

From the repository root, install the tested prompt-preparation dependencies into
the existing virtual environment and run the script:

```bash
uv pip install --python .venv/bin/python -r benchmark/requirements.txt
.venv/bin/python benchmark/prepare_prompts.py
```

This virtual environment was created with uv and does not include pip. Transformers
5.15.0 and Tokenizers 0.22.2 require `huggingface-hub < 2.0`, while Datasets 5.1.0
requires `huggingface-hub >= 1.31.0`. The requirements above pin a compatible set;
installing an unconstrained Hub 2.x causes the Transformers import to fail.
Check the environment with `uv pip check --python .venv/bin/python`.
The HTTPX SOCKS extra is included for the local proxy configuration; without it,
Hub requests fail with `Using SOCKS proxy, but the 'socksio' package is not installed`.

`prepare_prompts.py` selects the first three nonempty, non-heading test rows with
at least 128 tokens. Each row produces prefixes of 16, 64, and 128 tokens, for
nine samples in `cache/prompt_sets/wikitext2_smoke.jsonl`. Each JSONL line records
the sample ID, dataset commit, zero-based source row, prompt length, token IDs,
and `input_mode="raw_completion"`. No chat template or automatic special tokens
are added.

The model and output paths are resolved relative to the repository, so running
the script from another working directory uses the same files. Rerunning replaces
the output rather than appending duplicates. Source data and its commit are saved
as a local snapshot; later runs reuse it offline. Use `--refresh` to resolve and
download the current dataset version again. Every sample records the commit;
reuse the generated file for experiments with identical input IDs. The output is
ignored by Git under `cache/`.

To inspect one saved sample from the repository root:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path

path = Path("cache/prompt_sets/wikitext2_smoke.jsonl")
samples = [json.loads(line) for line in path.read_text().splitlines()]
print("samples:", len(samples))
print(json.dumps(samples[0], ensure_ascii=False, indent=2))
assert len(samples) == 9
assert all(len(sample["input_ids"]) == sample["prompt_tokens"] for sample in samples)
PY
```

## Run one saved prompt

Start with a short native generation to check the file-to-model path:

```bash
.venv/bin/python benchmark/run_prompt.py
```

The default is `row_3_len_16` on NVIDIA, requesting three new tokens. The script
passes the saved IDs directly to `Qwen2.generate(top_k=1)` without a chat template
or re-tokenization. It prints the new token IDs and decoded text, checks that the
returned sequence preserves the input prefix, and appends a record to
`benchmark/results/generation_smoke.jsonl`. EOS may stop generation early.

After checking that output, try a longer saved input:

```bash
.venv/bin/python benchmark/run_prompt.py --sample-id row_3_len_64
```

This is a generation smoke test. It does not compare against Hugging Face or
measure benchmark performance; saved records explicitly say `reference_check=not_run`.

## Compare native generation with Hugging Face

```bash
# One short case first.
.venv/bin/python benchmark/compare_prompt.py

# Compare all nine inputs, requesting up to eight new tokens per input.
.venv/bin/python benchmark/compare_prompt.py --all --max-new-tokens 8
```

Both implementations receive the exact saved token IDs and the same local weights.
The reference uses BF16, eager attention, eval/inference mode, and an explicit
argmax loop with KV Cache. It does not inherit the model's default sampling
settings. EOS comes from the same model config. The HF model is released before
loading LLAISYS to avoid holding both models on the GPU simultaneously.

Records are appended to `benchmark/results/generation_compare.jsonl`. Each record
contains both generated ID lists and `reference_check=pass/fail`; `first_mismatch`
reports the zero-based new-token index and both IDs. If one output ends earlier,
the missing side is recorded as `null`. Any mismatch makes the script exit with
status 1 after saving all completed comparisons. A token mismatch is diagnostic
evidence; it does not by itself establish the location or cause of a numerical
error. Matching tokens likewise does not prove that all logits match.

The installed PyTorch can invoke Triton even without explicit `torch.compile`.
Triton's compiler requires matching Python development headers. This machine's
Python 3.12 headers were extracted from Ubuntu's `libpython3.12-dev` package into
`cache/python-dev/`; `compare_prompt.py` uses that include tree if the system
`Python.h` is absent. This changes compiler include paths, not model calculations.
On another Ubuntu machine running Python 3.12, installing `python3.12-dev` provides
the system headers. The local cache is ignored by Git and must be recreated when
reproducing that environment elsewhere.

These comparisons establish a correctness checkpoint. They do not provide
benchmark timings or end-to-end quality scores.

The initial 2026-11-01 run found mismatches in all nine eight-token cases; see the
[recorded conditions and first differences](../docs/results/token_compare_20261101.md).

## Compare intermediate prefill tensors

Build the opt-in native snapshot support, then compare one prefill:

```bash
xmake -y
xmake install -y
.venv/bin/python benchmark/compare_tensors.py
```

The default input is `row_3_len_64`. HF forward hooks and native snapshot points
capture `embedding`, `layer0.input_norm`, and `layer0.output`. HF's batch dimension
is removed so both sides use `[token, hidden]`. Both models run BF16 with identical
input IDs, and only one new token is computed. No subsequent decode is involved.

Native snapshots are enabled by the process environment variable
`LLAISYS_QWEN2_DEBUG_DIR`. The script sets it only around native generation and
restores its previous value afterward. With the variable absent or empty, native
inference performs no snapshot copies or file writes. Snapshots are taken only
when cached length is zero, so later decode calls cannot overwrite the prefill
files. Another reset/new request can overwrite files if the same directory is
manually reused; the comparison script creates a fresh directory for every run.

Output layout:

```text
cache/tensor_debug/row_3_len_64_<unique suffix>/
  hf/
    embedding.pt                 # [T,H] float32 CPU values; source dtype recorded
    layer0.input_norm.pt
    layer0.output.pt
  native/
    embedding.bin                # Original BF16/F16/F32 bytes copied to the host
    embedding.json               # dtype, shape, byte order, byte count
    layer0.input_norm.bin
    layer0.input_norm.json
    layer0.output.bin
    layer0.output.json
  summary.json                   # Metrics, first difference index/values, input metadata
  summary.md                     # Readable comparison table
```

The native reader validates dtype, byte order, shape, and file length before
converting to float32 for comparison. BF16 conversion here preserves the stored
numeric values; it does not recover precision lost during inference. Exact-value
agreement is numeric equality, not a bitwise test of signed zero or NaN payloads.
Errors are computed over finite pairs; non-finite counts are reported separately.
No automatic tolerance classifies the first rounding difference as a model bug.

Inspect `summary.md` for the table and `summary.json` for the first differing
`[token, channel]` coordinate. To change inputs:

```bash
.venv/bin/python benchmark/compare_tensors.py --sample-id row_3_len_16
```

This command succeeds when snapshots and metrics are produced, even if their
values or generated tokens differ. Shape/format problems cause an error. These
diagnostic runs include extra synchronization, D2H copies, and file I/O, so their
timings must not be used as performance measurements.

See the [optimization report](../docs/optimization/README.md) for the current audit and measured results. The old two-generation TPOT estimate has been replaced with direct same-request prefill/decode timing. Legacy operator Python/event timing still includes dispatch effects; use the new C++ graph microbenchmark for device-only comparisons.

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

## Versioned CUDA microbenchmark and regression

```bash
.venv/bin/python test/test_cuda_edges.py
.venv/bin/python test/test_engine_regression.py --device nvidia
.venv/bin/python benchmark/cuda_experiments.py --op argmax --version 0 --dtype bf16
.venv/bin/python benchmark/cuda_experiments.py --op argmax --version 2 --dtype bf16
.venv/bin/python benchmark/run_optimization_suite.py --phase all \
  --output-dir benchmark/results/my_new_run
```

`cuda_experiments.py` validates outputs first, then captures repeated native calls in a private-stream CUDA Graph and measures replay using events. Lazy scratch/handles and graph creation are outside timing. A correctness failure is saved and makes the script fail. Native graph time excludes Python dispatch; the record's PyTorch reference time includes that boundary, so they are not used for cross-backend speedup ratios. Normal model inference does not use CUDA Graph.

V0 preserves basic algorithms, V1 adjusts low-precision intermediate rounding where needed, V2 is optimized/default. Global `LLAISYS_CUDA_VERSION` and per-op `LLAISYS_CUDA_RMS_NORM_VERSION` etc. are read on first operator call; restart the process to change them. The suite runs serially and clears inherited version overrides. Do not run simultaneous GPU benchmarks.

`test_engine_regression.py` creates a deterministic tiny local Qwen2, without downloading weights, and checks cached/full-prefix behavior, HF teacher-forced tokens, reset, capacity, invalid IDs/C API errors, stable workspace allocation and lifecycle. This is separate from full-model generation quality.

## Inference benchmark

```bash
# Real greedy: exact P/N, EOS respected.
.venv/bin/python benchmark/benchmark_infer.py \
  --prompt-tokens 128 --output-tokens 16 --warmup 2 --repeat 5

# Replay fixed tokens for controlled version ablation, explicitly ignoring EOS.
.venv/bin/python benchmark/benchmark_infer.py \
  --prompt-tokens 64 --output-tokens 8 --mode replay \
  --cuda-version 2 --engine-version 2 --cache-capacity 2048

# HF eager reference (separate process; same math controls).
.venv/bin/python benchmark/benchmark_infer.py \
  --backend hf --prompt-tokens 128 --output-tokens 16
```

The default model path is the local `DeepSeek-R1-Distill-Qwen-1.5B`. Existing saved prompt IDs are reused when length matches; `--sample-id` chooses exactly, `--token-id` forces repeated synthetic input, and records identify the source. `--matrix` tests lengths 1/32/128/512, using saved IDs where available and synthetic IDs otherwise. This is not a batch benchmark.

Public `prefill/decode/reset/state` APIs now exist. The timer measures host IDs to host token in the same request, with explicit synchronization; output N entails N-1 decode steps unless EOS stops early. TTFT/TPOT include native API work/copies, but exclude model loading, tokenizer, printing and service scheduling. Per-request TPOT median and all-step mean-throughput are distinct statistics; raw samples are retained. No two-generation subtraction or clipping is used.

In `replay` mode, every decode call receives the last prompt token ID, regardless of the previous prediction, and EOS is ignored. This fixes the inputs and cache lengths across versions; it is not a greedy text-quality test. Normal `greedy` mode feeds the model's prediction back and may stop at EOS. Generating one output involves no decode steps, so its TPOT is absent rather than zero.

Native allocator current/peak cover framework device Tensor storage only, excluding operator raw CUDA scratch, cuBLAS internals and context. HF statistics cover the PyTorch allocator. They are labelled separately and are not a fair total-process memory comparison. Cache capacity and model-owned cache/weight/workspace bytes are separately recorded. Default native capacity is 2048, limited by the model position bound; increase it explicitly for longer requests.

Use `--nvtx --profile` with Nsight Systems `--capture-range=cudaProfilerApi` to capture only measured requests after warmup. `LLAISYS_QWEN2_DEBUG_DIR` must be unset during timing; the script rejects it. Trace commands and interpreted results are in the optimization report.

## Extended tensor comparison

`compare_tensors.py --full-layer` captures Q/K/V projections, RoPE, attention, residuals and MLP stages in layer 0, plus the last-position final norm/logits. The summary includes top-5 logits, top1/top2 margin and cosine similarity. These diagnostic runs include copies and file I/O and are not performance measurements. The ordinary option keeps the three original checkpoints.

The final HF comparison uses `logits_to_keep=1`, eager attention, TF32 disabled and low-precision GEMM partial reductions disabled. `compare_prompt.py --hf-logits-to-keep 0` reproduces the earlier all-position projection setting if needed; shape-dependent GEMM differences can affect rounding. Math settings are recorded; do not silently compare old and new runs as identical configurations.

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
