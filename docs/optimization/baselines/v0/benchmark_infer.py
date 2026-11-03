#!/usr/bin/env python3
import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

import llaisys  # noqa: E402
import torch  # noqa: E402


DEFAULT_MATRIX = ((1, 32), (128, 32), (512, 128), (2048, 128))


def percentile(values, value):
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * value / 100
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summary(values, prefix):
    if not values:
        return {
            f"{prefix}_mean_ms": None,
            f"{prefix}_median_ms": None,
            f"{prefix}_p90_ms": None,
            f"{prefix}_p99_ms": None,
            f"{prefix}_min_ms": None,
            f"{prefix}_max_ms": None,
        }
    return {
        f"{prefix}_mean_ms": statistics.fmean(values),
        f"{prefix}_median_ms": statistics.median(values),
        f"{prefix}_p90_ms": percentile(values, 90),
        f"{prefix}_p99_ms": percentile(values, 99),
        f"{prefix}_min_ms": min(values),
        f"{prefix}_max_ms": max(values),
    }


def git_revision():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_model_config(model_path):
    config_path = model_path / "config.json"
    if not config_path.is_file():
        return {}
    with config_path.open(encoding="utf-8") as file:
        return json.load(file)


def timed_generate(model, inputs, output_tokens, synchronize):
    synchronize()
    start = time.perf_counter()
    output = model.generate(inputs, max_new_tokens=output_tokens, top_k=1)
    synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000
    generated_tokens = len(output) - len(inputs)
    if generated_tokens < 1:
        raise RuntimeError(
            f"generate returned {len(output)} tokens for a {len(inputs)}-token prompt; "
            "finish the Qwen2 generate implementation before benchmarking inference"
        )
    return elapsed_ms, generated_tokens


def benchmark_case(model, prompt_tokens, output_tokens, token_id, warmup, repeat, synchronize):
    inputs = [token_id] * prompt_tokens
    for _ in range(warmup):
        timed_generate(model, inputs, output_tokens, synchronize)

    ttft_samples = []
    total_samples = []
    tpot_samples = []
    generated_token_counts = []
    for _ in range(repeat):
        ttft_ms, _ = timed_generate(model, inputs, 1, synchronize)
        total_ms, generated_tokens = timed_generate(
            model, inputs, output_tokens, synchronize
        )
        ttft_samples.append(ttft_ms)
        total_samples.append(total_ms)
        generated_token_counts.append(generated_tokens)
        if generated_tokens > 1:
            tpot_samples.append(max(0.0, total_ms - ttft_ms) / (generated_tokens - 1))

    record = {}
    record.update(summary(ttft_samples, "ttft"))
    record.update(summary(tpot_samples, "tpot"))
    record.update(summary(total_samples, "end_to_end"))
    record["generated_tokens_mean"] = statistics.fmean(generated_token_counts)
    record["generated_tokens_min"] = min(generated_token_counts)
    record["generated_tokens_max"] = max(generated_token_counts)
    record["decode_tokens_per_s"] = (
        1000 / record["tpot_mean_ms"] if record["tpot_mean_ms"] else None
    )
    return record


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark LLAISYS Qwen2 inference.")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--device", default="cpu", choices=("cpu", "nvidia"))
    parser.add_argument("--prompt-tokens", default=128, type=int)
    parser.add_argument("--output-tokens", default=32, type=int)
    parser.add_argument("--token-id", default=None, type=int)
    parser.add_argument("--warmup", default=2, type=int)
    parser.add_argument("--repeat", default=10, type=int)
    parser.add_argument("--matrix", action="store_true", help="Run the guide's length matrix")
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="JSONL output (default: benchmark/results/infer_<device>.jsonl)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.model.is_dir():
        raise SystemExit(f"Model directory does not exist: {args.model}")
    if args.warmup < 0 or args.repeat <= 0:
        raise SystemExit("--warmup must be non-negative and --repeat must be positive")

    cases = DEFAULT_MATRIX if args.matrix else ((args.prompt_tokens, args.output_tokens),)
    for prompt_tokens, output_tokens in cases:
        if prompt_tokens <= 0 or output_tokens <= 0:
            raise SystemExit("Prompt and output token counts must be positive")

    config = load_model_config(args.model)
    token_id = args.token_id
    if token_id is None:
        token_id = config.get("bos_token_id") or 1
    device = llaisys.DeviceType.CPU if args.device == "cpu" else llaisys.DeviceType.NVIDIA
    runtime = llaisys.RuntimeAPI(device)
    if runtime.get_device_count() < 1:
        raise SystemExit(
            f"No LLAISYS {args.device} device is available. Build and install that backend "
            "before running inference benchmarks."
        )
    runtime.set_device(0)

    load_start = time.perf_counter()
    model = llaisys.models.Qwen2(args.model, device)
    runtime.device_synchronize()
    load_ms = (time.perf_counter() - load_start) * 1000

    output_path = args.json or REPO_ROOT / "benchmark" / "results" / f"infer_{args.device}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as output:
        for prompt_tokens, output_tokens in cases:
            metrics = benchmark_case(
                model,
                prompt_tokens,
                output_tokens,
                token_id,
                args.warmup,
                args.repeat,
                runtime.device_synchronize,
            )
            record = {
                "schema_version": 1,
                "benchmark": "inference",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "git_revision": git_revision(),
                "model": str(args.model.resolve()),
                "model_revision": config.get("_commit_hash"),
                "device": args.device,
                "device_id": 0,
                "device_name": (
                    torch.cuda.get_device_name(0)
                    if args.device == "nvidia" and torch.cuda.is_available()
                    else platform.processor() or platform.machine()
                ),
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "dtype": config.get("torch_dtype"),
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "warmup": args.warmup,
                "repeat": args.repeat,
                "load_ms": load_ms,
                "h2d_ms": None,
                "prefill_ms": None,
                "prefill_tokens_per_s": None,
                "peak_memory_bytes": None,
                "token_check": "not_run",
                "timer": "perf_counter_with_device_synchronize",
                "measurement_method": "generate_api_ttft_and_tpot_estimate",
                **metrics,
            }
            output.write(json.dumps(record, sort_keys=True) + "\n")
            tpot = record["tpot_mean_ms"]
            decode_rate = record["decode_tokens_per_s"]
            print(
                f"prompt={prompt_tokens} output={output_tokens} "
                f"TTFT={record['ttft_mean_ms']:.3f} ms "
                f"TPOT={f'{tpot:.3f}' if tpot is not None else 'n/a'} ms "
                f"decode={f'{decode_rate:.3f}' if decode_rate is not None else 'n/a'} tok/s"
            )

    print(f"Benchmark records appended to {output_path}")


if __name__ == "__main__":
    main()
