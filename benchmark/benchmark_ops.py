#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

import llaisys  # noqa: E402


SUPPORTED_OPS = (
    "add",
    "argmax",
    "embedding",
    "linear",
    "rms_norm",
    "rope",
    "self_attention",
    "swiglu",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run correctness checks and benchmarks for LLAISYS operators."
    )
    parser.add_argument("--device", default="cpu", choices=("cpu", "nvidia"))
    parser.add_argument("--ops", nargs="+", choices=SUPPORTED_OPS, default=SUPPORTED_OPS)
    parser.add_argument("--warmup", default=10, type=int)
    parser.add_argument("--repeat", default=100, type=int)
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="JSONL output (default: benchmark/results/ops_<device>.jsonl)",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue after an operator fails",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.warmup < 0 or args.repeat <= 0:
        raise SystemExit("--warmup must be non-negative and --repeat must be positive")

    device = llaisys.DeviceType.CPU if args.device == "cpu" else llaisys.DeviceType.NVIDIA
    device_count = llaisys.RuntimeAPI(device).get_device_count()
    if device_count < 1:
        raise SystemExit(
            f"No LLAISYS {args.device} device is available. Build and install that backend "
            "before running its operator benchmarks."
        )

    json_path = args.json or REPO_ROOT / "benchmark" / "results" / f"ops_{args.device}.jsonl"
    failures = []
    for op in args.ops:
        command = [
            sys.executable,
            str(REPO_ROOT / "test" / "ops" / f"{op}.py"),
            "--device",
            args.device,
            "--profile",
            "--warmup",
            str(args.warmup),
            "--repeat",
            str(args.repeat),
            "--json",
            str(json_path),
        ]
        print(f"\n=== {op} ({args.device}) ===", flush=True)
        result = subprocess.run(command, cwd=REPO_ROOT, check=False)
        if result.returncode != 0:
            failures.append(op)
            if not args.keep_going:
                break

    if failures:
        raise SystemExit(f"Failed operators: {', '.join(failures)}")
    print(f"\nBenchmark records appended to {json_path}")


if __name__ == "__main__":
    main()
