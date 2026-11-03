"""Run reproducible optimization measurements serially on one GPU (no competing jobs)."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OPS = (
    "add",
    "embedding",
    "linear",
    "rms_norm",
    "rope",
    "self_attention",
    "swiglu",
    "argmax",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("ops", "engine", "ablation", "all"), default="all"
    )
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "benchmark/results/optimization"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logs = ROOT / "cache/optimization_logs"
    logs.mkdir(parents=True, exist_ok=True)

    def run(name, script, flags, overrides=None):
        env = os.environ.copy()
        for key in list(env):
            if key.startswith("LLAISYS_CUDA_") or key in (
                "LLAISYS_ENGINE_VERSION",
                "LLAISYS_QWEN2_DEBUG_DIR",
                "LLAISYS_LINEAR_BIAS_MODE",
            ):
                env.pop(key)
        env.update(overrides or {})
        command = [
            sys.executable,
            "-u",
            str(ROOT / "benchmark" / script),
            *map(str, flags),
        ]
        print(name, " ".join(command), flush=True)
        with (logs / (name + ".log")).open("w") as file:
            subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                stdout=file,
                stderr=subprocess.STDOUT,
                check=True,
            )
        print((logs / (name + ".log")).read_text()[-1200:], flush=True)

    if args.phase in ("ops", "all"):
        for dtype, versions in [("bf16", (0, 1, 2)), ("f16", (2,)), ("f32", (2,))]:
            for version in versions:
                run(
                    f"ops-{dtype}-v{version}",
                    "cuda_experiments.py",
                    [
                        "--dtype",
                        dtype,
                        "--version",
                        version,
                        "--repeat",
                        15,
                        "--output",
                        args.output_dir / "suite_ops.jsonl",
                    ],
                )
    if args.phase in ("engine", "all"):
        for cuda, engine in [(0, 1), (1, 1), (2, 1), (2, 2)]:
            run(
                f"engine-c{cuda}-e{engine}",
                "benchmark_infer.py",
                [
                    "--cuda-version",
                    cuda,
                    "--engine-version",
                    engine,
                    "--prompt-tokens",
                    64,
                    "--output-tokens",
                    8,
                    "--mode",
                    "replay",
                    "--warmup",
                    1,
                    "--repeat",
                    args.repeat,
                    "--json",
                    args.output_dir / "engine_replay.jsonl",
                ],
            )
        for backend in ("llaisys", "hf"):
            for prompt in (16, 64, 128):
                run(
                    f"greedy-{backend}-p{prompt}",
                    "benchmark_infer.py",
                    [
                        "--backend",
                        backend,
                        "--prompt-tokens",
                        prompt,
                        "--output-tokens",
                        16,
                        "--mode",
                        "greedy",
                        "--warmup",
                        2,
                        "--repeat",
                        args.repeat,
                        "--json",
                        args.output_dir / "engine_greedy.jsonl",
                    ],
                )
    if args.phase in ("ablation", "all"):
        for op in ("all_v2", *OPS):
            run(
                f"ablation-{op}",
                "benchmark_infer.py",
                [
                    "--prompt-tokens",
                    64,
                    "--output-tokens",
                    8,
                    "--mode",
                    "replay",
                    "--warmup",
                    2,
                    "--repeat",
                    args.repeat,
                    "--json",
                    args.output_dir / "ablation.jsonl",
                ],
                {} if op == "all_v2" else {f"LLAISYS_CUDA_{op.upper()}_VERSION": "1"},
            )


if __name__ == "__main__":
    main()
