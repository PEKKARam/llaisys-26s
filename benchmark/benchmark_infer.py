#!/usr/bin/env python3
"""Single-request model-call latency. Time prefill/decode in the same request."""

import argparse
import ctypes
from contextlib import nullcontext
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=ROOT / "DeepSeek-R1-Distill-Qwen-1.5B")
    p.add_argument("--backend", choices=("llaisys", "hf"), default="llaisys")
    p.add_argument("--device", choices=("cpu", "nvidia"), default="nvidia")
    p.add_argument("--cuda-version", type=int, choices=(0, 1, 2), default=2)
    p.add_argument("--engine-version", type=int, choices=(0, 1, 2), default=2)
    p.add_argument("--cache-capacity", type=int, default=2048)
    p.add_argument("--prompt-tokens", type=int, default=128)
    p.add_argument("--output-tokens", type=int, default=32)
    p.add_argument("--mode", choices=("greedy", "replay"), default="greedy")
    p.add_argument("--sample-id")
    p.add_argument(
        "--prompts", type=Path, default=ROOT / "cache/prompt_sets/wikitext2_smoke.jsonl"
    )
    p.add_argument("--token-id", type=int)
    p.add_argument("--matrix", action="store_true")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument(
        "--nvtx", action="store_true", help="Annotate prefill/decode in Nsight Systems"
    )
    p.add_argument(
        "--profile",
        action="store_true",
        help="Start CUDA profiler after warmup; use with nsys capture-range",
    )
    p.add_argument(
        "--json",
        type=Path,
        default=ROOT / "benchmark/results/optimization/engine.jsonl",
    )
    args = p.parse_args()
    if (
        args.warmup < 0
        or args.repeat < 1
        or args.output_tokens < 1
        or args.prompt_tokens < 1
    ):
        p.error("Lengths/repeat must be positive; warmup nonnegative")
    if os.environ.get("LLAISYS_QWEN2_DEBUG_DIR"):
        p.error("Disable LLAISYS_QWEN2_DEBUG_DIR before timing")
    os.environ["LLAISYS_CUDA_VERSION"] = str(args.cuda_version)
    os.environ["LLAISYS_ENGINE_VERSION"] = str(args.engine_version)
    import torch
    from compare_prompt import configure_local_python_headers, configure_reference_math

    configure_reference_math(torch)
    if args.device == "nvidia" and not torch.cuda.is_available():
        p.error("CUDA unavailable")

    def sync():
        if args.device == "nvidia":
            torch.cuda.synchronize()

    config = json.loads((args.model / "config.json").read_text())
    saved = []
    if args.prompts.is_file():
        saved = [
            json.loads(line)
            for line in args.prompts.read_text().splitlines()
            if line.strip()
        ]
    cases = []
    lengths = (1, 32, 128, 512) if args.matrix else (args.prompt_tokens,)
    if args.sample_id:
        sample = next((x for x in saved if x["id"] == args.sample_id), None)
        if sample is None:
            p.error("sample ID not found")
        cases = [(sample["input_ids"], sample["id"])]
    else:
        for length in lengths:
            sample = next((x for x in saved if x["prompt_tokens"] == length), None)
            if sample and args.token_id is None:
                cases.append((sample["input_ids"], sample["id"]))
            else:
                cases.append(
                    (
                        [args.token_id if args.token_id is not None else 1] * length,
                        "synthetic_repeat",
                    )
                )
    for ids, _ in cases:
        if len(ids) + args.output_tokens - 1 > args.cache_capacity:
            p.error("Increase --cache-capacity for this request")
        if any(type(i) is not int or not 0 <= i < config["vocab_size"] for i in ids):
            p.error("Invalid token ID")
    t = time.perf_counter()
    if args.backend == "llaisys":
        import llaisys
        from llaisys.libllaisys import LIB_LLAISYS

        device = (
            llaisys.DeviceType.NVIDIA
            if args.device == "nvidia"
            else llaisys.DeviceType.CPU
        )
        model = llaisys.models.Qwen2(
            args.model, device, cache_capacity=args.cache_capacity
        )

        class Stats(ctypes.Structure):
            _fields_ = [
                (name, ctypes.c_size_t)
                for name in ("current_bytes", "peak_bytes", "allocations")
            ]

        LIB_LLAISYS.llaisysGetAllocatorStats.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(Stats),
        ]
        LIB_LLAISYS.llaisysGetAllocatorStats.restype = ctypes.c_int
        LIB_LLAISYS.llaisysResetAllocatorPeak.argtypes = [ctypes.c_int, ctypes.c_int]
        LIB_LLAISYS.llaisysResetAllocatorPeak.restype = ctypes.c_int

        def stats():
            value = Stats()
            if (
                LIB_LLAISYS.llaisysGetAllocatorStats(device, 0, ctypes.byref(value))
                != 0
            ):
                raise RuntimeError("Allocator stats failed")
            return {name: int(getattr(value, name)) for name, _ in value._fields_}

        def reset_peak():
            if LIB_LLAISYS.llaisysResetAllocatorPeak(device, 0) != 0:
                raise RuntimeError("Peak reset failed")

        reset = model.reset
        prefill = model.prefill
        decode = model.decode
        library_sha = hashlib.sha256(Path(LIB_LLAISYS._name).read_bytes()).hexdigest()
    else:
        configure_local_python_headers()
        from transformers import AutoModelForCausalLM

        model = (
            AutoModelForCausalLM.from_pretrained(
                args.model,
                dtype=torch.bfloat16,
                attn_implementation="eager",
                local_files_only=True,
            )
            .to("cuda:0" if args.device == "nvidia" else "cpu")
            .eval()
        )
        past = None

        def reset():
            nonlocal past
            past = None

        def hf_step(ids):
            nonlocal past
            with torch.inference_mode():
                output = model(
                    input_ids=torch.tensor(
                        [ids], dtype=torch.long, device=model.device
                    ),
                    past_key_values=past,
                    use_cache=True,
                    logits_to_keep=1,
                    return_dict=True,
                )
                past = output.past_key_values
                return int(output.logits[0, -1].argmax().item())

        def prefill(ids):
            reset()
            return hf_step(ids)

        def decode(token):
            return hf_step([token])

        def reset_peak():
            if args.device == "nvidia":
                torch.cuda.reset_peak_memory_stats()

        def stats():
            return dict(
                current_bytes=torch.cuda.memory_allocated()
                if args.device == "nvidia"
                else 0,
                peak_bytes=torch.cuda.max_memory_allocated()
                if args.device == "nvidia"
                else 0,
                allocations=0,
            )

        library_sha = None
    sync()
    load_ms = (time.perf_counter() - t) * 1000

    def request(ids):
        reset()
        sync()
        with (
            torch.cuda.nvtx.range("prefill")
            if args.nvtx and args.device == "nvidia"
            else nullcontext()
        ):
            begin = time.perf_counter()
            token = prefill(ids)
            sync()
            ttft = (time.perf_counter() - begin) * 1000
        predictions = [token]
        steps = []
        for _ in range(args.output_tokens - 1):
            if args.mode == "greedy" and token == config["eos_token_id"]:
                break
            feed = token if args.mode == "greedy" else ids[-1]
            with (
                torch.cuda.nvtx.range("decode")
                if args.nvtx and args.device == "nvidia"
                else nullcontext()
            ):
                begin = time.perf_counter()
                token = decode(feed)
                sync()
                steps.append((time.perf_counter() - begin) * 1000)
                predictions.append(token)
        return ttft, steps, predictions

    args.json.parent.mkdir(parents=True, exist_ok=True)
    for ids, source in cases:
        for _ in range(args.warmup):
            request(ids)
        reset()
        reset_peak()
        before = stats()
        first = []
        steps = []
        tpots = []
        counts = []
        predictions = []
        if args.profile and args.device == "nvidia":
            torch.cuda.profiler.start()
        for _ in range(args.repeat):
            ttft, decode_steps, predictions = request(ids)
            first.append(ttft)
            steps.extend(decode_steps)
            counts.append(len(predictions))
            if decode_steps:
                tpots.append(statistics.mean(decode_steps))
        if args.profile and args.device == "nvidia":
            torch.cuda.profiler.stop()
        after = stats()
        record = dict(
            backend=args.backend,
            device=args.device,
            dtype="bf16",
            batch=1,
            model=str(args.model.resolve()),
            source=source,
            prompt_tokens=len(ids),
            input_ids=ids,
            output_tokens_requested=args.output_tokens,
            output_tokens_actual=counts,
            mode=args.mode,
            replay_token=ids[-1] if args.mode == "replay" else None,
            ignore_eos=args.mode == "replay",
            predicted_ids_last=predictions,
            cache_capacity=args.cache_capacity,
            cuda_version=args.cuda_version if args.backend == "llaisys" else None,
            engine_version=args.engine_version if args.backend == "llaisys" else None,
            cuda_overrides={
                k: v
                for k, v in os.environ.items()
                if k.startswith("LLAISYS_CUDA_") and k.endswith("_VERSION")
            },
            warmup=args.warmup,
            repeat=args.repeat,
            load_ms=load_ms,
            ttft_samples_ms=first,
            decode_step_samples_ms=steps,
            tpot_request_samples_ms=tpots,
            ttft_median_ms=statistics.median(first),
            ttft_mean_ms=statistics.mean(first),
            tpot_median_ms=statistics.median(tpots) if tpots else None,
            tpot_mean_ms=statistics.mean(steps) if steps else None,
            decode_tokens_per_s=1000 / statistics.mean(steps) if steps else None,
            allocator_peak_bytes=after["peak_bytes"],
            allocator_current_bytes=after["current_bytes"],
            tensor_allocations_during_measurement=after["allocations"]
            - before["allocations"]
            if args.backend == "llaisys"
            else None,
            memory_scope="framework_tensor_allocator_excludes_operator_scratch_cublas_context"
            if args.backend == "llaisys"
            else "pytorch_allocator",
            model_state=model.state if args.backend == "llaisys" else None,
            cache_policy="preallocated_contiguous_per_layer"
            if args.backend == "llaisys"
            else "hf_dynamic_cache",
            timer="perf_counter_with_sync_model_api_host_ids_to_host_token",
            token_check="not_run_in_timed_region",
            nvtx=args.nvtx,
            profiler=args.profile,
            gpu=torch.cuda.get_device_name(0)
            if args.device == "nvidia"
            else platform.processor(),
            torch_version=torch.__version__,
            tf32=False,
            reduced_precision_gemm_reduction=False,
            git_revision=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            library_sha256=library_sha,
        )
        with args.json.open("a") as file:
            file.write(json.dumps(record) + "\n")
        print(
            f"{args.backend} cuda-v{args.cuda_version} engine-v{args.engine_version} {source} P={len(ids)} N={counts} TTFT={record['ttft_median_ms']:.3f}ms TPOT={record['tpot_median_ms']}ms decode={record['decode_tokens_per_s']}tok/s tensor_allocs={record['tensor_allocations_during_measurement']} peak={after['peak_bytes']}",
            flush=True,
        )
    if args.backend == "llaisys":
        model.close()


if __name__ == "__main__":
    main()
