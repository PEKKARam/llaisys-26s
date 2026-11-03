"""Reproducible single-operator correctness and timing experiments (batch dimension absent)."""

import argparse
import ctypes
import hashlib
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--op", choices=(*OPS, "all"), default="all")
    p.add_argument("--version", type=int, choices=(0, 1, 2), default=2)
    p.add_argument("--dtype", choices=("bf16", "f16", "f32"), default="bf16")
    p.add_argument("--repeat", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--timer", choices=("native", "python"), default="native")
    p.add_argument(
        "--output", type=Path, default=ROOT / "benchmark/results/optimization/ops.jsonl"
    )
    args = p.parse_args()
    if args.repeat < 1 or args.warmup < 0:
        p.error("repeat must be positive; warmup nonnegative")
    os.environ["LLAISYS_CUDA_VERSION"] = str(args.version)
    import torch
    import llaisys
    from llaisys.libllaisys import LIB_LLAISYS, llaisysTensor_t

    if args.timer == "native":
        native_bench = LIB_LLAISYS.llaisysBenchmarkCudaOp
        native_bench.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(llaisysTensor_t),
            ctypes.c_size_t,
            ctypes.c_float,
            ctypes.c_int,
        ]
        native_bench.restype = ctypes.c_float
    torch.manual_seed(20261102)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    device = llaisys.DeviceType.NVIDIA
    api = llaisys.RuntimeAPI(device)
    api.set_device(0)
    dt = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}[
        args.dtype
    ]
    ld = {
        "bf16": llaisys.DataType.BF16,
        "f16": llaisys.DataType.F16,
        "f32": llaisys.DataType.F32,
    }
    handles = []

    def tensor(x):
        dtype = llaisys.DataType.I64 if x.dtype == torch.int64 else ld[args.dtype]
        t = llaisys.Tensor(tuple(x.shape), dtype=dtype, device=device)
        api.memcpy_sync(
            t.data_ptr(),
            x.data_ptr(),
            x.numel() * x.element_size(),
            llaisys.MemcpyKind.D2D,
        )
        handles.append(t)
        return t

    def rand(shape):
        return torch.randn(shape, dtype=dt, device="cuda") * 0.2

    def output(shape, integer=False):
        x = torch.empty(shape, dtype=torch.int64 if integer else dt, device="cuda")
        return x, tensor(x)

    def read(t, like):
        out = torch.empty_like(like)
        api.memcpy_sync(
            out.data_ptr(),
            t.data_ptr(),
            out.numel() * out.element_size(),
            llaisys.MemcpyKind.D2D,
        )
        return out

    def cases(op):
        if op == "add":
            return [(1, 1536), (128, 1536), (1024, 1024)]
        if op == "embedding":
            return [(1, 1536), (128, 1536), (128, 155)]
        if op == "linear":
            return [
                (1, 1536, 1536),
                (32, 1536, 1536),
                (1, 8960, 1536),
                (32, 8960, 1536),
                (1, 1536, 8960),
                (1, 151936, 1536),
            ]
        if op == "rms_norm":
            return [(1, 1536), (128, 1536), (5, 157)]
        if op == "rope":
            return [(1, 12, 128), (128, 12, 128), (5, 2, 10)]
        if op == "self_attention":
            return [
                (1, 128, 12, 2, 128),
                (32, 32, 12, 2, 128),
                (128, 128, 12, 2, 128),
                (1, 1024, 12, 2, 128),
                (5, 11, 4, 2, 8),
            ]
        if op == "swiglu":
            return [(1, 8960), (128, 8960), (5, 157)]
        return [(1, 151936), (5, 157)]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    git_rev = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    lib = ROOT / "python/llaisys/libllaisys/libllaisys.so"
    lib_sha = hashlib.sha256(lib.read_bytes()).hexdigest()
    failures = 0
    for op in OPS if args.op == "all" else (args.op,):
        for shape in cases(op):
            handles.clear()
            if op == "add":
                a, b = rand(shape), rand(shape)
                aa, bb = tensor(a), tensor(b)
                out, oo = output(shape)
                call = lambda: llaisys.Ops.add(oo, aa, bb)
                reference = lambda: torch.add(a, b)
            elif op == "embedding":
                m, k = shape
                w = rand((4096, k))
                ids = torch.arange(m, device="cuda", dtype=torch.int64) % 4096
                ww, ii = tensor(w), tensor(ids)
                out, oo = output((m, k))
                call = lambda: llaisys.Ops.embedding(oo, ii, ww)
                reference = lambda: torch.nn.functional.embedding(ids, w)
            elif op == "linear":
                m, n, k = shape
                x, w, b = rand((m, k)), rand((n, k)), rand((n,))
                xx, ww, bb = tensor(x), tensor(w), tensor(b)
                out, oo = output((m, n))
                call = lambda: llaisys.Ops.linear(oo, xx, ww, bb)
                reference = lambda: torch.nn.functional.linear(x, w, b)
            elif op == "rms_norm":
                m, k = shape
                x, w = rand(shape), rand((k,))
                xx, ww = tensor(x), tensor(w)
                out, oo = output(shape)
                call = lambda: llaisys.Ops.rms_norm(oo, xx, ww, 1e-6)
                reference = lambda: (
                    (
                        x.float()
                        * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
                    ).to(dt)
                    * w
                )
            elif op == "rope":
                s, h, d = shape
                x = rand(shape)
                pos = torch.arange(19, 19 + s, dtype=torch.int64, device="cuda")
                xx, pp = tensor(x), tensor(pos)
                out, oo = output(shape)

                def reference():
                    angles = (
                        pos.float()[:, None]
                        * (
                            1
                            / (
                                10000.0
                                ** (torch.arange(0, d, 2, device="cuda").float() / d)
                            )
                        )[None, :]
                    )
                    c = angles.cos().to(dt)[:, None, :]
                    sn = angles.sin().to(dt)[:, None, :]
                    a, b = x[..., : d // 2], x[..., d // 2 :]
                    return torch.cat((a * c - b * sn, b * c + a * sn), dim=-1)

                call = lambda: llaisys.Ops.rope(oo, xx, pp, 10000.0)
            elif op == "self_attention":
                ql, kl, h, hk, d = shape
                q, k, v = rand((ql, h, d)), rand((kl, hk, d)), rand((kl, hk, d))
                qq, kk, vv = tensor(q), tensor(k), tensor(v)
                out, oo = output((ql, h, d))
                scale = d**-0.5

                def reference():
                    qt = q.transpose(0, 1)
                    kt = k.transpose(0, 1).repeat_interleave(h // hk, 0)
                    vt = v.transpose(0, 1).repeat_interleave(h // hk, 0)
                    scores = (qt @ kt.transpose(1, 2)) * scale
                    mask = (
                        torch.arange(kl, device="cuda")[None, :]
                        > (kl - ql + torch.arange(ql, device="cuda"))[:, None]
                    )
                    scores = scores.masked_fill(mask, -float("inf"))
                    probs = torch.softmax(scores.float(), dim=-1).to(dt)
                    return (probs @ vt).transpose(0, 1).contiguous()

                call = lambda: llaisys.Ops.self_attention(oo, qq, kk, vv, scale)
            elif op == "swiglu":
                g, u = rand(shape), rand(shape)
                gg, uu = tensor(g), tensor(u)
                out, oo = output(shape)
                call = lambda: llaisys.Ops.swiglu(oo, gg, uu)
                reference = lambda: torch.nn.functional.silu(g) * u
            else:
                m, k = shape
                x = rand(shape)
                x[:, 0] = 2.0
                x[:, 1] = 2.0  # explicit first-index tie
                xx = tensor(x)
                out, oo = output((m, 1))
                indices, ii = output((m, 1), integer=True)
                call = lambda: llaisys.Ops.argmax(ii, oo, xx)
                reference = lambda: x.max(-1, keepdim=True).values
            expected = reference()
            call()
            api.device_synchronize()
            actual = read(oo, out)
            diff = (actual.float() - expected.float()).abs()
            tol = {"bf16": (0.02, 0.02), "f16": (0.003, 0.003), "f32": (1e-4, 1e-4)}[
                args.dtype
            ]
            correct = bool(
                torch.isfinite(actual).all()
                and torch.allclose(actual, expected, atol=tol[0], rtol=tol[1])
            )
            if op == "argmax":
                correct = correct and bool(
                    torch.equal(read(ii, indices), x.argmax(-1, keepdim=True))
                )
            # Events use the legacy/default stream, matching all native kernels and cuBLAS.
            stream = torch.cuda.default_stream()

            def timing(fn):
                with torch.cuda.stream(stream):
                    for _ in range(args.warmup):
                        fn()
                    torch.cuda.synchronize()
                    a, b = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    a.record(stream)
                    fn()
                    b.record(stream)
                    b.synchronize()
                    pilot = a.elapsed_time(b)
                    repeat = min(args.repeat, max(3, int(200 / max(pilot, 0.001))))
                    samples = []
                    for _ in range(repeat):
                        a.record(stream)
                        fn()
                        b.record(stream)
                        b.synchronize()
                        samples.append(a.elapsed_time(b))
                return samples

            if args.timer == "native":
                if op == "argmax":
                    ordered = [ii, oo, xx]
                elif op == "self_attention":
                    ordered = [oo, qq, kk, vv]
                elif op == "linear":
                    ordered = [oo, xx, ww, bb]
                elif op == "embedding":
                    ordered = [oo, ii, ww]
                elif op == "rope":
                    ordered = [oo, xx, pp]
                elif op == "rms_norm":
                    ordered = [oo, xx, ww]
                elif op == "swiglu":
                    ordered = [oo, gg, uu]
                else:
                    ordered = [oo, aa, bb]
                pointers = (llaisysTensor_t * len(ordered))(
                    *(t.lib_tensor() for t in ordered)
                )
                scalar = {
                    "rms_norm": 1e-6,
                    "rope": 10000.0,
                    "self_attention": shape[-1] ** -0.5,
                }.get(op, 0.0)

                def native_time(rounds):
                    result = float(
                        native_bench(
                            op.encode(), pointers, len(ordered), scalar, rounds
                        )
                    )
                    if result < 0:
                        raise RuntimeError("native microbenchmark failed")
                    return result

                for _ in range(args.warmup):
                    call()
                pilot = native_time(1)
                inner = min(100, max(1, int(1.0 / max(pilot, 0.001))))
                count = min(args.repeat, max(3, int(200 / max(pilot * inner, 0.001))))
                samples = [native_time(inner) for _ in range(count)]
            else:
                inner = 1
                samples = timing(call)
            ref_samples = timing(reference)
            record = dict(
                op=op,
                version=args.version,
                dtype=args.dtype,
                shape=shape,
                seed=20261102,
                correct=correct,
                max_abs_error=float(diff.max()),
                mean_abs_error=float(diff.mean()),
                atol=tol[0],
                rtol=tol[1],
                median_ms=statistics.median(samples),
                mean_ms=statistics.mean(samples),
                samples_ms=samples,
                reference_median_ms=statistics.median(ref_samples),
                reference_samples_ms=ref_samples,
                timer="cuda_graph_device_events"
                if args.timer == "native"
                else "python_boundary_cuda_events",
                inner_repeats=inner,
                reference_timer="python_boundary_cuda_events",
                warmup=args.warmup,
                git_revision=git_rev,
                library_sha256=lib_sha,
                gpu=torch.cuda.get_device_name(0),
                torch_version=torch.__version__,
            )
            with args.output.open("a") as f:
                f.write(json.dumps(record) + "\n")
            print(
                f"{op:14} v{args.version} {args.dtype} {str(shape):28} ok={correct} native={record['median_ms']:.6f} ms torch_host_boundary={record['reference_median_ms']:.6f} ms maxerr={record['max_abs_error']:.6g}",
                flush=True,
            )
            if not correct:
                failures += 1
                print("  correctness failure retained in records", flush=True)
    if failures:
        raise SystemExit(f"{failures} correctness failures")


if __name__ == "__main__":
    main()
