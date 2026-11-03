"""Regression cases targeted at vector tails, alignment, GQA masking and reductions."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "benchmark"))
from compare_prompt import configure_reference_math
import torch
import llaisys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    configure_reference_math(torch)
    torch.manual_seed(20261102)
    api = llaisys.RuntimeAPI(llaisys.DeviceType.NVIDIA)
    api.set_device(0)
    checks = []
    for dt, ld, tol in [
        (torch.float32, llaisys.DataType.F32, 1e-5),
        (torch.float16, llaisys.DataType.F16, 0.003),
        (torch.bfloat16, llaisys.DataType.BF16, 0.02),
    ]:
        handles = []

        def wrap(x, offset=False):
            dtype = llaisys.DataType.I64 if x.dtype == torch.int64 else ld
            if offset:
                owner = llaisys.Tensor(
                    (x.numel() + 1,), dtype=dtype, device=llaisys.DeviceType.NVIDIA
                )
                result = owner.slice(0, 1, x.numel() + 1).view(*x.shape)
                handles.append(owner)
            else:
                result = llaisys.Tensor(
                    x.shape, dtype=dtype, device=llaisys.DeviceType.NVIDIA
                )
            api.memcpy_sync(
                result.data_ptr(),
                x.data_ptr(),
                x.numel() * x.element_size(),
                llaisys.MemcpyKind.D2D,
            )
            handles.append(result)
            return result

        def rand(shape):
            return torch.randn(shape, dtype=dt, device="cuda") * 0.2

        def read(t, expected):
            actual = torch.empty_like(expected)
            api.memcpy_sync(
                actual.data_ptr(),
                t.data_ptr(),
                actual.numel() * actual.element_size(),
                llaisys.MemcpyKind.D2D,
            )
            return actual

        def check(name, t, expected, exact=False):
            actual = read(t, expected)
            torch.testing.assert_close(
                actual,
                expected,
                atol=0 if exact else tol,
                rtol=0 if exact else tol,
                equal_nan=True,
            )
            checks.append(f"{dt}:{name}")
            print("PASS", checks[-1], flush=True)

        a, b = rand((3, 157)), rand((3, 157))
        aa, bb = wrap(a, True), wrap(b, True)
        oo = wrap(torch.empty_like(a), True)
        llaisys.Ops.add(oo, aa, bb)
        check("unaligned odd add", oo, a + b, True)
        llaisys.Ops.swiglu(oo, aa, bb)
        check("unaligned odd swiglu", oo, torch.nn.functional.silu(a) * b)
        g = torch.tensor([[-100.0, -20.0, 0.0, 20.0, 100.0]], dtype=dt, device="cuda")
        u = torch.ones_like(g)
        go, uo = wrap(g), wrap(u)
        out = wrap(u)
        llaisys.Ops.swiglu(out, go, uo)
        check("swiglu saturation", out, torch.nn.functional.silu(g) * u)
        w = rand((19, 155))
        ids = torch.tensor([18, 0, 5], device="cuda")
        wo, io = wrap(w, True), wrap(ids)
        out = wrap(torch.empty((3, 155), dtype=dt, device="cuda"), True)
        llaisys.Ops.embedding(out, io, wo)
        check("embedding unaligned row", out, w[ids], True)
        for bias in (None, rand((7,))):
            x, w = rand((3, 13)), rand((7, 13))
            xo, wo = wrap(x), wrap(w)
            bo = wrap(bias) if bias is not None else None
            expected = torch.nn.functional.linear(x, w, bias)
            out = wrap(torch.empty_like(expected))
            llaisys.Ops.linear(out, xo, wo, bo)
            check("linear odd dims " + str(bias is not None), out, expected)
        for kind in ("zero", "constant"):
            x = (
                torch.zeros((3, 157), dtype=dt, device="cuda")
                if kind == "zero"
                else torch.full((3, 157), 0.5, dtype=dt, device="cuda")
            )
            w = rand((157,))
            xo, wo = wrap(x), wrap(w)
            out = wrap(x)
            llaisys.Ops.rms_norm(out, xo, wo, 1e-6)
            expected = (
                x.float()
                * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
            ).to(dt) * w
            check("norm " + kind, out, expected)
        x = rand((3, 4, 10))
        pos = torch.zeros(3, dtype=torch.int64, device="cuda")
        xo, po = wrap(x), wrap(pos)
        out = wrap(x)
        llaisys.Ops.rope(out, xo, po, 10000.0)
        check("RoPE zero position", out, x, True)
        q, k, v = rand((3, 4, 8)), rand((7, 2, 8)), rand((7, 2, 3))
        qo, ko, vo = wrap(q), wrap(k), wrap(v)
        out = wrap(torch.empty((3, 4, 3), dtype=dt, device="cuda"))

        def attention(q, k, v):
            qt = q.transpose(0, 1)
            kt = k.transpose(0, 1).repeat_interleave(2, 0)
            vt = v.transpose(0, 1).repeat_interleave(2, 0)
            scores = (qt @ kt.transpose(1, 2)) * (8**-0.5)
            mask = (
                torch.arange(7, device="cuda")[None, :]
                > (4 + torch.arange(3, device="cuda"))[:, None]
            )
            probs = torch.softmax(
                scores.masked_fill(mask, -float("inf")).float(), -1
            ).to(dt)
            return (probs @ vt).transpose(0, 1).contiguous()

        llaisys.Ops.self_attention(out, qo, ko, vo, 8**-0.5)
        check("GQA offset causal Dv != D", out, attention(q, k, v))
        old = read(out, torch.empty((3, 4, 3), dtype=dt, device="cuda")).clone()
        k[-1] = 10
        v[-1] = 10
        ko, vo = wrap(k), wrap(v)
        llaisys.Ops.self_attention(out, qo, ko, vo, 8**-0.5)
        actual = read(out, old)
        torch.testing.assert_close(actual[:2], old[:2], atol=0, rtol=0)
        checks.append(f"{dt}:future key/value cannot affect earlier queries")
        print("PASS", checks[-1], flush=True)
        for kind in ("negative_inf", "ties", "nan", "wide_tail"):
            length = 151937 if kind == "wide_tail" else 157
            x = (
                torch.full((2, length), -float("inf"), dtype=dt, device="cuda")
                if kind == "negative_inf"
                else rand((2, length))
            )
            if kind == "ties":
                x[:, 31] = 3
                x[:, 129] = 3
            if kind == "nan":
                x[:, 2] = float("nan")
                x[:, 100] = float("nan")
            if kind == "wide_tail":
                x[:, -1] = 5
            xo = wrap(x)
            values = wrap(torch.empty((2, 1), dtype=dt, device="cuda"))
            idx = wrap(torch.empty((2, 1), dtype=torch.int64, device="cuda"))
            llaisys.Ops.argmax(idx, values, xo)
            check("argmax value " + kind, values, x.max(-1, keepdim=True).values, True)
            check("argmax index " + kind, idx, x.argmax(-1, keepdim=True), True)
    result = {"passed": len(checks), "tests": checks, "seed": 20261102}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{len(checks)} targeted CUDA checks passed")


if __name__ == "__main__":
    main()
