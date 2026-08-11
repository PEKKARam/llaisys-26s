import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
import llaisys
import torch
from test_utils import (
    add_benchmark_arguments,
    benchmark,
    benchmark_kwargs,
    check_equal,
    random_tensor,
)


def torch_rms_norm(ans, x, w, eps):
    torch.pow(x, 2, out=ans)
    mean = torch.mean(ans, dim=-1, keepdim=True)
    mean.add_(eps)
    torch.rsqrt(mean, out=mean)
    torch.mul(x, mean, out=ans)
    ans.mul_(w)


def test_op_rms_norm(
    shape,
    dtype_name="f32",
    atol=1e-5,
    rtol=1e-5,
    device_name="cpu",
    profile=False,
):
    print(f"   shape {shape} dtype <{dtype_name}>")
    x, x_ = random_tensor(shape, dtype_name, device_name)
    w, w_ = random_tensor((shape[1], ), dtype_name, device_name)
    eps = 1e-5

    c, c_ = random_tensor(shape, dtype_name, device_name)
    torch_rms_norm(c, x, w, eps)
    llaisys.Ops.rms_norm(c_, x_, w_, eps)

    assert check_equal(c_, c, atol=atol, rtol=rtol)

    if profile:
        benchmark(
            lambda: torch_rms_norm(c, x, w, eps),
            lambda: llaisys.Ops.rms_norm(c_, x_, w_, eps),
            device_name,
            op="rms_norm",
            shape=shape,
            dtype=dtype_name,
            metadata={"eps": eps},
            **(profile if isinstance(profile, dict) else {}),
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    add_benchmark_arguments(parser)
    args = parser.parse_args()
    profile = benchmark_kwargs(args)
    testShapes = [(1, 4), (512, 4096)]
    testDtypePrec = [
        # type, atol, rtol
        ("f32", 1e-5, 1e-5),
        ("f16", 1e-3, 1e-3),
        ("bf16", 1e-2, 1e-2),
    ]
    print(f"Testing Ops.rms_norm on {args.device}")
    for shape in testShapes:
        for dtype_name, atol, rtol in testDtypePrec:
            test_op_rms_norm(shape, dtype_name, atol, rtol, args.device, profile)

    print("\033[92mTest passed!\033[0m\n")
