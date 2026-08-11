import json
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import llaisys
import torch


def random_tensor(
    shape, dtype_name, device_name, device_id=0, scale=None, bias=None
) -> tuple[torch.Tensor, llaisys.Tensor]:
    torch_tensor = torch.rand(
        shape,
        dtype=torch_dtype(dtype_name),
        device=torch_device(device_name, device_id),
    )
    if scale is not None:
        torch_tensor *= scale
    if bias is not None:
        torch_tensor += bias

    llaisys_tensor = llaisys.Tensor(
        shape,
        dtype=llaisys_dtype(dtype_name),
        device=llaisys_device(device_name),
        device_id=device_id,
    )

    api = llaisys.RuntimeAPI(llaisys_device(device_name))
    bytes_ = torch_tensor.numel() * torch_tensor.element_size()
    api.memcpy_sync(
        llaisys_tensor.data_ptr(),
        torch_tensor.data_ptr(),
        bytes_,
        llaisys.MemcpyKind.D2D,
    )

    return torch_tensor, llaisys_tensor


def random_int_tensor(shape, device_name, dtype_name="i64", device_id=0, low=0, high=2):
    torch_tensor = torch.randint(
        low,
        high,
        shape,
        dtype=torch_dtype(dtype_name),
        device=torch_device(device_name, device_id),
    )

    llaisys_tensor = llaisys.Tensor(
        shape,
        dtype=llaisys_dtype(dtype_name),
        device=llaisys_device(device_name),
        device_id=device_id,
    )

    api = llaisys.RuntimeAPI(llaisys_device(device_name))
    bytes_ = torch_tensor.numel() * torch_tensor.element_size()
    api.memcpy_sync(
        llaisys_tensor.data_ptr(),
        torch_tensor.data_ptr(),
        bytes_,
        llaisys.MemcpyKind.D2D,
    )

    return torch_tensor, llaisys_tensor


def zero_tensor(
    shape, dtype_name, device_name, device_id=0
) -> tuple[torch.Tensor, llaisys.Tensor]:
    torch_tensor = torch.zeros(
        shape,
        dtype=torch_dtype(dtype_name),
        device=torch_device(device_name, device_id),
    )

    llaisys_tensor = llaisys.Tensor(
        shape,
        dtype=llaisys_dtype(dtype_name),
        device=llaisys_device(device_name),
        device_id=device_id,
    )

    api = llaisys.RuntimeAPI(llaisys_device(device_name))
    bytes_ = torch_tensor.numel() * torch_tensor.element_size()
    api.memcpy_sync(
        llaisys_tensor.data_ptr(),
        torch_tensor.data_ptr(),
        bytes_,
        llaisys.MemcpyKind.D2D,
    )

    return torch_tensor, llaisys_tensor


def arrange_tensor(
    start, end, device_name, device_id=0
) -> tuple[torch.Tensor, llaisys.Tensor]:
    torch_tensor = torch.arange(start, end, device=torch_device(device_name, device_id))
    llaisys_tensor = llaisys.Tensor(
        (end - start,),
        dtype=llaisys_dtype("i64"),
        device=llaisys_device(device_name),
        device_id=device_id,
    )

    api = llaisys.RuntimeAPI(llaisys_device(device_name))
    bytes_ = torch_tensor.numel() * torch_tensor.element_size()
    api.memcpy_sync(
        llaisys_tensor.data_ptr(),
        torch_tensor.data_ptr(),
        bytes_,
        llaisys.MemcpyKind.D2D,
    )

    return torch_tensor, llaisys_tensor


def check_equal(
    llaisys_result: llaisys.Tensor,
    torch_answer: torch.Tensor,
    atol=1e-5,
    rtol=1e-5,
    strict=False,
):
    shape = llaisys_result.shape()
    strides = llaisys_result.strides()
    assert shape == torch_answer.shape
    assert torch_dtype(dtype_name(llaisys_result.dtype())) == torch_answer.dtype

    right = 0
    for i in range(len(shape)):
        if strides[i] > 0:
            right += strides[i] * (shape[i] - 1)
        else:  # TODO: Support negative strides in the future
            raise ValueError("Negative strides are not supported yet")

    tmp = torch.zeros(
        (right + 1,),
        dtype=torch_answer.dtype,
        device=torch_device(
            device_name(llaisys_result.device_type()), llaisys_result.device_id()
        ),
    )
    result = torch.as_strided(tmp, shape, strides)
    api = llaisys.RuntimeAPI(llaisys_result.device_type())
    api.memcpy_sync(
        result.data_ptr(),
        llaisys_result.data_ptr(),
        (right + 1) * tmp.element_size(),
        llaisys.MemcpyKind.D2D,
    )

    if strict:
        if torch.equal(result, torch_answer):
            return True
    else:
        if torch.allclose(result, torch_answer, atol=atol, rtol=rtol):
            return True

    print(f"LLAISYS result: \n{result}")
    print(f"Torch answer: \n{torch_answer}")
    return False


def _percentile(values, percentile):
    if not values:
        raise ValueError("Cannot calculate a percentile from no samples")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile / 100
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _git_revision():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _json_value(value):
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    return value


def _device_description(device_name, device_id):
    if device_name == "nvidia" and torch.cuda.is_available():
        return torch.cuda.get_device_name(device_id)
    return platform.processor() or platform.machine()


def _summarize_samples(samples_ms):
    return {
        "mean_ms": statistics.fmean(samples_ms),
        "median_ms": statistics.median(samples_ms),
        "p90_ms": _percentile(samples_ms, 90),
        "p99_ms": _percentile(samples_ms, 99),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
    }


def _append_jsonl(path, records):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, sort_keys=True) + "\n")


def benchmark(
    torch_func,
    llaisys_func,
    device_name,
    warmup=10,
    repeat=100,
    *,
    op="unknown",
    shape=None,
    dtype=None,
    device_id=0,
    json_path=None,
    metadata=None,
):
    """Benchmark pre-created operator calls and return structured records.

    CUDA calls are measured with events. CPU calls use perf_counter. Input and
    output allocation must happen before calling this function.
    """
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if repeat <= 0:
        raise ValueError("repeat must be positive")

    api = llaisys.RuntimeAPI(llaisys_device(device_name))
    api.set_device(device_id)
    use_cuda_events = device_name == "nvidia"
    if use_cuda_events and not torch.cuda.is_available():
        raise RuntimeError("CUDA event timing requested, but PyTorch cannot access CUDA")

    def time_op(func):
        for _ in range(warmup):
            func()
        api.device_synchronize()

        samples_ms = []
        for _ in range(repeat):
            if use_cuda_events:
                start = torch.cuda.Event(enable_timing=True)
                stop = torch.cuda.Event(enable_timing=True)
                start.record()
                func()
                stop.record()
                api.device_synchronize()
                samples_ms.append(start.elapsed_time(stop))
            else:
                start = time.perf_counter()
                func()
                end = time.perf_counter()
                samples_ms.append((end - start) * 1000)
        return samples_ms

    common = {
        "schema_version": 1,
        "benchmark": "operator",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_revision": _git_revision(),
        "op": op,
        "shape": _json_value(shape),
        "dtype": dtype,
        "device": device_name,
        "device_id": device_id,
        "device_name": _device_description(device_name, device_id),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "warmup": warmup,
        "repeat": repeat,
        "timer": "cuda_event" if use_cuda_events else "perf_counter",
    }
    if metadata:
        common.update(_json_value(metadata))

    records = {}
    for implementation, func in (("torch", torch_func), ("llaisys", llaisys_func)):
        samples_ms = time_op(func)
        record = {**common, "implementation": implementation}
        record.update(_summarize_samples(samples_ms))
        records[implementation] = record
        print(
            f"        {implementation.upper():7s} "
            f"mean={record['mean_ms']:.5f} ms "
            f"median={record['median_ms']:.5f} ms "
            f"p90={record['p90_ms']:.5f} ms "
            f"p99={record['p99_ms']:.5f} ms"
        )

    if json_path:
        _append_jsonl(json_path, records.values())
    return records


def add_benchmark_arguments(parser):
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--warmup", default=10, type=int)
    parser.add_argument("--repeat", default=100, type=int)
    parser.add_argument(
        "--json",
        dest="json_path",
        default=None,
        help="Append benchmark records to this JSONL file",
    )


def benchmark_kwargs(args):
    if not args.profile:
        return False
    return {
        "warmup": args.warmup,
        "repeat": args.repeat,
        "json_path": args.json_path,
    }


def torch_device(device_name: str, device_id=0):
    if device_name == "cpu":
        return torch.device("cpu")
    elif device_name == "nvidia":
        return torch.device(f"cuda:{device_id}")
    else:
        raise ValueError(f"Unsupported device name: {device_name}")


def llaisys_device(device_name: str):
    if device_name == "cpu":
        return llaisys.DeviceType.CPU
    elif device_name == "nvidia":
        return llaisys.DeviceType.NVIDIA
    else:
        raise ValueError(f"Unsupported device name: {device_name}")


def device_name(llaisys_device: llaisys.DeviceType):
    if llaisys_device == llaisys.DeviceType.CPU:
        return "cpu"
    elif llaisys_device == llaisys.DeviceType.NVIDIA:
        return "nvidia"
    else:
        raise ValueError(f"Unsupported llaisys device: {llaisys_device}")


def torch_dtype(dtype_name: str):
    if dtype_name == "f16":
        return torch.float16
    elif dtype_name == "f32":
        return torch.float32
    elif dtype_name == "f64":
        return torch.float64
    elif dtype_name == "bf16":
        return torch.bfloat16
    elif dtype_name == "i32":
        return torch.int32
    elif dtype_name == "i64":
        return torch.int64
    elif dtype_name == "u32":
        return torch.uint32
    elif dtype_name == "u64":
        return torch.uint64
    elif dtype_name == "bool":
        return torch.bool
    else:
        raise ValueError(f"Unsupported dtype name: {dtype_name}")


def llaisys_dtype(dtype_name: str):
    if dtype_name == "f16":
        return llaisys.DataType.F16
    elif dtype_name == "f32":
        return llaisys.DataType.F32
    elif dtype_name == "f64":
        return llaisys.DataType.F64
    elif dtype_name == "bf16":
        return llaisys.DataType.BF16
    elif dtype_name == "i32":
        return llaisys.DataType.I32
    elif dtype_name == "i64":
        return llaisys.DataType.I64
    elif dtype_name == "u32":
        return llaisys.DataType.U32
    elif dtype_name == "u64":
        return llaisys.DataType.U64
    elif dtype_name == "bool":
        return llaisys.DataType.BOOL
    else:
        raise ValueError(f"Unsupported dtype name: {dtype_name}")


def dtype_name(llaisys_dtype: llaisys.DataType):
    if llaisys_dtype == llaisys.DataType.F16:
        return "f16"
    elif llaisys_dtype == llaisys.DataType.F32:
        return "f32"
    elif llaisys_dtype == llaisys.DataType.F64:
        return "f64"
    elif llaisys_dtype == llaisys.DataType.BF16:
        return "bf16"
    elif llaisys_dtype == llaisys.DataType.I32:
        return "i32"
    elif llaisys_dtype == llaisys.DataType.I64:
        return "i64"
    elif llaisys_dtype == llaisys.DataType.U32:
        return "u32"
    elif llaisys_dtype == llaisys.DataType.U64:
        return "u64"
    elif llaisys_dtype == llaisys.DataType.BOOL:
        return "bool"
    else:
        raise ValueError(f"Unsupported llaisys dtype: {llaisys_dtype}")
