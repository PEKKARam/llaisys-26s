import argparse
import ctypes

import llaisys
import torch
from test_utils import llaisys_device, torch_device


def test_sync_memcpy(api, size_bytes):
    source = torch.arange(size_bytes, dtype=torch.uint8, device=torch_device("cpu"))
    result = torch.zeros_like(source)
    device_source = None
    device_result = None
    try:
        device_source = api.malloc_device(size_bytes)
        device_result = api.malloc_device(size_bytes)
        if not device_source or not device_result:
            raise RuntimeError("Device allocation returned a null pointer")

        api.memcpy_sync(
            device_source,
            source.data_ptr(),
            size_bytes,
            llaisys.MemcpyKind.H2D,
        )
        api.memcpy_sync(
            device_result,
            device_source,
            size_bytes,
            llaisys.MemcpyKind.D2D,
        )
        api.memcpy_sync(
            result.data_ptr(),
            device_result,
            size_bytes,
            llaisys.MemcpyKind.D2H,
        )
        api.device_synchronize()
        torch.testing.assert_close(source, result, rtol=0, atol=0)
    finally:
        if device_result:
            api.free_device(device_result)
        if device_source:
            api.free_device(device_source)


def test_async_memcpy(api, size_bytes, stream_count=2):
    streams = []
    allocations = []
    try:
        for stream_index in range(stream_count):
            stream = api.create_stream()
            streams.append(stream)
            host_source = api.malloc_host(size_bytes)
            host_result = api.malloc_host(size_bytes)
            device_source = api.malloc_device(size_bytes)
            device_result = api.malloc_device(size_bytes)
            allocations.append(
                (host_source, host_result, device_source, device_result)
            )
            if not all((host_source, host_result, device_source, device_result)):
                raise RuntimeError("Runtime allocation returned a null pointer")

            fill = (stream_index + 1) * 37
            ctypes.memset(host_source, fill, size_bytes)
            ctypes.memset(host_result, 0, size_bytes)
            api.memcpy_async(
                device_source,
                host_source,
                size_bytes,
                llaisys.MemcpyKind.H2D,
                stream,
            )
            api.memcpy_async(
                device_result,
                device_source,
                size_bytes,
                llaisys.MemcpyKind.D2D,
                stream,
            )
            api.memcpy_async(
                host_result,
                device_result,
                size_bytes,
                llaisys.MemcpyKind.D2H,
                stream,
            )

        for stream, (host_source, host_result, _, _) in zip(streams, allocations):
            api.stream_synchronize(stream)
            assert ctypes.string_at(host_source, size_bytes) == ctypes.string_at(
                host_result, size_bytes
            )
        api.device_synchronize()
    finally:
        for host_source, host_result, device_source, device_result in reversed(
            allocations
        ):
            if device_result:
                api.free_device(device_result)
            if device_source:
                api.free_device(device_source)
            if host_result:
                api.free_host(host_result)
            if host_source:
                api.free_host(host_source)
        for stream in reversed(streams):
            api.destroy_stream(stream)


def test_runtime_api(device_name="cpu", size_bytes=1024 * 1024, extended=False):
    api = llaisys.RuntimeAPI(llaisys_device(device_name))
    device_count = api.get_device_count()
    print(f"Found {device_count} {device_name} device(s)")

    for device_id in range(device_count):
        print(f"Testing {device_name}:{device_id}...")
        api.set_device(device_id)
        test_sync_memcpy(api, size_bytes)
        if extended:
            test_async_memcpy(api, size_bytes)
        print("     Passed")
    return device_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=("cpu", "nvidia"))
    parser.add_argument("--size-mib", default=1, type=int)
    parser.add_argument(
        "--extended",
        action="store_true",
        help="Test pinned host memory, async copies, and multiple streams",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Exit successfully when the requested backend has no devices",
    )
    args = parser.parse_args()
    if args.size_mib <= 0:
        raise SystemExit("--size-mib must be positive")

    tested_devices = test_runtime_api(
        args.device,
        size_bytes=args.size_mib * 1024 * 1024,
        extended=args.extended,
    )
    if tested_devices == 0:
        message = f"No LLAISYS {args.device} device is available"
        if args.allow_missing:
            print(f"{message}; skipped by request")
        else:
            raise SystemExit(message)
    else:
        print("\033[92mTest passed!\033[0m\n")
