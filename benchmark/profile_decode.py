"""Single BF16 decode-sized Linear launch for Nsight Compute; no timing claims."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import torch
import llaisys

api = llaisys.RuntimeAPI(llaisys.DeviceType.NVIDIA)
api.set_device(0)
x = torch.ones((1, 1536), dtype=torch.bfloat16)
w = torch.ones((8960, 1536), dtype=torch.bfloat16)


def tensor(t):
    out = llaisys.Tensor(
        t.shape, dtype=llaisys.DataType.BF16, device=llaisys.DeviceType.NVIDIA
    )
    out.load(t.data_ptr())
    return out


xx, ww = tensor(x), tensor(w)
out = llaisys.Tensor(
    (1, 8960), dtype=llaisys.DataType.BF16, device=llaisys.DeviceType.NVIDIA
)
llaisys.Ops.linear(out, xx, ww, None)
api.device_synchronize()
