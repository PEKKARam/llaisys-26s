#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rope_cpu.hpp"

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(in, out, pos_ids);
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    CHECK_SAME_DTYPE(pos_ids->dtype(), LLAISYS_DTYPE_I64);

    CHECK_ARGUMENT(in->ndim() == 3, "RoPE: input must be a 3-D tenor.");
    CHECK_ARGUMENT(pos_ids->ndim() == 1, "RoPE: position id must be a 1-D tensor.");
    CHECK_ARGUMENT(pos_ids->numel() == in->shape()[0], "RoPE: position id doesnot match input.");
    CHECK_ARGUMENT(theta > 0, "RoPE: theta should be positive.");
    CHECK_ARGUMENT(in->shape()[2] % 2 == 0, "RoPE: input head dim must be even.");

    ASSERT(pos_ids->isContiguous() && in->isContiguous() && out->isContiguous(),
           "RoPE: all tensors must be contiguous.");

    switch (in->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(out, in, pos_ids, theta);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
