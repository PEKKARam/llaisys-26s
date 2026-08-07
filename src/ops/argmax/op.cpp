#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/argmax_cpu.hpp"

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    CHECK_SAME_DEVICE(vals, max_idx, max_val);
    ASSERT(vals->isContiguous() && max_idx->isContiguous() && max_val->isContiguous(),
           "Argmax: all tensors must be contiguous.");

    CHECK_ARGUMENT(vals->numel() != 0 && vals->ndim() != 0, "Argmax input must be a nonempty tensor.");
    CHECK_SAME_DTYPE(max_idx->dtype(), LLAISYS_DTYPE_I64);
    CHECK_SAME_DTYPE(max_val->dtype(), vals->dtype());

    auto expected_shape = vals->shape();
    expected_shape.back() = 1;
    CHECK_SAME_SHAPE(expected_shape, max_idx->shape(), max_val->shape());

    switch (vals->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::argmax(max_idx, max_val, vals);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        llaisys::core::context().setDevice(vals->deviceType(), vals->deviceId());
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
