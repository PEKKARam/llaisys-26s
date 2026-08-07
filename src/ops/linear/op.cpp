#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(in, out, weight);

    CHECK_ARGUMENT(weight->ndim() == 2 && weight->numel() != 0, "Linear weight must be 2D nonempty tensor.");
    CHECK_ARGUMENT(in->ndim() >= 1 && in->numel() != 0, "Linear input must be nonempty tensor.");

    CHECK_ARGUMENT(in->shape().back() == weight->shape()[1], "Linear input features do not match weight.");

    CHECK_ARGUMENT(out->ndim() == in->ndim(), "Linear output must have the same rank as input.");

    CHECK_ARGUMENT(out->shape().back() == weight->shape()[0], "Linear output shape and weight shape are not match.");

    auto expected_out_shape = in->shape();
    expected_out_shape.back() = weight->shape()[0];
    CHECK_SAME_SHAPE(expected_out_shape, out->shape());

    CHECK_SAME_DTYPE(in->dtype(), out->dtype(), weight->dtype());

    if (bias != nullptr) {
        CHECK_SAME_DTYPE(in->dtype(), bias->dtype());
        CHECK_SAME_DEVICE(in, bias);
    }

    ASSERT(weight->isContiguous() && in->isContiguous() && out->isContiguous()
               && (bias == nullptr || bias->isContiguous()),
           "Linear: all tensors must be contiguous.");

    switch (in->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out, in, weight, bias);
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
