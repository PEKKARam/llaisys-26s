#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/embedding_cpu.hpp"

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(weight, index, out);

    auto expected_shape = index->shape();
    expected_shape.push_back(weight->shape()[1]);
    CHECK_SAME_SHAPE(expected_shape, out->shape());

    CHECK_ARGUMENT(weight->ndim() == 2 && weight->numel() != 0, "Embedding weight must be a 2D nonempty tensor.");
    CHECK_SAME_DTYPE(index->dtype(), LLAISYS_DTYPE_I64);
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());

    ASSERT(weight->isContiguous() && index->isContiguous() && out->isContiguous(),
           "Embedding: all tensors must be contiguous.");

    switch (weight->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(out, index, weight);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        llaisys::core::context().setDevice(weight->deviceType(), weight->deviceId());
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
