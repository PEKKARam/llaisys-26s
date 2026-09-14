#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include <cstring>

namespace llaisys::ops {
void rearrange(tensor_t out, tensor_t in) {
    CHECK_SAME_DEVICE(out, in);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    CHECK_ARGUMENT(out->isContiguous(), "rearrange output must be contiguous");
    if (in->isContiguous()) {
        const size_t bytes = in->numel() * in->elementSize();
        if (in->deviceType() == LLAISYS_DEVICE_CPU) std::memcpy(out->data(), in->data(), bytes);
        else {
            llaisys::core::context().setDevice(in->deviceType(), in->deviceId());
            llaisys::core::context().runtime().api()->memcpy_sync(out->data(), in->data(), bytes, LLAISYS_MEMCPY_D2D);
        }
        return;
    }
    auto tmp = in->contiguous();
    const size_t bytes = tmp->numel() * tmp->elementSize();
    if (in->deviceType() == LLAISYS_DEVICE_CPU) std::memcpy(out->data(), tmp->data(), bytes);
    else {
        llaisys::core::context().setDevice(in->deviceType(), in->deviceId());
        llaisys::core::context().runtime().api()->memcpy_sync(out->data(), tmp->data(), bytes, LLAISYS_MEMCPY_D2D);
    }
}
} // namespace llaisys::ops
