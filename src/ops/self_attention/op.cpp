#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/self_atten_cpu.hpp"

#include <cmath>

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());

    CHECK_ARGUMENT(attn_val->ndim() == 3 && q->ndim() == 3 && k->ndim() == 3 && v->ndim() == 3,
                   "Self attention: all tensors must be 3-D.");
    CHECK_ARGUMENT(q->numel() != 0 && k->numel() != 0 && v->numel() != 0,
                   "Self attention: input tensors must be nonempty.");

    const size_t query_len = q->shape()[0];
    const size_t kv_len = k->shape()[0];
    const size_t query_heads = q->shape()[1];
    const size_t kv_heads = k->shape()[1];

    CHECK_ARGUMENT(kv_len >= query_len, "Self attention: KV length must not be shorter than query length.");
    CHECK_ARGUMENT(v->shape()[0] == kv_len, "Self attention: key and value lengths must match.");
    CHECK_ARGUMENT(v->shape()[1] == kv_heads, "Self attention: key and value head counts must match.");
    CHECK_ARGUMENT(q->shape()[2] == k->shape()[2], "Self attention: query and key head dimensions must match.");
    CHECK_ARGUMENT(query_heads % kv_heads == 0,
                   "Self attention: query head count must be divisible by KV head count.");
    CHECK_ARGUMENT(attn_val->shape()[0] == query_len && attn_val->shape()[1] == query_heads
                       && attn_val->shape()[2] == v->shape()[2],
                   "Self attention: output shape does not match query and value shapes.");
    CHECK_ARGUMENT(std::isfinite(scale), "Self attention: scale must be finite.");

    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(),
           "Self attention: all tensors must be contiguous.");

    switch (q->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_atten(attn_val, q, k, v, scale);
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
