#include "../../nvidia/common.cuh"
#include "add_nvidia.cuh"

#include <cmath>

namespace llaisys::ops::nvidia {
namespace {
using detail::load;
using detail::store;

template <int MODE>
__global__ void add_k(void *c, const void *a, const void *b, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        store<MODE>(c, i, load<MODE>(a, i) + load<MODE>(b, i));
    }
}

} // namespace

void add_v0(tensor_t c, tensor_t a, tensor_t b) {
    if (c->dtype() == LLAISYS_DTYPE_F32) {
        LLAISYS_CUDA_LAUNCH((add_k<0>), c->numel(), c->data(), a->data(),
                            b->data(), c->numel());
    } else if (c->dtype() == LLAISYS_DTYPE_BF16) {
        LLAISYS_CUDA_LAUNCH((add_k<1>), c->numel(), c->data(), a->data(),
                            b->data(), c->numel());
    } else {
        LLAISYS_CUDA_LAUNCH((add_k<2>), c->numel(), c->data(), a->data(),
                            b->data(), c->numel());
    }
}

namespace {
template <int MODE>
__global__ void add_pair_v2(void *out, const void *a, const void *b,
                            size_t count) {
    size_t i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (i + 1 < count) {
        float2 x = detail::loadPair<MODE>(a, i),
               y = detail::loadPair<MODE>(b, i);
        detail::storePair<MODE>(out, i, make_float2(x.x + y.x, x.y + y.y));
    } else if (i < count) {
        store<MODE>(out, i, load<MODE>(a, i) + load<MODE>(b, i));
    }
}
template <int MODE> void add_run(tensor_t o, tensor_t a, tensor_t b) {
    LLAISYS_CUDA_LAUNCH((add_pair_v2<MODE>), (o->numel() + 1) / 2, o->data(),
                        a->data(), b->data(), o->numel());
}
} // namespace
void add(tensor_t o, tensor_t a, tensor_t b) {
    static const int version = detail::selectedVersion("ADD");
    const size_t alignment = o->dtype() == LLAISYS_DTYPE_F32 ? 8 : 4;
    if (version < 2 || reinterpret_cast<uintptr_t>(o->data()) % alignment
        || reinterpret_cast<uintptr_t>(a->data()) % alignment
        || reinterpret_cast<uintptr_t>(b->data()) % alignment) {
        return add_v0(o, a, b);
    }
    if (o->dtype() == LLAISYS_DTYPE_F32) {
        return add_run<0>(o, a, b);
    }
    if (o->dtype() == LLAISYS_DTYPE_BF16) {
        return add_run<1>(o, a, b);
    }
    return add_run<2>(o, a, b);
}
} // namespace llaisys::ops::nvidia
