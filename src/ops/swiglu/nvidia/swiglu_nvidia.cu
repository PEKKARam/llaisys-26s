#include "../../nvidia/common.cuh"
#include "swiglu_nvidia.cuh"

#include <cmath>

namespace llaisys::ops::nvidia {
namespace {
using detail::load;
using detail::store;

template <int MODE>
__global__ void swiglu_k(void *o, const void *g, const void *u, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float gv = load<MODE>(g, i), uv = load<MODE>(u, i);
        store<MODE>(o, i, uv * gv / (1.f + expf(-gv)));
    }
}

} // namespace

void swiglu_v0(tensor_t o, tensor_t g, tensor_t u) {
    if (o->dtype() == LLAISYS_DTYPE_F32) {
        LLAISYS_CUDA_LAUNCH((swiglu_k<0>), o->numel(), o->data(), g->data(),
                            u->data(), o->numel());
    } else if (o->dtype() == LLAISYS_DTYPE_BF16) {
        LLAISYS_CUDA_LAUNCH((swiglu_k<1>), o->numel(), o->data(), g->data(),
                            u->data(), o->numel());
    } else {
        LLAISYS_CUDA_LAUNCH((swiglu_k<2>), o->numel(), o->data(), g->data(),
                            u->data(), o->numel());
    }
}

namespace {
template <int MODE>
__global__ void swiglu_serial_v1(void *out, const void *gate, const void *up,
                                 size_t count) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        float g = load<MODE>(gate, i);
        float silu = detail::rounded<MODE>(g / (1.f + expf(-g)));
        store<MODE>(out, i, silu * load<MODE>(up, i));
    }
}

template <int MODE>
__global__ void swiglu_pair_v2(void *out, const void *gate, const void *up,
                               size_t count) {
    size_t i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (i + 1 < count) {
        float2 g = detail::loadPair<MODE>(gate, i),
               u = detail::loadPair<MODE>(up, i);
        float2 result = make_float2(
            detail::rounded<MODE>(g.x / (1.f + expf(-g.x))) * u.x,
            detail::rounded<MODE>(g.y / (1.f + expf(-g.y))) * u.y);
        detail::storePair<MODE>(out, i, result);
    } else if (i < count) {
        float g = load<MODE>(gate, i);
        store<MODE>(out, i,
                    detail::rounded<MODE>(g / (1.f + expf(-g)))
                        * load<MODE>(up, i));
    }
}
template <int MODE>
void swiglu_run(tensor_t o, tensor_t g, tensor_t u, int version) {
    size_t alignment = MODE == 0 ? 8 : 4;
    if (version == 2 && reinterpret_cast<uintptr_t>(o->data()) % alignment == 0
        && reinterpret_cast<uintptr_t>(g->data()) % alignment == 0
        && reinterpret_cast<uintptr_t>(u->data()) % alignment == 0) {
        LLAISYS_CUDA_LAUNCH((swiglu_pair_v2<MODE>), (o->numel() + 1) / 2,
                            o->data(), g->data(), u->data(), o->numel());
        return;
    }
    LLAISYS_CUDA_LAUNCH((swiglu_serial_v1<MODE>), o->numel(), o->data(),
                        g->data(), u->data(), o->numel());
}
} // namespace
void swiglu(tensor_t o, tensor_t g, tensor_t u) {
    static const int version = detail::selectedVersion("SWIGLU");
    if (version == 0) {
        return swiglu_v0(o, g, u);
    }
    if (o->dtype() == LLAISYS_DTYPE_F32) {
        return swiglu_run<0>(o, g, u, version);
    }
    if (o->dtype() == LLAISYS_DTYPE_BF16) {
        return swiglu_run<1>(o, g, u, version);
    }
    return swiglu_run<2>(o, g, u, version);
}
} // namespace llaisys::ops::nvidia
