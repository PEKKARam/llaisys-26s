#include "../../nvidia/common.cuh"
#include "swiglu_nvidia.cuh"

#include <cmath>

namespace llaisys::ops::nvidia {
namespace {
using detail::load;
using detail::store;

template <int MODE> __global__ void swiglu_k(void *o, const void *g, const void *u, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float gv = load<MODE>(g, i), uv = load<MODE>(u, i);
        store<MODE>(o, i, uv * gv / (1.f + expf(-gv)));
    }
}

} // namespace

void swiglu(tensor_t o, tensor_t g, tensor_t u) {
    if (o->dtype() == LLAISYS_DTYPE_F32) {
        LLAISYS_CUDA_LAUNCH((swiglu_k<0>), o->numel(), o->data(), g->data(), u->data(), o->numel());
    } else if (o->dtype() == LLAISYS_DTYPE_BF16) {
        LLAISYS_CUDA_LAUNCH((swiglu_k<1>), o->numel(), o->data(), g->data(), u->data(), o->numel());
    } else {
        LLAISYS_CUDA_LAUNCH((swiglu_k<2>), o->numel(), o->data(), g->data(), u->data(), o->numel());
    }
}

} // namespace llaisys::ops::nvidia
