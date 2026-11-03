#include "../../nvidia/common.cuh"
#include "add_nvidia.cuh"

#include <cmath>

namespace llaisys::ops::nvidia {
namespace {
using detail::load;
using detail::store;

template <int MODE> __global__ void add_k(void *c, const void *a, const void *b, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        store<MODE>(c, i, load<MODE>(a, i) + load<MODE>(b, i));
    }
}

} // namespace

void add(tensor_t c, tensor_t a, tensor_t b) {
    if (c->dtype() == LLAISYS_DTYPE_F32) {
        LLAISYS_CUDA_LAUNCH((add_k<0>), c->numel(), c->data(), a->data(), b->data(), c->numel());
    } else if (c->dtype() == LLAISYS_DTYPE_BF16) {
        LLAISYS_CUDA_LAUNCH((add_k<1>), c->numel(), c->data(), a->data(), b->data(), c->numel());
    } else {
        LLAISYS_CUDA_LAUNCH((add_k<2>), c->numel(), c->data(), a->data(), b->data(), c->numel());
    }
}

} // namespace llaisys::ops::nvidia
