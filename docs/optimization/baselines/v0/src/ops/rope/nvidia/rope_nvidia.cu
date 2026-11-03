#include "../../nvidia/common.cuh"
#include "rope_nvidia.cuh"

#include <cmath>

namespace llaisys::ops::nvidia {
namespace {
using detail::load;
using detail::store;

// Input/output: [S, H, D]. Pair the first and second halves of each head.
// pos[S] contains absolute positions, including the existing cache length.
template <int MODE>
__global__ void rope_k(void *o, const void *x, const int64_t *pos, size_t S, size_t H, size_t D,
                       float theta) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < S * H * D) {
        size_t d = i % D, j = d < (D / 2) ? d : d - D / 2;
        float a = load<MODE>(x, i);
        size_t base = i - d;
        size_t pair = base + (d < (D / 2) ? d + (D / 2) : d - (D / 2));
        float b = load<MODE>(x, pair);
        float ang = (float)pos[i / (H * D)] / powf(theta, 2.f * j / D);
        float v = d < (D / 2) ? a * cosf(ang) - b * sinf(ang) : a * cosf(ang) + b * sinf(ang);
        store<MODE>(o, i, v);
    }
}

} // namespace

void rope(tensor_t o, tensor_t x, tensor_t p, float th) {
    size_t S = x->shape()[0], H = x->shape()[1], D = x->shape()[2];
    if (x->dtype() == LLAISYS_DTYPE_F32) {
        LLAISYS_CUDA_LAUNCH((rope_k<0>), o->numel(), o->data(), x->data(),
                            reinterpret_cast<const int64_t *>(p->data()), S, H, D, th);
    } else if (x->dtype() == LLAISYS_DTYPE_BF16) {
        LLAISYS_CUDA_LAUNCH((rope_k<1>), o->numel(), o->data(), x->data(),
                            reinterpret_cast<const int64_t *>(p->data()), S, H, D, th);
    } else {
        LLAISYS_CUDA_LAUNCH((rope_k<2>), o->numel(), o->data(), x->data(),
                            reinterpret_cast<const int64_t *>(p->data()), S, H, D, th);
    }
}

} // namespace llaisys::ops::nvidia
